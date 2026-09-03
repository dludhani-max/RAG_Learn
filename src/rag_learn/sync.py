"""Incremental document sync: detects new/modified/removed/renamed files
under DATA_DIR and updates the vector store accordingly, instead of the
blind "load everything, re-embed everything" flow in app.py.

See Phase 2.6 in the implementation plan for the full design/rationale.
"""

import difflib
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from rag_learn import config
from rag_learn.data_loader import LOADER_VERSION, LOADERS, load_document
from rag_learn.embedding import EmbeddingPipeline
from rag_learn.vectorstore import VectorStore

MANIFEST_PATH = Path(config.VECTOR_STORE_DIR) / "manifest.json"
PENDING_REVIEW_PATH = Path(config.VECTOR_STORE_DIR) / "pending_review.json"

FILENAME_SIMILARITY_THRESHOLD = 0.6

_DATE_PATTERNS = [
    (re.compile(r"(\d{4})[-_.](\d{2})[-_.](\d{2})"), "%Y-%m-%d"),
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), "%Y%m%d"),
]


def _pipeline_fingerprint() -> str:
    raw = f"{config.EMBEDDING_MODEL}|{config.CHUNK_SIZE}|{config.CHUNK_OVERLAP}|{LOADER_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _hash_file(path: Path) -> Optional[str]:
    """Content hash with a cheap stability check (file size read twice) to
    avoid hashing a file mid-write/mid-copy. Returns None if unstable."""
    size_before = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    size_after = path.stat().st_size
    if size_before != size_after:
        return None
    return digest


def _date_from_filename(name: str) -> Optional[date]:
    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.search(name)
        if m:
            try:
                return datetime.strptime("-".join(m.groups()), "%Y-%m-%d").date()
            except ValueError:
                continue
    return None


def _recency_signal(path: Path) -> tuple[date, str]:
    """A date to compare for auto-replace decisions: a date pattern in the
    filename takes priority (explicit user intent), falling back to the
    file's OS mtime."""
    from_name = _date_from_filename(path.stem)
    if from_name:
        return from_name, "filename"
    return datetime.fromtimestamp(path.stat().st_mtime).date(), "mtime"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _walk_data_dir(data_dir: Path):
    vector_store_dir = Path(config.VECTOR_STORE_DIR).resolve()
    for path in sorted(data_dir.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        if vector_store_dir in path.resolve().parents:
            continue  # never ingest the vector store's own state files (manifest, chroma db, etc.)
        yield path


def _ingest_file(path: Path, pipeline: EmbeddingPipeline, store: VectorStore) -> list[str]:
    docs = load_document(path)
    if not docs:
        return []
    chunks = pipeline.chunk_documents(docs)
    if not chunks:
        return []
    embeddings = pipeline.embed_chunks(chunks)
    return store.add_documents(chunks, embeddings)


def sync(data_dir: Optional[str] = None) -> dict[str, Any]:
    data_path = Path(data_dir or config.DATA_DIR).resolve()
    fingerprint = _pipeline_fingerprint()

    manifest: dict[str, Any] = _load_json(MANIFEST_PATH, {})
    pending_review: list[dict[str, Any]] = _load_json(PENDING_REVIEW_PATH, [])

    current_paths = list(_walk_data_dir(data_path))
    current_by_str = {str(p): p for p in current_paths}

    manifest_paths = set(manifest.keys())
    current_path_strs = set(current_by_str.keys())

    new_path_strs = current_path_strs - manifest_paths
    missing_path_strs = manifest_paths - current_path_strs
    common_path_strs = current_path_strs & manifest_paths

    summary = {"added": [], "updated": [], "removed": [], "renamed": [], "skipped": 0, "auto_replaced": [], "pending_review": []}

    pipeline: Optional[EmbeddingPipeline] = None
    store: Optional[VectorStore] = None

    def _ensure_clients():
        nonlocal pipeline, store
        if pipeline is None:
            pipeline = EmbeddingPipeline()
        if store is None:
            store = VectorStore()

    # --- Rename detection: a missing path + a new path sharing the same
    # content hash, in the same sync pass -> update the manifest key rather
    # than deleting+re-embedding identical content. ---
    unresolved_missing = set(missing_path_strs)
    unresolved_new = set(new_path_strs)
    for missing_str in list(unresolved_missing):
        old_entry = manifest[missing_str]
        old_path = Path(missing_str)
        try:
            old_hash = _hash_file(old_path) if old_path.exists() else old_entry.get("content_hash")
        except OSError:
            old_hash = old_entry.get("content_hash")
        for new_str in list(unresolved_new):
            new_hash = _hash_file(current_by_str[new_str])
            if new_hash and new_hash == old_entry.get("content_hash"):
                manifest[new_str] = {**old_entry, "last_synced": datetime.now().isoformat()}
                del manifest[missing_str]
                unresolved_missing.discard(missing_str)
                unresolved_new.discard(new_str)
                summary["renamed"].append({"from": missing_str, "to": new_str})
                print(f"[SYNC] Renamed (content unchanged): {missing_str} -> {new_str}")
                break

    # --- New-version detection among the still-unresolved new files:
    # filename-similarity match against existing manifest entries of the
    # same extension, then a recency signal decides auto-replace vs.
    # manual review. New files are always ingested either way -- only the
    # *old* file's chunks are ever removed, and only on a clear signal. ---
    auto_replace_targets: dict[str, str] = {}  # old_path_str -> new_path_str
    for new_str in list(unresolved_new):
        new_path = current_by_str[new_str]
        best_match, best_ratio = None, 0.0
        for candidate_str, candidate_entry in manifest.items():
            if candidate_str in unresolved_missing:
                continue  # already resolved as a rename target
            if candidate_entry.get("superseded_by"):
                continue  # match against the live version, not a tombstone
            candidate_path = Path(candidate_str)
            if candidate_path.suffix.lower() != new_path.suffix.lower():
                continue
            ratio = difflib.SequenceMatcher(None, candidate_path.stem.lower(), new_path.stem.lower()).ratio()
            if ratio > best_ratio:
                best_match, best_ratio = candidate_str, ratio

        if best_match is None or best_ratio < FILENAME_SIMILARITY_THRESHOLD:
            continue

        new_date, new_source = _recency_signal(new_path)
        old_path = Path(best_match)
        old_date, old_source = (
            _recency_signal(old_path) if old_path.exists() else (None, None)
        )
        if old_date is None:
            old_mtime = manifest[best_match].get("file_mtime")
            old_date = datetime.fromtimestamp(old_mtime).date() if old_mtime else None

        if old_date is not None and new_date > old_date:
            auto_replace_targets[best_match] = new_str
            summary["auto_replaced"].append(
                {"old": best_match, "new": new_str, "similarity": round(best_ratio, 2),
                 "new_date_source": new_source, "old_date_source": old_source}
            )
            print(f"[SYNC] Auto-replacing '{best_match}' with '{new_str}' (similarity={best_ratio:.2f}, {new_date} > {old_date})")
        else:
            entry = {
                "new_path": new_str,
                "existing_path": best_match,
                "similarity": round(best_ratio, 2),
                "reason": "no unambiguous newer date signal",
            }
            pending_review.append(entry)
            summary["pending_review"].append(entry)
            print(f"[SYNC] '{new_str}' looks similar to '{best_match}' (similarity={best_ratio:.2f}) but recency is unclear -- flagged for manual review, both kept.")

    # --- Apply removals: genuinely deleted files, and old sides of an
    # auto-replace. ---
    for missing_str in unresolved_missing:
        _ensure_clients()
        store.delete(manifest[missing_str].get("chunk_ids", []))
        del manifest[missing_str]
        summary["removed"].append(missing_str)
        print(f"[SYNC] Removed (deleted from disk): {missing_str}")

    for old_str, new_str in auto_replace_targets.items():
        _ensure_clients()
        store.delete(manifest[old_str].get("chunk_ids", []))
        # The old file may still be physically present on disk (the user
        # added a new version alongside it rather than deleting it) -- in
        # that case it stays in current_path_strs/common_path_strs, so a
        # plain `del` here would make the "changed files" loop below hit a
        # missing manifest key. Leave a tombstone instead: no chunk_ids (so
        # nothing to delete twice), marked superseded so future syncs skip
        # it rather than re-ingesting or re-flagging it against its own
        # replacement.
        manifest[old_str] = {
            "content_hash": manifest[old_str].get("content_hash"),
            "chunk_ids": [],
            "pipeline_fingerprint": fingerprint,
            "superseded_by": new_str,
            "last_synced": datetime.now().isoformat(),
        }
        common_path_strs.discard(old_str)
        print(f"[SYNC] Removed superseded version: {old_str} (superseded by {new_str})")

    # --- Apply changed files (content or pipeline fingerprint differs). ---
    for path_str in common_path_strs:
        entry = manifest[path_str]
        if entry.get("superseded_by"):
            continue  # tombstone from an earlier auto-replace -- intentionally not re-indexed
        path = current_by_str[path_str]
        content_hash = _hash_file(path)
        if content_hash is None:
            summary["skipped"] += 1
            print(f"[SYNC] Skipped (unstable read, will retry next sync): {path_str}")
            continue
        if content_hash == entry.get("content_hash") and entry.get("pipeline_fingerprint") == fingerprint:
            continue  # unchanged, nothing to do

        _ensure_clients()
        store.delete(entry.get("chunk_ids", []))
        chunk_ids = _ingest_file(path, pipeline, store)
        manifest[path_str] = {
            "content_hash": content_hash,
            "chunk_ids": chunk_ids,
            "pipeline_fingerprint": fingerprint,
            "file_mtime": path.stat().st_mtime,
            "last_synced": datetime.now().isoformat(),
        }
        summary["updated"].append(path_str)
        print(f"[SYNC] Updated: {path_str} ({len(chunk_ids)} chunks)")

    # --- Apply new files (including auto-replace winners). ---
    for path_str in unresolved_new:
        path = current_by_str[path_str]
        content_hash = _hash_file(path)
        if content_hash is None:
            summary["skipped"] += 1
            print(f"[SYNC] Skipped (unstable read, will retry next sync): {path_str}")
            continue

        _ensure_clients()
        chunk_ids = _ingest_file(path, pipeline, store)
        manifest[path_str] = {
            "content_hash": content_hash,
            "chunk_ids": chunk_ids,
            "pipeline_fingerprint": fingerprint,
            "file_mtime": path.stat().st_mtime,
            "last_synced": datetime.now().isoformat(),
        }
        summary["added"].append(path_str)
        print(f"[SYNC] Added: {path_str} ({len(chunk_ids)} chunks)")

    _save_json(MANIFEST_PATH, manifest)
    _save_json(PENDING_REVIEW_PATH, pending_review)

    print(
        f"[SYNC] Done. added={len(summary['added'])} updated={len(summary['updated'])} "
        f"removed={len(summary['removed'])} renamed={len(summary['renamed'])} "
        f"auto_replaced={len(summary['auto_replaced'])} pending_review={len(summary['pending_review'])} "
        f"skipped={summary['skipped']}"
    )
    return summary


if __name__ == "__main__":
    sync()
