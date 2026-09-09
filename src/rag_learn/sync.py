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

from rag_learn import config, vectorless_pageindex, vectorless_sql
from rag_learn.classifier import (
    ROUTING_PAGEINDEX,
    ROUTING_SQL,
    ROUTING_VECTOR,
    TABULAR_EXTENSIONS,
    classify_and_tag,
)
from rag_learn.data_loader import LOADER_VERSION, LOADERS, load_document
from rag_learn.embedding import EmbeddingPipeline
from rag_learn.vectorstore import VectorStore

MANIFEST_PATH = Path(config.VECTOR_STORE_DIR) / "manifest.json"
PENDING_REVIEW_PATH = Path(config.VECTOR_STORE_DIR) / "pending_review.json"

FILENAME_SIMILARITY_THRESHOLD = 0.6
# Filename similarity alone isn't enough signal that two files are the same
# document -- verified live: "100 LLM Interview Questions .pdf" vs. "RAG
# Interview Questions .pdf" scored 0.81 on filename alone (shared "Interview
# Questions" wording) despite being unrelated documents, which would have
# auto-deleted the older one's page-index tree. Content similarity on a text
# preview is the second, independent signal required before either an
# auto-replace or even a manual-review flag fires.
CONTENT_SIMILARITY_THRESHOLD = 0.5
CONTENT_PREVIEW_CHARS = 4000

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


def _preview_text(path: Path) -> Optional[str]:
    """A cheap text sample for content-similarity comparison -- not a full
    load, just enough (CONTENT_PREVIEW_CHARS) to tell two same-named-ish
    files apart by what they actually say. Returns None on any extraction
    failure so callers can fail safe (no auto-replace) rather than compare
    against empty/garbage text."""
    try:
        docs = load_document(path)
    except Exception:
        return None
    if not docs:
        return None
    text = "\n".join(d.page_content for d in docs)[:CONTENT_PREVIEW_CHARS]
    return text or None


def _content_similarity(old_path: Path, new_path: Path) -> Optional[float]:
    """None means "couldn't compare" (a file failed to load, or the old file
    no longer exists on disk) -- callers must treat that as "not confirmed
    similar," not as a pass."""
    if not old_path.exists():
        return None
    old_text = _preview_text(old_path)
    new_text = _preview_text(new_path)
    if old_text is None or new_text is None:
        return None
    return difflib.SequenceMatcher(None, old_text, new_text).ratio()


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


def list_indexed_documents() -> dict[str, list[str]]:
    """Group the manifest's live (non-superseded) entries by routing,
    returning basenames -- used by the Streamlit Chat page's "search this
    document" picker, so it doesn't need to reach into sync's private
    manifest format itself."""
    manifest = _load_json(MANIFEST_PATH, {})
    grouped: dict[str, list[str]] = {}
    for path_str, entry in manifest.items():
        if entry.get("superseded_by"):
            continue
        routing = entry.get("routing", ROUTING_VECTOR)
        grouped.setdefault(routing, []).append(Path(path_str).name)
    return grouped


def load_pending_review() -> list[dict[str, Any]]:
    return _load_json(PENDING_REVIEW_PATH, [])


def save_pending_review(entries: list[dict[str, Any]]) -> None:
    _save_json(PENDING_REVIEW_PATH, entries)


def _walk_data_dir(data_dir: Path):
    vector_store_dir = Path(config.VECTOR_STORE_DIR).resolve()
    for path in sorted(data_dir.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        if vector_store_dir in path.resolve().parents:
            continue  # never ingest the vector store's own state files (manifest, chroma db, etc.)
        yield path


def _ingest_file(
    path: Path, pipeline: EmbeddingPipeline, store: VectorStore
) -> tuple[list[str], str, dict[str, Any]]:
    """Route a file to the right ingestion mechanism based on its
    classification. Returns (chunk_ids, routing, extra) -- extra carries
    whatever the manifest needs to clean this entry up later (a DuckDB
    table name or a page-index tree path), since vectorless routes have no
    chunk_ids for VectorStore.delete to act on."""
    # Tabular files skip load_document/classification text analysis
    # entirely: extension alone is sufficient signal (matches
    # classifier.TABULAR_EXTENSIONS), and DuckDB reading the file directly
    # is both cheaper and more useful than a per-row Document list from
    # CSVLoader/UnstructuredExcelLoader, which vectorless_sql doesn't need.
    if path.suffix.lower() in TABULAR_EXTENSIONS:
        table = vectorless_sql.ingest_table(path)
        return [], ROUTING_SQL, {"sql_table": table}

    docs = load_document(path)
    if not docs:
        return [], ROUTING_VECTOR, {}

    # Classify before chunking -- routing is a whole-document decision
    # (e.g. "this PDF is sectioned"), not a per-chunk one, and the
    # page-index path needs the source document's own structure, not a
    # fragment of it.
    routing = classify_and_tag(path, docs)

    if routing == ROUTING_SQL:
        # Only reachable for images the classifier judged tabular-looking
        # (TABULAR_EXTENSIONS files already returned above) -- image-to-table
        # extraction isn't implemented (Phase 3b scoped to native tabular
        # files), so fall back to embedding rather than silently dropping
        # the content, and correct the tag to match what actually happened.
        print(f"[SQL] {path.name} classified tabular but image-to-table extraction isn't implemented -- embedding instead.")
        routing = ROUTING_VECTOR
        for doc in docs:
            doc.metadata["routing"] = routing

    if routing == ROUTING_PAGEINDEX:
        tree_path = vectorless_pageindex.build_tree(path, docs)
        return [], routing, {"tree_path": tree_path}

    chunks = pipeline.chunk_documents(docs)
    if not chunks:
        return [], routing, {}
    embeddings = pipeline.embed_chunks(chunks)
    return store.add_documents(chunks, embeddings), routing, {}


def _cleanup_entry(entry: dict[str, Any], store: VectorStore) -> None:
    """Undo whatever _ingest_file did for this manifest entry -- dispatches
    on the routing that was recorded at ingest time, since a vector entry's
    chunk_ids, a SQL entry's table, and a page-index entry's tree file each
    need a different removal call."""
    routing = entry.get("routing", ROUTING_VECTOR)
    if routing == ROUTING_SQL and entry.get("sql_table"):
        vectorless_sql.drop_table(entry["sql_table"])
    elif routing == ROUTING_PAGEINDEX and entry.get("tree_path"):
        vectorless_pageindex.delete_tree(entry["tree_path"])
    else:
        store.delete(entry.get("chunk_ids", []))


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

    summary = {
        "added": [], "updated": [], "removed": [], "renamed": [], "skipped": 0,
        "auto_replaced": [], "pending_review": [], "routing_counts": {},
    }

    def _record_routing(routing: str) -> None:
        summary["routing_counts"][routing] = summary["routing_counts"].get(routing, 0) + 1

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

        content_ratio = _content_similarity(Path(best_match), new_path)
        if content_ratio is None or content_ratio < CONTENT_SIMILARITY_THRESHOLD:
            # Filenames looked alike but the actual text doesn't -- these are
            # two different documents that happen to share wording in their
            # names, not versions of the same one. Don't auto-replace, and
            # don't even flag for manual review (that would just be noise for
            # every unrelated pair with a generic-ish filename).
            print(
                f"[SYNC] '{new_str}' filename resembles '{best_match}' (similarity={best_ratio:.2f}) "
                f"but content does not (content_similarity={content_ratio}) -- treating as a distinct document."
            )
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
        _cleanup_entry(manifest[missing_str], store)
        del manifest[missing_str]
        summary["removed"].append(missing_str)
        print(f"[SYNC] Removed (deleted from disk): {missing_str}")

    for old_str, new_str in auto_replace_targets.items():
        _ensure_clients()
        _cleanup_entry(manifest[old_str], store)
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
        _cleanup_entry(entry, store)
        chunk_ids, routing, extra = _ingest_file(path, pipeline, store)
        _record_routing(routing)
        manifest[path_str] = {
            "content_hash": content_hash,
            "chunk_ids": chunk_ids,
            "routing": routing,
            "pipeline_fingerprint": fingerprint,
            "file_mtime": path.stat().st_mtime,
            "last_synced": datetime.now().isoformat(),
            **extra,
        }
        summary["updated"].append(path_str)
        print(f"[SYNC] Updated: {path_str} ({len(chunk_ids)} chunks, routing={routing})")

    # --- Apply new files (including auto-replace winners). ---
    for path_str in unresolved_new:
        path = current_by_str[path_str]
        content_hash = _hash_file(path)
        if content_hash is None:
            summary["skipped"] += 1
            print(f"[SYNC] Skipped (unstable read, will retry next sync): {path_str}")
            continue

        _ensure_clients()
        chunk_ids, routing, extra = _ingest_file(path, pipeline, store)
        _record_routing(routing)
        manifest[path_str] = {
            "content_hash": content_hash,
            "chunk_ids": chunk_ids,
            "routing": routing,
            "pipeline_fingerprint": fingerprint,
            "file_mtime": path.stat().st_mtime,
            "last_synced": datetime.now().isoformat(),
            **extra,
        }
        summary["added"].append(path_str)
        print(f"[SYNC] Added: {path_str} ({len(chunk_ids)} chunks, routing={routing})")

    _save_json(MANIFEST_PATH, manifest)
    _save_json(PENDING_REVIEW_PATH, pending_review)

    # Content actually changed (renames don't count -- same content, cached
    # answers derived from it are still valid) -> the Q&A cache may now hold
    # stale answers, so wipe it rather than risk serving one.
    content_changed = summary["added"] or summary["updated"] or summary["removed"] or summary["auto_replaced"]
    if content_changed:
        from rag_learn.cache import QACache

        QACache().clear()

    print(
        f"[SYNC] Done. added={len(summary['added'])} updated={len(summary['updated'])} "
        f"removed={len(summary['removed'])} renamed={len(summary['renamed'])} "
        f"auto_replaced={len(summary['auto_replaced'])} pending_review={len(summary['pending_review'])} "
        f"skipped={summary['skipped']}"
    )
    if summary["routing_counts"]:
        print(f"[SYNC] Routing decisions this run: {summary['routing_counts']}")
    return summary


if __name__ == "__main__":
    sync()
