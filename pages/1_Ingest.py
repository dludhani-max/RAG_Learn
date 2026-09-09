"""Ingest page: thin orchestration only, no business logic duplicated here
-- the actual add/update/remove/rename/routing logic all lives in
rag_learn.sync. See Phase 4 in the implementation plan."""

from pathlib import Path

import streamlit as st

from rag_learn import config
from rag_learn.sync import load_pending_review, save_pending_review, sync

st.set_page_config(page_title="Ingest - RAG_Learn", page_icon="📥")
st.title("📥 Ingest")
st.write(
    "Sync new, modified, removed, or renamed files under a data directory into the vector "
    "store, SQL tables, or page-index trees -- whichever a file's classification routes it to. "
    "Unchanged files are skipped; this is cheap to re-run."
)

data_dir = st.text_input("Data directory", value=config.DATA_DIR)

if st.button("Run Sync", type="primary"):
    with st.spinner("Syncing... (first run downloads the embedding/reranker models, can take a while)"):
        try:
            summary = sync(data_dir)
        except Exception as e:
            st.error(f"Sync failed: {e}")
            summary = None

    if summary is not None:
        st.success(
            f"Added {len(summary['added'])} · updated {len(summary['updated'])} · "
            f"removed {len(summary['removed'])} · renamed {len(summary['renamed'])} · "
            f"auto-replaced {len(summary['auto_replaced'])} · skipped {summary['skipped']}"
        )
        if summary["routing_counts"]:
            st.write("**Routing decisions this run:**", summary["routing_counts"])

        for label, key in (("Added", "added"), ("Updated", "updated"), ("Removed", "removed")):
            if summary[key]:
                with st.expander(f"{label} ({len(summary[key])})"):
                    for f in summary[key]:
                        st.write(Path(f).name)

st.divider()
st.subheader("Pending review")
st.caption(
    "A new file that looked like a version of an existing one, but had no unambiguous recency "
    "signal (a date in the filename, or file modification time). Both versions are already "
    "indexed as-is -- nothing here blocks ingestion -- dismiss an entry once you've manually "
    "confirmed which one should be treated as current. To actually remove a stale version from "
    "the store, delete its file from the data directory and run Sync again; there's no "
    "one-click delete here since sync only ever removes what's no longer on disk."
)

pending = load_pending_review()
if not pending:
    st.info("Nothing pending review.")
else:
    for i, entry in enumerate(pending):
        with st.container(border=True):
            st.write(f"**New:** `{Path(entry['new_path']).name}`")
            st.write(f"**Existing:** `{Path(entry['existing_path']).name}`")
            st.caption(f"Filename similarity {entry['similarity']} -- {entry['reason']}")
            if st.button("Dismiss (stop flagging this pair)", key=f"dismiss_{i}"):
                remaining = [e for e in pending if e is not entry]
                save_pending_review(remaining)
                st.rerun()
