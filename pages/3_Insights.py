"""Insights page: LangSmith trace and evaluation data, pulled via the API
and rendered natively -- not an iframe embed of smith.langchain.com, which
that site's own security headers (X-Frame-Options/CSP frame-ancestors)
refuse to allow from any other origin. This is the practical alternative:
real data, inside the app, no tab-switching."""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from rag_learn import config, telemetry
from rag_learn.eval import local_store
from rag_learn.eval.golden_dataset import DATASET_NAME

st.set_page_config(page_title="Insights - RAG_Learn", page_icon="📊")
st.title("📊 Insights")

# --- LLM usage & cost (local telemetry, no LangSmith needed) ---------------
# Every LLM call (both providers -- Groq primary, Anthropic fallback --
# see llm_factory.py) is recorded locally regardless of whether LangSmith
# is configured, so this section renders unconditionally, before the
# LangSmith-only sections below (which do require a key and st.stop() if
# it's missing).
st.subheader("🔢 LLM usage & cost")
summary_rows = telemetry.summary()

if not summary_rows:
    st.info("No LLM calls recorded yet -- this fills in as the app is used (chat, ingestion, eval).")
else:
    df = pd.DataFrame(summary_rows)

    totals_by_provider = df.groupby("provider").agg(
        calls=("calls", "sum"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        cost_usd=("cost_usd", "sum"),
    )
    cols = st.columns(len(totals_by_provider) + 1)
    for col, (provider, row) in zip(cols, totals_by_provider.iterrows()):
        with col:
            st.metric(
                f"{provider} calls",
                int(row["calls"]),
                help=f"{int(row['input_tokens'])} in / {int(row['output_tokens'])} out tokens",
            )
            st.caption(f"${row['cost_usd']:.4f}" if row["cost_usd"] else "$0.00 (free tier)")

    fallback_calls = int(df.loc[df["is_fallback"], "calls"].sum()) if "is_fallback" in df else 0
    total_calls = int(df["calls"].sum())
    with cols[-1]:
        st.metric(
            "Fallback rate",
            f"{(fallback_calls / total_calls * 100):.1f}%" if total_calls else "0%",
            help="How often the Anthropic fallback actually fired instead of Groq (the primary).",
        )

    st.markdown("**By call site (purpose):**")
    display_df = df.rename(
        columns={
            "provider": "Provider",
            "purpose": "Purpose",
            "is_fallback": "Was fallback",
            "calls": "Calls",
            "input_tokens": "Input tokens",
            "output_tokens": "Output tokens",
            "cost_usd": "Cost ($)",
        }
    )
    display_df["Cost ($)"] = display_df["Cost ($)"].map(lambda v: round(v, 4))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("Recent individual calls"):
        recent_rows = telemetry.recent(100)
        if recent_rows:
            st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Nothing to show.")

st.divider()

# --- Local eval history (always available, regardless of LangSmith) -------
# run_eval.py writes every result here unconditionally and only
# *additionally* pushes to LangSmith on a best-effort basis -- this is the
# source of truth for "did my eval run actually happen and what did it
# score," independent of whether LangSmith's quota was available that day.
st.subheader("🧪 Local eval history")
experiments = local_store.list_experiments()
if not experiments:
    st.info(
        "No local eval runs yet -- run `uv run python3 -m rag_learn.eval.run_eval` "
        "(or the Insights/Evaluation section once experiments exist)."
    )
else:
    exp_df = pd.DataFrame(experiments)
    exp_df["started"] = pd.to_datetime(exp_df["started"]).dt.strftime("%Y-%m-%d %H:%M")
    for col in ("retrieval_relevance", "groundedness", "answer_correctness", "answer_relevancy"):
        exp_df[col] = exp_df[col].round(3)
    exp_df["synced_to_langsmith"] = exp_df.apply(
        lambda r: f"{r['synced_to_langsmith']}/{r['questions']}", axis=1
    )
    st.dataframe(exp_df, use_container_width=True, hide_index=True)

    st.markdown("**Drill into one local run:**")
    exp_choice = st.selectbox("Experiment", [e["experiment"] for e in experiments], key="local_exp_choice")
    if exp_choice:
        detail_rows = local_store.results_for(exp_choice)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Question": r["question"],
                        "Answer": (r["answer"] or "")[:150],
                        "retrieval_relevance": r["retrieval_relevance"],
                        "groundedness": r["groundedness"],
                        "answer_correctness": r["answer_correctness"],
                        "answer_relevancy": r["answer_relevancy"],
                        "On LangSmith": "yes" if r.get("langsmith_run_id") else "no",
                    }
                    for r in detail_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption("The sections below pull from LangSmith directly and require `LANGSMITH_API_KEY`.")

if not config.LANGSMITH_API_KEY:
    st.warning(
        "`LANGSMITH_API_KEY` isn't set, so there's no trace/eval data to show below. "
        "It's optional (see README) -- set it in `.env` and restart to enable the rest of this page."
    )
    st.stop()


@st.cache_resource
def get_client():
    from langsmith import Client

    return Client()


client = get_client()

st.caption(
    f"Pulled live from your LangSmith account via its API. Full detail (step-by-step traces, "
    f"raw run data) is still easiest at [smith.langchain.com](https://smith.langchain.com) -- "
    f"LangSmith blocks being embedded in a page like this one, so this shows a summary, not a mirror."
)

st.subheader("Recent chat queries")
try:
    runs = list(
        client.list_runs(
            project_name=config.LANGSMITH_PROJECT,
            is_root=True,
            limit=20,
        )
    )
except Exception as e:
    runs = []
    st.error(f"Couldn't reach LangSmith: {e}")

if not runs:
    st.info("No traced queries yet -- ask something in Chat, then check back here.")
else:
    rows = []
    for r in sorted(runs, key=lambda r: r.start_time or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        question = (r.inputs or {}).get("question", "")
        latency = (
            f"{(r.end_time - r.start_time).total_seconds():.1f}s" if r.start_time and r.end_time else ""
        )
        rows.append(
            {
                "Time": r.start_time.strftime("%Y-%m-%d %H:%M") if r.start_time else "",
                "Question": question[:80],
                "Latency": latency,
                "Error": "yes" if r.error else "",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Evaluation experiments")
st.caption(f"Runs of `rag-learn-golden` (Phase 6) -- retrieval relevance, groundedness, answer correctness, answer relevancy.")

try:
    experiments = list(client.list_projects(reference_dataset_name=DATASET_NAME, include_stats=True))
except Exception as e:
    experiments = []
    st.error(f"Couldn't load experiments: {e}")

if not experiments:
    st.info(
        "No evaluation runs yet -- see the README's \"Evaluation framework\" section "
        "(`uv run python3 -m rag_learn.eval.run_eval`)."
    )
else:
    experiments = sorted(experiments, key=lambda p: p.start_time or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    summary_rows = []
    for p in experiments:
        stats = getattr(p, "feedback_stats", None) or {}
        row = {"Experiment": p.name, "Date": p.start_time.strftime("%Y-%m-%d %H:%M") if p.start_time else ""}
        for metric in ("retrieval_relevance", "groundedness", "answer_correctness", "answer_relevancy"):
            avg = stats.get(metric, {}).get("avg")
            row[metric] = round(avg, 2) if avg is not None else None
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("**Drill into one experiment:**")
    choice = st.selectbox("Experiment", [p.name for p in experiments])
    if choice:
        exp_runs = list(client.list_runs(project_name=choice, is_root=True))
        detail_rows = []
        for r in exp_runs:
            question = (r.inputs or {}).get("question", "")
            answer = (r.outputs or {}).get("answer", "")
            feedback = {f.key: f.score for f in client.list_feedback(run_ids=[r.id])}
            detail_rows.append(
                {
                    "Question": question,
                    "Answer": answer[:150],
                    **{k: feedback.get(k) for k in ("retrieval_relevance", "groundedness", "answer_correctness", "answer_relevancy")},
                }
            )
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
