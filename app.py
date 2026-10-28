"""Review UI: streamlit run app.py"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

from labeler.cli import build_report
from labeler.export import to_csv
from labeler.labelers import LLMLabeler, KeywordLabeler
from labeler.pipeline import suggest_all
from labeler.store import Store
from labeler.task import LABELS, NAME

SAMPLE_CSV = Path(__file__).parent / "data" / "musiccaps_sample.csv"

st.set_page_config(page_title="AI Labeling", layout="wide")


@st.cache_resource
def get_store(path: str) -> Store:
    return Store(path)


store = get_store(os.environ.get("LABELER_DB", "labels.db"))


# --- sidebar: dataset + AI pre-labeling ---------------------------------------

with st.sidebar:
    st.header(NAME)
    counts = store.counts_by_status()
    total = sum(counts.values())
    st.metric("Reviewed", f"{counts['reviewed']} / {total}")
    st.progress(counts["reviewed"] / total if total else 0.0)
    st.caption(f"{counts['suggested']} awaiting review · {counts['pending']} not yet AI-labeled")

    if total == 0:
        if st.button("Load MusicCaps sample (200 items)"):
            store.import_csv(SAMPLE_CSV)
            st.rerun()

    if counts["pending"]:
        st.subheader("AI pre-labeling")
        choice = st.radio("Labeler", ("LLM", "Keyword baseline"), horizontal=True)
        if st.button(f"Label {counts['pending']} items", type="primary"):
            labeler = LLMLabeler() if choice == "LLM" else KeywordLabeler()
            bar = st.progress(0.0)
            try:
                result = suggest_all(store, labeler, on_progress=lambda d, t: bar.progress(d / t))
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                    anthropic.NotFoundError) as e:
                st.error(f"The LLM API rejected the request: {e}. Check ANTHROPIC_API_KEY and LABELER_MODEL.")
            else:
                if result.failed:
                    st.warning(f"{len(result.failed)} items failed and stay pending; run again to retry.")
                st.rerun()


# --- main area -------------------------------------------------------------------

review_tab, metrics_tab, export_tab = st.tabs(["Review", "Metrics", "Export"])

with review_tab:
    queue = store.review_queue()
    if not queue:
        st.info("Nothing to review. Load data and run AI pre-labeling from the sidebar."
                if counts["reviewed"] == 0 else "All suggested items are reviewed.")
    else:
        item = queue[0]
        st.caption(f"{len(queue)} left · lowest AI confidence first · item `{item.id}`")
        st.markdown(f"> {item.text}")
        st.markdown(
            f"**AI suggests `{item.ai_label}`** · confidence {item.ai_confidence:.0%}  \n"
            f"_{item.ai_rationale}_"
        )
        with st.form(f"review-{item.id}"):
            label = st.radio("Label", LABELS, index=LABELS.index(item.ai_label), horizontal=True)
            if st.form_submit_button("Save and next", type="primary"):
                store.save_review(item.id, label)
                st.rerun()

    reviewed = store.reviewed()
    if reviewed:
        with st.expander("Correct an earlier review"):
            ids = [it.id for it in reversed(reviewed)]  # most recent first
            item_id = st.selectbox("Item", ids)
            it = store.get(item_id)
            st.markdown(f"> {it.text}")
            st.caption(f"AI said `{it.ai_label}` · you said `{it.human_label}`")
            new = st.selectbox("New label", LABELS, index=LABELS.index(it.human_label), key="fix")
            if st.button("Update label", disabled=new == it.human_label):
                store.save_review(item_id, new)
                st.rerun()

with metrics_tab:
    r = build_report(store)
    if r.n == 0:
        st.info("Metrics appear once you have reviewed some items.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviewed", r.n)
        c2.metric("AI accuracy", f"{r.accuracy:.1%}")
        c3.metric("Cohen's kappa", f"{r.kappa:.2f}", help="Agreement beyond chance: 0 = chance, 1 = perfect.")

        st.subheader("Accuracy by AI confidence")
        st.caption("If the confidence is honest, accuracy rises band by band. "
                   "Low bands are where human review pays off most.")
        bands = pd.DataFrame([vars(b) for b in r.bands])
        st.dataframe(bands, hide_index=True, column_config={
            "accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})

        st.subheader("Per label")
        st.dataframe(pd.DataFrame([vars(s) for s in r.per_label]), hide_index=True, column_config={
            k: st.column_config.NumberColumn(format="percent") for k in ("precision", "recall", "f1")})

        st.subheader("Confusion matrix")
        st.caption("Rows: human label. Columns: AI label. Off-diagonal cells are AI errors.")
        st.dataframe(pd.DataFrame(r.confusion).T)

        st.subheader("Disagreements")
        wrong = [it for it in store.reviewed() if it.ai_label and it.ai_label != it.human_label]
        st.dataframe(
            pd.DataFrame([{"id": it.id, "human": it.human_label, "ai": it.ai_label,
                           "confidence": it.ai_confidence, "text": it.text} for it in wrong]),
            hide_index=True,
        )

with export_tab:
    include_all = st.toggle("Include unreviewed items", value=False)
    items = store.all_items() if include_all else store.reviewed()
    st.write(f"{len(items)} rows. The `label` column is the human label.")
    st.download_button("Download CSV", to_csv(items), file_name="labels.csv",
                       mime="text/csv", disabled=not items, type="primary")
