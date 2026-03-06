"""Streamlit UI for the Clinical Note Simplification System."""

import streamlit as st
import re
import html as html_module
import pandas as pd
import plotly.graph_objects as go
from pipeline.groq_client import GroqClient
from pipeline.sources.base import SimplificationResult
from pipeline.sources.freedictionary import FreeDictionarySource
from pipeline.sources.medlineplus import MedlinePlusSource
from pipeline.sources.wikipedia_source import WikipediaSource
from pipeline.rag_simplifier import run_rag_pipeline
from pipeline.baseline_simplifier import run_baseline_pipeline
from pipeline.evaluator import evaluate
from pipeline.abbreviations import load_abbreviations, load_abbreviations_csv
from config import GROQ_MODELS, GROQ_API_KEY

# ─── Page config ───
st.set_page_config(
    page_title="Clinical Note Simplifier",
    page_icon="🏥",
    layout="wide",
)

# ─── Custom CSS for highlighted words with tooltips ───
st.markdown("""
<style>
.simplified-text {
    font-size: 1.1rem;
    line-height: 1.8;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #dee2e6;
}
.highlight-medical {
    background-color: #d4edda;
    padding: 2px 4px;
    border-radius: 3px;
    cursor: help;
    position: relative;
    border-bottom: 2px solid #28a745;
}
.highlight-nonmedical {
    background-color: #fff3cd;
    padding: 2px 4px;
    border-radius: 3px;
    cursor: help;
    position: relative;
    border-bottom: 2px solid #ffc107;
}
.highlight-abbreviation {
    background-color: #cce5ff;
    padding: 2px 4px;
    border-radius: 3px;
    cursor: help;
    position: relative;
    border-bottom: 2px solid #007bff;
}
.highlight-medical:hover::after,
.highlight-nonmedical:hover::after,
.highlight-abbreviation:hover::after {
    content: attr(data-original);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.85rem;
    white-space: nowrap;
    z-index: 100;
    margin-bottom: 4px;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
}
.metric-delta-good { color: #28a745; }
.metric-delta-bad { color: #dc3545; }
.legend-item {
    display: inline-block;
    margin-right: 1rem;
    font-size: 0.9rem;
}
.legend-dot {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 3px;
    margin-right: 4px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Configuration")

    _default_key = st.session_state.get("groq_api_key", "") or GROQ_API_KEY
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=_default_key,
        help="Loaded from .env if available. Get a key at https://console.groq.com/keys",
    )
    if api_key:
        st.session_state["groq_api_key"] = api_key

    st.divider()

    model_name = st.selectbox(
        "LLM Model",
        options=list(GROQ_MODELS.keys()),
        index=0,
        help="Select which Groq model to use for simplification",
    )
    selected_model = GROQ_MODELS[model_name]

    st.divider()

    st.subheader("📚 External Sources")
    use_freedict = st.checkbox("FreeDictionary Medical", value=True)
    use_medline = st.checkbox("MedlinePlus", value=True)
    use_wikipedia = st.checkbox("Wikipedia", value=True)

    st.divider()

    st.subheader("📝 Abbreviations")
    use_csv_abbrevs = st.checkbox(
        "Use CSV abbreviation list (abr.csv)",
        value=True,
        help="Use the extended abbreviation list from abr.csv instead of the built-in JSON.",
    )

    st.divider()

    mode = st.radio(
        "Simplification Mode",
        options=["RAG Mode", "Baseline Mode", "Compare Both"],
        index=0,
        help="RAG uses external sources; Baseline uses only the LLM; Compare runs both",
    )

    st.divider()
    st.markdown("""
    <div>
        <div class="legend-item"><span class="legend-dot" style="background:#cce5ff;"></span> Abbreviation</div>
        <div class="legend-item"><span class="legend-dot" style="background:#d4edda;"></span> Medical term</div>
        <div class="legend-item"><span class="legend-dot" style="background:#fff3cd;"></span> Non-medical hard word</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Helper functions ───

def build_highlighted_html(result: SimplificationResult) -> str:
    """Build HTML with highlighted replaced words and tooltips."""
    text = result.simplified_text

    # Build a map: for each unique simplified word, track the original + type
    # We need to match replacements to positions in the simplified text
    replacement_map: dict[str, dict] = {}
    for r in result.replacements:
        key = r.simplified.lower()
        if key not in replacement_map:
            replacement_map[key] = {
                "original": r.original,
                "is_abbreviation": r.is_abbreviation,
                "is_medical": r.is_medical,
            }

    # Simple approach: replace each simplified word with its highlighted version
    # Process longest replacements first to avoid partial matches
    sorted_replacements = sorted(replacement_map.keys(), key=len, reverse=True)

    for simplified_word in sorted_replacements:
        info = replacement_map[simplified_word]
        if info["is_abbreviation"]:
            css_class = "highlight-abbreviation"
        elif info["is_medical"]:
            css_class = "highlight-medical"
        else:
            css_class = "highlight-nonmedical"

        original = info["original"]
        # Escape HTML special chars in the original for the tooltip
        safe_original = html_module.escape(original)

        pattern = re.compile(re.escape(simplified_word), re.IGNORECASE)
        replacement_html = (
            f'<span class="{css_class}" data-original="Original: {safe_original}" '
            f'title="Original: {safe_original}">{simplified_word}</span>'
        )
        text = pattern.sub(replacement_html, text, count=0)

    return f'<div class="simplified-text">{text}</div>'


def _render_single_result(result: SimplificationResult, label: str):
    """Render a complete result view for a single approach."""
    st.subheader(f"{label} — Simplified Note")
    st.markdown(build_highlighted_html(result), unsafe_allow_html=True)

    st.divider()
    st.subheader("Word Details")
    _render_word_details(result)

    if result.mode != "baseline":
        st.divider()
        st.subheader("Attribution")
        _render_attribution(result)


def _render_simplified_output(result: SimplificationResult, label: str):
    """Render the simplified note with highlights."""
    st.subheader(f"{label} — Simplified Note")
    st.markdown(build_highlighted_html(result), unsafe_allow_html=True)

    st.divider()
    st.caption(f"Model: {result.model_used} | Mode: {result.mode} | Replacements: {len(set(r.original.lower() for r in result.replacements))}")


def _render_word_details(result: SimplificationResult):
    """Render the word-level replacement details table."""
    df = build_word_detail_df(result)
    if df.empty:
        st.info("No word replacements detected.")
        return

    col1, col2, col3 = st.columns(3)
    abbr_count = len(df[df["Type"] == "Abbreviation"])
    med_count = len(df[df["Type"] == "Medical"])
    nonmed_count = len(df[df["Type"] == "Non-medical"])

    with col1:
        st.metric("Abbreviations Expanded", abbr_count)
    with col2:
        st.metric("Medical Terms Simplified", med_count)
    with col3:
        st.metric("Non-medical Words Simplified", nonmed_count)

    st.divider()
    st.dataframe(df, width="stretch", hide_index=True, height=400)


def _render_attribution(result: SimplificationResult):
    """Render the attribution / provenance view."""
    df = build_attribution_df(result)
    if df.empty:
        st.info("No attribution data available.")
        return

    sources = df["Source"].unique()
    for source in sources:
        with st.expander(f"📚 {source}", expanded=True):
            source_df = df[df["Source"] == source]
            for _, row in source_df.iterrows():
                st.markdown(f"**{row['Word']}** → {row['Simplified']}")
                if row["URL"] and row["URL"] != "N/A":
                    st.markdown(f"🔗 Source: [{row['URL']}]({row['URL']})")
                if row["Source Passage"] and row["Source Passage"] != "N/A":
                    st.caption(f"📖 *{row['Source Passage']}*")
                st.markdown(f"💡 **Why:** {row['Rationale']}")
                st.divider()


def _render_metrics(original_text: str, rag_result=None, baseline_result=None):
    """Render evaluation metrics comparison."""
    st.subheader("Readability Metrics")

    metrics_data = {}
    if rag_result:
        rag_metrics = evaluate(original_text, rag_result.simplified_text)
        metrics_data["RAG"] = rag_metrics
    if baseline_result:
        baseline_metrics = evaluate(original_text, baseline_result.simplified_text)
        metrics_data["Baseline"] = baseline_metrics

    if not metrics_data:
        st.info("No results to evaluate.")
        return

    first_metrics = list(metrics_data.values())[0]

    table_rows = []
    for metric in first_metrics:
        row = {"Metric": metric.metric_name, "Original": round(metric.original_score, 2)}
        for approach, data in metrics_data.items():
            match = next((m for m in data if m.metric_name == metric.metric_name), None)
            if match:
                row[approach] = round(match.simplified_score, 2)
                row[f"{approach} Δ"] = round(match.delta, 2)
        table_rows.append(row)

    metrics_df = pd.DataFrame(table_rows)
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    if metrics_data:
        fig = build_metrics_chart({"Original": first_metrics, **metrics_data})
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Interpretation")
    st.markdown("""
    - **FKGL (Flesch-Kincaid Grade Level):** Lower is simpler. Target: ≤ 8.0 (8th-grade level).
    - **Dale-Chall:** Lower is simpler. Scores below 5.0 are readable by 4th-graders; 9.0+ is college-level.
    - **Flesch Reading Ease:** Higher is easier. 60-70 is 8th-9th grade; 90-100 is 5th-grade level.
    - **SARI:** Higher is better simplification quality (0-100). Measures how well edits simplify the text.
    """)


def build_word_detail_df(result: SimplificationResult) -> pd.DataFrame:
    """Build a DataFrame of word-level replacement details.

    For baseline mode, only shows Original Word, Simplified Word, Type.
    For RAG mode, also includes Source and Reason.
    """
    seen = set()
    rows = []
    for r in result.replacements:
        key = r.original.lower()
        if key in seen:
            continue
        seen.add(key)
        row = {
            "Original Word": r.original,
            "Simplified Word": r.simplified,
            "Type": "Abbreviation" if r.is_abbreviation else ("Medical" if r.is_medical else "Non-medical"),
        }
        if result.mode != "baseline":
            row["Source"] = r.source_name
            row["Reason"] = r.reason
        rows.append(row)
    return pd.DataFrame(rows)


def build_attribution_df(result: SimplificationResult) -> pd.DataFrame:
    """Build a DataFrame grouping replacements by source with links and passages."""
    seen = set()
    rows = []
    for r in result.replacements:
        key = r.original.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "Word": r.original,
            "Simplified": r.simplified,
            "Source": r.source_name,
            "URL": r.source_url if r.source_url else "N/A",
            "Source Passage": r.source_passage[:200] if r.source_passage else "N/A",
            "Rationale": r.reason,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Source")
    return df


def build_metrics_chart(metrics_data: dict) -> go.Figure:
    """Build a grouped bar chart comparing metrics across approaches."""
    metrics = ["FKGL (Grade Level)", "Dale-Chall", "Flesch Reading Ease"]

    fig = go.Figure()

    for approach, data in metrics_data.items():
        values = []
        for m in metrics:
            match = next((r for r in data if r.metric_name == m), None)
            if match:
                values.append(match.simplified_score if approach != "Original" else match.original_score)
            else:
                values.append(0)
        fig.add_trace(go.Bar(name=approach, x=metrics, y=values))

    fig.update_layout(
        barmode="group",
        title="Readability Metrics Comparison",
        yaxis_title="Score",
        height=400,
    )
    return fig


# ─── Main area ───
st.title("🏥 Clinical Note Simplifier")
st.markdown("Simplify clinical notes for patient understanding using RAG with external medical sources or LLM-only baseline.")

# Input
clinical_note = st.text_area(
    "Paste your clinical note here:",
    height=200,
    placeholder="e.g., Pt is a 65 y/o male c/o SOB and CP. PMH significant for HTN, DM, and COPD. BP 145/92, HR 88 BPM. Dx: acute exacerbation of CHF...",
)

col_btn, col_sample = st.columns([1, 3])
with col_btn:
    simplify_btn = st.button("🔬 Simplify", type="primary", width="stretch")
with col_sample:
    if st.button("📋 Load Sample Note"):
        st.session_state["sample_note"] = (
            "Pt is a 65 y/o male presenting to the ER c/o SOB and CP radiating to the left arm. "
            "PMH significant for HTN, DM, COPD, and hyperlipidemia. PSH includes CABG in 2018. "
            "Current medications include metformin, lisinopril, atorvastatin, and albuterol PRN. "
            "VS: BP 145/92, HR 88 BPM, RR 22, T 98.6F, O2 sat 94% on RA. "
            "HEENT: PERRLA, oropharynx clear. Cardiovascular exam reveals RRR with S3 gallop, "
            "bilateral lower extremity edema. Lungs with bilateral basilar crackles. "
            "Labs: BUN 35, Cr 1.8, BNP elevated at 1200, troponin negative. "
            "CXR shows cardiomegaly with pulmonary congestion. ECG shows NSR with LVH. "
            "Assessment: Acute exacerbation of CHF, likely secondary to medication noncompliance. "
            "Plan: Admit to telemetry, IV furosemide, strict I&O, salt-restricted diet, "
            "cardiology consult, repeat echocardiogram, titrate ACE inhibitor."
        )
        st.rerun()

# Use sample note if loaded
if "sample_note" in st.session_state:
    clinical_note = st.session_state.pop("sample_note")
    st.session_state["clinical_note_value"] = clinical_note
    # Re-render with the sample text
    st.rerun()

if "clinical_note_value" in st.session_state and not clinical_note:
    clinical_note = st.session_state.pop("clinical_note_value")

# ─── Run simplification ───
if simplify_btn and clinical_note:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    groq = GroqClient(api_key=api_key, model=selected_model)

    # Build enabled sources
    enabled_sources = []
    if use_freedict:
        enabled_sources.append(FreeDictionarySource(groq_client=groq))
    if use_medline:
        enabled_sources.append(MedlinePlusSource(groq_client=groq))
    if use_wikipedia:
        enabled_sources.append(WikipediaSource(groq_client=groq))

    rag_result = None
    baseline_result = None

    # Build abbreviation dict based on checkbox
    if use_csv_abbrevs:
        abbreviations = load_abbreviations_csv()
    else:
        abbreviations = load_abbreviations()

    # --- RAG Mode ---
    if mode in ("RAG Mode", "Compare Both"):
        with st.spinner("Running RAG pipeline..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def rag_progress(step_name, pct):
                progress_bar.progress(pct / 100)
                status_text.text(step_name)

            rag_result = run_rag_pipeline(
                clinical_note, groq, enabled_sources,
                abbreviations=abbreviations,
                progress_callback=rag_progress,
            )
            progress_bar.empty()
            status_text.empty()

    # --- Baseline Mode ---
    if mode in ("Baseline Mode", "Compare Both"):
        with st.spinner("Running Baseline pipeline..."):
            progress_bar2 = st.progress(0)
            status_text2 = st.empty()

            def baseline_progress(step_name, pct):
                progress_bar2.progress(pct / 100)
                status_text2.text(step_name)

            baseline_result = run_baseline_pipeline(
                clinical_note, groq,
                abbreviations=abbreviations,
                progress_callback=baseline_progress,
            )
            progress_bar2.empty()
            status_text2.empty()

    # Store results in session state
    st.session_state["rag_result"] = rag_result
    st.session_state["baseline_result"] = baseline_result
    st.session_state["original_text"] = clinical_note

# ─── Display results ───
rag_result = st.session_state.get("rag_result")
baseline_result = st.session_state.get("baseline_result")
original_text = st.session_state.get("original_text", "")

if rag_result or baseline_result:
    # Determine which results to show
    results_to_show = {}
    if rag_result:
        results_to_show["RAG"] = rag_result
    if baseline_result:
        results_to_show["Baseline"] = baseline_result

    # --- Compare Both layout ---
    if len(results_to_show) == 2:
        tab_compare, tab_rag, tab_baseline, tab_metrics = st.tabs([
            "📊 Comparison", "🔍 RAG Details", "📝 Baseline Details", "📈 Metrics"
        ])

        with tab_compare:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("RAG Output")
                st.markdown(build_highlighted_html(rag_result), unsafe_allow_html=True)
            with col2:
                st.subheader("Baseline Output")
                st.markdown(
                    f'<div class="simplified-text">{baseline_result.simplified_text}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            st.subheader("Word Replacement Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("RAG Replacements")
                st.dataframe(build_word_detail_df(rag_result), width="stretch", hide_index=True)
            with col2:
                st.caption("Baseline Replacements")
                st.dataframe(build_word_detail_df(baseline_result), width="stretch", hide_index=True)

        with tab_rag:
            _render_single_result(rag_result, "RAG")

        with tab_baseline:
            _render_single_result(baseline_result, "Baseline")

        with tab_metrics:
            _render_metrics(original_text, rag_result, baseline_result)

    else:
        # Single mode
        result = list(results_to_show.values())[0]
        result_name = list(results_to_show.keys())[0]

        if result.mode == "baseline":
            tab_output, tab_words, tab_metrics = st.tabs([
                "📄 Simplified Note", "📝 Word Details", "📈 Metrics"
            ])
        else:
            tab_output, tab_words, tab_attribution, tab_metrics = st.tabs([
                "📄 Simplified Note", "📝 Word Details", "🔗 Attribution", "📈 Metrics"
            ])

        with tab_output:
            _render_simplified_output(result, result_name)

        with tab_words:
            _render_word_details(result)

        if result.mode != "baseline":
            with tab_attribution:
                _render_attribution(result)

        with tab_metrics:
            _render_metrics(original_text, rag_result, baseline_result)


elif not clinical_note:
    st.info("👆 Paste a clinical note above and click **Simplify** to get started.")
