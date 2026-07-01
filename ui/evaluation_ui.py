"""
ui/evaluation_ui.py
--------------------
Streamlit evaluation dashboard for the Contract Intelligence System.

Run with:
    streamlit run ui/evaluation_ui.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---- Add project root to sys.path so evaluation/ and docs/ are importable ----
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env")

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="Contract Intelligence Evaluator",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Global CSS ----
st.markdown(
    """
    <style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #4C9BE8;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logger = logging.getLogger(__name__)

# ---- Colour palette ----
COLOURS = {
    "KnowledgeRAG":    "#4C9BE8",
    "ComparisonAgent": "#F4A261",
    "ComplianceAgent": "#E63946",
    "GeneralQAAgent":  "#2A9D8F",
    "System":          "#8338EC",
    "IngestionPipeline": "#FB8500",
    "IntentDetection": "#264653",
}


def _status_emoji(score: float, inverted: bool = False) -> str:
    if inverted:
        score = 1.0 - score
    if score >= 0.80:
        return "🟢 Good"
    if score >= 0.60:
        return "🟡 Fair"
    return "🔴 Poor"


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "report": None,
        "samples": [],
        "chunks": [],
        "vector_store": None,
        "history": [],
        "page": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()

# ---------------------------------------------------------------------------
# Plotly imports (local, no CDN)
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")

    # --- Document Upload ---
    st.subheader("📄 Document Upload")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF contracts",
        type=["pdf"],
        accept_multiple_files=True,
        key="uploaded_pdfs",
    )
    if uploaded_pdfs:
        st.success(f"✅ {len(uploaded_pdfs)} PDF(s) ready")

    # --- API Settings ---
    st.subheader("🔑 API Settings")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        key="api_key_input",
    )
    if env_key:
        st.caption("✅ Loaded from .env")

    # --- Evaluation Settings ---
    st.subheader("⚡ Evaluation Settings")
    use_ragas = st.toggle("Enable RAGAS metrics", value=True, key="use_ragas")
    st.caption("Requires OpenAI API key. Disable for fast local run.")
    sla_ms = st.slider(
        "Latency SLA (ms)", min_value=500, max_value=5000,
        value=2000, step=100, key="sla_ms",
    )
    include_secondary = st.toggle("Include secondary metrics", value=True, key="include_secondary")
    top_k = st.number_input("Top-K retrieval", min_value=1, max_value=20, value=5, key="top_k")

    # --- Test Queries ---
    st.subheader("🧪 Test Queries")
    default_queries = (
        "What is the governing law in this contract?\n"
        "Compare the termination clauses across all uploaded contracts.\n"
        "Is this contract compliant with GDPR requirements?\n"
        "What are the payment terms and any late payment penalties?\n"
        "Which contracts expire within 90 days and do they have renewal options?"
    )
    test_queries_text = st.text_area(
        "Queries (one per line)",
        value=default_queries,
        height=160,
        key="test_queries_text",
    )
    run_button = st.button("▶ Run Evaluation", type="primary", key="run_button")

# ---------------------------------------------------------------------------
# Run pipeline on button click
# ---------------------------------------------------------------------------

if run_button:
    # --- Validation ---
    if not uploaded_pdfs:
        st.warning("Please upload at least one PDF contract.")
        st.stop()

    active_api_key = st.session_state.get("api_key_input", "") or env_key
    if not active_api_key:
        st.error("OpenAI API key not found. Add it to .env or the sidebar.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = active_api_key

    try:
        # ---- Step 1: Process PDFs ----
        with st.spinner("Processing PDFs…"):
            # Import only langchain_core.documents — safe, no torch dependency.
            # We do NOT import from docs.chunking because it imports
            # langchain_text_splitters, whose __init__ eagerly loads
            # SentenceTransformersTokenTextSplitter → sentence_transformers →
            # torch → c10.dll, which fails on this machine.
            import pypdf
            from langchain_core.documents import Document

            def _chunk_text(
                text: str, chunk_size: int = 1000, chunk_overlap: int = 200
            ) -> list:
                """
                Lightweight recursive-character chunker.
                Produces the same metadata-compatible Document list as
                DocumentChunker, with zero torch/sentence-transformers deps.
                """
                if not text.strip():
                    return []
                if len(text) <= chunk_size:
                    return [text.strip()]
                chunks = []
                start = 0
                n = len(text)
                while start < n:
                    end = min(start + chunk_size, n)
                    piece = text[start:end]
                    # Try to end at a natural boundary
                    if end < n:
                        for sep in ("\n\n", "\n", ". ", " "):
                            idx = piece.rfind(sep)
                            if idx > chunk_size // 2:
                                end = start + idx + len(sep)
                                piece = text[start:end]
                                break
                    stripped = piece.strip()
                    if stripped:
                        chunks.append(stripped)
                    # Advance with overlap, but always make forward progress
                    next_start = end - chunk_overlap
                    start = next_start if next_start > start else start + 1
                return chunks

            all_chunks = []
            for pdf_file in uploaded_pdfs:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name
                try:
                    reader = pypdf.PdfReader(tmp_path)
                    name_stem = pdf_file.name
                    if name_stem.lower().endswith(".pdf"):
                        name_stem = name_stem[:-4]
                    doc_id = name_stem.replace(" ", "_").lower()
                    chunk_idx = 0
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        for piece in _chunk_text(page_text):
                            all_chunks.append(
                                Document(
                                    page_content=piece,
                                    metadata={
                                        "doc_id":      doc_id,
                                        "filename":    pdf_file.name,
                                        "page":        page_num + 1,
                                        "chunk_index": chunk_idx,
                                        "chunk_size":  len(piece),
                                        "source":      pdf_file.name,
                                    },
                                )
                            )
                            chunk_idx += 1
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            st.session_state["chunks"] = all_chunks

        # ---- Step 2: Build FAISS index ----
        with st.spinner("Building FAISS index…"):
            from docs.embedding import OpenAIEmbedder, EmbeddingConfig
            from docs.vector_store import FAISSVectorStore

            embedder = OpenAIEmbedder(EmbeddingConfig(model="text-embedding-3-small"))
            vs = FAISSVectorStore(
                index_dir=str(_ROOT / "faiss_index_eval"),
                embedding_model=embedder.embedding_model,
            )
            if all_chunks:
                vs.add_documents(all_chunks)
            st.session_state["vector_store"] = vs

        # ---- Step 3: Run queries through pipeline ----
        with st.spinner("Running queries through pipeline…"):
            from openai import OpenAI
            from src.agents.intent_detection import IntentDetector, Workflow
            from src.agents.knowledge_rag import KnowledgeRAG
            from src.agents.agentic_rag import AgenticRAG
            from evaluation.base import EvalSample

            oai_client = OpenAI(api_key=active_api_key)
            detector = IntentDetector()
            knowledge_rag = KnowledgeRAG(oai_client, vs, top_k=int(top_k))
            agentic_rag = AgenticRAG(oai_client, vs, top_k=int(top_k))

            raw_queries = [
                q.strip() for q in test_queries_text.splitlines() if q.strip()
            ]
            samples = []
            ragas_warn_shown: set[str] = set()

            for query in raw_queries:
                t_start = time.perf_counter()
                try:
                    intent_result = detector.detect(query)
                    if intent_result.workflow == Workflow.KNOWLEDGE_RAG:
                        result = knowledge_rag.run(query)
                        subquestions = []
                    else:
                        result = agentic_rag.run(query)
                        subquestions = result.get("subquestions", [])

                    elapsed_ms = (time.perf_counter() - t_start) * 1000

                    # Normalise sources to always include doc_id + page
                    raw_sources = result.get("sources", [])
                    norm_sources = []
                    for src in raw_sources:
                        norm_sources.append({
                            "doc_id": src.get("doc_id", ""),
                            "page": str(src.get("page", "")),
                            "source": src.get("source", ""),
                            "content": src.get("content", ""),
                        })

                    contexts = [s.get("content", "") for s in norm_sources if s.get("content")]

                    sample = EvalSample(
                        query=query,
                        answer=result.get("answer", ""),
                        ground_truth="",
                        contexts=contexts,
                        sources=norm_sources,
                        predicted_intent=intent_result.workflow.value,
                        predicted_task=intent_result.task.value,
                        tool_calls=subquestions,
                        response_time_ms=elapsed_ms,
                        metadata={"confidence": intent_result.confidence},
                    )
                    samples.append(sample)

                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t_start) * 1000
                    st.warning(f"Query failed: '{query[:60]}…' — {exc}")
                    samples.append(
                        EvalSample(
                            query=query,
                            answer="",
                            response_time_ms=elapsed_ms,
                        )
                    )

        # ---- Step 4: Compute evaluation metrics ----
        with st.spinner("Computing evaluation metrics…"):
            from evaluation.run_evaluation import EvaluationRunner

            runner = EvaluationRunner(
                use_ragas=use_ragas,
                latency_sla_ms=float(sla_ms),
                verbose=False,
            )
            report = runner.run(
                samples=samples,
                chunks=st.session_state["chunks"],
                expected_doc_ids=list({
                    s.get("doc_id", "")
                    for spl in samples
                    for s in spl.sources
                    if s.get("doc_id")
                }),
                vector_store=st.session_state["vector_store"],
            )

        # ---- Step 5: Store results ----
        st.session_state["report"] = report
        st.session_state["samples"] = samples
        st.session_state["history"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "scores": report["scores"],
        })
        st.session_state["page"] = 0
        st.success(f"Evaluation complete! Ran {len(samples)} queries.")
        st.rerun()

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Helper: safely get a score from the report
# ---------------------------------------------------------------------------

def _get_score(report: dict, component: str, metric: str, default: float = 0.0) -> float:
    try:
        return float(
            report["results"][component]["metrics"].get(metric, default)
        )
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Main area — 5 tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🧩 Component Deep-Dive",
    "📝 Sample Inspector",
    "📈 Trends & Comparisons",
    "⬇️ Export",
])

# ===========================================================================
# TAB 1 — Overview
# ===========================================================================

with tab1:
    report = st.session_state.get("report")

    if report is None:
        st.info(
            "📋 Upload PDFs and click **▶ Run Evaluation** to get started.\n\n"
            "This dashboard evaluates your Contract Intelligence pipeline across "
            "intent detection, retrieval quality, citation accuracy, and more."
        )
    else:
        # ---- KPI cards ----
        intent_acc = _get_score(report, "IntentDetection", "workflow_accuracy")
        faithfulness = _get_score(report, "KnowledgeRAG", "faithfulness")
        citation_f1 = _get_score(report, "KnowledgeRAG", "citation_f1")
        hallucination = _get_score(report, "System", "hallucination_rate")

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            st.metric(
                "Intent Accuracy",
                f"{intent_acc*100:.1f}%",
                delta=None,
                help="Fraction of queries routed to correct workflow",
            )
        with col2:
            st.metric(
                "RAG Faithfulness",
                f"{faithfulness*100:.1f}%",
                help="RAGAS faithfulness (answer grounded in context)",
            )
        with col3:
            st.metric(
                "Citation F1",
                f"{citation_f1*100:.1f}%",
                help="Harmonic mean of citation precision & recall",
            )
        with col4:
            delta_color = "inverse"  # red if high hallucination
            st.metric(
                "Hallucination Rate",
                f"{hallucination*100:.1f}%",
                delta=f"{'⚠️' if hallucination > 0.2 else '✅'}",
                delta_color=delta_color,
                help="Fraction of answers with no supporting context",
            )

        if _PLOTLY:
            # ---- Radar chart ----
            radar_axes = [
                "Faithfulness", "Context Precision", "Context Recall",
                "Answer Relevancy", "Answer Correctness", "Citation F1",
                "Semantic Similarity",
            ]
            radar_keys = [
                "faithfulness", "context_precision", "context_recall",
                "answer_relevancy", "answer_correctness", "citation_f1",
                "semantic_similarity",
            ]
            component_radar = {
                "KnowledgeRAG":    COLOURS["KnowledgeRAG"],
                "ComparisonAgent": COLOURS["ComparisonAgent"],
                "ComplianceAgent": COLOURS["ComplianceAgent"],
                "GeneralQAAgent":  COLOURS["GeneralQAAgent"],
            }

            fig_radar = go.Figure()
            for comp, colour in component_radar.items():
                comp_metrics = report["results"].get(comp, {}).get("metrics", {})
                if not comp_metrics:
                    continue
                vals = [float(comp_metrics.get(k, 0.0)) for k in radar_keys]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=radar_axes + [radar_axes[0]],
                        name=comp,
                        fill="toself",
                        line=dict(color=colour),
                        opacity=0.6,
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_white",
                title="Component Quality Radar",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            )
            st.plotly_chart(fig_radar, width='stretch')

            # ---- System health bar chart ----
            sys_metrics_raw = report["results"].get("System", {}).get("metrics", {})
            bar_items = [
                ("Intent Routing", sys_metrics_raw.get("intent_routing_accuracy", 0.0), False),
                ("Citation Overall", sys_metrics_raw.get("citation_accuracy_overall", 0.0), False),
                ("SLA Compliance", sys_metrics_raw.get("sla_compliance_rate", 0.0), False),
                ("Error-free Rate", 1.0 - sys_metrics_raw.get("error_rate", 0.0), False),
                ("Hallucination-free", 1.0 - sys_metrics_raw.get("hallucination_rate", 0.0), False),
            ]
            bar_colours = [
                "#2A9D8F" if v >= 0.80 else ("#F4A261" if v >= 0.60 else "#E63946")
                for _, v, _ in bar_items
            ]
            fig_health = go.Figure(
                go.Bar(
                    x=[v for _, v, _ in bar_items],
                    y=[label for label, _, _ in bar_items],
                    orientation="h",
                    marker=dict(color=bar_colours),
                    text=[f"{v*100:.1f}%" for _, v, _ in bar_items],
                    textposition="outside",
                )
            )
            fig_health.update_layout(
                template="plotly_white",
                title="System Health",
                height=350,
                xaxis=dict(range=[0, 1.15]),
            )
            st.plotly_chart(fig_health, width='stretch')

            # ---- Response time ----
            p50 = sys_metrics_raw.get("response_time_p50_ms", 0.0)
            p95 = sys_metrics_raw.get("response_time_p95_ms", 0.0)
            p99 = sys_metrics_raw.get("response_time_p99_ms", 0.0)

            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric("p50 Latency", f"{p50:.0f} ms")
            with tc2:
                st.metric("p95 Latency", f"{p95:.0f} ms")
            with tc3:
                st.metric("p99 Latency", f"{p99:.0f} ms")

            fig_lat = go.Figure(
                go.Bar(
                    x=["p50", "p95", "p99"],
                    y=[p50, p95, p99],
                    marker=dict(color=["#4C9BE8", "#F4A261", "#E63946"]),
                    text=[f"{v:.0f} ms" for v in [p50, p95, p99]],
                    textposition="outside",
                )
            )
            fig_lat.add_hline(
                y=float(sla_ms),
                line_dash="dash",
                line_color="red",
                annotation_text=f"SLA {sla_ms} ms",
            )
            fig_lat.update_layout(
                template="plotly_white",
                title="Response Time Distribution",
                yaxis_title="ms",
                height=350,
            )
            st.plotly_chart(fig_lat, width='stretch')
        else:
            st.info("Install plotly for interactive charts: pip install plotly")

# ===========================================================================
# TAB 2 — Component Deep-Dive
# ===========================================================================

with tab2:
    report = st.session_state.get("report")
    if report is None:
        st.info("Run evaluation first.")
    else:
        component_choice = st.selectbox(
            "Select component",
            [
                "Intent Detection",
                "Knowledge RAG",
                "Comparison Agent",
                "Compliance Agent",
                "General QA Agent",
                "Ingestion Pipeline",
            ],
            key="component_choice",
        )

        comp_key_map = {
            "Intent Detection":   "IntentDetection",
            "Knowledge RAG":      "KnowledgeRAG",
            "Comparison Agent":   "ComparisonAgent",
            "Compliance Agent":   "ComplianceAgent",
            "General QA Agent":   "GeneralQAAgent",
            "Ingestion Pipeline": "IngestionPipeline",
        }
        comp_key = comp_key_map[component_choice]
        comp_data = report["results"].get(comp_key, {})
        comp_metrics = comp_data.get("metrics", {})
        comp_details = comp_data.get("details", {})
        comp_errors = comp_data.get("errors", [])

        if comp_errors and not comp_metrics:
            st.warning(f"No {component_choice} samples in this run: {comp_errors}")
        elif not comp_metrics:
            st.info(f"No metrics available for {component_choice}.")
        else:
            col_left, col_right = st.columns([3, 2])

            with col_left:
                # ---- Metrics table ----
                if _PANDAS:
                    INVERTED = {"error_rate", "hallucination_rate", "orphan_chunk_rate",
                                "undersized_chunk_rate", "oversized_chunk_rate",
                                "no_context_answer_rate", "issue_false_positive_rate",
                                "inter_chunk_separation"}
                    rows = []
                    for metric, val in comp_metrics.items():
                        inv = metric in INVERTED
                        status = _status_emoji(val, inverted=inv)
                        rows.append({
                            "Metric": metric,
                            "Score": round(float(val), 4),
                            "Status": status,
                        })
                    df = pd.DataFrame(rows).reset_index(drop=True)
                    st.subheader("Metrics Table")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # ---- Bar chart ----
                if _PLOTLY and comp_metrics:
                    sorted_items = sorted(
                        comp_metrics.items(), key=lambda x: float(x[1]), reverse=True
                    )
                    names = [k for k, _ in sorted_items]
                    vals = [float(v) for _, v in sorted_items]
                    bar_clrs = [
                        "#2A9D8F" if v >= 0.80 else ("#F4A261" if v >= 0.60 else "#E63946")
                        for v in vals
                    ]
                    fig_bar = go.Figure(
                        go.Bar(
                            x=vals,
                            y=names,
                            orientation="h",
                            marker=dict(color=bar_clrs),
                            text=[f"{v:.3f}" for v in vals],
                            textposition="outside",
                        )
                    )
                    fig_bar.add_vline(
                        x=0.8, line_dash="dash", line_color="red",
                        annotation_text="0.8 threshold",
                    )
                    fig_bar.update_layout(
                        template="plotly_white",
                        title=f"{component_choice} Metrics",
                        height=max(350, len(names) * 28),
                        xaxis=dict(range=[0, max(max(vals) * 1.2, 1.1)]),
                    )
                    st.plotly_chart(fig_bar, width='stretch')

            with col_right:
                # ---- Component-specific extras ----

                if comp_key == "IntentDetection" and _PLOTLY:
                    for cm_key, cm_title in [
                        ("workflow_confusion_matrix", "Workflow Confusion Matrix"),
                        ("task_confusion_matrix", "Task Confusion Matrix"),
                    ]:
                        cm_data = comp_details.get(cm_key)
                        if cm_data:
                            labels = cm_data.get("labels", [])
                            matrix = cm_data.get("matrix", [])
                            if matrix and labels:
                                fig_cm = go.Figure(
                                    go.Heatmap(
                                        z=matrix,
                                        x=labels,
                                        y=labels,
                                        colorscale="Blues",
                                        text=matrix,
                                        texttemplate="%{text}",
                                        showscale=False,
                                    )
                                )
                                fig_cm.update_layout(
                                    template="plotly_white",
                                    title=cm_title,
                                    height=350,
                                    xaxis_title="Predicted",
                                    yaxis_title="True",
                                )
                                st.plotly_chart(fig_cm, width='stretch')

                    for report_key in ["workflow_classification_report", "task_classification_report"]:
                        report_str = comp_details.get(report_key)
                        if report_str:
                            st.subheader(report_key.replace("_", " ").title())
                            st.code(report_str, language=None)

                elif comp_key == "KnowledgeRAG" and _PLOTLY:
                    sim_data = comp_details.get("semantic_similarity_per_sample", [])
                    if sim_data:
                        scores = [d.get("score", 0.0) for d in sim_data]
                        fig_hist = go.Figure(
                            go.Histogram(x=scores, nbinsx=20, marker_color="#4C9BE8")
                        )
                        fig_hist.update_layout(
                            template="plotly_white",
                            title="Semantic Similarity Distribution",
                            xaxis_title="Score",
                            height=350,
                        )
                        st.plotly_chart(fig_hist, width='stretch')

                    # Citation breakdown
                    cp = comp_metrics.get("citation_precision", 0.0)
                    cr = comp_metrics.get("citation_recall", 0.0)
                    cf = comp_metrics.get("citation_f1", 0.0)
                    fig_cit = go.Figure(
                        go.Bar(
                            x=["Precision", "Recall", "F1"],
                            y=[cp, cr, cf],
                            marker=dict(color=["#4C9BE8", "#F4A261", "#2A9D8F"]),
                            text=[f"{v:.3f}" for v in [cp, cr, cf]],
                            textposition="outside",
                        )
                    )
                    fig_cit.update_layout(
                        template="plotly_white",
                        title="Citation Accuracy",
                        yaxis=dict(range=[0, 1.15]),
                        height=350,
                    )
                    st.plotly_chart(fig_cit, width='stretch')

                elif comp_key == "ComparisonAgent" and _PLOTLY:
                    tc_p = comp_metrics.get("tool_call_precision", 0.0)
                    tc_r = comp_metrics.get("tool_call_recall", 0.0)
                    tc_f = comp_metrics.get("tool_call_f1", 0.0)
                    fig_gauges = go.Figure()
                    for i, (name, val) in enumerate([
                        ("Precision", tc_p), ("Recall", tc_r), ("F1", tc_f)
                    ]):
                        fig_gauges.add_trace(
                            go.Indicator(
                                mode="gauge+number",
                                value=val,
                                title={"text": name},
                                gauge={
                                    "axis": {"range": [0, 1]},
                                    "bar": {"color": COLOURS["ComparisonAgent"]},
                                    "threshold": {
                                        "line": {"color": "red", "width": 2},
                                        "thickness": 0.75,
                                        "value": 0.8,
                                    },
                                },
                                domain={
                                    "x": [i / 3, (i + 1) / 3 - 0.05],
                                    "y": [0, 1],
                                },
                            )
                        )
                    fig_gauges.update_layout(
                        template="plotly_white",
                        title="Tool Call Metrics",
                        height=350,
                    )
                    st.plotly_chart(fig_gauges, width='stretch')
                    st.metric(
                        "Multi-doc Citation Rate",
                        f"{comp_metrics.get('multi_doc_citation_rate', 0.0)*100:.1f}%",
                    )

                elif comp_key == "ComplianceAgent" and _PLOTLY:
                    sev_data = {
                        "High": comp_metrics.get("severity_accuracy_high", 0.0),
                        "Medium": comp_metrics.get("severity_accuracy_medium", 0.0),
                        "Low": comp_metrics.get("severity_accuracy_low", 0.0),
                    }
                    fig_sev = go.Figure(
                        go.Bar(
                            x=list(sev_data.values()),
                            y=list(sev_data.keys()),
                            orientation="h",
                            marker=dict(color=["#E63946", "#F4A261", "#2A9D8F"]),
                            text=[f"{v*100:.1f}%" for v in sev_data.values()],
                            textposition="outside",
                        )
                    )
                    fig_sev.update_layout(
                        template="plotly_white",
                        title="Severity Accuracy by Level",
                        xaxis=dict(range=[0, 1.2]),
                        height=350,
                    )
                    st.plotly_chart(fig_sev, width='stretch')

                    # Precision/Recall scatter
                    ip = comp_metrics.get("issue_precision", 0.0)
                    ir = comp_metrics.get("issue_recall", 0.0)
                    fig_scatter = go.Figure(
                        go.Scatter(
                            x=[ip],
                            y=[ir],
                            mode="markers+text",
                            text=["Issues P/R"],
                            textposition="top center",
                            marker=dict(size=18, color=COLOURS["ComplianceAgent"]),
                        )
                    )
                    fig_scatter.update_layout(
                        template="plotly_white",
                        title="Issue Detection P vs R",
                        xaxis=dict(title="Precision", range=[0, 1.1]),
                        yaxis=dict(title="Recall", range=[0, 1.1]),
                        height=350,
                    )
                    st.plotly_chart(fig_scatter, width='stretch')

                elif comp_key == "GeneralQAAgent":
                    mhc = comp_metrics.get("multi_hop_coverage", 0.0)
                    st.subheader("Multi-hop Coverage")
                    st.progress(float(mhc), text=f"{mhc*100:.1f}%")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric(
                            "Avg Subquestions / Query",
                            f"{comp_metrics.get('avg_subquestions_per_query', 0.0):.1f}",
                        )
                    with c2:
                        st.metric(
                            "No-Context Answer Rate",
                            f"{comp_metrics.get('no_context_answer_rate', 0.0)*100:.1f}%",
                        )

                elif comp_key == "IngestionPipeline" and _PLOTLY:
                    cbq = comp_metrics.get("chunk_boundary_quality", 0.0)
                    intra = comp_metrics.get("intra_chunk_coherence", 0.0)
                    inter = comp_metrics.get("inter_chunk_separation", 0.0)
                    fig_cbq = go.Figure(
                        go.Bar(
                            x=["CBQ", "Intra-Coherence", "Inter-Separation"],
                            y=[cbq, intra, inter],
                            marker=dict(color=["#FB8500", "#4C9BE8", "#E63946"]),
                            text=[f"{v:.3f}" for v in [cbq, intra, inter]],
                            textposition="outside",
                        )
                    )
                    fig_cbq.update_layout(
                        template="plotly_white",
                        title="Chunk Boundary Quality",
                        yaxis=dict(range=[0, 1.2]),
                        height=350,
                    )
                    st.plotly_chart(fig_cbq, width='stretch')

                    size_dist = comp_details.get("size_distribution", {})
                    if size_dist:
                        dist_keys = ["min", "p25", "p50", "p75", "max"]
                        dist_vals = [size_dist.get(k, 0) for k in dist_keys]
                        fig_dist = go.Figure(
                            go.Bar(
                                x=dist_keys,
                                y=dist_vals,
                                marker=dict(color="#FB8500"),
                                text=[f"{int(v)}" for v in dist_vals],
                                textposition="outside",
                            )
                        )
                        fig_dist.update_layout(
                            template="plotly_white",
                            title="Chunk Size Distribution",
                            yaxis_title="Characters",
                            height=350,
                        )
                        st.plotly_chart(fig_dist, width='stretch')

# ===========================================================================
# TAB 3 — Sample Inspector
# ===========================================================================

with tab3:
    samples_list = st.session_state.get("samples", [])
    report = st.session_state.get("report")

    if not samples_list:
        st.info("Run evaluation first to see sample details.")
    else:
        col_filter, col_search = st.columns([1, 2])
        with col_filter:
            filter_comp = st.selectbox(
                "Filter by component",
                ["All", "KnowledgeRAG", "AgenticRAG"],
                key="filter_comp",
            )
        with col_search:
            search_text = st.text_input("Search queries", key="search_text", placeholder="Type to filter...")

        # Filter
        filtered_samples = samples_list
        if filter_comp != "All":
            filtered_samples = [
                s for s in filtered_samples
                if s.predicted_intent == filter_comp
            ]
        if search_text:
            filtered_samples = [
                s for s in filtered_samples
                if search_text.lower() in s.query.lower()
            ]

        PAGE_SIZE = 10
        total_pages = max(1, (len(filtered_samples) + PAGE_SIZE - 1) // PAGE_SIZE)
        current_page = st.session_state.get("page", 0)
        current_page = min(current_page, total_pages - 1)

        page_samples = filtered_samples[current_page * PAGE_SIZE:(current_page + 1) * PAGE_SIZE]

        st.caption(
            f"Showing {len(page_samples)} of {len(filtered_samples)} samples "
            f"(page {current_page + 1}/{total_pages})"
        )

        for s in page_samples:
            with st.expander(f"🔍 {s.query[:80]}", expanded=False):
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**Query:** {s.query}")
                    intent_badge = s.predicted_intent or "—"
                    task_badge = s.predicted_task or "—"
                    st.markdown(
                        f"**Intent:** `{intent_badge}`  |  **Task:** `{task_badge}`"
                    )
                    if s.answer:
                        st.markdown(
                            f"<div style='background:#f0f2f6;padding:10px;"
                            f"border-radius:6px;font-size:0.9em'>{s.answer}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("No answer generated.")

                with right:
                    if s.ground_truth:
                        with st.expander("Ground Truth"):
                            st.write(s.ground_truth)

                    if report:
                        from evaluation.base import CitationChecker, SemanticSimilarity
                        # Per-sample scores
                        try:
                            sim = SemanticSimilarity.score(s.answer, s.ground_truth) if (s.answer and s.ground_truth) else 0.0
                            st.metric("Semantic Similarity", f"{sim:.3f}")
                        except Exception:
                            st.metric("Semantic Similarity", "—")

                        cit_score = 0.0
                        try:
                            cit = CitationChecker.score(s.answer, s.sources)
                            cit_score = cit.get("citation_f1", 0.0)
                            for src in s.sources:
                                doc_id = src.get("doc_id", "")
                                page = str(src.get("page", ""))
                                cited_docs = {c["doc_id"] for c in CitationChecker.parse_citations(s.answer)}
                                icon = "✅" if doc_id in cited_docs else "❌"
                                st.caption(f"{icon} {doc_id} p.{page}")
                        except Exception:
                            pass

                    if s.response_time_ms is not None:
                        colour = "green" if s.response_time_ms <= float(sla_ms) else "red"
                        st.markdown(
                            f"<span style='color:{colour};font-weight:bold'>"
                            f"⏱ {s.response_time_ms:.0f} ms</span>",
                            unsafe_allow_html=True,
                        )

        # Pagination buttons
        pn1, pn2, pn3 = st.columns([1, 6, 1])
        with pn1:
            if st.button("◀ Prev", key="page_prev", disabled=current_page == 0):
                st.session_state["page"] = current_page - 1
                st.rerun()
        with pn3:
            if st.button("Next ▶", key="page_next", disabled=current_page >= total_pages - 1):
                st.session_state["page"] = current_page + 1
                st.rerun()

# ===========================================================================
# TAB 4 — Trends & Comparisons
# ===========================================================================

with tab4:
    history = st.session_state.get("history", [])

    if len(history) < 2:
        st.info("Run evaluation at least twice to see trends.")
        if history:
            st.caption(f"Current run: {history[-1].get('timestamp', '—')}")
    else:
        trend_metrics = [
            "IntentDetection.workflow_accuracy",
            "KnowledgeRAG.faithfulness",
            "KnowledgeRAG.citation_f1",
            "System.hallucination_rate",
            "System.sla_compliance_rate",
        ]
        timestamps = [h["timestamp"] for h in history]

        if _PLOTLY:
            fig_trend = go.Figure()
            for metric in trend_metrics:
                vals = [h["scores"].get(metric, 0.0) for h in history]
                comp = metric.split(".")[0]
                colour = COLOURS.get(comp, "#555555")
                fig_trend.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=vals,
                        mode="lines+markers",
                        name=metric,
                        line=dict(color=colour),
                    )
                )
            fig_trend.update_layout(
                template="plotly_white",
                title="Metric Trends Over Runs",
                xaxis_title="Run",
                yaxis=dict(range=[0, 1.05]),
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            )
            st.plotly_chart(fig_trend, width='stretch')

        # Comparison table
        if _PANDAS:
            rows = []
            for h in history:
                row = {"Run": h["timestamp"]}
                row.update({m: round(h["scores"].get(m, 0.0), 4) for m in trend_metrics})
                rows.append(row)
            df_hist = pd.DataFrame(rows)

            # Highlight best per column
            numeric_cols = [c for c in df_hist.columns if c != "Run"]

            def highlight_best(col):
                inverted = "hallucination" in col.name
                if inverted:
                    best_idx = col.idxmin()
                else:
                    best_idx = col.idxmax()
                colours_col = ["background-color: #d4edda" if i == best_idx else "" for i in col.index]
                return colours_col

            styled = df_hist.style.apply(highlight_best, subset=numeric_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 5 — Export
# ===========================================================================

with tab5:
    report = st.session_state.get("report")
    samples_list = st.session_state.get("samples", [])

    if report is None:
        st.info("Run evaluation first to export results.")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Full JSON report
        json_bytes = json.dumps(report["results"], indent=2, default=str).encode()
        st.download_button(
            "⬇️ Download Full Report (JSON)",
            data=json_bytes,
            file_name=f"eval_report_{ts}.json",
            mime="application/json",
        )

        # 2. Metrics CSV
        if _PANDAS:
            scores_df = pd.DataFrame(
                [{"Metric": k, "Score": v} for k, v in report["scores"].items()]
            )
            csv_metrics = scores_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Metrics Summary (CSV)",
                data=csv_metrics,
                file_name=f"eval_metrics_{ts}.csv",
                mime="text/csv",
            )

        # 3. Sample details CSV
        if _PANDAS and samples_list:
            from evaluation.base import CitationChecker, SemanticSimilarity

            sample_rows = []
            for s in samples_list:
                try:
                    cit = CitationChecker.score(s.answer, s.sources)
                    cf1 = cit.get("citation_f1", 0.0)
                except Exception:
                    cf1 = 0.0
                try:
                    sim = SemanticSimilarity.score(s.answer, s.ground_truth) if (s.answer and s.ground_truth) else 0.0
                except Exception:
                    sim = 0.0
                sample_rows.append({
                    "query": s.query,
                    "predicted_intent": s.predicted_intent or "",
                    "predicted_task": s.predicted_task or "",
                    "answer_preview": s.answer[:80] if s.answer else "",
                    "faithfulness": report["scores"].get("KnowledgeRAG.faithfulness", 0.0),
                    "semantic_similarity": round(sim, 4),
                    "citation_f1": round(cf1, 4),
                    "response_time_ms": s.response_time_ms or 0.0,
                })
            df_samples = pd.DataFrame(sample_rows)
            csv_samples = df_samples.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Sample Details (CSV)",
                data=csv_samples,
                file_name=f"eval_samples_{ts}.csv",
                mime="text/csv",
            )

        # 4. Raw JSON expander
        with st.expander("📋 Raw JSON Report"):
            st.json(report["results"])
