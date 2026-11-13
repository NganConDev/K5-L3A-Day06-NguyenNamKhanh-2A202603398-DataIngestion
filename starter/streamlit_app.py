from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.embeddings import MyEmbedder
from src.evaluation import evaluate_query, summarize
from src.ingestion import load_pdf
from src.run import build_strategy_chunks, infer_question_section, load_benchmark
from src.store import PgVectorStore


CSS = """
<style>
    [data-testid="stAppViewContainer"] {
        background: #f4f7fb;
    }
    .stApp {
        color: #0f172a;
    }
    .stSidebar {
        background: #f8fafc;
        border-right: 1px solid rgba(148, 163, 184, 0.25);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.8rem;
    }
    .topbar {
        background: linear-gradient(90deg, #eef4ff 0%, #f8fbff 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.03);
    }
    .topbar h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #0f172a;
    }
    .topbar p {
        margin: 0.3rem 0 0 0;
        color: #475569;
    }
    .panel {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.025);
    }
    .metric-box {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        height: 100%;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.025);
    }
    .metric-box .small-label {
        font-size: 0.7rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #64748b;
    }
    .metric-box h3 {
        margin: 0.35rem 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .result-card {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-left: 4px solid #2563eb;
        border-radius: 14px;
        padding: 1rem 1rem 0.9rem 1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.025);
    }
    .result-tag {
        display: inline-block;
        background: rgba(37, 99, 235, 0.08);
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 999px;
        padding: 0.18rem 0.6rem;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #1d4ed8;
        margin-bottom: 0.65rem;
    }
    .result-card p {
        margin: 0.38rem 0;
        color: #1e293b;
        line-height: 1.6;
    }
    .small-label {
        font-size: 0.72rem;
        color: #64748b;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(148,163,184,0.15);
        color: #334155;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.08);
        border-color: rgba(37,99,235,0.15);
        color: #1d4ed8;
    }
    .stButton > button {
        background: #111827;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #1f2937;
        color: white;
    }
    .stButton > button,
    .stButton > button * {
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] * {
        color: inherit !important;
    }
    .stSidebar .stSelectbox label,
    .stSidebar .stTextInput label,
    .stSidebar .stCheckbox label,
    .stSidebar .stRadio label,
    .stSidebar [data-testid="stBaseSelectLabel"],
    .stSidebar [data-testid="stBaseTextInputLabel"],
    .stSidebar [data-testid="stBaseWidgetLabel"] {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    .stSidebar .stSelectbox [data-testid="stBaseSelectContainer"],
    .stSidebar .stSelectbox [role="combobox"],
    .stSidebar .stSelectbox [role="button"],
    .stSidebar .stSelectbox [data-baseweb="select"],
    .stSidebar .stSelectbox [data-baseweb="select"] * {
        color: #0f172a !important;
    }
    .stSelectbox [role="listbox"],
    .stSelectbox [role="option"],
    .stSelectbox [role="listbox"] *,
    .stSelectbox [role="option"] * {
        color: #0f172a !important;
    }
    .stInfo, .stInfo *,
    .stWarning, .stWarning *,
    .stError, .stError *,
    .stSuccess, .stSuccess *,
    .stAlert, .stAlert * {
        color: #0f172a !important;
    }
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(255,255,255,0.9);
        color: #0f172a !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stTextInput, .stSelectbox, .stNumberInput, .stRadio, .stCheckbox, .stFileUploader,
    .stDataFrame, .stMetric, .stMetricValue, .stMetricLabel,
    div, p, span, label, strong, em, li, a {
        color: #0f172a !important;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def prepare_project_index() -> tuple[dict, list, list, MyEmbedder, dict[str, list[dict]]]:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    manifest_path = base_dir / "data" / "manifest.json"
    benchmark_path = base_dir / "benchmark" / "questions.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_path = Path(manifest["source_path_or_url"])
    document_id = manifest["document_id"]
    source = manifest["title"]

    pages = load_pdf(corpus_path, document_id, source)
    questions = load_benchmark(benchmark_path)

    all_texts: list[str] = []
    for strategy in ["A", "B", "C"]:
        strategy_chunks = build_strategy_chunks(document_id, source, pages, strategy)
        all_texts.extend(chunk.text for chunk in strategy_chunks)

    embedder = MyEmbedder()
    embedder.fit(all_texts)

    database_url = read_database_url()
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    indices: dict[str, list[dict]] = {}
    for strategy in ["A", "B", "C"]:
        strategy_chunks = build_strategy_chunks(document_id, source, pages, strategy)
        vectors = embedder.embed_documents([chunk.text for chunk in strategy_chunks])
        store.upsert(strategy, strategy_chunks, vectors)
        indices[strategy] = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in strategy_chunks
        ]

    return manifest, pages, questions, embedder, indices


def read_database_url() -> str:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.getenv(
        "DATABASE_URL",
        "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab",
    )


def run_retrieval(strategy: str, question: str, embedder: MyEmbedder) -> list[dict]:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")
    db_url = read_database_url()
    store = PgVectorStore(db_url, embedder.dimensions)

    filters = None
    section_hint = infer_question_section(question)
    if strategy == "C" and section_hint:
        filters = {"section": section_hint}

    query_vector = embedder.embed_query(question)
    hits = store.search(strategy, query_vector, top_k=3, filters=filters)
    return hits


def compute_strategy_metrics(strategy: str, questions: list[dict], embedder: MyEmbedder) -> dict:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")
    db_url = read_database_url()
    store = PgVectorStore(db_url, embedder.dimensions)

    traces: list[dict] = []
    for question in questions:
        filters = None
        section_hint = infer_question_section(question["question"])
        if strategy == "C" and section_hint:
            filters = {"section": section_hint}
        query_vector = embedder.embed_query(question["question"])
        hits = store.search(strategy, query_vector, top_k=3, filters=filters)
        traces.append(evaluate_query(question, hits))

    summary = summarize(traces)
    summary["strategy"] = strategy
    return summary


def render_result_card(hit: dict, rank: int, show_evidence: bool = False) -> None:
    evidence = (hit.get("text", "") or "").strip()
    preview = evidence[:260] + ("..." if len(evidence) > 260 else "") if evidence else "No evidence text available."

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-tag">Top {rank}</div>
            <div class="small-label">Page</div>
            <p><strong>{hit.get('page', 'N/A')}</strong></p>
            <div class="small-label">Source</div>
            <p>{hit.get('source', 'N/A')}</p>
            <div class="small-label">Similarity</div>
            <p>{hit.get('score', 0):.4f}</p>
            <div class="small-label">Evidence</div>
            <p>{preview if not show_evidence else evidence}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Data Ingestion Lab Dashboard", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.header("Configuration")
    strategy = st.sidebar.selectbox("Strategy", ["A", "B", "C"], index=0)
    sample_questions = [
        "What does the book say a professional programmer must do to be called professional?",
        "What is the book's message about doing anything less than your best?",
        "What does the foreword say most of software work is?",
        "What does the introduction say about craftsmanship?",
    ]
    selected_question = st.sidebar.selectbox("Sample question", sample_questions, index=0)
    show_evidence = st.sidebar.checkbox("Show evidence passages", value=False)

    st.markdown(
        """
        <div class="topbar">
            <h1>Data Ingestion Benchmark</h1>
            <p>Retrieval analysis for corpus ingestion and benchmark evaluation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.5, 1.0])

    with left_col:
        question_input = st.text_input("Query", value=selected_question, help="Enter a question to evaluate retrieval performance.")
        run_button = st.button("Run retrieval", use_container_width=True)

        if run_button or question_input:
            try:
                manifest, pages, questions, embedder, _ = prepare_project_index()
                st.info(f"Corpus: {manifest['title']} • {len(pages)} pages loaded")

                with st.spinner("Running retrieval..."):
                    hits = run_retrieval(strategy, question_input, embedder)

                tabs = st.tabs(["Retrieval results", "Benchmark overview"])

                with tabs[0]:
                    if not hits:
                        st.warning("No chunks were returned for this query.")
                    else:
                        for idx, hit in enumerate(hits, start=1):
                            render_result_card(hit, idx, show_evidence=show_evidence)

                with tabs[1]:
                    dev_questions = [q for q in questions if q.get("split") == "dev"]
                    test_questions = [q for q in questions if q.get("split") == "test"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Dev questions", len(dev_questions))
                    c2.metric("Test questions", len(test_questions))
                    c3.metric("Total", len(questions))

                    with st.expander("View benchmark questions"):
                        for q in questions:
                            st.write(f"- {q['id']} [{q['split']}]: {q['question']}")
            except Exception as exc:
                st.error(f"Unable to initialize the app: {exc}")
                st.info("Ensure PostgreSQL + pgvector is running and the project `.env` file contains a valid `DATABASE_URL` before starting retrieval.")

    with right_col:
        try:
            manifest, pages, questions, embedder, _ = prepare_project_index()
            strategy_stats = []
            for name in ["A", "B", "C"]:
                summary = compute_strategy_metrics(name, questions, embedder)
                strategy_stats.append(
                    {
                        "Strategy": name,
                        "Hit@1": round(summary.get("hit1_rate", 0.0) * 100, 1),
                        "Evidence@3": round(summary.get("evidence_hit3_rate", 0.0) * 100, 1),
                        "Total": summary.get("total", 0),
                    }
                )

            df = pd.DataFrame(strategy_stats)
            chart_df = df.set_index("Strategy")[["Hit@1", "Evidence@3"]]

            c1, c2, c3 = st.columns(3)
            c1.markdown("<div class='metric-box'><div class='small-label'>Pages</div><h3>{}</h3></div>".format(len(pages)), unsafe_allow_html=True)
            c2.markdown("<div class='metric-box'><div class='small-label'>Corpus</div><h3>{}</h3></div>".format(manifest["document_id"][:12]), unsafe_allow_html=True)
            c3.markdown("<div class='metric-box'><div class='small-label'>Strategies</div><h3>3</h3></div>", unsafe_allow_html=True)

            st.markdown("<div class='panel'><div class='small-label'>Comparison chart</div></div>", unsafe_allow_html=True)
            st.bar_chart(chart_df)

            st.markdown("<div class='panel' style='margin-top: 1rem;'><div class='small-label'>Score table</div></div>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Summary metrics unavailable: {exc}")


if __name__ == "__main__":
    main()
