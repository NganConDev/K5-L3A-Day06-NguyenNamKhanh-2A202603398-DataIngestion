"""Run the data ingestion pipeline and benchmark comparisons."""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_strategy_chunks(document_id: str, source: str, pages: list, strategy: str) -> list:
    chunks: list = []
    for page in pages:
        if strategy == "A":
            chunks.extend(fixed_size(page, size=120, overlap=20))
        elif strategy == "B":
            chunks.extend(by_paragraph(page, max_size=80))
        else:
            chunks.extend(by_paragraph(page, max_size=80))
    for chunk in chunks:
        chunk.metadata.setdefault("section", "general")
        if "PAGE" in chunk.text.upper() or "SECTION:" in chunk.text.upper():
            continue
    return chunks


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    base_dir = Path(__file__).resolve().parents[1]
    manifest_path = base_dir / "data" / "manifest.json"
    benchmark_path = base_dir / "benchmark" / "questions.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_path = Path(manifest["source_path_or_url"])
    document_id = manifest["document_id"]
    source = manifest["title"]

    pages = load_pdf(corpus_path, document_id, source)
    if len(pages) < 30:
        raise ValueError(f"Corpus must contain at least 30 pages; found {len(pages)}")

    strategy_names = ["A", "B", "C"]
    questions = load_benchmark(benchmark_path)
    embedder = MyEmbedder()
    all_texts = []
    for strategy in strategy_names:
        for chunk in build_strategy_chunks(document_id, source, pages, strategy):
            all_texts.append(chunk.text)

    embedder.fit(all_texts)
    database_url = (base_dir / ".env").read_text(encoding="utf-8").split("DATABASE_URL=")[-1].strip().splitlines()[0]
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()

    results_by_strategy: dict[str, dict] = {}
    for strategy in strategy_names:
        strategy_chunks = build_strategy_chunks(document_id, source, pages, strategy)
        if strategy == "C":
            for chunk in strategy_chunks:
                chunk.metadata["section"] = "general"
                if chunk.page in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}:
                    chunk.metadata["section"] = "general"
                    if chunk.page in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}:
                        pass
        vectors = embedder.embed_documents([chunk.text for chunk in strategy_chunks])
        store.upsert(strategy, strategy_chunks, vectors)

        traces: list[dict] = []
        for question in questions:
            filter_value = question.get("filter")
            query_vector = embedder.embed_query(question["question"])
            hits = store.search(strategy, query_vector, top_k=3, filters=filter_value)
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            trace["question"] = question["question"]
            traces.append(trace)

        results_by_strategy[strategy] = summarize(traces)
        print(f"Strategy {strategy}: {results_by_strategy[strategy]}")

    print("Pipeline complete. Results are printed above.")


if __name__ == "__main__":
    main()
