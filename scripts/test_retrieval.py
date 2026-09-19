import json
from pathlib import Path

from app.retrieval.document_search import DocumentSearcher


def main() -> None:
    # --------------------------------------------------
    # 1. Build the baseline knowledge base
    # --------------------------------------------------
    searcher = DocumentSearcher("data/dspy.pdf")

    chunk_count = searcher.index_document()

    print(f"Indexed chunks: {chunk_count}")

    # --------------------------------------------------
    # 2. Baseline evaluation queries
    # --------------------------------------------------
    queries = [
        "What is DSPy?",
        "What are the main abstractions in DSPy?",
        "What is a DSPy signature?",
        "What is a teleprompter in DSPy?",
        "How does DSPy reduce manual prompt engineering?",
    ]

    top_k = 3

    # 用于最终保存到 JSON
    benchmark_results = {
        "version": "v0-vector-retrieval",
        "document": "dspy.pdf",
        "indexed_chunks": chunk_count,
        "top_k": top_k,
        "queries": [],
    }

    # --------------------------------------------------
    # 3. Run all queries
    # --------------------------------------------------
    for query_index, query in enumerate(queries, start=1):

        print("\n" + "=" * 100)
        print(f"Query {query_index}: {query}")
        print("=" * 100)

        results = searcher.search(
            query=query,
            limit=top_k,
        )

        query_result = {
            "query_id": query_index,
            "query": query,
            "results": [],
        }

        for rank, result in enumerate(results, start=1):

            print(f"\n--- Result {rank} ---")
            print(f"Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            print(result["text"][:500])

            query_result["results"].append(
                {
                    "rank": rank,
                    "score": result["score"],
                    "chunk_id": result["metadata"].get("chunk_id"),
                    "source": result["metadata"].get("source"),
                    "text": result["text"],
                }
            )

        benchmark_results["queries"].append(query_result)

    # --------------------------------------------------
    # 4. Save baseline experiment
    # --------------------------------------------------
    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "v0_retrieval_baseline.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            benchmark_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("Benchmark completed.")
    print(f"Results saved to: {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()