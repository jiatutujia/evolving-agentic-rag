from app.retrieval.document_search import DocumentSearcher
from app.retrieval.reranker import Reranker


def main() -> None:
    # --------------------------------------------------
    # 1. Build knowledge base
    # --------------------------------------------------
    searcher = DocumentSearcher(
        "data/dspy.pdf"
    )

    chunk_count = searcher.index_document()

    print(f"Indexed chunks: {chunk_count}")

    # --------------------------------------------------
    # 2. Query
    # --------------------------------------------------
    query = "What is DSPy?"

    print("\nQuery:")
    print(query)

    # --------------------------------------------------
    # 3. First-stage retrieval
    # --------------------------------------------------
    retrieved_results = searcher.search(
        query=query,
        limit=10,
    )

    print("\n" + "=" * 100)
    print("BEFORE RERANKING")
    print("=" * 100)

    for rank, result in enumerate(
        retrieved_results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")

        print(
            f"Retrieval Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['metadata']['chunk_id']}"
        )

        print(result["text"][:300])

    # --------------------------------------------------
    # 4. Reranking
    # --------------------------------------------------
    reranker = Reranker()

    reranked_results = reranker.rerank(
        query=query,
        documents=retrieved_results,
        top_k=5,
    )

    print("\n" + "=" * 100)
    print("AFTER RERANKING")
    print("=" * 100)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")

        print(
            f"Original Retrieval Score: "
            f"{result['retrieval_score']:.4f}"
        )

        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['metadata']['chunk_id']}"
        )

        print(result["text"][:300])


if __name__ == "__main__":
    main()