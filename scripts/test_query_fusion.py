from app.retrieval.document_search import DocumentSearcher
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import Reranker


def main() -> None:
    # 1. 初始化 Retriever
    searcher = DocumentSearcher(
        "data/dspy.pdf"
    )

    searcher.index_document()

    # 2. 初始化 Query Rewriter
    rewriter = QueryRewriter()

    # 3. 初始化 Reranker
    reranker = Reranker()

    query = (
        "I'm trying to understand DSPy and I saw something "
        "about signatures, modules and teleprompters, "
        "can you explain what these main abstractions are?"
    )

    # --------------------------------------------------
    # Query Rewrite
    # --------------------------------------------------
    rewritten_query = rewriter.rewrite(
        query
    )

    print("\nOriginal Query:")
    print(query)

    print("\nRewritten Query:")
    print(rewritten_query)

    # --------------------------------------------------
    # Original Query Retrieval
    # --------------------------------------------------
    original_results = searcher.search(
        query=query,
        limit=10,
    )

    # --------------------------------------------------
    # Rewritten Query Retrieval
    # --------------------------------------------------
    rewritten_results = searcher.search(
        query=rewritten_query,
        limit=10,
    )

    # --------------------------------------------------
    # RRF Fusion
    # --------------------------------------------------
    fused_results = reciprocal_rank_fusion(
        result_lists=[
            original_results,
            rewritten_results,
        ],
        top_k=10,
        rrf_k=60,
    )

    print("\n" + "=" * 100)
    print("AFTER RRF")
    print("=" * 100)

    for rank, result in enumerate(
        fused_results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")

        print(
            f"Fusion Score: "
            f"{result['fusion_score']:.6f}"
        )

        print(
            f"Ranks from retrieval lists: "
            f"{result['fusion_ranks']}"
        )

        print(
            f"Chunk ID: "
            f"{result['metadata']['chunk_id']}"
        )

        print(
            result["text"][:300]
        )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------
    reranked_results = reranker.rerank(
        query=query,
        documents=fused_results,
        top_k=5,
    )

    print("\n" + "=" * 100)
    print("AFTER RRF + RERANKER")
    print("=" * 100)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        print(f"\n--- Rank {rank} ---")

        print(
            f"Fusion Score: "
            f"{result['fusion_score']:.6f}"
        )

        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            f"Chunk ID: "
            f"{result['metadata']['chunk_id']}"
        )

        print(
            result["text"][:300]
        )


if __name__ == "__main__":
    main()