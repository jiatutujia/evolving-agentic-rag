from app.retrieval.adaptive_retrieval import (
    AdaptiveRetriever,
)


def main() -> None:
    retriever = AdaptiveRetriever(
        "data/dspy.pdf"
    )

    queries = [
        "What is DSPy?",

        "DSPy teleprompter how work?",

        (
            "So basically I want to know like "
            "how DSPy gets rid of writing "
            "prompts manually"
        ),
    ]

    for index, query in enumerate(
        queries,
        start=1,
    ):
        print(
            "\n"
            + "=" * 100
        )

        print(
            f"QUERY {index}"
        )

        print(
            "=" * 100
        )

        print(
            f"Original Query: {query}"
        )

        output = retriever.retrieve(
            query=query,
            retrieval_top_k=10,
            final_top_k=5,
        )

        print(
            "\nFinal Route:",
            output["route"],
        )

        print(
            "Rewritten Query:",
            output["rewritten_query"],
        )

        print(
            "\nFinal Results:"
        )

        for rank, result in enumerate(
            output["results"],
            start=1,
        ):
            print(
                f"\n--- Rank {rank} ---"
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