from pathlib import Path

from app.retrieval.adaptive_retrieval import (
    AdaptiveRetriever,
)


def print_result(
    result: dict,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL EXECUTION STATE"
    )

    print(
        "=" * 100
    )

    print(
        f"Query           : "
        f"{result['query']}"
    )

    print(
        f"Initial Route   : "
        f"{result['initial_route']}"
    )

    print(
        f"Final Route     : "
        f"{result['final_route']}"
    )

    print(
        f"Retry Triggered : "
        f"{result['retry_triggered']}"
    )

    print(
        f"Rewritten Query : "
        f"{result['rewritten_query']}"
    )

    print(
        f"Initial Grade   : "
        f"{result['initial_grade'].is_sufficient}"
    )

    print(
        f"Final Grade     : "
        f"{result['final_grade'].is_sufficient}"
    )

    print(
        f"Initial Score   : "
        f"{result['initial_grade'].best_score:.4f}"
    )

    print(
        f"Final Score     : "
        f"{result['final_grade'].best_score:.4f}"
    )

    print(
        "\nTop Results:"
    )

    for rank, item in enumerate(
        result["results"],
        start=1,
    ):

        print(
            f"\n--- Rank {rank} ---"
        )

        rerank_score = (
            item.get(
                "rerank_score"
            )
        )

        print(
            f"Rerank Score: "
            f"{rerank_score}"
        )

        print(
            f"Chunk ID: "
            f"{item.get('chunk_id')}"
        )

        text = item.get(
            "text",
            "",
        )

        print(
            text[:400]
        )


def main() -> None:

    document_path = Path(
        "data/dspy.pdf"
    )

    retriever = (
        AdaptiveRetriever(
            document_path=document_path
        )
    )

    queries = [

        # --------------------------------------------------
        # Case 1:
        # Clean query.
        #
        # Expected:
        # direct → sufficient
        # --------------------------------------------------
        "What is DSPy?",

        # --------------------------------------------------
        # Case 2:
        # Malformed query.
        #
        # Expected:
        # router → rewrite
        # --------------------------------------------------
        "DSPy teleprompter how work?",

        # --------------------------------------------------
        # Case 3:
        # Out-of-domain query.
        #
        # DSPy PDF should not contain the answer.
        #
        # Expected:
        #
        # direct
        # ↓
        # insufficient
        # ↓
        # rewrite retry
        # --------------------------------------------------
        (
            "What is the population "
            "of Tokyo in 2025?"
        ),
    ]

    for index, query in enumerate(
        queries,
        start=1,
    ):

        print(
            "\n\n"
            + "#" * 100
        )

        print(
            f"QUERY {index}"
        )

        print(
            "#" * 100
        )

        print(
            f"Original Query: "
            f"{query}"
        )

        result = (
            retriever.retrieve(
                query
            )
        )

        print_result(
            result
        )


if __name__ == "__main__":
    main()