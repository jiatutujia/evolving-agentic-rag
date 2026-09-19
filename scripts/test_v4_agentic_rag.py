from pathlib import Path

from app.retrieval.adaptive_retrieval import (
    AdaptiveRetriever,
)


def main() -> None:

    rag = AdaptiveRetriever(
        document_path=Path(
            "data/dspy.pdf"
        )
    )

    queries = [
        "What is DSPy?",

        (
            "DSPy teleprompter "
            "how work?"
        ),

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
            + "=" * 100
        )

        print(
            f"QUERY {index}"
        )

        print(
            "=" * 100
        )

        print(
            f"Question: {query}"
        )

        result = (
            rag.retrieve(
                query
            )
        )

        print(
            "\n"
            + "-" * 100
        )

        print(
            "FINAL RESULT"
        )

        print(
            "-" * 100
        )

        print(
            f"Initial Route : "
            f"{result['initial_route']}"
        )

        print(
            f"Final Route   : "
            f"{result['final_route']}"
        )

        print(
            f"Retry         : "
            f"{result['retry_triggered']}"
        )

        print(
            f"Final Grade   : "
            f"{result['final_grade'].is_sufficient}"
        )

        print(
            f"Answer Status : "
            f"{result['answer_status']}"
        )

        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )


if __name__ == "__main__":
    main()