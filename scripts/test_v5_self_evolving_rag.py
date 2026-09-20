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
        "FINAL EXECUTION"
    )

    print(
        "=" * 100
    )

    print(
        f"Query           : "
        f"{result['query']}"
    )

    print(
        f"Strategy Source : "
        f"{result['strategy_source']}"
    )

    print(
        f"Router Baseline : "
        f"{result['router_decision'].route}"
    )

    memory_strategy = (
        result["memory_strategy"]
    )

    if memory_strategy is None:

        print(
            "Memory Strategy : None"
        )

    else:

        print(
            f"Memory Strategy : "
            f"{memory_strategy.strategy}"
        )

        print(
            f"Memory Similarity: "
            f"{memory_strategy.similarity:.4f}"
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
        f"Answer Status   : "
        f"{result['answer_status']}"
    )

    print(
        "\nAnswer:"
    )

    print(
        result["answer"]
    )

    reflection = (
        result["reflection"]
    )

    print(
        "\nReflection:"
    )

    print(
        f"Outcome  : "
        f"{reflection.outcome}"
    )

    print(
        f"Strategy : "
        f"{reflection.recommended_strategy}"
    )

    print(
        f"Lesson   : "
        f"{reflection.lesson}"
    )

    print(
        f"\nMemory Size: "
        f"{result['memory_size']}"
    )


def main() -> None:

    # ==================================================
    # Use an isolated memory file so the experiment
    # remains deterministic.
    # ==================================================

    memory_path = Path(
        "memory/test_v5_self_evolving.json"
    )

    if memory_path.exists():

        memory_path.unlink()

    rag = AdaptiveRetriever(
        document_path=Path(
            "data/dspy.pdf"
        ),

        memory_path=memory_path,

        memory_min_similarity=0.75,
    )

    queries = [

        # ==================================================
        # Query 1
        #
        # No experience exists yet.
        #
        # Expected:
        #
        # Router → direct
        # direct fails
        # rewrite retry fails
        # Reflection → external_search
        # Write experience to memory
        # ==================================================

        (
            "What is the population "
            "of Tokyo in 2025?"
        ),

        # ==================================================
        # Query 2
        #
        # Semantically similar to Query 1.
        #
        # Expected:
        #
        # Memory finds Query 1
        # → external_search
        #
        # It should avoid wasting time on the same
        # local retrieval + rewrite failure again.
        # ==================================================

        (
            "What is Tokyo's population?"
        ),

        # ==================================================
        # Query 3
        #
        # No DSPy success experience exists yet.
        #
        # Expected:
        #
        # Router → direct
        # success
        # Reflection → prefer_direct
        # Write to memory
        # ==================================================

        "What is DSPy?",

        # ==================================================
        # Query 4
        #
        # Static router may prefer rewrite because:
        #
        # "can you tell me"
        #
        # But memory should find:
        #
        # What is DSPy?
        #
        # and override router with direct.
        # ==================================================

        (
            "Can you tell me "
            "what DSPy is?"
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
            f"Question: {query}"
        )

        result = rag.retrieve(
            query
        )

        print_result(
            result
        )


if __name__ == "__main__":
    main()