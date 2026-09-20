from pathlib import Path

from app.memory.experience_memory import (
    ExperienceMemory,
)

from app.strategy.memory_strategy import (
    MemoryStrategyAdvisor,
)


def test_query(
    advisor: MemoryStrategyAdvisor,
    query: str,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Query: {query}"
    )

    print(
        "=" * 100
    )

    decision = advisor.advise(
        query
    )

    if decision is None:

        print(
            "Memory Decision: None"
        )

        print(
            "Fallback: use normal QueryRouter"
        )

        return

    print(
        f"Memory Strategy : "
        f"{decision.strategy}"
    )

    print(
        f"Similarity      : "
        f"{decision.similarity:.4f}"
    )

    print(
        f"Past Query      : "
        f"{decision.past_query}"
    )

    print(
        f"Past Outcome    : "
        f"{decision.past_outcome}"
    )

    print(
        f"Lesson          : "
        f"{decision.lesson}"
    )

    print(
        f"Reason          : "
        f"{decision.reason}"
    )


def main() -> None:

    # Reuse the memory created by the previous test.
    memory = ExperienceMemory(
        memory_path=Path(
            "memory/test_experience_memory.json"
        )
    )

    advisor = MemoryStrategyAdvisor(
        memory=memory,
        min_similarity=0.75,
    )

    # ==================================================
    # Similar to known Tokyo failure
    #
    # Expected:
    # external_search
    # ==================================================

    test_query(
        advisor,
        "What is Tokyo's population?",
    )

    # ==================================================
    # Similar to known DSPy success
    #
    # Expected:
    # direct
    # ==================================================

    test_query(
        advisor,
        "Can you tell me what DSPy is?",
    )

    # ==================================================
    # Completely unrelated query
    #
    # Expected:
    # no memory override
    # → normal router
    # ==================================================

    test_query(
        advisor,
        (
            "How does photosynthesis "
            "work in plants?"
        ),
    )


if __name__ == "__main__":
    main()