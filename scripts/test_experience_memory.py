from dataclasses import dataclass
from pathlib import Path

from app.memory.experience_memory import (
    ExperienceMemory,
)


@dataclass
class FakeReflection:
    outcome: str
    success: bool
    initial_route: str
    final_route: str
    retry_triggered: bool
    initial_score: float
    final_score: float
    score_delta: float
    recommended_strategy: str
    lesson: str


def main() -> None:

    memory_path = Path(
        "memory/test_experience_memory.json"
    )

    memory = ExperienceMemory(
        memory_path=memory_path
    )

    # Make test deterministic.
    memory.clear()

    # ==================================================
    # Experience 1
    # Direct retrieval worked
    # ==================================================

    memory.add(
        query="What is DSPy?",

        reflection=FakeReflection(
            outcome="success_first_pass",
            success=True,
            initial_route="direct",
            final_route="direct",
            retry_triggered=False,
            initial_score=5.97,
            final_score=5.97,
            score_delta=0.0,
            recommended_strategy=(
                "prefer_direct"
            ),
            lesson=(
                "The original query produced "
                "sufficient retrieval evidence."
            ),
        ),
    )

    # ==================================================
    # Experience 2
    # Knowledge base did not contain information
    # ==================================================

    memory.add(
        query=(
            "What is the population "
            "of Tokyo in 2025?"
        ),

        reflection=FakeReflection(
            outcome="failed_after_retry",
            success=False,
            initial_route="direct",
            final_route="rewrite_retry",
            retry_triggered=True,
            initial_score=-7.17,
            final_score=-7.17,
            score_delta=0.0,
            recommended_strategy=(
                "external_search"
            ),
            lesson=(
                "Query rewriting did not "
                "materially improve retrieval. "
                "The knowledge base likely lacks "
                "the requested information."
            ),
        ),
    )

    print(
        "\nMemory Size:"
    )

    print(
        memory.size()
    )

    # ==================================================
    # Test semantic memory retrieval
    # ==================================================

    test_query = (
        "What is Tokyo's population?"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"New Query: "
        f"{test_query}"
    )

    print(
        "=" * 100
    )

    matches = memory.search(
        query=test_query,
        top_k=3,
        min_similarity=0.3,
    )

    for rank, match in enumerate(
        matches,
        start=1,
    ):

        experience = (
            match.experience
        )

        print(
            f"\n--- Memory Rank "
            f"{rank} ---"
        )

        print(
            f"Similarity : "
            f"{match.similarity:.4f}"
        )

        print(
            f"Past Query : "
            f"{experience.query}"
        )

        print(
            f"Outcome    : "
            f"{experience.outcome}"
        )

        print(
            f"Strategy   : "
            f"{experience.recommended_strategy}"
        )

        print(
            f"Lesson     : "
            f"{experience.lesson}"
        )


if __name__ == "__main__":
    main()