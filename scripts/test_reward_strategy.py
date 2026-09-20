from dataclasses import dataclass
from pathlib import Path

from app.memory.experience_memory import (
    ExperienceMemory,
)

from app.reward.reward_history import (
    RewardHistory,
)

from app.strategy.reward_strategy import (
    RewardAwareStrategyAdvisor,
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
        "memory/test_v6_experience.json"
    )

    reward_path = Path(
        "memory/test_v6_rewards.json"
    )

    # ==================================================
    # Clean test data
    # ==================================================

    memory = ExperienceMemory(
        memory_path=memory_path
    )

    memory.clear()

    history = RewardHistory(
        path=reward_path
    )

    history.clear()

    # ==================================================
    # Experience A:
    # DSPy direct success
    # ==================================================

    memory.add(
        query=(
            "What is DSPy?"
        ),

        reflection=FakeReflection(
            outcome=(
                "success_first_pass"
            ),

            success=True,

            initial_route="direct",

            final_route="direct",

            retry_triggered=False,

            initial_score=5.9,

            final_score=5.9,

            score_delta=0.0,

            recommended_strategy=(
                "prefer_direct"
            ),

            lesson=(
                "Direct retrieval was "
                "sufficient."
            ),
        ),
    )

    # ==================================================
    # Experience B:
    # Similar DSPy query succeeded using rewrite
    # ==================================================

    memory.add(
        query=(
            "Please explain what DSPy is."
        ),

        reflection=FakeReflection(
            outcome=(
                "success_first_pass"
            ),

            success=True,

            initial_route="rewrite",

            final_route="rewrite",

            retry_triggered=False,

            initial_score=5.0,

            final_score=5.0,

            score_delta=0.0,

            recommended_strategy=(
                "prefer_rewrite"
            ),

            lesson=(
                "Query rewriting produced "
                "sufficient evidence."
            ),
        ),
    )

    # ==================================================
    # Historical Reward
    #
    # direct performs better overall
    # ==================================================

    history.add(
        query="What is DSPy?",
        strategy="direct",
        reward=1.0,
        answer_status="generated",
    )

    history.add(
        query="Explain DSPy",
        strategy="direct",
        reward=1.0,
        answer_status="generated",
    )

    history.add(
        query="Tell me about DSPy",
        strategy="rewrite",
        reward=0.9,
        answer_status="generated",
    )

    history.add(
        query="DSPy please explain",
        strategy="rewrite",
        reward=0.6,
        answer_status="generated",
    )

    advisor = (
        RewardAwareStrategyAdvisor(
            memory=memory,

            reward_history=history,

            min_similarity=0.70,

            top_k=5,

            similarity_weight=0.8,

            reward_weight=0.2,
        )
    )

    # ==================================================
    # New Query
    # ==================================================

    query = (
        "Can you explain DSPy?"
    )

    decision = (
        advisor.advise(
            query
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "V6 REWARD-AWARE STRATEGY"
    )

    print(
        "=" * 100
    )

    print(
        f"Query: {query}"
    )

    if decision is None:

        print(
            "\nNo reward-aware "
            "strategy available."
        )

        return

    print(
        f"\nStrategy        : "
        f"{decision.strategy}"
    )

    print(
        f"Similarity      : "
        f"{decision.similarity:.4f}"
    )

    print(
        f"Average Reward  : "
        f"{decision.average_reward}"
    )

    print(
        f"Combined Score  : "
        f"{decision.combined_score:.4f}"
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


if __name__ == "__main__":
    main()