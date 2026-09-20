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
        "FINAL V6 EXECUTION"
    )

    print(
        "=" * 100
    )

    print(
        f"Query            : "
        f"{result['query']}"
    )

    print(
        f"Strategy Source  : "
        f"{result['strategy_source']}"
    )

    print(
        f"Selected Strategy: "
        f"{result['selected_strategy']}"
    )

    print(
        f"Router Baseline  : "
        f"{result['router_decision'].route}"
    )

    policy = (
        result[
            "policy_decision"
        ]
    )

    if policy is None:

        print(
            "Policy Decision   : None"
        )

    else:

        print(
            f"Policy Strategy   : "
            f"{policy.strategy}"
        )

        print(
            f"Similarity        : "
            f"{policy.similarity:.4f}"
        )

        print(
            f"Average Reward    : "
            f"{policy.average_reward}"
        )

        print(
            f"Combined Score    : "
            f"{policy.combined_score:.4f}"
        )

    print(
        f"Initial Route    : "
        f"{result['initial_route']}"
    )

    print(
        f"Final Route      : "
        f"{result['final_route']}"
    )

    print(
        f"Retry Triggered  : "
        f"{result['retry_triggered']}"
    )

    print(
        f"Answer Status    : "
        f"{result['answer_status']}"
    )

    print(
        "\nAnswer:"
    )

    print(
        result[
            "answer"
        ]
    )

    reflection = (
        result[
            "reflection"
        ]
    )

    print(
        "\nReflection:"
    )

    print(
        f"Outcome          : "
        f"{reflection.outcome}"
    )

    print(
        f"Recommendation   : "
        f"{reflection.recommended_strategy}"
    )

    print(
        f"Lesson           : "
        f"{reflection.lesson}"
    )

    reward = (
        result[
            "reward"
        ]
    )

    print(
        "\nReward:"
    )

    print(
        f"Executed Strategy: "
        f"{reward.strategy}"
    )

    print(
        f"Quality Reward   : "
        f"{reward.quality_reward:.2f}"
    )

    print(
        f"Rewrite Penalty  : "
        f"{reward.rewrite_penalty:.2f}"
    )

    print(
        f"Retry Penalty    : "
        f"{reward.retry_penalty:.2f}"
    )

    print(
        f"Total Reward     : "
        f"{reward.total_reward:.2f}"
    )

    print(
        f"\nMemory Size      : "
        f"{result['memory_size']}"
    )

    print(
        f"Reward History   : "
        f"{result['reward_history_size']}"
    )


def main() -> None:

    # ==================================================
    # Isolated test state
    # ==================================================

    memory_path = Path(
        "memory/test_v6_experience.json"
    )

    reward_path = Path(
        "memory/test_v6_reward_history.json"
    )

    if memory_path.exists():

        memory_path.unlink()

    if reward_path.exists():

        reward_path.unlink()

    rag = AdaptiveRetriever(
        document_path=Path(
            "data/dspy.pdf"
        ),

        memory_path=(
            memory_path
        ),

        reward_history_path=(
            reward_path
        ),

        memory_min_similarity=0.75,
    )

    queries = [

        # ==================================================
        # Query 1
        #
        # No history.
        #
        # Router:
        # direct
        #
        # Local retrieval fails.
        # Rewrite retry fails.
        #
        # Expected reward:
        # safe abstention
        # 0.20 - 0.10 - 0.15
        # = -0.05
        # ==================================================

        (
            "What is the population "
            "of Tokyo in 2025?"
        ),

        # ==================================================
        # Query 2
        #
        # Similar to Query 1.
        #
        # Experience Memory says:
        # external_search.
        #
        # Expected:
        # skip repeated local failure.
        #
        # Reward:
        # +0.30
        # ==================================================

        (
            "What is Tokyo's population?"
        ),

        # ==================================================
        # Query 3
        #
        # Another Tokyo-like query.
        #
        # Now external_search also has reward history.
        #
        # Policy should combine:
        #
        # semantic similarity
        # +
        # external_search reward
        # ==================================================

        (
            "How many people live "
            "in Tokyo?"
        ),

        # ==================================================
        # Query 4
        #
        # First DSPy experience.
        #
        # direct succeeds.
        #
        # Reward = 1.0
        # ==================================================

        "What is DSPy?",

        # ==================================================
        # Query 5
        #
        # Static router prefers rewrite because of:
        #
        # can you tell me
        #
        # Reward-aware memory should prefer direct.
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
            f"Question: "
            f"{query}"
        )

        result = (
            rag.retrieve(
                query
            )
        )

        print_result(
            result
        )

    # ==================================================
    # Final Strategy Summary
    # ==================================================

    print(
        "\n\n"
        + "=" * 100
    )

    print(
        "FINAL REWARD SUMMARY"
    )

    print(
        "=" * 100
    )

    summary = (
        rag.reward_history.strategy_summary()
    )

    for strategy, stats in (
        summary.items()
    ):

        print(
            f"\nStrategy: "
            f"{strategy}"
        )

        print(
            f"Count: "
            f"{stats['count']}"
        )

        print(
            f"Average Reward: "
            f"{stats['average_reward']:.4f}"
        )

        print(
            f"Best Reward: "
            f"{stats['best_reward']:.4f}"
        )

        print(
            f"Worst Reward: "
            f"{stats['worst_reward']:.4f}"
        )


if __name__ == "__main__":
    main()