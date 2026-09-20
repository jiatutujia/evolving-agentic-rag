from pathlib import Path

from app.reward.reward_history import (
    RewardHistory,
)


def main() -> None:

    path = Path(
        "memory/test_reward_history.json"
    )

    history = RewardHistory(
        path=path
    )

    history.clear()

    # ==================================================
    # Simulated historical executions
    # ==================================================

    history.add(
        query="What is DSPy?",
        strategy="direct",
        reward=1.0,
        answer_status="generated",
    )

    history.add(
        query="Explain DSPy.",
        strategy="direct",
        reward=1.0,
        answer_status="generated",
    )

    history.add(
        query="Tell me about DSPy.",
        strategy="rewrite",
        reward=0.9,
        answer_status="generated",
    )

    history.add(
        query="Malformed DSPy query",
        strategy="rewrite_retry",
        reward=0.75,
        answer_status="generated",
    )

    history.add(
        query="Tokyo population",
        strategy="rewrite_retry",
        reward=-1.25,
        answer_status=(
            "insufficient_evidence"
        ),
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REWARD HISTORY"
    )

    print(
        "=" * 100
    )

    print(
        f"Total Records: "
        f"{history.size()}"
    )

    print(
        "\nAverage direct reward:"
    )

    print(
        history.average_reward(
            "direct"
        )
    )

    print(
        "\nStrategy Summary:"
    )

    summary = (
        history.strategy_summary()
    )

    for (
        strategy,
        statistics,
    ) in summary.items():

        print(
            f"\nStrategy: "
            f"{strategy}"
        )

        print(
            f"Count: "
            f"{statistics['count']}"
        )

        print(
            f"Average Reward: "
            f"{statistics['average_reward']:.4f}"
        )

        print(
            f"Best Reward: "
            f"{statistics['best_reward']:.4f}"
        )

        print(
            f"Worst Reward: "
            f"{statistics['worst_reward']:.4f}"
        )


if __name__ == "__main__":
    main()