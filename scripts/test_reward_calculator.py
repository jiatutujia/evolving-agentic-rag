from app.reward.reward_calculator import (
    RewardCalculator,
)


def run_case(
    name: str,
    execution: dict,
) -> None:

    calculator = (
        RewardCalculator()
    )

    reward = (
        calculator.calculate(
            execution
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(name)

    print(
        "=" * 100
    )

    print(
        f"Strategy        : "
        f"{reward.strategy}"
    )

    print(
        f"Quality Reward  : "
        f"{reward.quality_reward:.2f}"
    )

    print(
        f"Rewrite Penalty : "
        f"{reward.rewrite_penalty:.2f}"
    )

    print(
        f"Retry Penalty   : "
        f"{reward.retry_penalty:.2f}"
    )

    print(
        f"Failure Penalty : "
        f"{reward.failure_penalty:.2f}"
    )

    print(
        f"Total Reward    : "
        f"{reward.total_reward:.2f}"
    )

    print(
        f"Reason          : "
        f"{reward.reason}"
    )


def main() -> None:

    direct_success = {
        "selected_strategy": "direct",

        "initial_route": "direct",

        "final_route": "direct",

        "retry_triggered": False,

        "answer_status": "generated",
    }

    rewrite_success = {
        "selected_strategy": "rewrite",

        "initial_route": "rewrite",

        "final_route": "rewrite",

        "retry_triggered": False,

        "answer_status": "generated",
    }

    retry_success = {
        "selected_strategy": "direct",

        "initial_route": "direct",

        "final_route": "rewrite_retry",

        "retry_triggered": True,

        "answer_status": "generated",
    }

    safe_abstention = {
        "selected_strategy": "direct",

        "initial_route": "direct",

        "final_route": "rewrite_retry",

        "retry_triggered": True,

        "answer_status": (
            "insufficient_evidence"
        ),
    }

    external_search = {
        "selected_strategy": (
            "external_search"
        ),

        "initial_route": (
            "external_search"
        ),

        "final_route": (
            "external_search"
        ),

        "retry_triggered": False,

        "answer_status": (
            "external_search_required"
        ),
    }

    run_case(
        "CASE 1 - Direct Success",
        direct_success,
    )

    run_case(
        "CASE 2 - Rewrite Success",
        rewrite_success,
    )

    run_case(
        "CASE 3 - Retry Success",
        retry_success,
    )

    run_case(
        "CASE 4 - Safe Abstention",
        safe_abstention,
    )

    run_case(
        "CASE 5 - External Search",
        external_search,
    )


if __name__ == "__main__":
    main()