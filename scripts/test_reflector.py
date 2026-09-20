from dataclasses import dataclass

from app.reflection.reflector import (
    Reflector,
)


@dataclass
class FakeGrade:
    is_sufficient: bool
    best_score: float


def run_case(
    name: str,
    execution: dict,
) -> None:

    reflector = Reflector()

    reflection = (
        reflector.reflect(
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
        f"Outcome              : "
        f"{reflection.outcome}"
    )

    print(
        f"Success              : "
        f"{reflection.success}"
    )

    print(
        f"Initial Route        : "
        f"{reflection.initial_route}"
    )

    print(
        f"Final Route          : "
        f"{reflection.final_route}"
    )

    print(
        f"Retry Triggered      : "
        f"{reflection.retry_triggered}"
    )

    print(
        f"Initial Score        : "
        f"{reflection.initial_score:.4f}"
    )

    print(
        f"Final Score          : "
        f"{reflection.final_score:.4f}"
    )

    print(
        f"Score Delta          : "
        f"{reflection.score_delta:.4f}"
    )

    print(
        f"Recommended Strategy : "
        f"{reflection.recommended_strategy}"
    )

    print(
        f"Lesson               : "
        f"{reflection.lesson}"
    )


def main() -> None:

    # ==================================================
    # Case 1:
    # Direct retrieval succeeds immediately
    # ==================================================

    direct_success = {
        "initial_route": "direct",

        "final_route": "direct",

        "retry_triggered": False,

        "initial_grade": FakeGrade(
            is_sufficient=True,
            best_score=5.97,
        ),

        "final_grade": FakeGrade(
            is_sufficient=True,
            best_score=5.97,
        ),

        "answer_status": "generated",
    }

    # ==================================================
    # Case 2:
    # Direct retrieval failed,
    # rewrite retry recovered
    # ==================================================

    retry_success = {
        "initial_route": "direct",

        "final_route": "rewrite_retry",

        "retry_triggered": True,

        "initial_grade": FakeGrade(
            is_sufficient=False,
            best_score=-2.5,
        ),

        "final_grade": FakeGrade(
            is_sufficient=True,
            best_score=2.8,
        ),

        "answer_status": "generated",
    }

    # ==================================================
    # Case 3:
    # Retry does not improve retrieval
    # ==================================================

    retry_failure = {
        "initial_route": "direct",

        "final_route": "rewrite_retry",

        "retry_triggered": True,

        "initial_grade": FakeGrade(
            is_sufficient=False,
            best_score=-7.17,
        ),

        "final_grade": FakeGrade(
            is_sufficient=False,
            best_score=-7.17,
        ),

        "answer_status": (
            "insufficient_evidence"
        ),
    }

    run_case(
        "CASE 1 - Direct Success",
        direct_success,
    )

    run_case(
        "CASE 2 - Retry Recovery",
        retry_success,
    )

    run_case(
        "CASE 3 - Retry Failure",
        retry_failure,
    )


if __name__ == "__main__":
    main()