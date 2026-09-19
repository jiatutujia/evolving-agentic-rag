from app.grading.retrieval_grader import (
    RetrievalGrader,
)


def run_case(
    name: str,
    results: list[dict],
) -> None:

    grader = RetrievalGrader()

    grade = grader.grade(
        results
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
        f"Sufficient      : "
        f"{grade.is_sufficient}"
    )

    print(
        f"Best Score      : "
        f"{grade.best_score:.4f}"
    )

    print(
        f"Relevant Count  : "
        f"{grade.relevant_count}"
    )

    print(
        f"Inspected Count : "
        f"{grade.inspected_count}"
    )

    print(
        f"Threshold       : "
        f"{grade.threshold:.4f}"
    )

    print(
        f"Reason          : "
        f"{grade.reason}"
    )


def main() -> None:

    strong_results = [
        {
            "rerank_score": 6.4121
        },
        {
            "rerank_score": 3.3658
        },
        {
            "rerank_score": 2.0478
        },
    ]

    borderline_results = [
        {
            "rerank_score": -0.0476
        },
        {
            "rerank_score": -0.5599
        },
        {
            "rerank_score": -1.3547
        },
    ]

    bad_results = [
        {
            "rerank_score": -2.1
        },
        {
            "rerank_score": -3.4
        },
        {
            "rerank_score": -5.0
        },
    ]

    run_case(
        "CASE 1 - Strong Retrieval",
        strong_results,
    )

    run_case(
        "CASE 2 - Borderline Retrieval",
        borderline_results,
    )

    run_case(
        "CASE 3 - Poor Retrieval",
        bad_results,
    )


if __name__ == "__main__":
    main()