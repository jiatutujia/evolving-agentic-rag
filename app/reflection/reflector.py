from dataclasses import dataclass


@dataclass
class ReflectionResult:
    """
    Structured reflection over one RAG execution.
    """

    outcome: str
    success: bool

    initial_route: str
    final_route: str

    retry_triggered: bool

    initial_score: float
    final_score: float
    score_delta: float

    lesson: str
    recommended_strategy: str


class Reflector:
    """
    Analyze a completed RAG execution and extract
    a reusable strategy lesson.

    V5 baseline uses deterministic reflection rules.
    """

    def reflect(
        self,
        execution: dict,
    ) -> ReflectionResult:

        initial_route = execution[
            "initial_route"
        ]

        final_route = execution[
            "final_route"
        ]

        retry_triggered = execution[
            "retry_triggered"
        ]

        answer_status = execution[
            "answer_status"
        ]

        # ==================================================
        # Case 0:
        # Memory says local KB should be bypassed and an
        # external-search strategy should be used.
        #
        # No retrieval score exists in this case.
        # ==================================================

        if (
            answer_status
            == "external_search_required"
        ):

            return ReflectionResult(
                outcome=(
                    "external_search_required"
                ),

                success=False,

                initial_route=(
                    initial_route
                ),

                final_route=(
                    final_route
                ),

                retry_triggered=(
                    retry_triggered
                ),

                initial_score=0.0,
                final_score=0.0,
                score_delta=0.0,

                lesson=(
                    "A highly similar past experience "
                    "indicates that the local knowledge "
                    "base is unlikely to contain enough "
                    "evidence. External search should be "
                    "preferred."
                ),

                recommended_strategy=(
                    "external_search"
                ),
            )

        # ==================================================
        # Normal retrieval execution
        # ==================================================

        initial_grade = execution[
            "initial_grade"
        ]

        final_grade = execution[
            "final_grade"
        ]

        initial_score = float(
            initial_grade.best_score
        )

        final_score = float(
            final_grade.best_score
        )

        score_delta = (
            final_score
            - initial_score
        )

        # ==================================================
        # Case 1:
        # First-pass retrieval succeeds
        # ==================================================

        if (
            final_grade.is_sufficient
            and not retry_triggered
        ):

            success = True

            outcome = (
                "success_first_pass"
            )

            if initial_route == "direct":

                lesson = (
                    "The original query produced "
                    "sufficient retrieval evidence "
                    "without query rewriting."
                )

                recommended_strategy = (
                    "prefer_direct"
                )

            else:

                lesson = (
                    "Query rewriting produced "
                    "sufficient retrieval evidence."
                )

                recommended_strategy = (
                    "prefer_rewrite"
                )

        # ==================================================
        # Case 2:
        # Direct retrieval failed,
        # rewrite retry recovered
        # ==================================================

        elif (
            final_grade.is_sufficient
            and retry_triggered
        ):

            success = True

            outcome = (
                "success_after_retry"
            )

            lesson = (
                "Direct retrieval was insufficient, "
                "but query rewriting and retry "
                "recovered useful evidence."
            )

            recommended_strategy = (
                "prefer_rewrite"
            )

        # ==================================================
        # Case 3:
        # Retry happened but still failed
        # ==================================================

        elif (
            not final_grade.is_sufficient
            and retry_triggered
        ):

            success = False

            outcome = (
                "failed_after_retry"
            )

            if score_delta > 0.5:

                lesson = (
                    "Query rewriting improved retrieval "
                    "quality, but the evidence was still "
                    "insufficient for grounded generation."
                )

                recommended_strategy = (
                    "rewrite_then_external_search"
                )

            else:

                lesson = (
                    "Query rewriting did not materially "
                    "improve retrieval. The knowledge "
                    "base likely lacks the information "
                    "needed to answer the query."
                )

                recommended_strategy = (
                    "external_search"
                )

        # ==================================================
        # Case 4:
        # Rewrite route itself failed
        # ==================================================

        elif (
            not final_grade.is_sufficient
            and initial_route == "rewrite"
        ):

            success = False

            outcome = (
                "failed_rewrite_path"
            )

            lesson = (
                "The query was already rewritten, "
                "but retrieval evidence remained "
                "insufficient."
            )

            recommended_strategy = (
                "external_search"
            )

        # ==================================================
        # Fallback
        # ==================================================

        else:

            success = (
                answer_status
                == "generated"
            )

            outcome = "unknown"

            lesson = (
                "The execution does not match a "
                "known reflection pattern."
            )

            recommended_strategy = (
                "default"
            )

        return ReflectionResult(
            outcome=outcome,

            success=success,

            initial_route=(
                initial_route
            ),

            final_route=(
                final_route
            ),

            retry_triggered=(
                retry_triggered
            ),

            initial_score=(
                initial_score
            ),

            final_score=(
                final_score
            ),

            score_delta=(
                score_delta
            ),

            lesson=lesson,

            recommended_strategy=(
                recommended_strategy
            ),
        )