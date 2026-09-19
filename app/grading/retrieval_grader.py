from dataclasses import dataclass


@dataclass
class RetrievalGrade:
    """
    Result returned by RetrievalGrader.
    """

    is_sufficient: bool
    best_score: float
    relevant_count: int
    inspected_count: int
    threshold: float
    reason: str


class RetrievalGrader:
    """
    Lightweight retrieval-quality grader.

    The grader uses CrossEncoder reranker scores to
    determine whether the retrieved context is good
    enough for downstream answer generation.

    Note:
        CrossEncoder scores are raw relevance logits,
        not calibrated probabilities.

        Therefore, score_threshold is currently a
        heuristic baseline and should later be
        calibrated using labeled evaluation data.
    """

    def __init__(
        self,
        score_threshold: float = -0.5,
        min_relevant_docs: int = 1,
        inspect_top_k: int = 3,
    ) -> None:

        self.score_threshold = score_threshold
        self.min_relevant_docs = min_relevant_docs
        self.inspect_top_k = inspect_top_k

    def grade(
        self,
        results: list[dict],
    ) -> RetrievalGrade:
        """
        Judge whether retrieval results are sufficient.

        Parameters
        ----------
        results:
            Reranked retrieval results.

            Each result is expected to contain:

                {
                    ...
                    "rerank_score": float
                }

        Returns
        -------
        RetrievalGrade
        """

        # --------------------------------------------------
        # No retrieval results
        # --------------------------------------------------
        if not results:
            return RetrievalGrade(
                is_sufficient=False,
                best_score=float("-inf"),
                relevant_count=0,
                inspected_count=0,
                threshold=self.score_threshold,
                reason="no retrieval results",
            )

        # --------------------------------------------------
        # Only inspect the highest-ranked documents
        # --------------------------------------------------
        inspected = results[
            : self.inspect_top_k
        ]

        scores: list[float] = []

        for result in inspected:

            score = result.get(
                "rerank_score"
            )

            if score is None:
                continue

            scores.append(
                float(score)
            )

        # --------------------------------------------------
        # No reranker scores available
        # --------------------------------------------------
        if not scores:
            return RetrievalGrade(
                is_sufficient=False,
                best_score=float("-inf"),
                relevant_count=0,
                inspected_count=len(
                    inspected
                ),
                threshold=self.score_threshold,
                reason=(
                    "retrieval results contain "
                    "no rerank_score"
                ),
            )

        # --------------------------------------------------
        # Count sufficiently relevant documents
        # --------------------------------------------------
        relevant_scores = [
            score
            for score in scores
            if score
            >= self.score_threshold
        ]

        best_score = max(
            scores
        )

        relevant_count = len(
            relevant_scores
        )

        is_sufficient = (
            relevant_count
            >= self.min_relevant_docs
        )

        # --------------------------------------------------
        # Human-readable reason
        # --------------------------------------------------
        if is_sufficient:

            reason = (
                f"{relevant_count} of "
                f"{len(scores)} inspected "
                f"documents passed the "
                f"rerank threshold "
                f"{self.score_threshold:.2f}"
            )

        else:

            reason = (
                f"only {relevant_count} of "
                f"{len(scores)} inspected "
                f"documents passed the "
                f"rerank threshold "
                f"{self.score_threshold:.2f}"
            )

        return RetrievalGrade(
            is_sufficient=is_sufficient,
            best_score=best_score,
            relevant_count=relevant_count,
            inspected_count=len(
                scores
            ),
            threshold=self.score_threshold,
            reason=reason,
        )