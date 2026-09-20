from dataclasses import dataclass

from app.memory.experience_memory import (
    ExperienceMemory,
)


@dataclass
class MemoryStrategyDecision:
    """
    Strategy recommendation derived from
    a semantically similar past experience.
    """

    strategy: str

    similarity: float

    past_query: str

    past_outcome: str

    lesson: str

    reason: str


class MemoryStrategyAdvisor:
    """
    Use episodic / experience memory to recommend
    an execution strategy for a new query.

    The advisor only overrides the normal router when
    a sufficiently similar historical experience exists.
    """

    def __init__(
        self,
        memory: ExperienceMemory,
        min_similarity: float = 0.75,
    ) -> None:

        self.memory = memory

        self.min_similarity = (
            min_similarity
        )

    def advise(
        self,
        query: str,
    ) -> MemoryStrategyDecision | None:
        """
        Search experience memory and return a strategy
        recommendation when a sufficiently similar
        past experience exists.
        """

        matches = self.memory.search(
            query=query,
            top_k=1,
            min_similarity=(
                self.min_similarity
            ),
        )

        if not matches:
            return None

        match = matches[0]

        experience = (
            match.experience
        )

        recommended = (
            experience.recommended_strategy
        )

        # ==================================================
        # Map experience-level recommendations to
        # executable strategies.
        # ==================================================

        if recommended == "prefer_direct":

            strategy = "direct"

        elif recommended == "prefer_rewrite":

            strategy = "rewrite"

        elif recommended == "external_search":

            strategy = "external_search"

        elif (
            recommended
            == "rewrite_then_external_search"
        ):

            strategy = (
                "rewrite_then_external_search"
            )

        else:

            # Unknown historical recommendation should
            # not override the normal router.
            return None

        reason = (
            "A semantically similar past query "
            f"was found with similarity "
            f"{match.similarity:.4f}. "
            f"Its recommended strategy was "
            f"{recommended}."
        )

        return MemoryStrategyDecision(
            strategy=strategy,

            similarity=(
                match.similarity
            ),

            past_query=(
                experience.query
            ),

            past_outcome=(
                experience.outcome
            ),

            lesson=(
                experience.lesson
            ),

            reason=reason,
        )