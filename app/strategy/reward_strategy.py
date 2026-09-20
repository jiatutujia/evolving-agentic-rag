from dataclasses import dataclass

from app.memory.experience_memory import (
    ExperienceMemory,
)

from app.reward.reward_history import (
    RewardHistory,
)


@dataclass
class RewardAwareStrategyDecision:
    """
    Strategy decision derived from:

    semantic similarity
    +
    historical strategy reward
    """

    strategy: str

    combined_score: float

    similarity: float

    average_reward: float | None

    past_query: str

    past_outcome: str

    lesson: str

    reason: str


class RewardAwareStrategyAdvisor:
    """
    Reward-aware policy selector.

    V5:
        similar experience
        → strategy

    V6:
        similar experience
        +
        historical reward
        → strategy

    This is an online reward-aware strategy baseline,
    not neural RL training.
    """

    def __init__(
        self,
        memory: ExperienceMemory,
        reward_history: RewardHistory,
        min_similarity: float = 0.75,
        top_k: int = 5,
        similarity_weight: float = 0.8,
        reward_weight: float = 0.2,
    ) -> None:

        self.memory = memory

        self.reward_history = (
            reward_history
        )

        self.min_similarity = (
            min_similarity
        )

        self.top_k = (
            top_k
        )

        self.similarity_weight = (
            similarity_weight
        )

        self.reward_weight = (
            reward_weight
        )

    # ======================================================
    # Experience strategy
    # →
    # executable strategy
    # ======================================================

    def _map_strategy(
        self,
        recommended_strategy: str,
    ) -> str | None:

        mapping = {
            "prefer_direct": (
                "direct"
            ),

            "prefer_rewrite": (
                "rewrite"
            ),

            "external_search": (
                "external_search"
            ),

            "rewrite_then_external_search": (
                "rewrite_then_external_search"
            ),
        }

        return mapping.get(
            recommended_strategy
        )

    # ======================================================
    # Main policy
    # ======================================================

    def advise(
        self,
        query: str,
    ) -> RewardAwareStrategyDecision | None:

        matches = (
            self.memory.search(
                query=query,

                top_k=(
                    self.top_k
                ),

                min_similarity=(
                    self.min_similarity
                ),
            )
        )

        if not matches:
            return None

        candidates = []

        for match in matches:

            experience = (
                match.experience
            )

            strategy = (
                self._map_strategy(
                    experience.recommended_strategy
                )
            )

            if strategy is None:
                continue

            # ==============================================
            # Historical reward
            # ==============================================

            average_reward = (
                self.reward_history.average_reward(
                    strategy
                )
            )

            # Unexplored strategy:
            # use neutral reward.
            if average_reward is None:

                reward_signal = 0.0

            else:

                reward_signal = float(
                    average_reward
                )

            # ==============================================
            # Combined policy score
            # ==============================================

            combined_score = (
                self.similarity_weight
                * match.similarity
                +
                self.reward_weight
                * reward_signal
            )

            candidates.append(
                {
                    "strategy": (
                        strategy
                    ),

                    "combined_score": (
                        combined_score
                    ),

                    "similarity": (
                        match.similarity
                    ),

                    "average_reward": (
                        average_reward
                    ),

                    "experience": (
                        experience
                    ),
                }
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[
                    "combined_score"
                ]
            ),
            reverse=True,
        )

        best = (
            candidates[0]
        )

        experience = (
            best[
                "experience"
            ]
        )

        average_reward = (
            best[
                "average_reward"
            ]
        )

        if average_reward is None:

            reward_text = (
                "no historical reward"
            )

        else:

            reward_text = (
                f"average reward "
                f"{average_reward:.4f}"
            )

        reason = (
            f"Selected strategy "
            f"{best['strategy']} using "
            f"semantic similarity "
            f"{best['similarity']:.4f} "
            f"and {reward_text}. "
            f"Combined score was "
            f"{best['combined_score']:.4f}."
        )

        return (
            RewardAwareStrategyDecision(
                strategy=(
                    best[
                        "strategy"
                    ]
                ),

                combined_score=float(
                    best[
                        "combined_score"
                    ]
                ),

                similarity=float(
                    best[
                        "similarity"
                    ]
                ),

                average_reward=(
                    average_reward
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
        )