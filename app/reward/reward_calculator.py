from dataclasses import dataclass


@dataclass
class RewardResult:
    """
    Reward assigned to one completed RAG execution.
    """

    total_reward: float

    quality_reward: float
    rewrite_penalty: float
    retry_penalty: float
    failure_penalty: float

    strategy: str
    reason: str


class RewardCalculator:
    """
    Reward function for adaptive RAG strategy optimization.

    Goals
    -----
    1. Reward grounded successful answers.
    2. Reward safe abstention when evidence is insufficient.
    3. Reward correct routing toward external search.
    4. Penalize unnecessary query rewriting.
    5. Penalize unnecessary retries.
    6. Penalize unknown / invalid execution failures.

    This is a deterministic reward baseline,
    not reinforcement-learning model training.
    """

    def __init__(
        self,
        success_reward: float = 1.0,
        abstention_reward: float = 0.20,
        external_search_reward: float = 0.30,
        rewrite_cost: float = 0.10,
        retry_cost: float = 0.15,
        failure_cost: float = 1.0,
    ) -> None:

        self.success_reward = (
            success_reward
        )

        self.abstention_reward = (
            abstention_reward
        )

        self.external_search_reward = (
            external_search_reward
        )

        self.rewrite_cost = (
            rewrite_cost
        )

        self.retry_cost = (
            retry_cost
        )

        self.failure_cost = (
            failure_cost
        )

    def calculate(
        self,
        execution: dict,
    ) -> RewardResult:

        answer_status = execution[
            "answer_status"
        ]

        initial_route = execution[
            "initial_route"
        ]

        final_route = execution[
            "final_route"
        ]

        retry_triggered = execution[
            "retry_triggered"
        ]

        selected_strategy = execution.get(
            "selected_strategy"
        )

        # ==================================================
        # Determine the actually executed strategy
        # ==================================================

        if final_route == "external_search":

            strategy = (
                "external_search"
            )

        elif retry_triggered:

            strategy = (
                "rewrite_retry"
            )

        elif (
            selected_strategy
            == "rewrite_then_external_search"
            and answer_status
            == "external_search_required"
        ):

            strategy = (
                "rewrite_then_external_search"
            )

        elif initial_route == "rewrite":

            strategy = (
                "rewrite"
            )

        else:

            strategy = (
                "direct"
            )

        # ==================================================
        # Outcome reward
        # ==================================================

        failure_penalty = 0.0

        if answer_status == "generated":

            quality_reward = (
                self.success_reward
            )

            reason = (
                "A grounded answer was generated."
            )

        elif (
            answer_status
            == "insufficient_evidence"
        ):

            quality_reward = (
                self.abstention_reward
            )

            reason = (
                "The system correctly abstained "
                "because the local knowledge base "
                "did not contain sufficient evidence."
            )

        elif (
            answer_status
            == "external_search_required"
        ):

            quality_reward = (
                self.external_search_reward
            )

            reason = (
                "The system correctly routed the "
                "query toward external search instead "
                "of forcing an unsupported answer."
            )

        else:

            quality_reward = 0.0

            failure_penalty = (
                self.failure_cost
            )

            reason = (
                "The execution ended in an "
                "unknown or invalid failure state."
            )

        # ==================================================
        # Rewrite cost
        # ==================================================

        used_rewrite = (
            initial_route == "rewrite"
            or retry_triggered
        )

        if used_rewrite:

            rewrite_penalty = (
                self.rewrite_cost
            )

        else:

            rewrite_penalty = 0.0

        # ==================================================
        # Retry cost
        # ==================================================

        if retry_triggered:

            retry_penalty = (
                self.retry_cost
            )

        else:

            retry_penalty = 0.0

        # ==================================================
        # Final Reward
        # ==================================================

        total_reward = (
            quality_reward
            - rewrite_penalty
            - retry_penalty
            - failure_penalty
        )

        return RewardResult(
            total_reward=round(
                total_reward,
                4,
            ),

            quality_reward=(
                quality_reward
            ),

            rewrite_penalty=(
                rewrite_penalty
            ),

            retry_penalty=(
                retry_penalty
            ),

            failure_penalty=(
                failure_penalty
            ),

            strategy=strategy,

            reason=reason,
        )