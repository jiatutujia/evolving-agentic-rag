from pathlib import Path

from app.grading.retrieval_grader import (
    RetrievalGrade,
    RetrievalGrader,
)

from app.memory.experience_memory import (
    ExperienceMemory,
)

from app.reflection.reflector import (
    Reflector,
)

from app.reward.reward_calculator import (
    RewardCalculator,
)

from app.reward.reward_history import (
    RewardHistory,
)

from app.retrieval.document_search import (
    DocumentSearcher,
)

from app.retrieval.fusion import (
    reciprocal_rank_fusion,
)

from app.retrieval.reranker import (
    Reranker,
)

from app.routing.query_router import (
    QueryRouter,
)

from app.strategy.reward_strategy import (
    RewardAwareStrategyAdvisor,
)


class AdaptiveRetriever:
    """
    Self-evolving adaptive RAG pipeline.

    V6 Pipeline
    -----------

    Query
      ↓
    Experience Memory
      +
    Reward History
      ↓
    Reward-Aware Strategy
      ↓
    direct / rewrite / external_search
      ↓
    Retrieval
      ↓
    Reranker
      ↓
    Retrieval Grader
      ↓
    Retry / Generation
      ↓
    Reflection
      ↓
    ┌───────────────────────┐
    │                       │
    Experience Memory    Reward History
    │                       │
    └────────────┬──────────┘
                 ↓
          Future Strategy
    """

    def __init__(
        self,
        document_path: str | Path,

        memory_path: str | Path = (
            "memory/experience_memory.json"
        ),

        reward_history_path: str | Path = (
            "memory/reward_history.json"
        ),

        memory_min_similarity: float = 0.75,
    ) -> None:

        self.document_path = Path(
            document_path
        )

        # ==================================================
        # Knowledge Retrieval
        # ==================================================

        self.searcher = (
            DocumentSearcher(
                file_path=(
                    self.document_path
                )
            )
        )

        print(
            f"Indexing document: "
            f"{self.document_path}"
        )

        self.searcher.index_document()

        # ==================================================
        # Static Query Router
        # ==================================================

        self.router = (
            QueryRouter()
        )

        # ==================================================
        # CrossEncoder Reranker
        # ==================================================

        self.reranker = (
            Reranker()
        )

        # ==================================================
        # Retrieval Grader
        # ==================================================

        self.grader = (
            RetrievalGrader(
                score_threshold=-0.5,
                min_relevant_docs=1,
                inspect_top_k=3,
            )
        )

        # ==================================================
        # Reflection
        # ==================================================

        self.reflector = (
            Reflector()
        )

        # ==================================================
        # Experience Memory
        # ==================================================

        self.memory = (
            ExperienceMemory(
                memory_path=(
                    memory_path
                )
            )
        )

        # ==================================================
        # Reward
        # ==================================================

        self.reward_calculator = (
            RewardCalculator()
        )

        self.reward_history = (
            RewardHistory(
                path=(
                    reward_history_path
                )
            )
        )

        # ==================================================
        # Reward-aware Strategy
        # ==================================================

        self.strategy_advisor = (
            RewardAwareStrategyAdvisor(
                memory=(
                    self.memory
                ),

                reward_history=(
                    self.reward_history
                ),

                min_similarity=(
                    memory_min_similarity
                ),

                top_k=5,

                similarity_weight=0.8,

                reward_weight=0.2,
            )
        )

        # ==================================================
        # Lazy Models
        # ==================================================

        self._rewriter = None

        self._answer_generator = None

    # ======================================================
    # Lazy Query Rewriter
    # ======================================================

    def _get_rewriter(
        self,
    ):

        if self._rewriter is None:

            print(
                "Loading QueryRewriter..."
            )

            from app.retrieval.query_rewriter import (
                QueryRewriter,
            )

            self._rewriter = (
                QueryRewriter()
            )

        return (
            self._rewriter
        )

    # ======================================================
    # Lazy Answer Generator
    # ======================================================

    def _get_answer_generator(
        self,
    ):

        if self._answer_generator is None:

            print(
                "Loading AnswerGenerator..."
            )

            from app.generation.answer_generator import (
                AnswerGenerator,
            )

            self._answer_generator = (
                AnswerGenerator()
            )

        return (
            self._answer_generator
        )

    # ======================================================
    # Direct Retrieval
    # ======================================================

    def _retrieve_direct(
        self,
        query: str,
    ) -> list[dict]:

        candidates = (
            self.searcher.search(
                query=query,
                limit=10,
            )
        )

        results = (
            self.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=5,
            )
        )

        return results

    # ======================================================
    # Rewrite Retrieval
    # ======================================================

    def _retrieve_with_rewrite(
        self,
        query: str,
    ) -> tuple[
        str,
        list[dict],
    ]:

        rewriter = (
            self._get_rewriter()
        )

        rewritten_query = (
            rewriter.rewrite(
                query
            )
        )

        print(
            f"Rewritten Query: "
            f"{rewritten_query}"
        )

        # --------------------------------------------------
        # Original query retrieval
        # --------------------------------------------------

        original_results = (
            self.searcher.search(
                query=query,
                limit=10,
            )
        )

        # --------------------------------------------------
        # Rewritten query retrieval
        # --------------------------------------------------

        rewritten_results = (
            self.searcher.search(
                query=(
                    rewritten_query
                ),
                limit=10,
            )
        )

        # --------------------------------------------------
        # RRF
        # --------------------------------------------------

        fused_results = (
            reciprocal_rank_fusion(
                [
                    original_results,
                    rewritten_results,
                ],
                top_k=10,
            )
        )

        # --------------------------------------------------
        # Final reranking always uses ORIGINAL query
        # --------------------------------------------------

        results = (
            self.reranker.rerank(
                query=query,
                documents=(
                    fused_results
                ),
                top_k=5,
            )
        )

        return (
            rewritten_query,
            results,
        )

    # ======================================================
    # Grade used when local retrieval is skipped
    # ======================================================

    def _external_search_grade(
        self,
    ) -> RetrievalGrade:

        return RetrievalGrade(
            is_sufficient=False,

            best_score=float(
                "-inf"
            ),

            relevant_count=0,

            inspected_count=0,

            threshold=(
                self.grader.score_threshold
            ),

            reason=(
                "Local retrieval was skipped "
                "because the learned strategy "
                "recommended external search."
            ),
        )

    # ======================================================
    # Reflection
    # +
    # Experience Memory
    # +
    # Reward
    # ======================================================

    def _reflect_and_store(
        self,
        execution: dict,
    ) -> dict:

        # --------------------------------------------------
        # Reflection
        # --------------------------------------------------

        reflection = (
            self.reflector.reflect(
                execution
            )
        )

        # --------------------------------------------------
        # Experience Memory
        # --------------------------------------------------

        experience = (
            self.memory.add(
                query=(
                    execution[
                        "query"
                    ]
                ),

                reflection=(
                    reflection
                ),
            )
        )

        # --------------------------------------------------
        # Reward
        # --------------------------------------------------

        reward = (
            self.reward_calculator.calculate(
                execution
            )
        )

        # --------------------------------------------------
        # Reward History
        # --------------------------------------------------

        reward_record = (
            self.reward_history.add(
                query=(
                    execution[
                        "query"
                    ]
                ),

                strategy=(
                    reward.strategy
                ),

                reward=(
                    reward.total_reward
                ),

                answer_status=(
                    execution[
                        "answer_status"
                    ]
                ),
            )
        )

        # --------------------------------------------------
        # Add learning state
        # --------------------------------------------------

        execution[
            "reflection"
        ] = reflection

        execution[
            "stored_experience"
        ] = experience

        execution[
            "reward"
        ] = reward

        execution[
            "reward_record"
        ] = reward_record

        execution[
            "memory_size"
        ] = (
            self.memory.size()
        )

        execution[
            "reward_history_size"
        ] = (
            self.reward_history.size()
        )

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        print(
            "\nReflection"
        )

        print(
            "-" * 50
        )

        print(
            f"Outcome  : "
            f"{reflection.outcome}"
        )

        print(
            f"Strategy : "
            f"{reflection.recommended_strategy}"
        )

        print(
            f"Lesson   : "
            f"{reflection.lesson}"
        )

        print(
            "\nReward"
        )

        print(
            "-" * 50
        )

        print(
            f"Executed Strategy : "
            f"{reward.strategy}"
        )

        print(
            f"Quality Reward    : "
            f"{reward.quality_reward:.2f}"
        )

        print(
            f"Rewrite Penalty   : "
            f"{reward.rewrite_penalty:.2f}"
        )

        print(
            f"Retry Penalty     : "
            f"{reward.retry_penalty:.2f}"
        )

        print(
            f"Failure Penalty   : "
            f"{reward.failure_penalty:.2f}"
        )

        print(
            f"Total Reward      : "
            f"{reward.total_reward:.2f}"
        )

        print(
            f"\nMemory Size       : "
            f"{self.memory.size()}"
        )

        print(
            f"Reward History    : "
            f"{self.reward_history.size()}"
        )

        return execution

    # ======================================================
    # Main Pipeline
    # ======================================================

    def retrieve(
        self,
        query: str,
    ) -> dict:

        # ==================================================
        # Static Router Baseline
        # ==================================================

        router_decision = (
            self.router.route(
                query
            )
        )

        print(
            f"\nRouter Baseline: "
            f"{router_decision.route}"
        )

        print(
            f"Router Reasons: "
            f"{router_decision.reasons}"
        )

        # ==================================================
        # Reward-Aware Policy
        # ==================================================

        policy_decision = (
            self.strategy_advisor.advise(
                query
            )
        )

        if policy_decision is not None:

            strategy_source = (
                "reward_memory"
            )

            selected_strategy = (
                policy_decision.strategy
            )

            print(
                "\nReward-Aware Strategy Found"
            )

            print(
                "-" * 50
            )

            print(
                f"Strategy       : "
                f"{policy_decision.strategy}"
            )

            print(
                f"Similarity     : "
                f"{policy_decision.similarity:.4f}"
            )

            if (
                policy_decision.average_reward
                is None
            ):

                reward_text = (
                    "None"
                )

            else:

                reward_text = (
                    f"{policy_decision.average_reward:.4f}"
                )

            print(
                f"Average Reward : "
                f"{reward_text}"
            )

            print(
                f"Combined Score : "
                f"{policy_decision.combined_score:.4f}"
            )

            print(
                f"Past Query     : "
                f"{policy_decision.past_query}"
            )

            print(
                f"Past Outcome   : "
                f"{policy_decision.past_outcome}"
            )

            print(
                f"Lesson         : "
                f"{policy_decision.lesson}"
            )

        else:

            strategy_source = (
                "router"
            )

            selected_strategy = (
                router_decision.route
            )

            print(
                "\nNo reward-aware "
                "strategy found."
            )

            print(
                "Using QueryRouter."
            )

        # ==================================================
        # External Search Strategy
        #
        # Real Web Search is NOT connected yet.
        # ==================================================

        if (
            selected_strategy
            == "external_search"
        ):

            print(
                "\nPolicy recommends "
                "external search."
            )

            print(
                "Skipping redundant "
                "local retrieval."
            )

            grade = (
                self._external_search_grade()
            )

            execution = {
                "query": query,

                "strategy_source": (
                    strategy_source
                ),

                "selected_strategy": (
                    selected_strategy
                ),

                "router_decision": (
                    router_decision
                ),

                "policy_decision": (
                    policy_decision
                ),

                "initial_route": (
                    "external_search"
                ),

                "final_route": (
                    "external_search"
                ),

                "rewritten_query": None,

                "retry_triggered": False,

                "initial_grade": (
                    grade
                ),

                "final_grade": (
                    grade
                ),

                "answer_status": (
                    "external_search_required"
                ),

                "answer": (
                    "A similar historical "
                    "experience indicates that "
                    "the local knowledge base "
                    "is unlikely to contain "
                    "enough evidence. "
                    "External search is recommended."
                ),

                "results": [],
            }

            return (
                self._reflect_and_store(
                    execution
                )
            )

        # ==================================================
        # First-pass Retrieval
        # ==================================================

        rewritten_query = None

        retry_triggered = False

        # --------------------------------------------------
        # Direct
        # --------------------------------------------------

        if selected_strategy == "direct":

            initial_route = (
                "direct"
            )

            results = (
                self._retrieve_direct(
                    query
                )
            )

        # --------------------------------------------------
        # Rewrite
        # --------------------------------------------------

        elif selected_strategy in {
            "rewrite",
            "rewrite_then_external_search",
        }:

            initial_route = (
                "rewrite"
            )

            (
                rewritten_query,
                results,
            ) = (
                self._retrieve_with_rewrite(
                    query
                )
            )

        # --------------------------------------------------
        # Unknown strategy:
        # fall back to Router
        # --------------------------------------------------

        else:

            selected_strategy = (
                router_decision.route
            )

            initial_route = (
                selected_strategy
            )

            if (
                initial_route
                == "rewrite"
            ):

                (
                    rewritten_query,
                    results,
                ) = (
                    self._retrieve_with_rewrite(
                        query
                    )
                )

            else:

                results = (
                    self._retrieve_direct(
                        query
                    )
                )

        # ==================================================
        # Initial Grade
        # ==================================================

        initial_grade = (
            self.grader.grade(
                results
            )
        )

        print(
            "\nInitial Retrieval Grade"
        )

        print(
            "-" * 50
        )

        print(
            f"Sufficient : "
            f"{initial_grade.is_sufficient}"
        )

        print(
            f"Best Score : "
            f"{initial_grade.best_score:.4f}"
        )

        print(
            f"Reason     : "
            f"{initial_grade.reason}"
        )

        final_grade = (
            initial_grade
        )

        # ==================================================
        # Retry
        #
        # Direct retrieval receives one rewrite retry.
        # ==================================================

        if (
            not initial_grade.is_sufficient
            and initial_route == "direct"
        ):

            print(
                "\nRetrieval insufficient."
            )

            print(
                "Triggering rewrite retry..."
            )

            retry_triggered = True

            (
                rewritten_query,
                results,
            ) = (
                self._retrieve_with_rewrite(
                    query
                )
            )

            final_grade = (
                self.grader.grade(
                    results
                )
            )

            print(
                "\nRetry Retrieval Grade"
            )

            print(
                "-" * 50
            )

            print(
                f"Sufficient : "
                f"{final_grade.is_sufficient}"
            )

            print(
                f"Best Score : "
                f"{final_grade.best_score:.4f}"
            )

            print(
                f"Reason     : "
                f"{final_grade.reason}"
            )

        # ==================================================
        # Final Route
        # ==================================================

        if retry_triggered:

            final_route = (
                "rewrite_retry"
            )

        else:

            final_route = (
                initial_route
            )

        # ==================================================
        # Generation
        # ==================================================

        if final_grade.is_sufficient:

            print(
                "\nRetrieval evidence "
                "is sufficient."
            )

            print(
                "Generating grounded answer..."
            )

            generator = (
                self._get_answer_generator()
            )

            answer = (
                generator.generate(
                    query=query,
                    results=results,
                )
            )

            answer_status = (
                "generated"
            )

        # ==================================================
        # Rewrite → external-search policy
        # ==================================================

        elif (
            selected_strategy
            == "rewrite_then_external_search"
        ):

            print(
                "\nRewrite retrieval remained "
                "insufficient."
            )

            print(
                "External search is recommended."
            )

            answer_status = (
                "external_search_required"
            )

            answer = (
                "The local knowledge base "
                "still does not provide "
                "enough evidence after "
                "query rewriting. "
                "External search is recommended."
            )

        # ==================================================
        # Safe Abstention
        # ==================================================

        else:

            print(
                "\nRetrieval evidence "
                "is insufficient."
            )

            print(
                "Skipping answer generation."
            )

            answer_status = (
                "insufficient_evidence"
            )

            answer = (
                "I do not have enough evidence "
                "in the knowledge base to answer "
                "this question."
            )

        # ==================================================
        # Execution State
        # ==================================================

        execution = {
            "query": query,

            "strategy_source": (
                strategy_source
            ),

            "selected_strategy": (
                selected_strategy
            ),

            "router_decision": (
                router_decision
            ),

            "policy_decision": (
                policy_decision
            ),

            "initial_route": (
                initial_route
            ),

            "final_route": (
                final_route
            ),

            "rewritten_query": (
                rewritten_query
            ),

            "retry_triggered": (
                retry_triggered
            ),

            "initial_grade": (
                initial_grade
            ),

            "final_grade": (
                final_grade
            ),

            "answer_status": (
                answer_status
            ),

            "answer": (
                answer
            ),

            "results": (
                results
            ),
        }

        # ==================================================
        # Reflection
        # +
        # Experience Memory
        # +
        # Reward
        # ==================================================

        return (
            self._reflect_and_store(
                execution
            )
        )