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

from app.strategy.memory_strategy import (
    MemoryStrategyAdvisor,
)


class AdaptiveRetriever:
    """
    Self-improving adaptive RAG pipeline.

    V5 pipeline:

        Query
          ↓
        Experience Memory
          ↓
        Memory Strategy?
        /              \
      Yes               No
       ↓                 ↓
    Memory            QueryRouter
    Strategy
       \                /
        \              /
            Retrieval
               ↓
            Reranker
               ↓
             Grader
               ↓
        Retry / Generation
               ↓
           Reflection
               ↓
        Experience Memory
    """

    def __init__(
        self,
        document_path: str | Path,
        memory_path: str | Path = (
            "memory/experience_memory.json"
        ),
        memory_min_similarity: float = 0.75,
    ) -> None:

        self.document_path = Path(
            document_path
        )

        # ==================================================
        # Knowledge Retrieval
        # ==================================================

        self.searcher = DocumentSearcher(
            file_path=self.document_path
        )

        print(
            f"Indexing document: "
            f"{self.document_path}"
        )

        self.searcher.index_document()

        # ==================================================
        # Static Router
        # ==================================================

        self.router = QueryRouter()

        # ==================================================
        # Reranker
        # ==================================================

        self.reranker = Reranker()

        # ==================================================
        # Retrieval Grader
        # ==================================================

        self.grader = RetrievalGrader(
            score_threshold=-0.5,
            min_relevant_docs=1,
            inspect_top_k=3,
        )

        # ==================================================
        # Reflection
        # ==================================================

        self.reflector = Reflector()

        # ==================================================
        # Experience Memory
        # ==================================================

        self.memory = ExperienceMemory(
            memory_path=memory_path
        )

        self.memory_advisor = (
            MemoryStrategyAdvisor(
                memory=self.memory,
                min_similarity=(
                    memory_min_similarity
                ),
            )
        )

        # ==================================================
        # Lazy-loaded models
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

        return self._rewriter

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

        return self._answer_generator

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

        return self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=5,
        )

    # ======================================================
    # Rewrite Retrieval
    # ======================================================

    def _retrieve_with_rewrite(
        self,
        query: str,
    ) -> tuple[str, list[dict]]:

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

        original_results = (
            self.searcher.search(
                query=query,
                limit=10,
            )
        )

        rewritten_results = (
            self.searcher.search(
                query=rewritten_query,
                limit=10,
            )
        )

        fused_results = (
            reciprocal_rank_fusion(
                [
                    original_results,
                    rewritten_results,
                ],
                top_k=10,
            )
        )

        results = (
            self.reranker.rerank(
                query=query,
                documents=fused_results,
                top_k=5,
            )
        )

        return (
            rewritten_query,
            results,
        )

    # ======================================================
    # Empty Grade
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
                "because experience memory "
                "recommended external search."
            ),
        )

    # ======================================================
    # Reflection + Memory Write
    # ======================================================

    def _reflect_and_store(
        self,
        execution: dict,
    ) -> dict:

        reflection = (
            self.reflector.reflect(
                execution
            )
        )

        experience = (
            self.memory.add(
                query=execution[
                    "query"
                ],
                reflection=reflection,
            )
        )

        execution[
            "reflection"
        ] = reflection

        execution[
            "stored_experience"
        ] = experience

        execution[
            "memory_size"
        ] = self.memory.size()

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
            f"Memory Size: "
            f"{self.memory.size()}"
        )

        return execution

    # ======================================================
    # Main Retrieval
    # ======================================================

    def retrieve(
        self,
        query: str,
    ) -> dict:

        # ==================================================
        # V3 Router baseline
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
        # V5 Experience Memory
        # ==================================================

        memory_decision = (
            self.memory_advisor.advise(
                query
            )
        )

        strategy_source = "router"

        if memory_decision is not None:

            strategy_source = "memory"

            selected_strategy = (
                memory_decision.strategy
            )

            print(
                "\nMemory Strategy Found"
            )

            print(
                "-" * 50
            )

            print(
                f"Strategy   : "
                f"{memory_decision.strategy}"
            )

            print(
                f"Similarity : "
                f"{memory_decision.similarity:.4f}"
            )

            print(
                f"Past Query : "
                f"{memory_decision.past_query}"
            )

            print(
                f"Lesson     : "
                f"{memory_decision.lesson}"
            )

        else:

            selected_strategy = (
                router_decision.route
            )

            print(
                "\nNo sufficiently similar "
                "experience found."
            )

            print(
                "Using QueryRouter."
            )

        # ==================================================
        # Memory says:
        # local KB is not worth searching again.
        # ==================================================

        if (
            selected_strategy
            == "external_search"
        ):

            print(
                "\nExperience memory recommends "
                "external search."
            )

            print(
                "Skipping redundant local retrieval."
            )

            grade = (
                self._external_search_grade()
            )

            execution = {
                "query": query,

                "strategy_source": (
                    strategy_source
                ),

                "router_decision": (
                    router_decision
                ),

                "memory_strategy": (
                    memory_decision
                ),

                "initial_route": (
                    "external_search"
                ),

                "final_route": (
                    "external_search"
                ),

                "rewritten_query": None,

                "retry_triggered": False,

                "initial_grade": grade,

                "final_grade": grade,

                "answer_status": (
                    "external_search_required"
                ),

                "answer": (
                    "A similar past experience "
                    "indicates that the local "
                    "knowledge base is unlikely "
                    "to contain enough evidence. "
                    "External search is recommended."
                ),

                "results": [],
            }

            return self._reflect_and_store(
                execution
            )

        # ==================================================
        # First-pass retrieval
        # ==================================================

        rewritten_query = None
        retry_triggered = False

        if selected_strategy == "direct":

            initial_route = "direct"

            results = (
                self._retrieve_direct(
                    query
                )
            )

        elif selected_strategy in {
            "rewrite",
            "rewrite_then_external_search",
        }:

            initial_route = "rewrite"

            (
                rewritten_query,
                results,
            ) = (
                self._retrieve_with_rewrite(
                    query
                )
            )

        else:

            # Unknown memory strategy should never break
            # the pipeline. Fall back to QueryRouter.

            initial_route = (
                router_decision.route
            )

            if initial_route == "rewrite":

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
        # First Grade
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
        # Only DIRECT retrieval gets rewrite retry.
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
        # Generation / Abstention
        # ==================================================

        if final_grade.is_sufficient:

            print(
                "\nRetrieval evidence is sufficient."
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

        else:

            # ----------------------------------------------
            # Memory explicitly recommends:
            # rewrite, then external search if rewrite fails.
            # ----------------------------------------------

            if (
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
                    "The local knowledge base still "
                    "does not provide enough evidence "
                    "after query rewriting. "
                    "External search is recommended."
                )

            else:

                print(
                    "\nRetrieval evidence is insufficient."
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
        # Complete Execution State
        # ==================================================

        execution = {
            "query": query,

            "strategy_source": (
                strategy_source
            ),

            "router_decision": (
                router_decision
            ),

            "memory_strategy": (
                memory_decision
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
        # Experience Memory Write
        # ==================================================

        return self._reflect_and_store(
            execution
        )