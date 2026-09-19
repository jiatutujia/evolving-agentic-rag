from pathlib import Path

from app.grading.retrieval_grader import (
    RetrievalGrader,
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


class AdaptiveRetriever:
    """
    Adaptive retrieval pipeline.

    Query
      ↓
    Query Router
      ↓
    direct / rewrite
      ↓
    Retrieval
      ↓
    Reranker
      ↓
    Retrieval Grader
      ↓
    insufficient?
      ↓
    optional rewrite retry
    """

    def __init__(
        self,
        document_path: str | Path,
    ) -> None:

        self.document_path = Path(
            document_path
        )

        # ==================================================
        # Dense Retriever
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
        # Query Router
        # ==================================================

        self.router = QueryRouter()

        # ==================================================
        # CrossEncoder Reranker
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
        # Lazy-loaded Query Rewriter
        # ==================================================

        self._rewriter = None
        self._answer_generator = None

    # ======================================================
    # Lazy Query Rewriter
    # ======================================================
    def _get_answer_generator(
        self,
    ):
        """
        Lazy-load AnswerGenerator only when retrieval
        evidence is sufficient and answer generation
        is actually required.
        """

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
    
    def _get_rewriter(
        self,
    ):
        """
        Load FLAN-T5 only when rewriting is needed.
        """

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
                query=rewritten_query,
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
        # Final reranking uses ORIGINAL query
        # --------------------------------------------------

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
    # Main Adaptive Retrieval
    # ======================================================

    def retrieve(
        self,
        query: str,
    ) -> dict:

        # ==================================================
        # Router
        # ==================================================

        decision = (
            self.router.route(
                query
            )
        )

        print(
            f"\nRoute: "
            f"{decision.route}"
        )

        print(
            f"Reasons: "
            f"{decision.reasons}"
        )

        rewritten_query = None

        retry_triggered = False

        # ==================================================
        # First Retrieval
        # ==================================================

        if decision.route == "rewrite":

            (
                rewritten_query,
                results,
            ) = (
                self._retrieve_with_rewrite(
                    query
                )
            )

            initial_route = "rewrite"

        else:

            results = (
                self._retrieve_direct(
                    query
                )
            )

            initial_route = "direct"

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
        # Retry Logic
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

            # ----------------------------------------------
            # Grade again
            # ----------------------------------------------

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
        # Answer Generation
        # ==================================================

        if final_grade.is_sufficient:

            print(
                "\nRetrieval evidence is sufficient."
            )

            print(
                "Generating grounded answer..."
            )

            answer_generator = (
                self._get_answer_generator()
            )

            answer = (
                answer_generator.generate(
                    query=query,
                    results=results,
                )
            )

            answer_status = "generated"

        else:

            print(
                "\nRetrieval evidence is insufficient."
            )

            print(
                "Skipping answer generation."
            )

            answer = (
                "I do not have enough evidence "
                "in the knowledge base to answer "
                "this question."
            )

            answer_status = (
                "insufficient_evidence"
            )
        # ==================================================
        # Return State
        # ==================================================
        return {
            "query": query,

            "router_decision": (
                decision
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