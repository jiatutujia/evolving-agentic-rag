from app.retrieval.document_search import DocumentSearcher
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import Reranker
from app.routing.query_router import QueryRouter


class AdaptiveRetriever:
    def __init__(
        self,
        document_path: str,
    ) -> None:
        self.searcher = DocumentSearcher(
            document_path
        )

        self.searcher.index_document()

        self.router = QueryRouter()

        self.reranker = Reranker()

        # QueryRewriter 使用 lazy loading
        # direct query 不需要加载 FLAN-T5
        self._rewriter = None

    def _get_rewriter(
        self,
    ) -> QueryRewriter:
        if self._rewriter is None:
            print(
                "Loading QueryRewriter..."
            )

            self._rewriter = (
                QueryRewriter()
            )

        return self._rewriter

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
    ) -> dict:
        # --------------------------------------------------
        # 1. Routing
        # --------------------------------------------------
        decision = self.router.route(
            query
        )

        print(
            f"\nRoute: {decision.route}"
        )

        print(
            f"Reasons: {decision.reasons}"
        )

        # ==================================================
        # DIRECT ROUTE
        # ==================================================
        if decision.route == "direct":

            retrieved_results = (
                self.searcher.search(
                    query=query,
                    limit=retrieval_top_k,
                )
            )

            final_results = (
                self.reranker.rerank(
                    query=query,
                    documents=retrieved_results,
                    top_k=final_top_k,
                )
            )

            return {
                "query": query,
                "route": "direct",
                "rewritten_query": None,
                "results": final_results,
            }

        # ==================================================
        # REWRITE ROUTE
        # ==================================================
        rewriter = self._get_rewriter()

        rewritten_query = (
            rewriter.rewrite(
                query
            )
        )

        print(
            f"Rewritten Query: "
            f"{rewritten_query}"
        )

        # Original Query Retrieval
        original_results = (
            self.searcher.search(
                query=query,
                limit=retrieval_top_k,
            )
        )

        # Rewritten Query Retrieval
        rewritten_results = (
            self.searcher.search(
                query=rewritten_query,
                limit=retrieval_top_k,
            )
        )

        # --------------------------------------------------
        # RRF
        # --------------------------------------------------
        fused_results = (
            reciprocal_rank_fusion(
                result_lists=[
                    original_results,
                    rewritten_results,
                ],
                top_k=retrieval_top_k,
            )
        )

        # --------------------------------------------------
        # Reranking
        # --------------------------------------------------
        final_results = (
            self.reranker.rerank(
                query=query,
                documents=fused_results,
                top_k=final_top_k,
            )
        )

        return {
            "query": query,
            "route": "rewrite",
            "rewritten_query": rewritten_query,
            "results": final_results,
        }