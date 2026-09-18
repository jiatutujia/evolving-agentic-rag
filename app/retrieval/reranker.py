from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name

        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved documents according to
        query-document relevance.

        Parameters
        ----------
        query:
            User query.

        documents:
            Candidate documents returned by retriever.

        top_k:
            Number of documents to keep after reranking.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        # Cross-Encoder receives:
        #
        # [query, document]
        #
        # instead of independently embedding them.
        pairs = [
            [
                query,
                document["text"],
            ]
            for document in documents
        ]

        rerank_scores = self.model.predict(
            pairs
        )

        reranked_results = []

        for document, rerank_score in zip(
            documents,
            rerank_scores,
        ):
            result = document.copy()

            # Preserve original retrieval score
            result["retrieval_score"] = result.get(
                "score"
            )

            result["rerank_score"] = float(
                rerank_score
            )

            reranked_results.append(result)

        # Higher reranking score = more relevant
        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked_results[:top_k]