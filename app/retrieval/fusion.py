from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    top_k: int = 20,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked retrieval result lists using
    Reciprocal Rank Fusion (RRF).

    RRF score:
        score(d) = sum(1 / (rrf_k + rank))

    Parameters
    ----------
    result_lists:
        Multiple ranked retrieval result lists.

        Example:
        [
            original_query_results,
            rewritten_query_results,
        ]

    top_k:
        Number of fused results to return.

    rrf_k:
        RRF smoothing constant.
        60 is a commonly used default.

    Returns
    -------
    list[dict]
        Fused retrieval results sorted by RRF score.
    """

    if not result_lists:
        return []

    fusion_scores: dict[tuple, float] = defaultdict(float)
    document_map: dict[tuple, dict[str, Any]] = {}
    rank_history: dict[tuple, list[int]] = defaultdict(list)

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})

            # 用 source + chunk_id 唯一标识一个 chunk
            document_key = (
                metadata.get("source"),
                metadata.get("chunk_id"),
            )

            # RRF 核心公式
            fusion_scores[document_key] += (
                1.0 / (rrf_k + rank)
            )

            rank_history[document_key].append(rank)

            # 保存原始文档内容
            if document_key not in document_map:
                document_map[document_key] = result.copy()

    fused_results = []

    for document_key, fusion_score in fusion_scores.items():
        result = document_map[document_key].copy()

        result["fusion_score"] = fusion_score
        result["fusion_ranks"] = rank_history[document_key]

        fused_results.append(result)

    fused_results.sort(
        key=lambda item: item["fusion_score"],
        reverse=True,
    )

    return fused_results[:top_k]