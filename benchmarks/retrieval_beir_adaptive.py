import json
from pathlib import Path
from time import perf_counter

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import CrossEncoder

from app.routing.query_router import QueryRouter


DATASET_NAME = "scifact"

DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{DATASET_NAME}.zip"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RETRIEVAL_TOP_K = 20
RRF_K = 60

K_VALUES = [1, 3, 5, 10, 20]


# ==========================================================
# Dataset
# ==========================================================

def download_dataset() -> str:
    dataset_dir = (
        Path("benchmarks")
        / "datasets"
    )

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return util.download_and_unzip(
        DATASET_URL,
        str(dataset_dir),
    )


def load_dataset(
    data_path: str,
):
    corpus, queries, qrels = (
        GenericDataLoader(
            data_folder=data_path
        ).load(
            split="test"
        )
    )

    print(
        f"Corpus size : {len(corpus)}"
    )

    print(
        f"Query count : {len(queries)}"
    )

    print(
        f"Qrels count : {len(qrels)}"
    )

    return corpus, queries, qrels


# ==========================================================
# Dense Retriever
# ==========================================================

def build_retriever():
    dense_model = DRES(
        models.SentenceBERT(
            EMBEDDING_MODEL
        ),
        batch_size=32,
    )

    return EvaluateRetrieval(
        dense_model,
        score_function="cos_sim",
        k_values=K_VALUES,
    )


# ==========================================================
# Router
# ==========================================================

def route_queries(
    queries: dict,
):
    router = QueryRouter()

    direct_queries = {}
    rewrite_queries = {}

    route_records = {}

    start_time = perf_counter()

    for query_id, query in queries.items():

        decision = router.route(
            query
        )

        route_records[query_id] = {
            "query": query,
            "route": decision.route,
            "score": decision.score,
            "reasons": decision.reasons,
        }

        if decision.route == "rewrite":
            rewrite_queries[
                query_id
            ] = query

        else:
            direct_queries[
                query_id
            ] = query

    elapsed = (
        perf_counter()
        - start_time
    )

    total = len(queries)

    rewrite_count = len(
        rewrite_queries
    )

    direct_count = len(
        direct_queries
    )

    rewrite_rate = (
        rewrite_count / total
        if total
        else 0.0
    )

    print("\n" + "=" * 100)
    print("Routing Statistics")
    print("=" * 100)

    print(
        f"Total Queries   : {total}"
    )

    print(
        f"Direct Queries  : {direct_count}"
    )

    print(
        f"Rewrite Queries : {rewrite_count}"
    )

    print(
        f"Rewrite Rate    : "
        f"{rewrite_rate:.2%}"
    )

    print(
        f"Routing Time    : "
        f"{elapsed:.4f}s"
    )

    return (
        direct_queries,
        rewrite_queries,
        route_records,
        elapsed,
    )


# ==========================================================
# Query Rewrite
# ==========================================================

def rewrite_selected_queries(
    queries: dict,
):
    if not queries:
        print(
            "\nNo queries require rewriting."
        )

        return {}, 0.0

    cache_path = (
        Path("experiments")
        / "beir"
        / "scifact_adaptive_rewrites.json"
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {}

    if cache_path.exists():

        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            cache = json.load(file)

    rewritten_queries = {}

    missing_queries = {}

    # --------------------------------------------------
    # Read valid cached rewrites
    # --------------------------------------------------
    for query_id, query in queries.items():

        cached = cache.get(
            query_id
        )

        if (
            cached
            and cached.get("original")
            == query
        ):
            rewritten_queries[
                query_id
            ] = cached[
                "rewritten"
            ]

        else:
            missing_queries[
                query_id
            ] = query

    print(
        f"\nCached rewrites : "
        f"{len(rewritten_queries)}"
    )

    print(
        f"New rewrites    : "
        f"{len(missing_queries)}"
    )

    if not missing_queries:
        return (
            rewritten_queries,
            0.0,
        )

    # Lazy import:
    # FLAN-T5 / torch only loads when needed.
    from app.retrieval.query_rewriter import (
        QueryRewriter,
    )

    print(
        "\nLoading QueryRewriter..."
    )

    rewriter = QueryRewriter()

    start_time = perf_counter()

    total = len(
        missing_queries
    )

    for index, (
        query_id,
        query,
    ) in enumerate(
        missing_queries.items(),
        start=1,
    ):

        rewritten = rewriter.rewrite(
            query
        )

        rewritten_queries[
            query_id
        ] = rewritten

        cache[
            query_id
        ] = {
            "original": query,
            "rewritten": rewritten,
        }

        if (
            index <= 5
            or index % 25 == 0
            or index == total
        ):
            print(
                f"\n[{index}/{total}]"
            )

            print(
                f"Original : {query}"
            )

            print(
                f"Rewritten: {rewritten}"
            )

    elapsed = (
        perf_counter()
        - start_time
    )

    with cache_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cache,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nRewrite Time: "
        f"{elapsed:.2f}s"
    )

    print(
        f"Rewrite cache saved to:"
        f"\n{cache_path}"
    )

    return (
        rewritten_queries,
        elapsed,
    )


# ==========================================================
# RRF
# ==========================================================

def fuse_one_query(
    original_results: dict,
    rewritten_results: dict,
):
    fusion_scores = {}

    result_lists = [
        original_results,
        rewritten_results,
    ]

    for results in result_lists:

        ranked_documents = sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:RETRIEVAL_TOP_K]

        for rank, (
            document_id,
            _,
        ) in enumerate(
            ranked_documents,
            start=1,
        ):

            fusion_scores[
                document_id
            ] = (
                fusion_scores.get(
                    document_id,
                    0.0,
                )
                + 1.0
                / (RRF_K + rank)
            )

    sorted_results = sorted(
        fusion_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:RETRIEVAL_TOP_K]

    return dict(
        sorted_results
    )


def build_adaptive_candidates(
    queries: dict,
    route_records: dict,
    original_results: dict,
    rewritten_results: dict,
):
    """
    DIRECT:
        Original Retrieval Top-20

    REWRITE:
        Original Retrieval
        +
        Rewritten Retrieval
        -> RRF Top-20
    """

    adaptive_results = {}

    for query_id in queries:

        route = (
            route_records[
                query_id
            ]["route"]
        )

        original = (
            original_results.get(
                query_id,
                {},
            )
        )

        if route == "direct":

            ranked = sorted(
                original.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:RETRIEVAL_TOP_K]

            adaptive_results[
                query_id
            ] = dict(
                ranked
            )

            continue

        rewritten = (
            rewritten_results.get(
                query_id,
                {},
            )
        )

        adaptive_results[
            query_id
        ] = fuse_one_query(
            original,
            rewritten,
        )

    return adaptive_results


# ==========================================================
# Reranker
# ==========================================================

def document_to_text(
    document: dict,
) -> str:
    title = document.get(
        "title",
        "",
    ).strip()

    text = document.get(
        "text",
        "",
    ).strip()

    if title:
        return (
            f"{title}\n{text}"
        )

    return text


def rerank_results(
    corpus: dict,
    queries: dict,
    candidate_results: dict,
    model: CrossEncoder,
):
    pairs = []

    metadata = []

    for query_id, query in queries.items():

        candidates = (
            candidate_results.get(
                query_id,
                {},
            )
        )

        ranked_candidates = sorted(
            candidates.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:RETRIEVAL_TOP_K]

        for (
            document_id,
            candidate_score,
        ) in ranked_candidates:

            document_text = (
                document_to_text(
                    corpus[
                        document_id
                    ]
                )
            )

            # Always use original query
            # for final reranking.
            pairs.append(
                [
                    query,
                    document_text,
                ]
            )

            metadata.append(
                {
                    "query_id": (
                        query_id
                    ),
                    "document_id": (
                        document_id
                    ),
                    "candidate_score": (
                        candidate_score
                    ),
                }
            )

    print(
        f"\nCrossEncoder pairs: "
        f"{len(pairs)}"
    )

    start_time = perf_counter()

    scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=True,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    print(
        f"Reranking Time: "
        f"{elapsed:.2f}s"
    )

    reranked_results = {}

    for info, score in zip(
        metadata,
        scores,
    ):

        query_id = info[
            "query_id"
        ]

        document_id = info[
            "document_id"
        ]

        if query_id not in (
            reranked_results
        ):
            reranked_results[
                query_id
            ] = {}

        reranked_results[
            query_id
        ][document_id] = float(
            score
        )

    return (
        reranked_results,
        elapsed,
    )


# ==========================================================
# Evaluation
# ==========================================================

def evaluate(
    evaluator,
    qrels,
    results,
):
    (
        ndcg,
        map_score,
        recall,
        precision,
    ) = evaluator.evaluate(
        qrels,
        results,
        K_VALUES,
    )

    mrr = evaluator.evaluate_custom(
        qrels,
        results,
        K_VALUES,
        metric="mrr",
    )

    return (
        ndcg,
        map_score,
        recall,
        precision,
        mrr,
    )


def print_results(
    metrics,
):
    (
        ndcg,
        map_score,
        recall,
        precision,
        mrr,
    ) = metrics

    print(
        "\n"
        + "=" * 100
    )

    print(
        "V3 Adaptive Router Results"
    )

    print(
        "=" * 100
    )

    for k in [
        1,
        3,
        5,
        10,
    ]:

        print(
            f"\n@{k}"
        )

        print(
            f"NDCG      : "
            f"{ndcg[f'NDCG@{k}']:.5f}"
        )

        print(
            f"MAP       : "
            f"{map_score[f'MAP@{k}']:.5f}"
        )

        print(
            f"Recall    : "
            f"{recall[f'Recall@{k}']:.5f}"
        )

        print(
            f"Precision : "
            f"{precision[f'P@{k}']:.5f}"
        )

        print(
            f"MRR       : "
            f"{mrr[f'MRR@{k}']:.5f}"
        )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    total_start = perf_counter()

    print(
        "=" * 100
    )

    print(
        "BEIR SciFact "
        "Adaptive Router Benchmark"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    data_path = download_dataset()

    corpus, queries, qrels = (
        load_dataset(
            data_path
        )
    )

    # --------------------------------------------------
    # Router
    # --------------------------------------------------
    (
        direct_queries,
        rewrite_queries,
        route_records,
        routing_time,
    ) = route_queries(
        queries
    )

    # --------------------------------------------------
    # Dense Retriever
    # --------------------------------------------------
    print(
        "\nLoading Dense Retriever..."
    )

    retriever = build_retriever()

    print(
        "\nRunning Original Query Retrieval..."
    )

    dense_start = perf_counter()

    original_results = (
        retriever.retrieve(
            corpus,
            queries,
        )
    )

    original_retrieval_time = (
        perf_counter()
        - dense_start
    )

    print(
        f"Original Retrieval Time: "
        f"{original_retrieval_time:.2f}s"
    )

    # --------------------------------------------------
    # Rewrite only selected queries
    # --------------------------------------------------
    (
        rewritten_queries,
        rewrite_time,
    ) = rewrite_selected_queries(
        rewrite_queries
    )

    rewritten_results = {}

    rewrite_retrieval_time = 0.0

    if rewritten_queries:

        print(
            "\nRunning Rewritten Query Retrieval..."
        )

        start_time = (
            perf_counter()
        )

        rewritten_results = (
            retriever.retrieve(
                corpus,
                rewritten_queries,
            )
        )

        rewrite_retrieval_time = (
            perf_counter()
            - start_time
        )

        print(
            f"Rewrite Retrieval Time: "
            f"{rewrite_retrieval_time:.2f}s"
        )

    # --------------------------------------------------
    # Adaptive Candidate Construction
    # --------------------------------------------------
    candidate_results = (
        build_adaptive_candidates(
            queries=queries,
            route_records=route_records,
            original_results=original_results,
            rewritten_results=rewritten_results,
        )
    )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------
    print(
        "\nLoading CrossEncoder..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    (
        reranked_results,
        rerank_time,
    ) = rerank_results(
        corpus=corpus,
        queries=queries,
        candidate_results=candidate_results,
        model=reranker,
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    metrics = evaluate(
        retriever,
        qrels,
        reranked_results,
    )

    print_results(
        metrics
    )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------
    total_elapsed = (
        perf_counter()
        - total_start
    )

    total_queries = len(
        queries
    )

    rewrite_count = len(
        rewrite_queries
    )

    rewrite_rate = (
        rewrite_count
        / total_queries
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Adaptive Routing Statistics"
    )

    print(
        "=" * 100
    )

    print(
        f"Total Queries            : "
        f"{total_queries}"
    )

    print(
        f"Direct Queries           : "
        f"{len(direct_queries)}"
    )

    print(
        f"Rewrite Queries          : "
        f"{rewrite_count}"
    )

    print(
        f"Rewrite Rate             : "
        f"{rewrite_rate:.2%}"
    )

    print(
        f"Routing Time             : "
        f"{routing_time:.4f}s"
    )

    print(
        f"Query Rewrite Time       : "
        f"{rewrite_time:.2f}s"
    )

    print(
        f"Original Retrieval Time  : "
        f"{original_retrieval_time:.2f}s"
    )

    print(
        f"Rewrite Retrieval Time   : "
        f"{rewrite_retrieval_time:.2f}s"
    )

    print(
        f"Reranking Time           : "
        f"{rerank_time:.2f}s"
    )

    print(
        f"Total Benchmark Time     : "
        f"{total_elapsed:.2f}s"
    )

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------
    output_dir = (
        Path("experiments")
        / "beir"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        ndcg,
        map_score,
        recall,
        precision,
        mrr,
    ) = metrics

    result_path = (
        output_dir
        / "scifact_adaptive_router.json"
    )

    result_data = {
        "method": (
            "adaptive_router"
        ),
        "total_queries": (
            total_queries
        ),
        "direct_queries": (
            len(direct_queries)
        ),
        "rewrite_queries": (
            rewrite_count
        ),
        "rewrite_rate": (
            rewrite_rate
        ),
        "timing": {
            "routing_seconds": (
                routing_time
            ),
            "rewrite_seconds": (
                rewrite_time
            ),
            "original_retrieval_seconds": (
                original_retrieval_time
            ),
            "rewrite_retrieval_seconds": (
                rewrite_retrieval_time
            ),
            "rerank_seconds": (
                rerank_time
            ),
            "total_seconds": (
                total_elapsed
            ),
        },
        "metrics": {
            "ndcg": ndcg,
            "map": map_score,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
        },
        "routes": route_records,
    }

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nResults saved to:"
    )

    print(
        result_path
    )


if __name__ == "__main__":
    main()