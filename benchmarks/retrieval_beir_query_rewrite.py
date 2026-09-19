import json
from pathlib import Path
from time import perf_counter

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import CrossEncoder

from app.retrieval.query_rewriter import QueryRewriter


DATASET_NAME = "scifact"

DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{DATASET_NAME}.zip"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RETRIEVAL_TOP_K = 20
RRF_K = 60

K_VALUES = [1, 3, 5, 10, 20]


def download_dataset() -> str:
    dataset_dir = Path("benchmarks") / "datasets"

    dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return util.download_and_unzip(
        DATASET_URL,
        str(dataset_dir),
    )


def load_dataset(data_path: str):
    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(
        split="test"
    )

    print(f"Corpus size : {len(corpus)}")
    print(f"Query count : {len(queries)}")
    print(f"Qrels count : {len(qrels)}")

    return corpus, queries, qrels


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


def document_to_text(document: dict) -> str:
    title = document.get(
        "title",
        "",
    ).strip()

    text = document.get(
        "text",
        "",
    ).strip()

    if title:
        return f"{title}\n{text}"

    return text


# ==========================================================
# Query Rewrite
# ==========================================================

def rewrite_queries(
    queries: dict,
) -> dict:
    """
    Rewrite all SciFact queries.

    Rewrites are cached so we do not need to run
    FLAN-T5 every time the benchmark is executed.
    """

    cache_path = (
        Path("experiments")
        / "beir"
        / "scifact_rewritten_queries.json"
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Use cache if it already exists
    # --------------------------------------------------
    if cache_path.exists():
        print(
            f"\nLoading rewritten queries from:"
            f"\n{cache_path}"
        )

        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # --------------------------------------------------
    # Otherwise run FLAN-T5
    # --------------------------------------------------
    print("\nLoading Query Rewriter...")

    rewriter = QueryRewriter()

    rewritten_queries = {}

    total = len(queries)

    print(
        f"\nRewriting {total} queries..."
    )

    start_time = perf_counter()

    for index, (
        query_id,
        query,
    ) in enumerate(
        queries.items(),
        start=1,
    ):
        rewritten = rewriter.rewrite(
            query
        )

        rewritten_queries[query_id] = rewritten

        if (
            index <= 5
            or index % 25 == 0
            or index == total
        ):
            print(
                f"[{index}/{total}]"
            )
            print(
                f"Original : {query}"
            )
            print(
                f"Rewritten: {rewritten}"
            )

    elapsed = perf_counter() - start_time

    print(
        f"\nQuery rewriting time: "
        f"{elapsed:.2f} seconds"
    )

    # --------------------------------------------------
    # Save cache
    # --------------------------------------------------
    with cache_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            rewritten_queries,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nRewritten queries saved to:"
        f"\n{cache_path}"
    )

    return rewritten_queries


# ==========================================================
# RRF
# ==========================================================

def reciprocal_rank_fusion(
    original_results: dict,
    rewritten_results: dict,
) -> dict:
    """
    Fuse Original Query retrieval and
    Rewritten Query retrieval using RRF.

    BEIR format:

    {
        query_id: {
            document_id: score
        }
    }
    """

    fused_results = {}

    query_ids = set(
        original_results.keys()
    ) | set(
        rewritten_results.keys()
    )

    for query_id in query_ids:

        fusion_scores = {}

        result_lists = [
            original_results.get(
                query_id,
                {},
            ),
            rewritten_results.get(
                query_id,
                {},
            ),
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

        # Only keep the strongest candidates
        sorted_fused = sorted(
            fusion_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:RETRIEVAL_TOP_K]

        fused_results[query_id] = dict(
            sorted_fused
        )

    return fused_results


# ==========================================================
# Cross-Encoder
# ==========================================================

def rerank_results(
    corpus: dict,
    queries: dict,
    fused_results: dict,
    model: CrossEncoder,
) -> dict:

    pairs = []
    metadata = []

    for query_id, query in queries.items():

        candidate_results = (
            fused_results.get(
                query_id,
                {},
            )
        )

        ranked_candidates = sorted(
            candidate_results.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:RETRIEVAL_TOP_K]

        for (
            document_id,
            fusion_score,
        ) in ranked_candidates:

            document_text = (
                document_to_text(
                    corpus[document_id]
                )
            )

            # IMPORTANT:
            # Reranker uses the ORIGINAL query
            pairs.append(
                [
                    query,
                    document_text,
                ]
            )

            metadata.append(
                {
                    "query_id": query_id,
                    "document_id": (
                        document_id
                    ),
                    "fusion_score": (
                        fusion_score
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

    elapsed = perf_counter() - start_time

    print(
        f"Reranking time: "
        f"{elapsed:.2f} seconds"
    )

    reranked_results = {}

    for info, score in zip(
        metadata,
        scores,
    ):
        query_id = info["query_id"]
        document_id = (
            info["document_id"]
        )

        if (
            query_id
            not in reranked_results
        ):
            reranked_results[
                query_id
            ] = {}

        reranked_results[
            query_id
        ][document_id] = float(
            score
        )

    return reranked_results


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
) -> None:

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
        "V2 Query Rewrite + RRF + "
        "Reranker Results"
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
        print(f"\n@{k}")

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


def main() -> None:

    print(
        "=" * 100
    )

    print(
        "BEIR SciFact "
        "Query Rewrite + RRF + Reranker"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------
    data_path = download_dataset()

    corpus, queries, qrels = (
        load_dataset(
            data_path
        )
    )

    # --------------------------------------------------
    # 2. Dense retriever
    # --------------------------------------------------
    print(
        "\nLoading Dense Retriever..."
    )

    retriever = build_retriever()

    # --------------------------------------------------
    # 3. Original Query Retrieval
    # --------------------------------------------------
    print(
        "\nRunning Original Query "
        "Retrieval..."
    )

    original_results = (
        retriever.retrieve(
            corpus,
            queries,
        )
    )

    # --------------------------------------------------
    # 4. Query Rewrite
    # --------------------------------------------------
    rewritten_queries = (
        rewrite_queries(
            queries
        )
    )

    # --------------------------------------------------
    # 5. Rewrite Retrieval
    # --------------------------------------------------
    print(
        "\nRunning Rewritten Query "
        "Retrieval..."
    )

    rewritten_results = (
        retriever.retrieve(
            corpus,
            rewritten_queries,
        )
    )

    # --------------------------------------------------
    # 6. RRF
    # --------------------------------------------------
    print(
        "\nRunning RRF Fusion..."
    )

    fused_results = (
        reciprocal_rank_fusion(
            original_results,
            rewritten_results,
        )
    )

    # --------------------------------------------------
    # 7. Reranker
    # --------------------------------------------------
    print(
        "\nLoading CrossEncoder..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    reranked_results = rerank_results(
        corpus=corpus,
        queries=queries,
        fused_results=fused_results,
        model=reranker,
    )

    # --------------------------------------------------
    # 8. Evaluation
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
    # 9. Save
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
        / "scifact_query_rewrite_rrf_reranker.json"
    )

    util.save_results(
        str(result_path),
        ndcg,
        map_score,
        recall,
        precision,
        mrr,
    )

    runfile_path = (
        output_dir
        / "scifact_query_rewrite_rrf_reranker.run.trec"
    )

    util.save_runfile(
        str(runfile_path),
        reranked_results,
    )

    print(
        "\nResults saved to:"
    )
    print(
        result_path
    )

    print(
        "\nRunfile saved to:"
    )
    print(
        runfile_path
    )


if __name__ == "__main__":
    main()