from pathlib import Path
from time import perf_counter

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import CrossEncoder


DATASET_NAME = "scifact"

DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{DATASET_NAME}.zip"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 第一阶段先召回 20 个
RETRIEVAL_TOP_K = 20

# 最终重点评估到 Top-10
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


def build_dense_retriever():
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
    """
    Convert a BEIR document into text for the CrossEncoder.
    """

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


def rerank_results(
    corpus: dict,
    queries: dict,
    dense_results: dict,
    model: CrossEncoder,
) -> dict:
    """
    Rerank the top-20 dense retrieval candidates
    using a CrossEncoder.
    """

    pairs = []
    pair_metadata = []

    # --------------------------------------------------
    # Build all query-document pairs
    # --------------------------------------------------
    for query_id, query in queries.items():

        query_results = dense_results.get(
            query_id,
            {},
        )

        # Dense results sorted from high score to low score
        sorted_candidates = sorted(
            query_results.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:RETRIEVAL_TOP_K]

        for doc_id, retrieval_score in sorted_candidates:

            document_text = document_to_text(
                corpus[doc_id]
            )

            pairs.append(
                [
                    query,
                    document_text,
                ]
            )

            pair_metadata.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "retrieval_score": retrieval_score,
                }
            )

    print(
        f"CrossEncoder pairs: {len(pairs)}"
    )

    # --------------------------------------------------
    # CrossEncoder inference
    # --------------------------------------------------
    print("\nRunning CrossEncoder reranking...")

    start_time = perf_counter()

    rerank_scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=True,
    )

    elapsed = perf_counter() - start_time

    print(
        f"Reranking time: {elapsed:.2f} seconds"
    )

    # --------------------------------------------------
    # Convert back to BEIR result format
    #
    # {
    #     query_id: {
    #         doc_id: rerank_score
    #     }
    # }
    # --------------------------------------------------
    reranked_results = {}

    for metadata, rerank_score in zip(
        pair_metadata,
        rerank_scores,
    ):

        query_id = metadata["query_id"]
        doc_id = metadata["doc_id"]

        if query_id not in reranked_results:
            reranked_results[query_id] = {}

        reranked_results[query_id][doc_id] = float(
            rerank_score
        )

    return reranked_results


def evaluate(
    evaluator: EvaluateRetrieval,
    qrels: dict,
    results: dict,
):
    ndcg, map_score, recall, precision = (
        evaluator.evaluate(
            qrels,
            results,
            K_VALUES,
        )
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


def print_comparison(
    dense_metrics,
    rerank_metrics,
) -> None:

    (
        dense_ndcg,
        dense_map,
        dense_recall,
        dense_precision,
        dense_mrr,
    ) = dense_metrics

    (
        rerank_ndcg,
        rerank_map,
        rerank_recall,
        rerank_precision,
        rerank_mrr,
    ) = rerank_metrics

    print("\n" + "=" * 100)
    print("Dense Retriever vs Dense + Reranker")
    print("=" * 100)

    for k in [1, 3, 5, 10]:

        print(f"\n@{k}")

        print(
            f"{'Metric':<15}"
            f"{'Dense':>15}"
            f"{'Reranker':>15}"
            f"{'Delta':>15}"
        )

        print("-" * 60)

        metrics = [
            (
                "NDCG",
                dense_ndcg[f"NDCG@{k}"],
                rerank_ndcg[f"NDCG@{k}"],
            ),
            (
                "MAP",
                dense_map[f"MAP@{k}"],
                rerank_map[f"MAP@{k}"],
            ),
            (
                "Recall",
                dense_recall[f"Recall@{k}"],
                rerank_recall[f"Recall@{k}"],
            ),
            (
                "Precision",
                dense_precision[f"P@{k}"],
                rerank_precision[f"P@{k}"],
            ),
            (
                "MRR",
                dense_mrr[f"MRR@{k}"],
                rerank_mrr[f"MRR@{k}"],
            ),
        ]

        for name, before, after in metrics:

            delta = after - before

            print(
                f"{name:<15}"
                f"{before:>15.5f}"
                f"{after:>15.5f}"
                f"{delta:>+15.5f}"
            )


def main() -> None:

    print("=" * 100)
    print("BEIR SciFact: Dense Retrieval + CrossEncoder Reranking")
    print("=" * 100)

    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------
    data_path = download_dataset()

    corpus, queries, qrels = load_dataset(
        data_path
    )

    # --------------------------------------------------
    # 2. Dense Retrieval
    # --------------------------------------------------
    print("\nLoading Dense Retriever...")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    retriever = build_dense_retriever()

    print("\nRunning Dense Retrieval...")

    dense_results = retriever.retrieve(
        corpus,
        queries,
    )

    dense_metrics = evaluate(
        retriever,
        qrels,
        dense_results,
    )

    # --------------------------------------------------
    # 3. CrossEncoder Reranker
    # --------------------------------------------------
    print("\nLoading CrossEncoder...")
    print(f"Reranker model: {RERANKER_MODEL}")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    reranked_results = rerank_results(
        corpus=corpus,
        queries=queries,
        dense_results=dense_results,
        model=reranker,
    )

    # --------------------------------------------------
    # 4. Evaluate reranked results
    # --------------------------------------------------
    rerank_metrics = evaluate(
        retriever,
        qrels,
        reranked_results,
    )

    # --------------------------------------------------
    # 5. Compare
    # --------------------------------------------------
    print_comparison(
        dense_metrics,
        rerank_metrics,
    )

    # --------------------------------------------------
    # 6. Save experiment results
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
        rerank_ndcg,
        rerank_map,
        rerank_recall,
        rerank_precision,
        rerank_mrr,
    ) = rerank_metrics

    result_path = (
        output_dir
        / "scifact_dense_reranker.json"
    )

    util.save_results(
        str(result_path),
        rerank_ndcg,
        rerank_map,
        rerank_recall,
        rerank_precision,
        rerank_mrr,
    )

    runfile_path = (
        output_dir
        / "scifact_dense_reranker.run.trec"
    )

    util.save_runfile(
        str(runfile_path),
        reranked_results,
    )

    print("\nResults saved to:")
    print(result_path)

    print("\nRunfile saved to:")
    print(runfile_path)


if __name__ == "__main__":
    main()