from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


DATASET_NAME = "scifact"

DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{DATASET_NAME}.zip"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def download_dataset() -> str:
    """
    Download BEIR SciFact if it is not already available.
    """
    dataset_dir = Path("benchmarks") / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data_path = util.download_and_unzip(
        DATASET_URL,
        str(dataset_dir),
    )

    return data_path


def load_dataset(data_path: str):
    """
    Load SciFact test corpus, queries and relevance judgments.
    """
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
    """
    Build a dense retriever using Sentence-BERT.
    """
    model = DRES(
        models.SentenceBERT(
            EMBEDDING_MODEL
        ),
        batch_size=32,
    )

    retriever = EvaluateRetrieval(
        model,
        score_function="cos_sim",
    )

    return retriever


def run_benchmark():
    print("=" * 80)
    print("BEIR SciFact Retrieval Benchmark")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Download dataset
    # --------------------------------------------------
    data_path = download_dataset()

    print(f"\nDataset path: {data_path}")

    # --------------------------------------------------
    # 2. Load SciFact
    # --------------------------------------------------
    corpus, queries, qrels = load_dataset(
        data_path
    )

    # --------------------------------------------------
    # 3. Build retriever
    # --------------------------------------------------
    print("\nLoading embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")

    retriever = build_retriever()

    # --------------------------------------------------
    # 4. Retrieval
    # --------------------------------------------------
    print("\nRunning retrieval...")

    results = retriever.retrieve(
        corpus,
        queries,
    )

    # --------------------------------------------------
    # 5. Evaluation
    # --------------------------------------------------
    print("\nEvaluating...")

    ndcg, map_score, recall, precision = (
        retriever.evaluate(
            qrels,
            results,
            retriever.k_values,
        )
    )

    mrr = retriever.evaluate_custom(
        qrels,
        results,
        retriever.k_values,
        metric="mrr",
    )

    # --------------------------------------------------
    # 6. Print important metrics
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)

    metrics_to_show = [1, 3, 5, 10]

    for k in metrics_to_show:
        print(f"\n@{k}")
        print(f"NDCG      : {ndcg.get(f'NDCG@{k}')}")
        print(f"MAP       : {map_score.get(f'MAP@{k}')}")
        print(f"Recall    : {recall.get(f'Recall@{k}')}")
        print(
            f"Precision : "
            f"{precision.get(f'P@{k}')}"
        )
        print(f"MRR       : {mrr.get(f'MRR@{k}')}")

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------
    results_dir = (
        Path("experiments")
        / "beir"
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        results_dir
        / "scifact_dense_baseline.json"
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
        results_dir
        / "scifact_dense_baseline.run.trec"
    )

    util.save_runfile(
        str(runfile_path),
        results,
    )

    print("\nResults saved to:")
    print(result_path)

    print("\nRunfile saved to:")
    print(runfile_path)


if __name__ == "__main__":
    run_benchmark()