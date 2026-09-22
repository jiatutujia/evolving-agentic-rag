import argparse

from pathlib import Path

from app.retrieval.document_search import (
    DocumentSearcher,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build or reuse the persistent "
            "Qdrant knowledge-base index."
        )
    )

    parser.add_argument(
        "--file",
        type=str,
        default="data/dspy.pdf",
        help="Document to ingest.",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="knowledge_base",
        help="Qdrant collection name.",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help=(
            "Qdrant server URL. "
            "If omitted, persistent local "
            "storage is used."
        ),
    )

    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help=(
            "Persistent local Qdrant path. "
            "Defaults to qdrant_storage."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete and rebuild an "
            "existing collection."
        ),
    )

    parser.add_argument(
        "--smoke-query",
        type=str,
        default=None,
        help=(
            "Optional retrieval query "
            "after ingestion."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    searcher = DocumentSearcher(
        file_path=Path(
            args.file
        ),

        collection_name=(
            args.collection
        ),

        qdrant_url=(
            args.qdrant_url
        ),

        qdrant_path=(
            args.qdrant_path
        ),
    )

    try:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "PERSISTENT QDRANT INGESTION"
        )

        print(
            "=" * 100
        )

        count = (
            searcher.index_document(
                force_reindex=(
                    args.force
                )
            )
        )

        info = (
            searcher.get_index_info()
        )

        print(
            "\nIndex Information"
        )

        print(
            "-" * 50
        )

        print(
            f"Collection     : "
            f"{info['collection_name']}"
        )

        print(
            f"Storage Mode   : "
            f"{info['storage_mode']}"
        )

        print(
            f"Points         : "
            f"{info['points_count']}"
        )

        print(
            f"Ready          : "
            f"{info['ready']}"
        )

        print(
            f"Document       : "
            f"{info['document']}"
        )

        print(
            f"Embedding      : "
            f"{info['embedding_model']}"
        )

        print(
            f"Indexed Chunks : "
            f"{count}"
        )

        if args.smoke_query:

            print(
                "\nSmoke Retrieval"
            )

            print(
                "-" * 50
            )

            print(
                f"Query: "
                f"{args.smoke_query}"
            )

            results = (
                searcher.search(
                    query=(
                        args.smoke_query
                    ),

                    limit=3,
                )
            )

            for index, result in enumerate(
                results,
                start=1,
            ):

                print(
                    f"\nRank {index}"
                )

                print(
                    f"Score: "
                    f"{result['score']:.4f}"
                )

                print(
                    f"Source: "
                    f"{result['metadata']['source']}"
                )

                print(
                    f"Chunk ID: "
                    f"{result['metadata']['chunk_id']}"
                )

                preview = (
                    result[
                        "text"
                    ]
                    .replace(
                        "\n",
                        " ",
                    )[:300]
                )

                print(
                    f"Text: "
                    f"{preview}"
                )

    finally:

        searcher.close()


if __name__ == "__main__":
    main()