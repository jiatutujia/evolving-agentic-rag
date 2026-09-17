from app.retrieval.document_search import DocumentSearcher


def main() -> None:
    searcher = DocumentSearcher("data/dspy.pdf")

    chunk_count = searcher.index_document()

    print(f"Indexed chunks: {chunk_count}")

    results = searcher.search(
        query="What is DSPy?",
        limit=3,
    )

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['metadata']}")
        print(result["text"][:500])


if __name__ == "__main__":
    main()