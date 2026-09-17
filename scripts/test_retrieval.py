from app.retrieval.document_search import DocumentSearcher


def main() -> None:
    searcher = DocumentSearcher("data/dspy.pdf")

    chunk_count = searcher.index_document()
    print(f"Indexed chunks: {chunk_count}")

    queries = [
        "What is DSPy?",
        "What are the main abstractions in DSPy?",
        "What is a DSPy signature?",
        "What is a teleprompter in DSPy?",
        "How does DSPy reduce manual prompt engineering?",
    ]

    for query_index, query in enumerate(queries, start=1):
        print("\n" + "=" * 100)
        print(f"Query {query_index}: {query}")
        print("=" * 100)

        results = searcher.search(
            query=query,
            limit=3,
        )

        for result_index, result in enumerate(results, start=1):
            print(f"\n--- Result {result_index} ---")
            print(f"Score: {result['score']}")
            print(f"Metadata: {result['metadata']}")
            print(result["text"][:500])


if __name__ == "__main__":
    main()