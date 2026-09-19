from app.retrieval.query_rewriter import QueryRewriter


def main() -> None:
    rewriter = QueryRewriter()

    queries = [
        # 已经很标准，理论上可以不改
        "What is DSPy?",

        # 口语化
        "I don't really understand DSPy, can you tell me what the main ideas behind it are?",

        # 很短、表达不完整
        "DSPy teleprompter how work?",

        # 带大量口语噪声
        "So basically I want to know like how DSPy gets rid of writing prompts manually",

        # 冗长
        (
            "I'm trying to understand DSPy and I saw something about signatures, "
            "modules and teleprompters, can you explain what these main abstractions are?"
        ),
    ]

    for index, query in enumerate(
        queries,
        start=1,
    ):
        rewritten = rewriter.rewrite(query)

        print("\n" + "=" * 100)
        print(f"Query {index}")
        print("=" * 100)

        print(f"Original : {query}")
        print(f"Rewritten: {rewritten}")


if __name__ == "__main__":
    main()