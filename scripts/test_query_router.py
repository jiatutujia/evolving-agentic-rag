from app.routing.query_router import QueryRouter


def main() -> None:
    router = QueryRouter()

    queries = [
        "What is DSPy?",

        "What are the main abstractions in DSPy?",

        "DSPy teleprompter how work?",

        (
            "So basically I want to know like how DSPy "
            "gets rid of writing prompts manually"
        ),

        (
            "I'm trying to understand DSPy and I saw something "
            "about signatures, modules and teleprompters, "
            "can you explain what these main abstractions are?"
        ),
    ]

    for index, query in enumerate(
        queries,
        start=1,
    ):
        decision = router.route(query)

        print("\n" + "=" * 100)
        print(f"Query {index}")
        print("=" * 100)

        print(f"Query   : {query}")
        print(f"Route   : {decision.route}")
        print(f"Score   : {decision.score}")
        print(f"Reasons : {decision.reasons}")


if __name__ == "__main__":
    main()