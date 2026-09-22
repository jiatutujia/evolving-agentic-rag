from app.llm.vllm_provider import (
    VLLMProvider,
)


def main() -> None:

    provider = (
        VLLMProvider()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "VLLM PROVIDER TEST"
    )

    print(
        "=" * 100
    )

    print(
        f"Base URL : "
        f"{provider.base_url}"
    )

    print(
        f"Model    : "
        f"{provider.model}"
    )

    healthy = (
        provider.health_check()
    )

    print(
        f"Healthy  : "
        f"{healthy}"
    )

    if not healthy:

        print(
            "\n"
            "vLLM server is not running yet. "
            "Provider implementation is ready."
        )

        return

    result = (
        provider.generate(
            system_prompt=(
                "You are a concise "
                "helpful assistant."
            ),

            user_prompt=(
                "Explain RAG in one sentence."
            ),

            max_tokens=64,

            temperature=0.0,
        )
    )

    print(
        "\nGeneration"
    )

    print(
        "-" * 50
    )

    print(
        f"Text              : "
        f"{result.text}"
    )

    print(
        f"Provider          : "
        f"{result.provider}"
    )

    print(
        f"Model             : "
        f"{result.model}"
    )

    print(
        f"Prompt Tokens     : "
        f"{result.prompt_tokens}"
    )

    print(
        f"Completion Tokens : "
        f"{result.completion_tokens}"
    )

    print(
        f"Total Tokens      : "
        f"{result.total_tokens}"
    )

    print(
        f"Latency           : "
        f"{result.latency_ms} ms"
    )


if __name__ == "__main__":
    main()