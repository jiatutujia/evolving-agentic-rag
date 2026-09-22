import os
import time

import httpx

from app.llm.base import (
    GenerationResult,
    LLMProvider,
)


class VLLMProvider(LLMProvider):
    """
    Client for a vLLM OpenAI-compatible server.

    The RAG application does not load the LLM itself.
    It sends generation requests to vLLM over HTTP.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:

        self.base_url = (
            base_url
            or os.getenv(
                "VLLM_BASE_URL",
                "http://127.0.0.1:8000/v1",
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "VLLM_MODEL",
                "Qwen/Qwen3-1.7B",
            )
        )

        self.api_key = (
            api_key
            or os.getenv(
                "VLLM_API_KEY",
                "EMPTY",
            )
        )

        self.timeout = timeout

        self.client = httpx.Client(
            timeout=self.timeout,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerationResult:

        url = (
            f"{self.base_url}"
            "/chat/completions"
        )

        headers = {
            "Content-Type": (
                "application/json"
            ),
        }

        if self.api_key:

            headers[
                "Authorization"
            ] = (
                f"Bearer "
                f"{self.api_key}"
            )

        payload = {
            "model": (
                self.model
            ),

            "messages": [
                {
                    "role": "system",
                    "content": (
                        system_prompt
                    ),
                },

                {
                    "role": "user",
                    "content": (
                        user_prompt
                    ),
                },
            ],

            "temperature": (
                temperature
            ),

            "max_tokens": (
                max_tokens
            ),
        }

        start = (
            time.perf_counter()
        )

        response = (
            self.client.post(
                url,
                headers=headers,
                json=payload,
            )
        )

        latency_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000.0
        )

        response.raise_for_status()

        data = (
            response.json()
        )

        text = (
            data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        )

        usage = (
            data.get(
                "usage",
                {},
            )
        )

        return GenerationResult(
            text=(
                text.strip()
            ),

            model=(
                data.get(
                    "model",
                    self.model,
                )
            ),

            provider="vllm",

            prompt_tokens=(
                usage.get(
                    "prompt_tokens"
                )
            ),

            completion_tokens=(
                usage.get(
                    "completion_tokens"
                )
            ),

            total_tokens=(
                usage.get(
                    "total_tokens"
                )
            ),

            latency_ms=round(
                latency_ms,
                2,
            ),
        )

    def health_check(
        self,
    ) -> bool:

        try:

            response = (
                self.client.get(
                    f"{self.base_url}"
                    "/models"
                )
            )

            return (
                response.status_code
                == 200
            )

        except httpx.HTTPError:

            return False

    def close(
        self,
    ) -> None:

        self.client.close()