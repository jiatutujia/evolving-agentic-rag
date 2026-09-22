from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """
    Normalized result returned by any LLM provider.
    """

    text: str

    model: str

    provider: str

    prompt_tokens: int | None = None

    completion_tokens: int | None = None

    total_tokens: int | None = None

    latency_ms: float | None = None


class LLMProvider(ABC):
    """
    Common interface for all LLM backends.

    Examples
    --------
    - Transformers local model
    - vLLM
    - OpenAI-compatible APIs
    - TensorRT-LLM
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerationResult:
        raise NotImplementedError