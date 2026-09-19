import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class AnswerGenerator:
    """
    Baseline grounded answer generator.

    V4 uses FLAN-T5 as a lightweight local generator.

    Later versions can replace this implementation with
    Qwen + vLLM without changing the retrieval pipeline.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_input_length: int = 512,
        max_new_tokens: int = 128,
    ) -> None:

        self.model_name = model_name
        self.max_input_length = (
            max_input_length
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"AnswerGenerator device: "
            f"{self.device}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def _build_context(
        self,
        results: list[dict],
        top_k: int = 3,
    ) -> str:
        """
        Build generation context from the highest-ranked
        retrieval results.
        """

        contexts = []

        for index, result in enumerate(
            results[:top_k],
            start=1,
        ):

            text = result.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            contexts.append(
                f"[Document {index}]\n{text}"
            )

        return "\n\n".join(
            contexts
        )

    def generate(
        self,
        query: str,
        results: list[dict],
    ) -> str:
        """
        Generate an answer grounded only in retrieved
        context.
        """

        context = (
            self._build_context(
                results
            )
        )

        if not context:
            return (
                "I do not have enough evidence "
                "in the knowledge base to answer "
                "this question."
            )

        prompt = f"""
Answer the question using only the provided context.

If the context does not contain enough evidence,
say that there is not enough evidence.

Do not use outside knowledge.
Do not invent information.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        inputs = (
            self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=(
                    self.max_input_length
                ),
            )
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.no_grad():

            output_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    do_sample=False,
                    num_beams=4,
                )
            )

        answer = (
            self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            )
        )

        return answer.strip()