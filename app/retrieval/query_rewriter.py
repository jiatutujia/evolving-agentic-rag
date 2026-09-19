import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class QueryRewriter:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
    ) -> None:
        self.model_name = model_name

        # 自动选择 GPU / CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"QueryRewriter device: {self.device}")

        # Tokenizer：文本 → token ids
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # FLAN-T5 是 Seq2Seq 模型
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def rewrite(
        self,
        query: str,
    ) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        prompt = (
            "Rewrite user questions into concise search queries for information retrieval.\n"
            "Remove conversational filler and unnecessary words.\n"
            "Preserve technical terms, named entities, and the original intent.\n"
            "Do not answer the question.\n"
            "If the query is already concise and clear, keep it unchanged.\n\n"

            "Examples:\n\n"

            "User input: I don't really understand transformers, "
            "can you tell me what the main ideas behind them are?\n"
            "Search query: main concepts of transformers\n\n"

            "User input: So basically I want to know like how RAG retrieves documents\n"
            "Search query: how RAG retrieves documents\n\n"

            "User input: vector database how work?\n"
            "Search query: how vector databases work\n\n"

            "User input: What is BERT?\n"
            "Search query: What is BERT?\n\n"

            f"User input: {query}\n"
            "Search query:"
        )

        # 文本 → token
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # 推理时不计算梯度
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                num_beams=4,
            )

        # token → 文本
        rewritten_query = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        ).strip()

        # 防止模型生成空字符串
        if not rewritten_query:
            return query

        return rewritten_query