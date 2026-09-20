import json

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from sentence_transformers import (
    SentenceTransformer,
)


@dataclass
class ExperienceRecord:
    """
    One reusable experience extracted from
    a completed RAG execution.
    """

    query: str

    outcome: str

    success: bool

    initial_route: str

    final_route: str

    retry_triggered: bool

    initial_score: float

    final_score: float

    score_delta: float

    recommended_strategy: str

    lesson: str

    created_at: str


@dataclass
class MemoryMatch:
    """
    One semantic memory retrieval result.
    """

    similarity: float

    experience: ExperienceRecord


class ExperienceMemory:
    """
    Persistent semantic memory for RAG experiences.

    Storage:
        JSON file

    Retrieval:
        SentenceTransformer embedding
        +
        cosine similarity

    V5 baseline intentionally keeps the memory layer
    lightweight and interpretable.

    Later versions can migrate this storage to Qdrant.
    """

    def __init__(
        self,
        memory_path: str | Path = (
            "memory/experience_memory.json"
        ),
        embedding_model: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    ) -> None:

        self.memory_path = Path(
            memory_path
        )

        self.memory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"ExperienceMemory device: "
            f"{self.device}"
        )

        self.model = (
            SentenceTransformer(
                embedding_model,
                device=self.device,
            )
        )

        self.records: list[
            ExperienceRecord
        ] = []

        self._load()

    # ======================================================
    # Persistence
    # ======================================================

    def _load(
        self,
    ) -> None:

        if not self.memory_path.exists():
            return

        with self.memory_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            raw_records = json.load(
                file
            )

        self.records = [
            ExperienceRecord(
                **record
            )
            for record in raw_records
        ]

        print(
            f"Loaded "
            f"{len(self.records)} "
            f"experiences."
        )

    def _save(
        self,
    ) -> None:

        data = [
            asdict(record)
            for record in self.records
        ]

        with self.memory_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    # ======================================================
    # Write
    # ======================================================

    def add(
        self,
        query: str,
        reflection,
    ) -> ExperienceRecord:
        """
        Convert ReflectionResult into persistent
        ExperienceRecord.
        """

        record = ExperienceRecord(
            query=query,

            outcome=(
                reflection.outcome
            ),

            success=(
                reflection.success
            ),

            initial_route=(
                reflection.initial_route
            ),

            final_route=(
                reflection.final_route
            ),

            retry_triggered=(
                reflection.retry_triggered
            ),

            initial_score=(
                reflection.initial_score
            ),

            final_score=(
                reflection.final_score
            ),

            score_delta=(
                reflection.score_delta
            ),

            recommended_strategy=(
                reflection.recommended_strategy
            ),

            lesson=(
                reflection.lesson
            ),

            created_at=(
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
        )

        self.records.append(
            record
        )

        self._save()

        return record

    # ======================================================
    # Semantic Retrieval
    # ======================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[MemoryMatch]:
        """
        Retrieve semantically similar past experiences.
        """

        if not self.records:
            return []

        memory_queries = [
            record.query
            for record in self.records
        ]

        query_embedding = (
            self.model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        )

        memory_embeddings = (
            self.model.encode(
                memory_queries,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        )

        similarities = (
            memory_embeddings
            @ query_embedding
        )

        matches = []

        for index, similarity in enumerate(
            similarities.tolist()
        ):

            if similarity < min_similarity:
                continue

            matches.append(
                MemoryMatch(
                    similarity=float(
                        similarity
                    ),
                    experience=(
                        self.records[
                            index
                        ]
                    ),
                )
            )

        matches.sort(
            key=lambda item: (
                item.similarity
            ),
            reverse=True,
        )

        return matches[:top_k]

    # ======================================================
    # Utilities
    # ======================================================

    def size(
        self,
    ) -> int:

        return len(
            self.records
        )

    def clear(
        self,
    ) -> None:

        self.records = []

        self._save()