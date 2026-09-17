from pathlib import Path
from typing import Any

from chonkie import SemanticChunker
from markitdown import MarkItDown
from qdrant_client import QdrantClient, models

class DocumentSearcher:
    """
    Responsible for:

    1. Parsing a document
    2. Splitting it into semantic chunks
    3. Indexing chunks into Qdrant
    4. Retrieving relevant chunks for a query

    V0 uses an in-memory Qdrant instance.
    Persistent Qdrant will be introduced later with Docker.
    """

    def __init__(
        self,
        file_path: str | Path,
        collection_name: str = "knowledge_base",
    ) -> None:
        self.file_path = Path(file_path).resolve()
        self.collection_name = collection_name

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Document does not exist: {self.file_path}"
            )

        # V0: local in-memory Qdrant.
        # Later we will replace this with Docker Qdrant.
        self.client = QdrantClient(":memory:")
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        self._indexed = False

    def extract_text(self) -> str:
        """Extract text from the document."""
        converter = MarkItDown()
        result = converter.convert(str(self.file_path))

        text = result.text_content

        if not text or not text.strip():
            raise ValueError(
                f"No text could be extracted from {self.file_path}"
            )

        return text

    def create_chunks(self, text: str) -> list[Any]:
        """Split raw text into semantic chunks."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )

        chunks = chunker.chunk(text)

        if not chunks:
            raise ValueError("Document chunking returned no chunks.")

        return chunks

    def index_document(self) -> int:
        """
        Parse, chunk and index the document.
        """
        raw_text = self.extract_text()
        chunks = self.create_chunks(raw_text)

        documents = [chunk.text for chunk in chunks]

        # Create collection explicitly
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.client.get_embedding_size(
                        self.embedding_model
                    ),
                    distance=models.Distance.COSINE,
                ),
            )

        points = [
            models.PointStruct(
                id=index,
                vector=models.Document(
                    text=document,
                    model=self.embedding_model,
                ),
                payload={
                    "text": document,
                    "source": self.file_path.name,
                    "chunk_id": index,
                },
            )
            for index, document in enumerate(documents)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        self._indexed = True

        return len(documents)

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve relevant document chunks."""

        if not self._indexed:
            raise RuntimeError(
                "Document has not been indexed. "
                "Call index_document() first."
            )

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=models.Document(
                text=query,
                model=self.embedding_model,
            ),
            limit=limit,
        ).points

        return [
            {
                "text": (result.payload or {}).get("text", ""),
                "score": result.score,
                "metadata": {
                    "source": (result.payload or {}).get("source"),
                    "chunk_id": (result.payload or {}).get("chunk_id"),
                },
            }
            for result in results
        ]