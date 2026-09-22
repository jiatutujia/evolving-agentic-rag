import hashlib
import os

from pathlib import Path
from typing import Any

from chonkie import SemanticChunker
from markitdown import MarkItDown
from qdrant_client import QdrantClient
from qdrant_client import models


class DocumentSearcher:
    """
    Persistent document retrieval layer.

    Responsibilities
    ----------------
    1. Parse documents.
    2. Split documents into semantic chunks.
    3. Index chunks into Qdrant.
    4. Reuse an existing persistent collection.
    5. Retrieve relevant chunks for a query.

    Storage modes
    -------------
    Local development:
        QdrantClient(path="qdrant_storage")

    Qdrant server / Docker:
        QdrantClient(url="http://qdrant:6333")

    Existing public API remains compatible with V0-V6.
    """

    def __init__(
        self,
        file_path: str | Path,
        collection_name: str = "knowledge_base",
        qdrant_url: str | None = None,
        qdrant_path: str | Path | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:

        # ==================================================
        # Document
        # ==================================================

        self.file_path = Path(
            file_path
        ).resolve()

        self.collection_name = (
            collection_name
        )

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"Document does not exist: "
                f"{self.file_path}"
            )

        # ==================================================
        # Embedding Model
        # ==================================================

        self.embedding_model = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )

        # ==================================================
        # Document Fingerprint
        #
        # Used to detect:
        # - changed PDF
        # - stale collection
        # ==================================================

        self.document_hash = (
            self._calculate_file_hash()
        )

        # ==================================================
        # Qdrant Configuration
        #
        # Priority:
        #
        # explicit argument
        # →
        # environment variable
        # →
        # default local path
        # ==================================================

        env_qdrant_url = (
            os.getenv(
                "QDRANT_URL"
            )
        )

        env_qdrant_path = (
            os.getenv(
                "QDRANT_PATH",
                "qdrant_storage",
            )
        )

        env_qdrant_api_key = (
            os.getenv(
                "QDRANT_API_KEY"
            )
        )

        self.qdrant_url = (
            qdrant_url
            or env_qdrant_url
        )

        self.qdrant_api_key = (
            qdrant_api_key
            or env_qdrant_api_key
        )

        # ==================================================
        # Server Mode
        # ==================================================

        if self.qdrant_url:

            client_kwargs = {
                "url": (
                    self.qdrant_url
                )
            }

            if self.qdrant_api_key:

                client_kwargs[
                    "api_key"
                ] = (
                    self.qdrant_api_key
                )

            self.client = (
                QdrantClient(
                    **client_kwargs
                )
            )

            self.storage_mode = (
                "server"
            )

            self.qdrant_path = None

            print(
                "Qdrant mode: server"
            )

            print(
                f"Qdrant URL : "
                f"{self.qdrant_url}"
            )

        # ==================================================
        # Persistent Local Mode
        # ==================================================

        else:

            local_path = (
                qdrant_path
                or env_qdrant_path
            )

            self.qdrant_path = (
                Path(
                    local_path
                ).resolve()
            )

            self.qdrant_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.client = (
                QdrantClient(
                    path=str(
                        self.qdrant_path
                    )
                )
            )

            self.storage_mode = (
                "local"
            )

            print(
                "Qdrant mode: "
                "persistent local"
            )

            print(
                f"Qdrant path: "
                f"{self.qdrant_path}"
            )

        # ==================================================
        # Determine whether a reusable index already exists
        # ==================================================

        self._indexed = (
            self._collection_matches_current_document()
        )

    # ======================================================
    # File Fingerprint
    # ======================================================

    def _calculate_file_hash(
        self,
    ) -> str:
        """
        Calculate SHA256 of the source document.

        If the file changes, its hash changes and the
        existing collection will no longer be considered
        valid for this document.
        """

        digest = (
            hashlib.sha256()
        )

        with self.file_path.open(
            "rb"
        ) as file:

            while True:

                block = file.read(
                    1024 * 1024
                )

                if not block:
                    break

                digest.update(
                    block
                )

        return digest.hexdigest()

    # ======================================================
    # Collection Helpers
    # ======================================================

    def _collection_exists(
        self,
    ) -> bool:

        return bool(
            self.client.collection_exists(
                self.collection_name
            )
        )

    def _point_count(
        self,
    ) -> int:

        if not self._collection_exists():

            return 0

        result = (
            self.client.count(
                collection_name=(
                    self.collection_name
                ),
                exact=True,
            )
        )

        return int(
            result.count
        )

    def _sample_payload(
        self,
    ) -> dict:

        if not self._collection_exists():

            return {}

        records, _ = (
            self.client.scroll(
                collection_name=(
                    self.collection_name
                ),

                limit=1,

                with_payload=True,

                with_vectors=False,
            )
        )

        if not records:

            return {}

        return (
            records[0].payload
            or {}
        )

    def _collection_matches_current_document(
        self,
    ) -> bool:
        """
        Return True only when:

        - collection exists
        - collection contains points
        - source document hash matches
        - embedding model matches
        """

        if not self._collection_exists():

            return False

        if self._point_count() <= 0:

            return False

        payload = (
            self._sample_payload()
        )

        stored_hash = payload.get(
            "document_hash"
        )

        stored_model = payload.get(
            "embedding_model"
        )

        if (
            stored_hash
            != self.document_hash
        ):

            return False

        if (
            stored_model
            != self.embedding_model
        ):

            return False

        return True

    # ======================================================
    # Public Index Information
    # ======================================================

    def get_index_info(
        self,
    ) -> dict:
        """
        Return basic index diagnostics.
        """

        exists = (
            self._collection_exists()
        )

        points = (
            self._point_count()
            if exists
            else 0
        )

        ready = (
            self._collection_matches_current_document()
            if exists
            else False
        )

        return {
            "collection_name": (
                self.collection_name
            ),

            "storage_mode": (
                self.storage_mode
            ),

            "qdrant_url": (
                self.qdrant_url
            ),

            "qdrant_path": (
                None
                if self.qdrant_path is None
                else str(
                    self.qdrant_path
                )
            ),

            "exists": exists,

            "ready": ready,

            "points_count": (
                points
            ),

            "document": (
                self.file_path.name
            ),

            "document_hash": (
                self.document_hash
            ),

            "embedding_model": (
                self.embedding_model
            ),
        }

    # ======================================================
    # Text Extraction
    # ======================================================

    def extract_text(
        self,
    ) -> str:
        """
        Extract text from the document.
        """

        converter = (
            MarkItDown()
        )

        result = (
            converter.convert(
                str(
                    self.file_path
                )
            )
        )

        text = (
            result.text_content
        )

        if (
            not text
            or not text.strip()
        ):

            raise ValueError(
                "No text could be "
                "extracted from "
                f"{self.file_path}"
            )

        return text

    # ======================================================
    # Semantic Chunking
    # ======================================================

    def create_chunks(
        self,
        text: str,
    ) -> list[Any]:
        """
        Split raw text into semantic chunks.
        """

        chunker = (
            SemanticChunker(
                embedding_model=(
                    "minishlab/"
                    "potion-base-8M"
                ),

                threshold=0.5,

                chunk_size=512,

                min_sentences=1,
            )
        )

        chunks = (
            chunker.chunk(
                text
            )
        )

        if not chunks:

            raise ValueError(
                "Document chunking "
                "returned no chunks."
            )

        return chunks

    # ======================================================
    # Collection Creation
    # ======================================================

    def _create_collection(
        self,
    ) -> None:

        vector_size = (
            self.client.get_embedding_size(
                self.embedding_model
            )
        )

        self.client.create_collection(
            collection_name=(
                self.collection_name
            ),

            vectors_config=(
                models.VectorParams(
                    size=(
                        vector_size
                    ),

                    distance=(
                        models.Distance.COSINE
                    ),
                )
            ),
        )

    # ======================================================
    # Index Document
    # ======================================================

    def index_document(
        self,
        force_reindex: bool = False,
        batch_size: int = 64,
    ) -> int:
        """
        Parse, chunk and index the document.

        Important behavior
        ------------------
        If a valid persistent collection already exists,
        the collection is reused immediately.

        The PDF will NOT be:
        - reparsed
        - rechunked
        - re-embedded

        unless force_reindex=True.
        """

        # ==================================================
        # Existing compatible collection
        # ==================================================

        if (
            not force_reindex
            and
            self._collection_matches_current_document()
        ):

            count = (
                self._point_count()
            )

            self._indexed = True

            print(
                f"Reusing existing Qdrant "
                f"collection "
                f"'{self.collection_name}'."
            )

            print(
                f"Indexed chunks: "
                f"{count}"
            )

            return count

        # ==================================================
        # Existing incompatible collection
        # ==================================================

        if self._collection_exists():

            if not force_reindex:

                payload = (
                    self._sample_payload()
                )

                stored_hash = (
                    payload.get(
                        "document_hash"
                    )
                )

                stored_model = (
                    payload.get(
                        "embedding_model"
                    )
                )

                raise RuntimeError(
                    "\nExisting Qdrant collection "
                    "does not match the current "
                    "document or embedding model.\n\n"
                    f"Collection: "
                    f"{self.collection_name}\n"
                    f"Current document: "
                    f"{self.file_path.name}\n"
                    f"Current hash: "
                    f"{self.document_hash[:12]}...\n"
                    f"Stored hash: "
                    f"{stored_hash}\n"
                    f"Current embedding model: "
                    f"{self.embedding_model}\n"
                    f"Stored embedding model: "
                    f"{stored_model}\n\n"
                    "Run ingestion again with "
                    "force_reindex=True."
                )

            print(
                f"Deleting existing collection "
                f"'{self.collection_name}'..."
            )

            self.client.delete_collection(
                collection_name=(
                    self.collection_name
                )
            )

        # ==================================================
        # Parse
        # ==================================================

        print(
            f"Parsing document: "
            f"{self.file_path}"
        )

        raw_text = (
            self.extract_text()
        )

        # ==================================================
        # Chunk
        # ==================================================

        print(
            "Creating semantic chunks..."
        )

        chunks = (
            self.create_chunks(
                raw_text
            )
        )

        documents = [
            chunk.text
            for chunk
            in chunks
        ]

        print(
            f"Chunks created: "
            f"{len(documents)}"
        )

        # ==================================================
        # Create Collection
        # ==================================================

        self._create_collection()

        print(
            f"Created Qdrant collection: "
            f"{self.collection_name}"
        )

        # ==================================================
        # Build Points
        # ==================================================

        points = [
            models.PointStruct(
                id=index,

                vector=(
                    models.Document(
                        text=document,

                        model=(
                            self.embedding_model
                        ),
                    )
                ),

                payload={
                    "text": (
                        document
                    ),

                    "source": (
                        self.file_path.name
                    ),

                    "chunk_id": (
                        index
                    ),

                    "document_hash": (
                        self.document_hash
                    ),

                    "embedding_model": (
                        self.embedding_model
                    ),
                },
            )

            for index, document
            in enumerate(
                documents
            )
        ]

        # ==================================================
        # Batched Upsert
        # ==================================================

        total_points = len(
            points
        )

        print(
            "Embedding and uploading "
            "chunks to Qdrant..."
        )

        for start in range(
            0,
            total_points,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total_points,
            )

            batch = (
                points[
                    start:end
                ]
            )

            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),

                points=batch,

                wait=True,
            )

            print(
                f"Indexed "
                f"{end}/"
                f"{total_points}"
            )

        self._indexed = True

        print(
            "\nQdrant indexing complete."
        )

        return total_points

    # ======================================================
    # Search
    # ======================================================

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Retrieve relevant document chunks.
        """

        # ==================================================
        # Recover state after process restart
        # ==================================================

        if not self._indexed:

            if (
                self._collection_matches_current_document()
            ):

                self._indexed = True

            else:

                raise RuntimeError(
                    "Document has not been indexed. "
                    "Run index_document() first."
                )

        query = (
            query.strip()
        )

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )

        results = (
            self.client.query_points(
                collection_name=(
                    self.collection_name
                ),

                query=(
                    models.Document(
                        text=query,

                        model=(
                            self.embedding_model
                        ),
                    )
                ),

                limit=limit,

                with_payload=True,
            ).points
        )

        return [
            {
                "text": (
                    result.payload
                    or {}
                ).get(
                    "text",
                    "",
                ),

                "score": (
                    result.score
                ),

                "metadata": {
                    "source": (
                        result.payload
                        or {}
                    ).get(
                        "source"
                    ),

                    "chunk_id": (
                        result.payload
                        or {}
                    ).get(
                        "chunk_id"
                    ),
                },
            }

            for result
            in results
        ]

    # ======================================================
    # Cleanup
    # ======================================================

    def close(
        self,
    ) -> None:

        self.client.close()