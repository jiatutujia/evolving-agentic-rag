import os

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request

from app.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)

from app.services.rag_service import (
    RAGService,
)


# ==========================================================
# Configuration
# ==========================================================

DOCUMENT_PATH = Path(
    os.getenv(
        "RAG_DOCUMENT_PATH",
        "data/dspy.pdf",
    )
)

MEMORY_PATH = Path(
    os.getenv(
        "RAG_MEMORY_PATH",
        "memory/experience_memory.json",
    )
)

REWARD_HISTORY_PATH = Path(
    os.getenv(
        "RAG_REWARD_HISTORY_PATH",
        "memory/reward_history.json",
    )
)

MEMORY_MIN_SIMILARITY = float(
    os.getenv(
        "RAG_MEMORY_MIN_SIMILARITY",
        "0.75",
    )
)


# ==========================================================
# Lifespan
#
# Build the RAG system once when the API process starts.
# ==========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Starting Self-Evolving Agentic RAG API"
    )

    print(
        "=" * 100
    )

    print(
        f"Document Path : "
        f"{DOCUMENT_PATH}"
    )

    print(
        f"Memory Path   : "
        f"{MEMORY_PATH}"
    )

    print(
        f"Reward Path   : "
        f"{REWARD_HISTORY_PATH}"
    )

    if not DOCUMENT_PATH.exists():

        raise FileNotFoundError(
            f"RAG document does not exist: "
            f"{DOCUMENT_PATH}"
        )

    service = RAGService(
        document_path=(
            DOCUMENT_PATH
        ),

        memory_path=(
            MEMORY_PATH
        ),

        reward_history_path=(
            REWARD_HISTORY_PATH
        ),

        memory_min_similarity=(
            MEMORY_MIN_SIMILARITY
        ),
    )

    app.state.rag_service = (
        service
    )

    print(
        "\nRAG API is ready."
    )

    yield

    print(
        "\nShutting down RAG API."
    )


# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title=(
        "Self-Evolving Agentic RAG"
    ),

    description=(
        "Reward-aware adaptive RAG service "
        "with retrieval grading, reflection, "
        "experience memory and online "
        "strategy optimization."
    ),

    version="7.1.0",

    lifespan=lifespan,
)


# ==========================================================
# Root
# ==========================================================

@app.get("/")
async def root() -> dict:

    return {
        "service": (
            "Self-Evolving Agentic RAG"
        ),

        "version": "7.1.0",

        "docs": "/docs",

        "health": "/health",

        "stats": "/stats",
    }


# ==========================================================
# Health
# ==========================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health(
    request: Request,
) -> HealthResponse:

    ready = hasattr(
        request.app.state,
        "rag_service",
    )

    return HealthResponse(
        status=(
            "ok"
            if ready
            else "starting"
        ),

        ready=ready,

        service=(
            "self-evolving-agentic-rag"
        ),
    )


# ==========================================================
# Query
# ==========================================================

@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:

    service: RAGService = (
        request.app.state.rag_service
    )

    try:

        result = await service.query(
            payload.query
        )

        return QueryResponse(
            **result
        )

    except Exception as exc:

        print(
            f"Query execution failed: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,

            detail=(
                "RAG query execution failed."
            ),
        ) from exc


# ==========================================================
# Runtime Statistics
# ==========================================================

@app.get(
    "/stats",
    response_model=StatsResponse,
)
async def stats(
    request: Request,
) -> StatsResponse:

    service: RAGService = (
        request.app.state.rag_service
    )

    return StatsResponse(
        **service.stats()
    )