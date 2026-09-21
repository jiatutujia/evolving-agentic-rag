from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Public request schema for the RAG API.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question sent to the RAG system.",
    )


class QueryResponse(BaseModel):
    """
    Public response schema.

    We expose important execution information so that
    routing / reward / latency behavior can be benchmarked.
    """

    query: str

    answer: str

    answer_status: str

    strategy_source: str

    selected_strategy: str

    router_baseline: str

    initial_route: str

    final_route: str

    retry_triggered: bool

    rewritten_query: str | None

    reward_strategy: str

    reward_total: float

    latency_ms: float

    memory_size: int

    reward_history_size: int


class HealthResponse(BaseModel):
    status: str

    ready: bool

    service: str


class StatsResponse(BaseModel):
    request_count: int

    success_count: int

    failure_count: int

    p50_latency_ms: float | None

    p95_latency_ms: float | None

    mean_latency_ms: float | None

    strategy_counts: dict[str, int]

    answer_status_counts: dict[str, int]

    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str