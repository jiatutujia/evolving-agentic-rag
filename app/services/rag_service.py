import asyncio
import math
import time

from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

from app.retrieval.adaptive_retrieval import (
    AdaptiveRetriever,
)


class RAGService:
    """
    Production-facing service wrapper around AdaptiveRetriever.

    Responsibilities
    ----------------
    1. Keep one RAG pipeline alive across requests.
    2. Prevent repeated model / index initialization.
    3. Serialize access to current JSON-backed memory state.
    4. Collect runtime latency and execution metrics.
    """

    def __init__(
        self,
        document_path: str | Path,
        memory_path: str | Path,
        reward_history_path: str | Path,
        memory_min_similarity: float = 0.75,
        metric_window_size: int = 1000,
    ) -> None:

        self.document_path = Path(
            document_path
        )

        self.memory_path = Path(
            memory_path
        )

        self.reward_history_path = Path(
            reward_history_path
        )

        # ==================================================
        # Initialize the full RAG pipeline once.
        # ==================================================

        self.retriever = AdaptiveRetriever(
            document_path=self.document_path,

            memory_path=self.memory_path,

            reward_history_path=self.reward_history_path,

            memory_min_similarity=(
                memory_min_similarity
            ),
        )

        # ==================================================
        # Current implementation writes JSON memory files.
        #
        # These writes are not designed for concurrent
        # mutation, so requests are serialized for now.
        #
        # Later:
        # persistent Qdrant / database-backed state.
        # ==================================================

        self._execution_lock = (
            asyncio.Lock()
        )

        # ==================================================
        # Runtime Metrics
        # ==================================================

        self.started_at = (
            time.time()
        )

        self.request_count = 0

        self.success_count = 0

        self.failure_count = 0

        self.strategy_counts = (
            Counter()
        )

        self.answer_status_counts = (
            Counter()
        )

        self.latencies_ms = deque(
            maxlen=metric_window_size
        )

    # ======================================================
    # Percentile helper
    # ======================================================

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: float,
    ) -> float | None:

        if not values:
            return None

        ordered = sorted(
            values
        )

        index = (
            math.ceil(
                percentile
                / 100.0
                * len(ordered)
            )
            - 1
        )

        index = max(
            0,
            min(
                index,
                len(ordered) - 1,
            ),
        )

        return float(
            ordered[index]
        )

    # ======================================================
    # Query execution
    # ======================================================

    async def query(
        self,
        query: str,
    ) -> dict[str, Any]:

        self.request_count += 1

        start_time = (
            time.perf_counter()
        )

        try:

            # ==============================================
            # AdaptiveRetriever is synchronous and performs
            # CPU / model work.
            #
            # Run it outside the FastAPI event loop.
            # ==============================================

            async with self._execution_lock:

                result = (
                    await asyncio.to_thread(
                        self.retriever.retrieve,
                        query,
                    )
                )

            latency_ms = (
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000.0
            )

            self.latencies_ms.append(
                latency_ms
            )

            self.success_count += 1

            strategy = result.get(
                "selected_strategy",
                "unknown",
            )

            answer_status = result.get(
                "answer_status",
                "unknown",
            )

            self.strategy_counts[
                strategy
            ] += 1

            self.answer_status_counts[
                answer_status
            ] += 1

            reward = result[
                "reward"
            ]

            router_decision = result[
                "router_decision"
            ]

            response = {
                "query": result[
                    "query"
                ],

                "answer": result[
                    "answer"
                ],

                "answer_status": (
                    answer_status
                ),

                "strategy_source": result[
                    "strategy_source"
                ],

                "selected_strategy": (
                    strategy
                ),

                "router_baseline": (
                    router_decision.route
                ),

                "initial_route": result[
                    "initial_route"
                ],

                "final_route": result[
                    "final_route"
                ],

                "retry_triggered": result[
                    "retry_triggered"
                ],

                "rewritten_query": result[
                    "rewritten_query"
                ],

                "reward_strategy": (
                    reward.strategy
                ),

                "reward_total": float(
                    reward.total_reward
                ),

                "latency_ms": round(
                    latency_ms,
                    2,
                ),

                "memory_size": result[
                    "memory_size"
                ],

                "reward_history_size": result[
                    "reward_history_size"
                ],
            }

            return response

        except Exception:

            self.failure_count += 1

            latency_ms = (
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000.0
            )

            self.latencies_ms.append(
                latency_ms
            )

            raise

    # ======================================================
    # Runtime Statistics
    # ======================================================

    def stats(
        self,
    ) -> dict[str, Any]:

        latencies = list(
            self.latencies_ms
        )

        if latencies:

            mean_latency = (
                sum(latencies)
                / len(latencies)
            )

        else:

            mean_latency = None

        p50 = self._percentile(
            latencies,
            50,
        )

        p95 = self._percentile(
            latencies,
            95,
        )

        return {
            "request_count": (
                self.request_count
            ),

            "success_count": (
                self.success_count
            ),

            "failure_count": (
                self.failure_count
            ),

            "p50_latency_ms": (
                None
                if p50 is None
                else round(
                    p50,
                    2,
                )
            ),

            "p95_latency_ms": (
                None
                if p95 is None
                else round(
                    p95,
                    2,
                )
            ),

            "mean_latency_ms": (
                None
                if mean_latency is None
                else round(
                    mean_latency,
                    2,
                )
            ),

            "strategy_counts": dict(
                self.strategy_counts
            ),

            "answer_status_counts": dict(
                self.answer_status_counts
            ),

            "uptime_seconds": round(
                time.time()
                - self.started_at,
                2,
            ),
        }