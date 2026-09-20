import json

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RewardRecord:
    """
    Persistent record for one executed strategy.
    """

    query: str

    strategy: str

    reward: float

    answer_status: str

    created_at: str


class RewardHistory:
    """
    Persistent reward history.

    The history is used to estimate the historical
    utility of different strategies.
    """

    def __init__(
        self,
        path: str | Path = (
            "memory/reward_history.json"
        ),
    ) -> None:

        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records: list[
            RewardRecord
        ] = []

        self._load()

    # ======================================================
    # Persistence
    # ======================================================

    def _load(
        self,
    ) -> None:

        if not self.path.exists():
            return

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as file:

            raw_records = (
                json.load(
                    file
                )
            )

        self.records = [
            RewardRecord(
                **record
            )
            for record
            in raw_records
        ]

        print(
            f"Loaded "
            f"{len(self.records)} "
            f"reward records."
        )

    def _save(
        self,
    ) -> None:

        data = [
            asdict(record)
            for record
            in self.records
        ]

        with self.path.open(
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
        strategy: str,
        reward: float,
        answer_status: str,
    ) -> RewardRecord:

        record = RewardRecord(
            query=query,

            strategy=strategy,

            reward=float(
                reward
            ),

            answer_status=(
                answer_status
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
    # Statistics
    # ======================================================

    def average_reward(
        self,
        strategy: str,
    ) -> float | None:

        rewards = [
            record.reward
            for record
            in self.records
            if record.strategy
            == strategy
        ]

        if not rewards:
            return None

        return (
            sum(rewards)
            / len(rewards)
        )

    def strategy_summary(
        self,
    ) -> dict:

        strategies = {
            record.strategy
            for record
            in self.records
        }

        summary = {}

        for strategy in strategies:

            rewards = [
                record.reward
                for record
                in self.records
                if record.strategy
                == strategy
            ]

            summary[strategy] = {
                "count": len(
                    rewards
                ),

                "average_reward": (
                    sum(rewards)
                    / len(rewards)
                ),

                "best_reward": (
                    max(rewards)
                ),

                "worst_reward": (
                    min(rewards)
                ),
            }

        return summary

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