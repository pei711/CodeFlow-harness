from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PairCoverage:
    planned_pairs: int
    valid_pairs: int
    covered_tasks: int
    required_valid_pairs: int
    valid: bool


def assess_pair_coverage(
    *,
    expected_pairs: int,
    planned_pair_keys: Iterable[tuple[str, int]],
    valid_pair_keys: Iterable[tuple[str, int]],
    minimum_valid_pairs_per_task: int = 2,
) -> PairCoverage:
    if minimum_valid_pairs_per_task < 1:
        raise ValueError(
            "minimum_valid_pairs_per_task must be positive",
        )
    planned = set(planned_pair_keys)
    valid = set(valid_pair_keys)
    planned_by_task = Counter(task_id for task_id, _repetition in planned)
    valid_by_task = Counter(task_id for task_id, _repetition in valid)
    required_valid_pairs = (expected_pairs * 9 + 9) // 10
    coverage_valid = (
        expected_pairs > 0
        and len(planned) == expected_pairs
        and valid <= planned
        and len(valid) >= required_valid_pairs
        and all(
            valid_by_task[task_id]
            >= min(
                minimum_valid_pairs_per_task,
                repetitions,
            )
            for task_id, repetitions in planned_by_task.items()
        )
    )
    return PairCoverage(
        planned_pairs=len(planned),
        valid_pairs=len(valid),
        covered_tasks=len(valid_by_task),
        required_valid_pairs=required_valid_pairs,
        valid=coverage_valid,
    )


__all__ = ["PairCoverage", "assess_pair_coverage"]
