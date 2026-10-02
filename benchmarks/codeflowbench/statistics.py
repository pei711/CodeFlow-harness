from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    tasks: int
    samples: int
    seed: int
    unit: str = "task"
    exploratory: bool = False


def clustered_bootstrap_interval(
    per_task_repetitions: dict[str, tuple[float, ...]],
    *,
    samples: int,
    seed: int,
) -> BootstrapInterval:
    if not per_task_repetitions:
        raise ValueError("clustered bootstrap requires at least one task")
    if samples < 1:
        raise ValueError("samples must be positive")
    task_means = {task_id: mean(values) for task_id, values in per_task_repetitions.items() if values}
    if len(task_means) != len(per_task_repetitions):
        raise ValueError("every task requires at least one repetition")
    task_ids = sorted(task_means)
    estimate = mean(task_means.values())
    rng = random.Random(seed)
    distribution = sorted(mean(task_means[rng.choice(task_ids)] for _ in task_ids) for _ in range(samples))
    return BootstrapInterval(
        estimate=estimate,
        lower=_quantile(distribution, 0.025),
        upper=_quantile(distribution, 0.975),
        tasks=len(task_ids),
        samples=samples,
        seed=seed,
        exploratory=len(task_ids) < 30,
    )


def _quantile(values: list[float], quantile: float) -> float:
    index = round((len(values) - 1) * quantile)
    return values[index]
