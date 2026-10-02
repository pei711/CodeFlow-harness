from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TokenWiseCostMeasurement:
    task_id: str
    workload_class: str
    repetition: int
    cache_policy: str
    task_passed: bool
    usage_complete: bool
    cost_complete: bool
    requested_model: str
    actual_model: str | None
    fallback_used: bool
    fresh_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    cost_usd: float

    def __post_init__(self) -> None:
        if self.repetition < 0:
            raise ValueError("repetition must not be negative")
        for field_name in (
            "fresh_input_tokens",
            "cache_write_tokens",
            "cache_read_tokens",
            "output_tokens",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")
        if not isfinite(self.cost_usd) or self.cost_usd < 0:
            raise ValueError("cost_usd must be finite and non-negative")


@dataclass(frozen=True)
class TokenWiseArmSummary:
    cache_policy: str
    trials: int
    verified_successes: int
    task_pass_rate: float
    total_cost_usd: float
    cost_per_verified_success_usd: float | None
    conservative_cache_hit_rate: float


@dataclass(frozen=True)
class TokenWiseCostClaim:
    claim_eligible: bool
    expected_blocks: int
    valid_blocks: int
    arms: dict[str, TokenWiseArmSummary]
    cost_reduction_vs_disrupted: float | None
    cache_hit_rate_lift: float | None
    findings: tuple[str, ...]
    cv_metrics: dict[str, int | float]


__all__ = [
    "TokenWiseArmSummary",
    "TokenWiseCostClaim",
    "TokenWiseCostMeasurement",
]
