from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrajectoryMetrics:
    tool_rationality: float
    error_recovery: float
    constraint_preservation: float
    composite_judge_score: float
    early_constraint_retained: int
    active_constraint_applied: int
    latest_decision_applied: int
    key_constraint_retention_rate: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_strategy: str
    token_reduction_pct: float
    end_to_end_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_rationality": self.tool_rationality,
            "error_recovery": self.error_recovery,
            "constraint_preservation": self.constraint_preservation,
            "composite_judge_score": self.composite_judge_score,
            "early_constraint_retained": self.early_constraint_retained,
            "active_constraint_applied": self.active_constraint_applied,
            "latest_decision_applied": self.latest_decision_applied,
            "key_constraint_retention_rate": self.key_constraint_retention_rate,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_strategy": self.context_strategy,
            "token_reduction_pct": self.token_reduction_pct,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
        }
