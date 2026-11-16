from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.codeflowbench.schema import TaskSpec


class TrajectoryCategory(str, Enum):
    LONG_SESSION = "long_session"
    CROSS_TURN_DEPENDENCY = "cross_turn_dependency"
    COMPRESSION_DEGRADATION = "compression_degradation"


class TrajectoryTrack(str, Enum):
    FORMAL = "formal"
    SMOKE = "smoke"


@dataclass(frozen=True)
class TrajectoryTask:
    task_id: str
    category: TrajectoryCategory
    title: str
    message_count: int
    early_constraint: str
    active_constraint: str
    latest_decision: str
    noise_topic: str
    turns: tuple[dict[str, Any], ...]
    track: TrajectoryTrack = TrajectoryTrack.FORMAL

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=self.task_id,
            payload={
                "category": self.category.value,
                "title": self.title,
                "message_count": self.message_count,
                "verifier": "trajectory_llm_judge_and_context_v1",
            },
        )


@dataclass(frozen=True)
class JudgeScore:
    tool_rationality: float
    error_recovery: float
    constraint_preservation: float
    composite_score: float
    rationale: str
    is_synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_rationality": self.tool_rationality,
            "error_recovery": self.error_recovery,
            "constraint_preservation": self.constraint_preservation,
            "composite_score": self.composite_score,
            "rationale": self.rationale,
            "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class ContextQualityResult:
    scenario_id: str
    category: str
    strategy: str
    early_constraint_retained: bool
    active_constraint_applied: bool
    latest_decision_applied: bool
    key_constraint_retention_rate: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_reduction_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "strategy": self.strategy,
            "early_constraint_retained": self.early_constraint_retained,
            "active_constraint_applied": self.active_constraint_applied,
            "latest_decision_applied": self.latest_decision_applied,
            "key_constraint_retention_rate": self.key_constraint_retention_rate,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "token_reduction_pct": self.token_reduction_pct,
        }
