from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.codeflowbench.records import VerifierResult


@dataclass(frozen=True)
class SwebenchMetrics:
    resolved: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_strategy: str
    patch_generated: bool
    patch_lines_added: int
    patch_lines_removed: int
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int
    turns_taken: int
    end_to_end_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_strategy": self.context_strategy,
            "patch_generated": self.patch_generated,
            "patch_lines_added": self.patch_lines_added,
            "patch_lines_removed": self.patch_lines_removed,
            "fail_to_pass_passed": self.fail_to_pass_passed,
            "fail_to_pass_total": self.fail_to_pass_total,
            "pass_to_pass_passed": self.pass_to_pass_passed,
            "pass_to_pass_total": self.pass_to_pass_total,
            "turns_taken": self.turns_taken,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
        }


@dataclass(frozen=True)
class SwebenchComparisonMeasurement:
    task_id: str
    instance_id: str
    control_resolved: bool
    treatment_resolved: bool
    control_input_tokens: int
    treatment_input_tokens: int
    input_token_saving_pct: float
    treatment_advantage: int  # +1 if treatment resolved and control did not, 0 if equal, -1 if reverse
