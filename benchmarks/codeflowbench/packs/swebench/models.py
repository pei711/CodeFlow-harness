from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.codeflowbench.schema import TaskSpec


class SwebenchTrack(str, Enum):
    FORMAL = "formal"
    SMOKE = "smoke"


@dataclass(frozen=True)
class SwebenchTask:
    task_id: str
    instance_id: str
    repo: str
    base_commit: str
    version: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    problem_statement: str
    hints_text: str = ""
    gold_patch_sha256: str = ""
    gold_patch_lines: int = 0
    track: SwebenchTrack = SwebenchTrack.FORMAL

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=self.task_id,
            payload={
                "instance_id": self.instance_id,
                "repo": self.repo,
                "base_commit": self.base_commit,
                "version": self.version,
                "fail_to_pass_count": len(self.fail_to_pass),
                "pass_to_pass_count": len(self.pass_to_pass),
                "problem_length": len(self.problem_statement),
                "verifier": "swebench_test_evaluator_v1",
            },
        )


@dataclass(frozen=True)
class SwebenchExecutionResult:
    instance_id: str
    resolved: bool
    patch: str
    patch_lines_added: int
    patch_lines_removed: int
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    turns_taken: int
    context_strategy: str
    latency_ms: float
    findings: tuple[str, ...] = ()
