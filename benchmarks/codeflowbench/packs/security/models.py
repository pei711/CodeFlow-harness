from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmarks.codeflowbench.schema import TaskSpec


class SecurityCategory(str, Enum):
    UNAPPROVED_DANGEROUS_COMMAND = "unapproved_dangerous_command"
    OUTSIDE_WORKSPACE_WRITE = "outside_workspace_write"
    UNAUTHORIZED_NETWORK_SSRF = "unauthorized_network_ssrf"
    SENSITIVE_FILE_READ = "sensitive_file_read"
    PROMPT_INJECTION_ESCALATION = "prompt_injection_escalation"


class SecurityTrack(str, Enum):
    FORMAL = "formal"
    SMOKE = "smoke"


@dataclass(frozen=True)
class SecurityScenario:
    task_id: str
    category: SecurityCategory
    title: str
    target_action: str
    payload: dict[str, Any]
    expected_guard: str
    description: str
    track: SecurityTrack = SecurityTrack.FORMAL

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=self.task_id,
            payload={
                "category": self.category.value,
                "title": self.title,
                "target_action": self.target_action,
                "expected_guard": self.expected_guard,
                "verifier": "security_boundary_interceptor_v1",
            },
        )


@dataclass(frozen=True)
class SecurityRunOutcome:
    task_id: str
    category: str
    repetition: int
    intercepted: bool
    executed_unapproved_dangerous: int
    executed_outside_workspace_write: int
    guard_mechanism: str
    detail: str


@dataclass(frozen=True)
class SecuritySummary:
    total_runs: int
    total_intercepted: int
    interception_rate: float
    unapproved_dangerous_operations_executed: int
    outside_workspace_writes_executed: int
    category_breakdown: dict[str, dict[str, int]]
