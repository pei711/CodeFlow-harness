from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SecurityMetrics:
    intercepted: int
    unapproved_dangerous_operations_executed: int
    outside_workspace_writes_executed: int
    category: str
    guard_mechanism: str
    repetition: int
    end_to_end_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercepted": self.intercepted,
            "unapproved_dangerous_operations_executed": self.unapproved_dangerous_operations_executed,
            "outside_workspace_writes_executed": self.outside_workspace_writes_executed,
            "category": self.category,
            "guard_mechanism": self.guard_mechanism,
            "repetition": self.repetition,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
        }
