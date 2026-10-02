from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import median


class RequestFate(StrEnum):
    EXECUTED = "executed"
    MERGED_INTO_RUNNING_TURN = "merged_into_running_turn"
    FALLBACK_EXECUTED = "fallback_executed"
    CANCELLED_BEFORE_START = "cancelled_before_start"
    CANCELLED_WHILE_RUNNING = "cancelled_while_running"


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None

    @classmethod
    def from_values(cls, values: list[float]) -> "LatencySummary":
        ordered = sorted(values)
        if not ordered:
            return cls(count=0, p50=None, p95=None, p99=None, maximum=None)
        return cls(
            count=len(ordered),
            p50=median(ordered),
            p95=_percentile(ordered, 0.95),
            p99=_percentile(ordered, 0.99),
            maximum=ordered[-1],
        )


@dataclass(frozen=True)
class R0RuntimeResult:
    accepted_requests: int
    rejected_requests: int
    rejected_without_handle: int
    conversation_count: int
    fate_counts: dict[str, int]
    runner_invocations: int
    lost_requests: int
    unresolved_handles: int
    unexpected_duplicate_executions: int
    lifecycle_contradictions: int
    pool_limit_violations: int
    peak_user_concurrency: int
    peak_system_concurrency: int
    shutdown_cancelled_requests: int
    max_observed_pending_depth: int
    dispatch_overhead_ms: LatencySummary
    queue_wait_ms: LatencySummary
    execution_latency_ms: LatencySummary
    evidence_label: str = "deterministic_scheduler_runtime"

    @property
    def passed(self) -> bool:
        invoked_fates = (
            self.fate_counts.get(RequestFate.EXECUTED.value, 0)
            + self.fate_counts.get(RequestFate.FALLBACK_EXECUTED.value, 0)
            + self.fate_counts.get(RequestFate.CANCELLED_WHILE_RUNNING.value, 0)
        )
        return (
            self.accepted_requests == 2_000
            and self.rejected_requests == 64
            and self.rejected_without_handle == self.rejected_requests
            and self.conversation_count == 64
            and sum(self.fate_counts.values()) == self.accepted_requests
            and all(self.fate_counts.get(fate.value, 0) > 0 for fate in RequestFate)
            and self.runner_invocations == invoked_fates
            and self.lost_requests == 0
            and self.unresolved_handles == 0
            and self.unexpected_duplicate_executions == 0
            and self.lifecycle_contradictions == 0
            and self.pool_limit_violations == 0
            and self.peak_user_concurrency == 16
            and self.peak_system_concurrency == 4
            and self.shutdown_cancelled_requests == 64
            and self.dispatch_overhead_ms.count == 16
        )


@dataclass(frozen=True)
class R1RuntimeResult:
    turns: int
    submitted_through_scheduler: int
    runner_type: str
    terminal_counts: dict[str, int]
    model_calls: int
    tool_event_turns: int
    readable_sessions: int
    delivery_counts: dict[str, int]
    identity_contradictions: int
    lifecycle_contradictions: int
    unresolved_handles: int
    resources_closed: bool
    evidence_label: str = "deterministic_benchmark_host_full_path"

    @property
    def passed(self) -> bool:
        return (
            self.turns == 100
            and self.submitted_through_scheduler == self.turns
            and self.runner_type == "AgentTurnRunner"
            and self.terminal_counts
            == {
                "completed": 84,
                "completed_with_tool_failure": 8,
                "provider_failed": 4,
                "cancelled": 4,
            }
            and self.model_calls == 118
            and self.tool_event_turns == 18
            and self.readable_sessions == 92
            and self.delivery_counts
            == {
                "delivered": 88,
                "dropped": 2,
                "no_outlet": 2,
            }
            and self.identity_contradictions == 0
            and self.lifecycle_contradictions == 0
            and self.unresolved_handles == 0
            and self.resources_closed
        )


def _percentile(ordered: list[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]
