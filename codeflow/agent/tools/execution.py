from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeflow.agent.tools.base import ToolResult


class ToolEffect(StrEnum):
    UNKNOWN = "unknown"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ToolCapability:
    effect: ToolEffect = ToolEffect.UNKNOWN
    concurrency_safe: bool = False


@dataclass(frozen=True)
class ToolExecutionContext:
    call_id: str | None = None
    session_key: str = ""
    iteration: int | None = None
    origin: str | None = None
    parent_call_id: str | None = None

    def child(self, call_id: str | None) -> ToolExecutionContext:
        return replace(self, call_id=call_id, parent_call_id=self.call_id)


@dataclass(frozen=True)
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    context: ToolExecutionContext = field(default_factory=ToolExecutionContext)


@dataclass(frozen=True)
class ToolExecution:
    invocation: ToolInvocation
    result: ToolResult
    duration_ms: float


__all__ = [
    "ToolCapability",
    "ToolEffect",
    "ToolExecution",
    "ToolExecutionContext",
    "ToolInvocation",
]
