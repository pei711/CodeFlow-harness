from __future__ import annotations

import asyncio
from typing import Any

import pytest

from codeflow.agent.tools import (
    Tool,
    ToolCapability,
    ToolEffect,
    ToolExecution,
    ToolInvocation,
    ToolRegistry,
)
from codeflow.agent.tools.file_search import FindTool, GrepTool
from codeflow.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from codeflow.agent.tools.web import WebFetchTool, WebSearchTool


class _ConcurrencyProbe(Tool):
    capability = ToolCapability(effect=ToolEffect.READ, concurrency_safe=True)

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    @property
    def name(self) -> str:
        return "concurrency_probe"

    @property
    def description(self) -> str:
        return "records active calls"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        }

    async def execute(self, label: str, **kwargs: Any) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            return label
        finally:
            self.active -= 1


class _UnsafeConcurrencyProbe(_ConcurrencyProbe):
    capability = ToolCapability(effect=ToolEffect.WRITE)

    @property
    def name(self) -> str:
        return "unsafe_concurrency_probe"


class _MisdeclaredWriteProbe(_ConcurrencyProbe):
    capability = ToolCapability(effect=ToolEffect.WRITE, concurrency_safe=True)

    @property
    def name(self) -> str:
        return "misdeclared_write_probe"


class _CancellationProbe(_ConcurrencyProbe):
    def __init__(self) -> None:
        super().__init__()
        self.sibling_cancelled = False

    @property
    def name(self) -> str:
        return "cancellation_probe"

    async def execute(self, label: str, **kwargs: Any) -> str:
        if label == "cancel":
            await asyncio.sleep(0)
            raise asyncio.CancelledError()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            self.sibling_cancelled = True
            raise
        return label


def test_duplicate_registration_requires_explicit_replacement() -> None:
    first = _ConcurrencyProbe()
    second = _ConcurrencyProbe()
    registry = ToolRegistry()
    registry.register(first)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(second)

    registry.register(second, replace=True)
    assert registry.get(second.name) is second


async def test_concurrency_safe_calls_overlap_and_keep_result_order() -> None:
    tool = _ConcurrencyProbe()
    registry = ToolRegistry()
    registry.register(tool)

    executions = await registry.execute_many(
        [
            ToolInvocation(tool.name, {"label": "first"}),
            ToolInvocation(tool.name, {"label": "second"}),
        ]
    )

    assert tool.peak == 2
    assert [str(execution.result) for execution in executions] == ["first", "second"]


async def test_calls_without_concurrency_capability_remain_serial() -> None:
    tool = _UnsafeConcurrencyProbe()
    registry = ToolRegistry()
    registry.register(tool)

    executions = await registry.execute_many(
        [
            ToolInvocation(tool.name, {"label": "first"}),
            ToolInvocation(tool.name, {"label": "second"}),
        ]
    )

    assert tool.peak == 1
    assert [str(execution.result) for execution in executions] == ["first", "second"]


async def test_write_effect_is_serial_even_if_tool_claims_concurrency_safety() -> None:
    tool = _MisdeclaredWriteProbe()
    registry = ToolRegistry()
    registry.register(tool)

    await registry.execute_many(
        [
            ToolInvocation(tool.name, {"label": "first"}),
            ToolInvocation(tool.name, {"label": "second"}),
        ]
    )

    assert tool.peak == 1


async def test_parallel_batches_obey_registry_limit() -> None:
    tool = _ConcurrencyProbe()
    registry = ToolRegistry(max_parallel=2)
    registry.register(tool)

    executions = await registry.execute_many([ToolInvocation(tool.name, {"label": str(index)}) for index in range(5)])

    assert tool.peak == 2
    assert [str(execution.result) for execution in executions] == ["0", "1", "2", "3", "4"]


async def test_unsafe_call_is_a_barrier_between_safe_batches() -> None:
    safe = _ConcurrencyProbe()
    unsafe = _UnsafeConcurrencyProbe()
    registry = ToolRegistry()
    registry.register(safe)
    registry.register(unsafe)
    events: list[tuple[str, str]] = []

    async def on_start(invocation: ToolInvocation) -> None:
        events.append(("start", str(invocation.arguments["label"])))

    async def on_complete(execution: ToolExecution) -> None:
        events.append(("complete", str(execution.result)))

    await registry.execute_many(
        [
            ToolInvocation(safe.name, {"label": "safe-1"}),
            ToolInvocation(safe.name, {"label": "safe-2"}),
            ToolInvocation(unsafe.name, {"label": "write"}),
            ToolInvocation(safe.name, {"label": "safe-3"}),
            ToolInvocation(safe.name, {"label": "safe-4"}),
        ],
        on_start=on_start,
        on_complete=on_complete,
    )

    write_start = events.index(("start", "write"))
    write_complete = events.index(("complete", "write"))
    assert events.index(("complete", "safe-1")) < write_start
    assert events.index(("complete", "safe-2")) < write_start
    assert write_complete < events.index(("start", "safe-3"))
    assert write_complete < events.index(("start", "safe-4"))
    assert safe.peak == 2
    assert unsafe.peak == 1


async def test_cancelled_parallel_call_cancels_and_joins_siblings() -> None:
    tool = _CancellationProbe()
    registry = ToolRegistry()
    registry.register(tool)

    with pytest.raises(asyncio.CancelledError):
        await registry.execute_many(
            [
                ToolInvocation(tool.name, {"label": "cancel"}),
                ToolInvocation(tool.name, {"label": "sibling"}),
            ]
        )

    assert tool.sibling_cancelled is True


@pytest.mark.parametrize(
    ("tool_type", "effect", "concurrency_safe"),
    [
        (ReadFileTool, ToolEffect.READ, True),
        (ListDirTool, ToolEffect.READ, True),
        (GrepTool, ToolEffect.READ, True),
        (FindTool, ToolEffect.READ, True),
        (WriteFileTool, ToolEffect.WRITE, False),
        (EditFileTool, ToolEffect.WRITE, False),
        (WebSearchTool, ToolEffect.EXTERNAL, False),
        (WebFetchTool, ToolEffect.EXTERNAL, False),
    ],
)
def test_builtin_tools_declare_effect_and_concurrency(tool_type, effect, concurrency_safe) -> None:
    assert tool_type.capability.effect is effect
    assert tool_type.capability.concurrency_safe is concurrency_safe
