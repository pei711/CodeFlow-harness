"""SubagentManager concurrency gate.

Isolates the gate: build_executor and _run_subagent_inner are stubbed, so the
test drives only the Semaphore in _run_subagent (no real VM, no real LLM). A
stubbed inner holds each subagent inside the gate on an Event, letting the test
observe the concurrent peak.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from codeflow.agent.subagent import manager as manager_mod
from codeflow.agent.subagent.manager import SubagentManager, SubagentOutcome, SubagentStatus
from codeflow.agent.tools.registry import ToolRegistry
from codeflow.agent.tools.spawn import SpawnTool
from codeflow.config.schema import AgentDefaults
from codeflow.providers.base import LLMResponse, ToolCallRequest
from codeflow.tracing import spans as _spans


class _StubProvider:
    def get_default_model(self) -> str:
        return "stub-model"


class _DummyExecutor:
    async def __aenter__(self) -> "_DummyExecutor":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FailingExitExecutor(_DummyExecutor):
    async def __aexit__(self, *exc: object) -> bool:
        raise RuntimeError("executor cleanup failed")


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "1")
    monkeypatch.setenv("CODEFLOW_TRACING_DIR", str(tmp_path))
    _spans._store = None
    yield tmp_path
    _spans._store = None


def _trace_spans(trace_dir: Path) -> list[dict]:
    log = trace_dir / "logs" / "audit-spans.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


async def _settle(predicate, *, tries: int = 2000) -> None:
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never reached")


def _make_manager(max_concurrent: int) -> SubagentManager:
    return SubagentManager(
        provider=_StubProvider(),
        workspace=Path("/tmp"),
        max_concurrent=max_concurrent,
    )


async def _drive(monkeypatch, *, max_concurrent: int, spawn_n: int) -> int:
    """Spawn spawn_n subagents against a gate of max_concurrent; return the peak
    number that were ever inside the gate at once."""
    mgr = _make_manager(max_concurrent)
    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _DummyExecutor())

    state = {"current": 0, "peak": 0}
    release = asyncio.Event()

    async def _stub_inner(task_id, task, label, origin, executor) -> SubagentOutcome:
        state["current"] += 1
        state["peak"] = max(state["peak"], state["current"])
        await release.wait()
        state["current"] -= 1
        return SubagentOutcome(SubagentStatus.COMPLETED, "done")

    monkeypatch.setattr(mgr, "_run_subagent_inner", _stub_inner)
    mgr.set_submit(lambda req: None)

    for i in range(spawn_n):
        await mgr.spawn(task=f"task-{i}")
    tasks = list(mgr._running_tasks.values())

    await _settle(lambda: state["current"] == max_concurrent)
    await asyncio.sleep(0)
    peak = state["peak"]

    release.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    return peak


async def test_gate_caps_concurrent_subagents(monkeypatch):
    peak = await _drive(monkeypatch, max_concurrent=2, spawn_n=5)
    assert peak == 2


async def test_gate_of_one_serializes_subagents(monkeypatch):
    peak = await _drive(monkeypatch, max_concurrent=1, spawn_n=4)
    assert peak == 1


@pytest.mark.parametrize("bad", [0, -1])
def test_max_concurrent_subagents_must_be_positive(bad):
    with pytest.raises(ValidationError):
        AgentDefaults(max_concurrent_subagents=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_max_subagent_spawns_per_hour_must_be_positive(bad):
    with pytest.raises(ValidationError):
        AgentDefaults(max_subagent_spawns_per_hour=bad)


def _fixed_clock(monkeypatch, start: float = 1000.0) -> list[float]:
    """Pin manager's monotonic clock to a mutable value (advance via holder[0])."""
    holder = [start]
    monkeypatch.setattr(manager_mod.time, "monotonic", lambda: holder[0])
    return holder


def _stub_mgr(monkeypatch, **kw) -> SubagentManager:
    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _DummyExecutor())
    mgr = SubagentManager(provider=_StubProvider(), workspace=Path("/tmp"), **kw)

    async def _noop_inner(*a, **k) -> SubagentOutcome:
        return SubagentOutcome(SubagentStatus.COMPLETED, "done")

    monkeypatch.setattr(mgr, "_run_subagent_inner", _noop_inner)
    mgr.set_submit(lambda req: None)
    return mgr


async def test_spawn_rate_limit_refuses_within_window(monkeypatch):
    """N spawns/window allowed; the next is refused even as concurrency frees up."""
    _fixed_clock(monkeypatch)
    mgr = _stub_mgr(monkeypatch, max_spawns_per_hour=2)

    assert "started" in await mgr.spawn(task="a")
    assert "started" in await mgr.spawn(task="b")
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)

    r3 = await mgr.spawn(task="c")
    assert "Spawn refused" in r3
    assert "2 per hour" in r3


async def test_spawn_tool_reports_rate_limit_as_failed_without_starting_task(monkeypatch, trace_dir):
    _fixed_clock(monkeypatch)
    mgr = _stub_mgr(monkeypatch, max_spawns_per_hour=1)
    spawn_tool = SpawnTool(mgr)
    spawn_tool.set_context("cli", "direct", "cli:test")
    tools = ToolRegistry()
    tools.register(spawn_tool)

    accepted = await tools.execute("spawn", {"task": "first"}, call_id="spawn-accepted")
    assert accepted.failed is False
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)
    await asyncio.sleep(0)

    refused = await tools.execute("spawn", {"task": "second"}, call_id="spawn-refused")

    assert "Spawn refused" in refused
    assert refused.failed is True
    assert mgr.get_running_count() == 0
    refused_span = next(
        span for span in _trace_spans(trace_dir) if span["attributes"].get("tool.call_id") == "spawn-refused"
    )
    assert refused_span["status"]["code"] == "ERROR"
    assert refused_span["status"]["message"] == refused_span["attributes"]["tool.error"]
    assert refused_span["attributes"]["tool.error"].startswith("Spawn refused:")


async def test_spawn_rate_limit_recovers_after_window(monkeypatch):
    """Older spawns age out of the rolling window, so the limit auto-recovers
    without any explicit /stop."""
    clock = _fixed_clock(monkeypatch)
    mgr = _stub_mgr(monkeypatch, max_spawns_per_hour=1)

    assert "started" in await mgr.spawn(task="a")
    assert "Spawn refused" in await mgr.spawn(task="b")

    clock[0] += manager_mod._SPAWN_WINDOW_SECONDS + 1
    assert "started" in await mgr.spawn(task="c")


async def test_spawn_rate_limit_is_per_session(monkeypatch):
    """One session hitting the limit must not throttle others."""
    _fixed_clock(monkeypatch)
    mgr = _stub_mgr(monkeypatch, max_spawns_per_hour=1)

    assert "started" in await mgr.spawn(task="a", session_key="sessA")
    assert "Spawn refused" in await mgr.spawn(task="a2", session_key="sessA")
    assert "started" in await mgr.spawn(task="b", session_key="sessB")


async def test_cancel_by_session_clears_spawn_history(monkeypatch):
    """Session teardown drops its rate-limit history (bounds the dict)."""
    _fixed_clock(monkeypatch)
    mgr = _stub_mgr(monkeypatch, max_spawns_per_hour=1)

    assert "started" in await mgr.spawn(task="a", session_key="sessA")
    assert "Spawn refused" in await mgr.spawn(task="a2", session_key="sessA")
    await asyncio.gather(*mgr._running_tasks.values(), return_exceptions=True)

    await mgr.cancel_by_session("sessA")
    assert "sessA" not in mgr._session_spawn_times


async def test_cancel_by_session_cancels_live_task(monkeypatch, trace_dir):
    """cancel_by_session cancels a still-running asyncio.Task registered under
    the session (not just the rate-limit bookkeeping)."""
    mgr = _make_manager(max_concurrent=1)
    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _DummyExecutor())

    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_inner(task_id, task, label, origin, executor) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(mgr, "_run_subagent_inner", _blocking_inner)
    submitted = []
    mgr.set_submit(submitted.append)

    assert "started" in await mgr.spawn(task="long", session_key="sessLive")
    await _settle(entered.is_set)
    assert mgr.get_running_count() == 1
    (live_task,) = list(mgr._running_tasks.values())

    cancelled = await mgr.cancel_by_session("sessLive")
    assert cancelled == 1
    assert live_task.cancelled()
    await _settle(lambda: mgr.get_running_count() == 0)
    assert submitted == []
    run_span = next(span for span in _trace_spans(trace_dir) if span["name"] == "subagent.run")
    assert run_span["attributes"]["subagent.status"] == "cancelled"
    assert run_span["status"] == {"code": "ERROR", "message": "cancelled"}


async def test_announce_result_routes_to_tui_session_key(monkeypatch):
    """TUI origins pass an authoritative session_key distinct from channel:chat_id
    (chat_id falls back to "default" while the live subscription is keyed by
    session_key); the re-injected TurnRequest must land on that session, not on
    the derived channel:chat_id."""
    mgr = _make_manager(max_concurrent=1)
    submitted = []
    mgr.set_submit(lambda req: submitted.append(req))

    await mgr._announce_result(
        task_id="t1",
        label="label",
        task="task",
        result="result",
        origin={"channel": "tui", "chat_id": "default", "session_key": "tui:sess123"},
        status=SubagentStatus.COMPLETED,
    )

    assert len(submitted) == 1
    assert submitted[0].conversation == "tui:sess123"


async def test_announce_result_routes_non_tui_origin_unchanged(monkeypatch):
    """Non-TUI origins (e.g. a channel with a real chat_id) still announce on
    their existing channel:chat_id conversation — no regression."""
    mgr = _make_manager(max_concurrent=1)
    submitted = []
    mgr.set_submit(lambda req: submitted.append(req))

    await mgr._announce_result(
        task_id="t2",
        label="label",
        task="task",
        result="result",
        origin={"channel": "feishu", "chat_id": "ou_12345", "session_key": "feishu:ou_12345"},
        status=SubagentStatus.COMPLETED,
    )

    assert len(submitted) == 1
    assert submitted[0].conversation == "feishu:ou_12345"


async def test_iteration_exhaustion_is_announced_as_failure(monkeypatch, tmp_path, trace_dir):
    class _ToolOnlyProvider:
        def __init__(self) -> None:
            self.calls = 0

        def get_default_model(self) -> str:
            return "stub-model"

        async def chat_with_retry(self, **kwargs) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"call-{self.calls}",
                        name="missing",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            )

    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _DummyExecutor())
    provider = _ToolOnlyProvider()
    mgr = SubagentManager(provider=provider, workspace=tmp_path, max_concurrent=1)
    submitted = []
    mgr.set_submit(submitted.append)

    await mgr.spawn(task="keep using tools", label="loop", session_key="cli:test")
    (background_task,) = list(mgr._running_tasks.values())
    outcome = await background_task

    assert outcome.status is SubagentStatus.EXHAUSTED
    assert provider.calls == 15
    assert len(submitted) == 1
    assert "completed successfully" not in submitted[0].text
    assert "exhausted" in submitted[0].text.lower()
    run_span = next(span for span in _trace_spans(trace_dir) if span["name"] == "subagent.run")
    assert run_span["attributes"]["subagent.status"] == "exhausted"
    assert run_span["status"] == {"code": "ERROR", "message": "exhausted"}


async def test_completed_subagent_returns_and_traces_typed_status(monkeypatch, tmp_path, trace_dir):
    class _FinalProvider:
        def get_default_model(self) -> str:
            return "stub-model"

        async def chat_with_retry(self, **kwargs) -> LLMResponse:
            return LLMResponse(content="done")

    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _DummyExecutor())
    mgr = SubagentManager(provider=_FinalProvider(), workspace=tmp_path, max_concurrent=1)
    submitted = []
    mgr.set_submit(submitted.append)

    await mgr.spawn(task="finish", label="done", session_key="cli:test")
    (background_task,) = list(mgr._running_tasks.values())
    outcome = await background_task

    assert outcome.status is SubagentStatus.COMPLETED
    assert "completed successfully" in submitted[0].text
    run_span = next(span for span in _trace_spans(trace_dir) if span["name"] == "subagent.run")
    assert run_span["attributes"]["subagent.status"] == "completed"
    assert run_span["status"] == {"code": "OK", "message": ""}


async def test_failed_subagent_returns_and_traces_typed_status(monkeypatch, tmp_path, trace_dir):
    class _FailingProvider:
        def get_default_model(self) -> str:
            return "stub-model"

        async def chat_with_retry(self, **kwargs) -> LLMResponse:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _DummyExecutor())
    mgr = SubagentManager(provider=_FailingProvider(), workspace=tmp_path, max_concurrent=1)
    submitted = []
    mgr.set_submit(submitted.append)

    await mgr.spawn(task="fail", label="failed", session_key="cli:test")
    (background_task,) = list(mgr._running_tasks.values())
    outcome = await background_task

    assert outcome.status is SubagentStatus.FAILED
    assert "[Subagent 'failed' failed]" in submitted[0].text
    run_span = next(span for span in _trace_spans(trace_dir) if span["name"] == "subagent.run")
    assert run_span["attributes"]["subagent.status"] == "failed"
    assert run_span["status"] == {"code": "ERROR", "message": "failed"}


@pytest.mark.parametrize("inner_status", ["completed", "exhausted", "failed"])
async def test_executor_exit_failure_announces_one_final_failure(monkeypatch, tmp_path, trace_dir, inner_status):
    class _Provider:
        def __init__(self) -> None:
            self.calls = 0

        def get_default_model(self) -> str:
            return "stub-model"

        async def chat_with_retry(self, **kwargs) -> LLMResponse:
            self.calls += 1
            if inner_status == "completed":
                return LLMResponse(content="done")
            if inner_status == "failed":
                raise RuntimeError("provider unavailable")
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"call-{self.calls}",
                        name="missing",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            )

    monkeypatch.setattr(manager_mod, "build_executor", lambda *a, **k: _FailingExitExecutor())
    mgr = SubagentManager(provider=_Provider(), workspace=tmp_path, max_concurrent=1)
    submitted = []
    mgr.set_submit(submitted.append)

    await mgr.spawn(task="finish before cleanup", label="cleanup", session_key="cli:test")
    (background_task,) = list(mgr._running_tasks.values())
    outcome = await background_task

    assert len(submitted) == 1
    assert outcome.status is SubagentStatus.FAILED
    assert "[Subagent 'cleanup' failed]" in submitted[0].text
    assert "executor cleanup failed" in submitted[0].text
    run_span = next(span for span in _trace_spans(trace_dir) if span["name"] == "subagent.run")
    assert run_span["attributes"]["subagent.status"] == "failed"
    assert run_span["status"] == {"code": "ERROR", "message": "failed"}


def test_build_subagent_prompt_does_not_start_skill_watcher(monkeypatch):
    """_build_subagent_prompt uses a transient ContextBuilder just for
    _build_runtime_context; it must not leave a skill-catalog file watcher
    running behind it (one leaked watchfiles/inotify thread per spawn)."""
    import codeflow.agent.context as context_mod

    calls = []
    real_init = context_mod.ContextBuilder.__init__

    def _spy_init(self, workspace, *args, **kwargs):
        calls.append(kwargs.get("start_watcher", True))
        return real_init(self, workspace, *args, **kwargs)

    monkeypatch.setattr(context_mod.ContextBuilder, "__init__", _spy_init)

    mgr = _make_manager(max_concurrent=1)
    mgr._build_subagent_prompt()

    assert calls == [False]
