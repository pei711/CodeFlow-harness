"""Deterministic correlation contract for one Turn.

Drives real ``Lane`` turns and asserts what the evidence model in
The Turn evidence contract requires one trace per Turn, the
``spine.turn -> session.turn -> llm.call/tool.call`` chain, a usage row that
joins back by trace id, and five pairwise-distinct terminal states.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.agent.loop.main import ProviderTurnError
from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.providers.base import LLMProvider, LLMResponse
from codeflow.spine import ChatType, Origin, Source, Text, TurnOutcome, TurnRequest, Usage
from codeflow.spine.scheduler import Lane, OriginPools
from codeflow.token_wise.usage_tracker import UsageTracker
from codeflow.tracing import context as trace_context
from codeflow.tracing import spans as _spans
from codeflow.tracing import trace


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "1")
    monkeypatch.setenv("CODEFLOW_TRACING_DIR", str(tmp_path / "traces"))
    _spans._store = None
    yield tmp_path
    _spans._store = None


def _rows(trace_dir):
    log = trace_dir / "traces" / "logs" / "audit-spans.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _by_name(trace_dir):

    out: dict[str, dict] = {}
    for row in _rows(trace_dir):
        out[row["name"]] = row
    return out


def _req(text: str = "hi") -> TurnRequest:
    src = Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    return TurnRequest(origin=Origin.USER, source=src, text=text)


def _lane(runner, events: list) -> Lane:
    async def sink(event) -> None:
        events.append(event)

    return Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="t:c")


async def _run_one(runner, events: list) -> None:
    lane = _lane(runner, events)
    await lane.submit(_req())


class ScriptedRunner:
    """Opens the spans a real turn would, then returns the scripted outcome."""

    def __init__(self, *, tool_failures: int = 0, raises: BaseException | None = None):
        self._tool_failures = tool_failures
        self._raises = raises

    async def run(self, req, emit, drain) -> TurnOutcome:
        with trace.span("session.turn", {"turn.input_preview": req.text}):
            with trace.span("llm.call", {"llm.provider": "p", "llm.model": "m"}):
                pass
            with trace.span("tool.call", {"tool.name": "grep"}):
                pass
            if self._raises is not None:
                raise self._raises
        await emit(Text(content="reply"))
        return TurnOutcome(
            usage=Usage(1, 2, 3),
            explicit_reply=True,
            tool_calls=1,
            tool_failures=self._tool_failures,
        )


class HangingRunner:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        with trace.span("session.turn"):
            self.started.set()
            await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _spine(trace_dir) -> dict:
    return _by_name(trace_dir)["spine.turn"]


def test_trace_context_pins_the_session_turn_across_nested_model_spans() -> None:
    with trace_context.turn_scope(
        session_key="s1",
        channel="test",
        chat_id="c1",
        root_span_id="spine-span",
    ):
        assert trace_context.current().turn_span_id is None
        session_token = trace_context.push(
            trace_id="trace-1",
            span_id="session-span",
            name="session.turn",
            kind="session",
        )
        try:
            assert trace_context.current().turn_span_id == "session-span"
            model_token = trace_context.push(
                trace_id="trace-1",
                span_id="model-span",
                name="llm.call",
                kind="model",
            )
            try:
                assert trace_context.current().turn_span_id == "session-span"
            finally:
                trace_context.reset(model_token)
        finally:
            trace_context.reset(session_token)


async def test_a_completed_turn_chains_spine_session_and_leaf_spans(trace_dir):
    events: list = []
    await _run_one(ScriptedRunner(), events)

    by = _by_name(trace_dir)
    assert {sp["traceId"] for sp in _rows(trace_dir)} == {by["spine.turn"]["traceId"]}
    assert by["spine.turn"]["parentSpanId"] is None
    assert by["session.turn"]["parentSpanId"] == by["spine.turn"]["spanId"]
    assert by["llm.call"]["parentSpanId"] == by["session.turn"]["spanId"]
    assert by["tool.call"]["parentSpanId"] == by["session.turn"]["spanId"]

    attrs = by["spine.turn"]["attributes"]
    assert attrs["span.type"] == "session"
    assert attrs["spine.outcome"] == "completed"
    assert attrs["spine.terminal_event"] == "TurnEnded"
    assert attrs["spine.conversation_id"] == "t:c"
    assert attrs["spine.origin"] == "user"
    assert attrs["spine.channel"] == "t"
    assert by["spine.turn"]["status"]["code"] == "OK"


async def test_a_tool_failure_is_a_completion_with_its_own_outcome(trace_dir):
    await _run_one(ScriptedRunner(tool_failures=1), [])
    attrs = _spine(trace_dir)["attributes"]
    assert attrs["spine.outcome"] == "completed_with_tool_failure"
    assert attrs["spine.terminal_event"] == "TurnEnded"
    assert attrs["spine.tool_failures"] == 1
    assert attrs["spine.tool_calls"] == 1


async def test_a_provider_failure_is_distinguishable_from_a_generic_error(trace_dir):
    await _run_one(ScriptedRunner(raises=ProviderTurnError("rate_limit")), [])
    attrs = _spine(trace_dir)["attributes"]
    assert attrs["spine.outcome"] == "provider_failed"
    assert attrs["spine.error_class"] == "ProviderTurnError"
    assert attrs["spine.provider_error_category"] == "rate_limit"
    assert attrs["spine.terminal_event"] == "TurnFailed"
    assert _spine(trace_dir)["status"]["code"] == "ERROR"


async def test_a_generic_runner_error_records_its_class(trace_dir):
    await _run_one(ScriptedRunner(raises=ValueError("boom")), [])
    attrs = _spine(trace_dir)["attributes"]
    assert attrs["spine.outcome"] == "error"
    assert attrs["spine.error_class"] == "ValueError"
    assert "spine.provider_error_category" not in attrs


async def test_a_cancelled_turn_reaches_the_span(trace_dir):
    runner = HangingRunner()
    lane = _lane(runner, [])
    fut = lane.submit(_req())
    await runner.started.wait()
    assert lane.cancel_running() == 1
    await fut
    for _ in range(2000):
        if "spine.turn" in _by_name(trace_dir):
            break
        await asyncio.sleep(0)

    spine = _spine(trace_dir)
    assert spine["attributes"]["spine.outcome"] == "cancelled"
    assert spine["attributes"]["spine.terminal_event"] == "TurnFailed"
    assert spine["status"]["code"] == "ERROR"
    assert spine["status"]["message"] == "cancelled"


async def test_every_terminal_state_is_pairwise_distinct(trace_dir):
    """The five states must not collapse; a collapse is what hides regressions."""
    outcomes = []
    for runner in (
        ScriptedRunner(),
        ScriptedRunner(tool_failures=2),
        ScriptedRunner(raises=ProviderTurnError("timeout")),
        ScriptedRunner(raises=ValueError("boom")),
    ):
        await _run_one(runner, [])
        outcomes.append(_spine(trace_dir)["attributes"]["spine.outcome"])
    assert len(set(outcomes)) == len(outcomes) == 4
    assert set(outcomes) == {
        "completed",
        "completed_with_tool_failure",
        "provider_failed",
        "error",
    }


async def test_a_turn_submitted_inside_a_span_still_owns_its_own_trace(trace_dir):
    """The subagent re-injection shape: submit() starts the lane worker on the
    caller's context, so without a root span the Turn would join the caller's
    trace and the run would show two Turns under one trace id."""
    lane = _lane(ScriptedRunner(), [])
    with trace.span("subagent.run", kind="subagent") as caller:
        fut = lane.submit(_req())
    await fut

    by = _by_name(trace_dir)
    assert by["spine.turn"]["parentSpanId"] is None
    assert by["spine.turn"]["traceId"] != caller.trace_id
    assert by["session.turn"]["traceId"] == by["spine.turn"]["traceId"]
    assert by["llm.call"]["traceId"] == by["spine.turn"]["traceId"]


async def test_tracing_disabled_leaves_the_turn_untouched(trace_dir, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "0")
    events: list = []
    await _run_one(ScriptedRunner(), events)
    assert _rows(trace_dir) == []
    assert [type(e).__name__ for e in events] == ["TurnStarted", "Text", "TurnEnded"]


class _StubProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        return LLMResponse(
            content="ok",
            finish_reason="stop",
            usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        )

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


async def test_the_agent_loop_turn_span_nests_under_the_spine_root(trace_dir, workspace):
    loop = AgentLoop(
        provider=_StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
    )
    await _run_one(AgentTurnRunner(loop, stream=False), [])

    by = _by_name(trace_dir)
    assert by["session.turn"]["parentSpanId"] == by["spine.turn"]["spanId"]
    assert by["session.turn"]["traceId"] == by["spine.turn"]["traceId"]
    assert by["llm.call"]["traceId"] == by["spine.turn"]["traceId"]
    assert by["llm.call"]["attributes"]["llm.call_id"] == by["llm.call"]["spanId"]
    assert by["spine.turn"]["attributes"]["spine.outcome"] == "completed"


async def test_a_usage_snapshot_carries_the_turn_trace_ids(trace_dir, workspace):
    tracker = UsageTracker(telemetry_dir=trace_dir / "telemetry", persist=True)
    loop = AgentLoop(
        provider=_StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
    )
    loop.strategies.register(tracker)
    await _run_one(AgentTurnRunner(loop, stream=False), [])
    tracker.close()

    rows = [
        json.loads(line)
        for path in sorted((trace_dir / "telemetry").glob("usage-*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    by = _by_name(trace_dir)
    assert rows, "the tracker persisted no usage row"
    assert all(row["trace_id"] == by["spine.turn"]["traceId"] for row in rows)
    assert all(row["turn_span_id"] == by["session.turn"]["spanId"] for row in rows)


async def test_usage_ids_default_to_none_without_tracing(workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("CODEFLOW_TRACING", "0")
    tracker = UsageTracker(telemetry_dir=tmp_path / "telemetry", persist=True)
    loop = AgentLoop(
        provider=_StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
    )
    loop.strategies.register(tracker)
    await _run_one(AgentTurnRunner(loop, stream=False), [])
    tracker.close()

    rows = [
        json.loads(line)
        for path in sorted((tmp_path / "telemetry").glob("usage-*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert rows
    assert all(row["trace_id"] is None and row["turn_span_id"] is None for row in rows)
