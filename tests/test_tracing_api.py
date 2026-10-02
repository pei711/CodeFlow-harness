"""Contract tests for the ``codeflow.tracing.trace`` facade (standard-api.v1).

Exercises the public ``trace.span`` API and asserts it emits well-formed
``audit.span.v1`` records: correct nesting, kinds, attributes, artifact refs,
error status, and no-op when disabled.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from codeflow.tracing import spans as _spans
from codeflow.tracing import trace


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "1")
    monkeypatch.setenv("CODEFLOW_TRACING_DIR", str(tmp_path))
    _spans._store = None
    yield tmp_path
    _spans._store = None


def _spans_written(trace_dir):
    log = trace_dir / "logs" / "audit-spans.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def test_nesting_kinds_and_attributes(trace_dir):
    with trace.span("session.turn", {"turn.input_preview": "hi"}) as root:
        root_id = root.span_id
        with trace.span("llm.call", {"llm.provider": "openrouter", "llm.model": "m"}) as s:
            s.set({"llm.usage.total_tokens": 42})

    spans = _spans_written(trace_dir)
    by = {sp["name"]: sp for sp in spans}
    assert len(spans) == 2
    assert len({sp["traceId"] for sp in spans}) == 1
    assert by["session.turn"]["parentSpanId"] is None
    assert by["llm.call"]["parentSpanId"] == root_id
    assert by["session.turn"]["attributes"]["span.type"] == "session"
    assert by["llm.call"]["attributes"]["span.type"] == "model"
    assert by["llm.call"]["attributes"]["llm.provider"] == "openrouter"
    assert by["llm.call"]["attributes"]["llm.usage.total_tokens"] == 42
    assert all(sp["schemaVersion"] == "audit.span.v1" for sp in spans)


def test_invocation_source_derives_from_enclosing_purpose(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("skill.gate", kind="skill"):
        with trace.span("llm.call") as s:
            assert s.invocation_source == "skill.gate"
            semconv.llm_call(s, {"self": None, "messages": [], "tools": None, "model": "openrouter/m"}, None, None)
    sp = next(x for x in _spans_written(trace_dir) if x["name"] == "llm.call")
    assert sp["attributes"]["llm.invocation_source"] == "skill.gate"


def test_invocation_source_is_none_at_root(trace_dir):
    with trace.span("session.turn") as root:
        assert root.invocation_source is None


def test_purpose_spans_record_input_and_output(trace_dir):
    from codeflow.tracing import semconv

    class _R:
        need_retrieval = True
        rewritten_query = "frontend design skills"

    with trace.span("skill.rewrite", kind="skill") as s:
        semconv.skill_rewrite(s, {"query": "help me install a frontend skill"}, _R(), None)
    sp = _spans_written(trace_dir)[0]
    a = sp["attributes"]
    assert a["skill.rewrite.need_retrieval"] is True

    assert "skill.rewrite.input.artifact_path" in a
    assert "skill.rewrite.output.artifact_path" in a


def test_personalize_extractor_records_step_io(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("personalize.classify", kind="memory") as s:
        semconv.personalize(
            s, {"self": object(), "message": "hi", "history": None}, {"needs_clarification": False}, None
        )
    a = _spans_written(trace_dir)[0]["attributes"]
    assert a["span.type"] == "memory"
    assert a["personalize.step"] == "classify"
    assert a["personalize.ok"] is True
    assert "personalize.input.artifact_path" in a
    assert "personalize.output.artifact_path" in a


def test_error_marks_status_and_reraises(trace_dir):
    with pytest.raises(ValueError):
        with trace.span("tool.call", {"tool.name": "read_file"}):
            raise ValueError("boom")

    spans = _spans_written(trace_dir)
    assert len(spans) == 1
    assert spans[0]["status"]["code"] == "ERROR"
    assert "boom" in spans[0]["status"]["message"]


def test_cancellation_marks_error_and_reraises(trace_dir):
    async def cancelled_body():
        with trace.span("spine.turn", {"spine.conversation_id": "t:c"}):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled_body())

    spans = _spans_written(trace_dir)
    assert len(spans) == 1
    assert spans[0]["status"]["code"] == "ERROR"
    assert spans[0]["status"]["message"] == "cancelled"


def test_cancellation_of_an_awaiting_task_marks_error(trace_dir):
    async def scenario():
        started = asyncio.Event()

        async def body():
            with trace.span("spine.turn"):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(body())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    spans = _spans_written(trace_dir)
    assert [sp["status"]["code"] for sp in spans] == ["ERROR"]
    assert spans[0]["status"]["message"] == "cancelled"


def test_base_exception_marks_error_and_reraises(trace_dir):
    with pytest.raises(KeyboardInterrupt):
        with trace.span("tool.call"):
            raise KeyboardInterrupt()

    assert _spans_written(trace_dir)[0]["status"]["code"] == "ERROR"


def test_attach_rejoins_a_trace_from_another_task(trace_dir):
    async def scenario():
        with trace.span("spine.turn") as root:
            captured = (root.trace_id, root.span_id)

        async def worker():
            with trace.attach(*captured):
                with trace.span("channel.deliver", kind="channel"):
                    pass

        await asyncio.create_task(worker())
        return captured

    trace_id, span_id = asyncio.run(scenario())
    by = {sp["name"]: sp for sp in _spans_written(trace_dir)}
    assert by["channel.deliver"]["traceId"] == trace_id
    assert by["channel.deliver"]["parentSpanId"] == span_id


def test_attach_without_a_trace_id_is_a_noop(trace_dir):
    with trace.attach(None, None):
        with trace.span("channel.deliver", kind="channel") as s:
            attached_trace = s.trace_id
    assert attached_trace
    assert _spans_written(trace_dir)[0]["parentSpanId"] is None


def test_root_refuses_inherited_context(trace_dir):
    with trace.span("subagent.run", kind="subagent") as caller:
        with trace.span("spine.turn", root=True, session_key="c") as turn:
            with trace.span("llm.call"):
                pass

    by = {sp["name"]: sp for sp in _spans_written(trace_dir)}
    assert turn.trace_id != caller.trace_id
    assert by["spine.turn"]["parentSpanId"] is None
    assert by["llm.call"]["traceId"] == turn.trace_id
    assert by["llm.call"]["parentSpanId"] == turn.span_id


def test_llm_and_tool_call_ids_are_emitted(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("llm.call") as s:
        semconv.llm_call(s, {"self": None, "messages": [], "tools": None, "model": "openrouter/m"}, None, None)
        llm_span_id = s.span_id
    with trace.span("tool.call") as s:
        semconv.tool_call(s, {"name": "grep", "params": {}, "call_id": "call_abc"}, "ok", None)
    by = {sp["name"]: sp for sp in _spans_written(trace_dir)}
    assert by["llm.call"]["attributes"]["llm.call_id"] == llm_span_id
    assert by["tool.call"]["attributes"]["tool.call_id"] == "call_abc"


def test_tool_call_id_is_omitted_when_the_caller_has_none(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("tool.call") as s:
        semconv.tool_call(s, {"name": "grep", "params": {}}, "ok", None)
    assert "tool.call_id" not in _spans_written(trace_dir)[0]["attributes"]


def test_artifact_reference_attached(trace_dir):
    with trace.span("llm.call", {"llm.provider": "p", "llm.model": "m"}) as s:
        s.artifact("llm.input", {"messages": [{"role": "user", "content": "hi"}]})

    spans = _spans_written(trace_dir)
    assert "llm.input.artifact_path" in spans[0]["attributes"]


def test_artifact_omits_inline_image_data_without_mutating_live_payload(trace_dir):
    data_uri = "data:image/png;base64,QUJDRA=="
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
            }
        ],
        "serialized_prompt": f'before {{"url":"{data_uri}"}} after',
    }

    with trace.span("llm.call") as span:
        span.artifact("llm.input", payload)

    artifact_path = _spans_written(trace_dir)[0]["attributes"]["llm.input.artifact_path"]
    persisted = Path(artifact_path).read_text(encoding="utf-8")
    assert payload["messages"][0]["content"][0]["image_url"]["url"] == data_uri
    assert data_uri not in persisted
    assert "QUJDRA==" not in persisted
    assert persisted.count("[image data omitted]") == 2


def test_custom_node_uses_explicit_kind(trace_dir):
    with trace.span("codeflow.plugin.refresh", {"plugin.reason": "x"}, kind="plugin"):
        pass

    spans = _spans_written(trace_dir)
    assert spans[0]["name"] == "codeflow.plugin.refresh"
    assert spans[0]["attributes"]["span.type"] == "plugin"


def test_disabled_is_noop(trace_dir, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "0")
    with trace.span("should.noop") as n:
        n.set({"x": 1})
    assert _spans_written(trace_dir) == []


def test_tool_call_extractor(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("tool.call") as s:
        semconv.tool_call(s, {"name": "list_dir", "params": {"path": "."}}, "a\nb", None)
    sp = _spans_written(trace_dir)[0]
    assert sp["name"] == "tool.call"
    assert sp["attributes"]["span.type"] == "tool"
    assert sp["attributes"]["tool.name"] == "list_dir"
    assert "tool.input.artifact_path" in sp["attributes"]
    assert "tool.output.artifact_path" in sp["attributes"]


def test_read_file_of_skill_retypes_to_skill(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("tool.call") as s:
        semconv.tool_call(
            s,
            {"name": "read_file", "params": {"path": "/ws/skills/weather/SKILL.md"}},
            "## weather\nbody",
            None,
        )
    sp = _spans_written(trace_dir)[0]
    assert sp["name"] == "skill.read"
    assert sp["attributes"]["span.type"] == "skill"
    assert sp["attributes"]["skill.name"] == "weather"
    assert sp["attributes"]["skill.path"] == "/ws/skills/weather/SKILL.md"
    assert sp["attributes"]["skill.read.via_tool"] == "read_file"


def test_skill_read_tool_retypes_to_skill(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("tool.call") as s:
        semconv.tool_call(
            s,
            {"name": "skill_read", "params": {"name": "release-helper"}},
            "## release-helper\nbody",
            None,
        )
    sp = _spans_written(trace_dir)[0]
    assert sp["name"] == "skill.read"
    assert sp["attributes"]["span.type"] == "skill"
    assert sp["attributes"]["skill.name"] == "release-helper"
    assert sp["attributes"]["skill.read.via_tool"] == "skill_read"


def test_tool_error_result_marks_status(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("tool.call") as s:
        semconv.tool_call(s, {"name": "read_file", "params": {"path": "x"}}, "Error: no such file", None)
    sp = _spans_written(trace_dir)[0]
    assert sp["status"]["code"] == "ERROR"


def test_tool_result_explicit_success_overrides_error_prefix(trace_dir):
    from codeflow.agent.tools.base import ToolResult
    from codeflow.tracing import semconv

    result = ToolResult("Error: quoted source text", failed=False)
    with trace.span("tool.call") as s:
        semconv.tool_call(s, {"name": "read_file", "params": {"path": "x"}}, result, None)

    sp = _spans_written(trace_dir)[0]
    assert sp["status"] == {"code": "OK", "message": ""}
    assert sp["attributes"]["tool.error"] is None


def test_memory_extract_extractor(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("memory.extract") as s:
        semconv.memory_extract(
            s, {"messages": [{"role": "user", "content": "x"}], "model": "m", "enable_foresight": True}, True, None
        )
    a = _spans_written(trace_dir)[0]["attributes"]
    assert a["span.type"] == "memory"
    assert a["memory.message_count"] == 1
    assert a["memory.annotated"] is True


def test_memory_consolidate_extractor(trace_dir):
    from codeflow.tracing import semconv

    class _S:
        key = "cli:abc"
        last_consolidated = 3
        messages = [1, 2, 3]

    with trace.span("memory.consolidate") as s:
        semconv.memory_consolidate(s, {"session": _S()}, None, None)
    a = _spans_written(trace_dir)[0]["attributes"]
    assert a["memory.session_key"] == "cli:abc"
    assert a["memory.message_count"] == 3


def test_subagent_children_nest(trace_dir):
    from codeflow.tracing import semconv

    with trace.span("subagent.run") as sa:
        semconv.subagent(
            sa, {"task_id": "t1", "task": "do x", "label": "worker", "origin": {"session_key": "cli:p"}}, None, None
        )
        with trace.span("llm.call", {"llm.provider": "p", "llm.model": "m"}) as inner:
            inner_parent = inner._parent
        sa_id = sa.span_id
    spans = _spans_written(trace_dir)
    by = {sp["name"]: sp for sp in spans}
    assert by["subagent.run"]["attributes"]["span.type"] == "subagent"
    assert by["subagent.run"]["attributes"]["subagent.label"] == "worker"
    assert inner_parent == sa_id


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


def test_audit_span_v1_record_shape_is_frozen(trace_dir):
    with trace.span("llm.call", {"llm.provider": "p", "llm.model": "m"}):
        pass
    sp = _spans_written(trace_dir)[0]
    assert set(sp.keys()) == {
        "schemaVersion",
        "traceId",
        "spanId",
        "parentSpanId",
        "name",
        "kind",
        "startTime",
        "endTime",
        "status",
        "events",
        "attributes",
    }
    assert sp["schemaVersion"] == "audit.span.v1"
    assert set(sp["status"].keys()) == {"code", "message"}
    for key in ("span.type", "framework", "session.id", "channel.id", "audit.schema_version"):
        assert key in sp["attributes"]
    assert sp["attributes"]["framework"] == "codeflow"


def test_span_kind_vocabulary_is_frozen():
    from codeflow.tracing import trace as _t

    assert set(_t._KIND_BY_DOMAIN.values()) == {
        "session",
        "model",
        "tool",
        "subagent",
        "skill",
        "memory",
        "plugin",
        "channel",
    }


def test_spine_and_channel_domains_derive_their_kind(trace_dir):
    with trace.span("spine.turn"):
        pass
    with trace.span("channel.deliver"):
        pass
    by = {sp["name"]: sp for sp in _spans_written(trace_dir)}
    assert by["spine.turn"]["attributes"]["span.type"] == "session"
    assert by["channel.deliver"]["attributes"]["span.type"] == "channel"


def test_standard_span_required_attributes(trace_dir):
    from codeflow.tracing import semconv

    class _Resp:
        content = "hi"
        tool_calls: list = []
        usage = None
        finish_reason = "stop"
        reasoning_content = None

    with trace.span("llm.call") as s:
        semconv.llm_call(s, {"self": None, "messages": [], "tools": None, "model": "openrouter/x"}, _Resp(), None)
    with trace.span("tool.call") as s:
        semconv.tool_call(s, {"name": "grep", "params": {}}, "ok", None)
    by = {sp["name"]: sp for sp in _spans_written(trace_dir)}
    assert by["llm.call"]["attributes"]["llm.provider"]
    assert by["llm.call"]["attributes"]["llm.model"]
    assert by["tool.call"]["attributes"]["tool.name"] == "grep"


def test_tracing_disabled_is_passthrough(monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "0")
    calls = {"n": 0}

    @trace.instrument("llm.call")
    async def f(x):
        calls["n"] += 1
        return x * 2

    assert asyncio.run(f(21)) == 42
    assert calls["n"] == 1
    assert trace.current() is None


def test_tracing_internal_failure_never_breaks_host(trace_dir, monkeypatch):
    from codeflow.tracing import spans as _spans

    def _boom(*_a, **_k):
        raise RuntimeError("tracing store down")

    monkeypatch.setattr(_spans, "emit", _boom)

    @trace.instrument("llm.call")
    async def ok(x):
        return x + 1

    @trace.instrument("tool.call")
    async def app_error():
        raise ValueError("APP")

    assert asyncio.run(ok(41)) == 42

    with pytest.raises(ValueError, match="APP"):
        asyncio.run(app_error())
