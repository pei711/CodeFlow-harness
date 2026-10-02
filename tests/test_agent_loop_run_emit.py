"""Tests pinning AgentLoop's spine-native ``run_turn(req, emit)`` output behavior.

run_turn fans the agent's output (streamed token deltas, reasoning, tool events,
notices, media) onto a single ``emit`` and returns a TurnOutcome. These pin that
observable behavior per category, plus origin gating and metadata reconstruction.
Driven against a real AgentLoop with only the LLM provider + sandbox edges faked
(never the output path itself).
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from codeflow.agent.hook import AgentHook, AgentHookContext, CompositeHook, HookDecision
from codeflow.agent.loop import AgentLoop
from codeflow.agent.loop.main import ProviderTurnError
from codeflow.agent.tools.base import Tool
from codeflow.agent.tools.execution import ToolCapability, ToolEffect
from codeflow.config.schema import ToolSearchConfig
from codeflow.providers.base import ErrorClassification, LLMProvider, LLMResponse, StreamDelta, ToolCallRequest
from codeflow.sandbox import SandboxInitError
from codeflow.spine.events import MediaOut as EvMediaOut
from codeflow.spine.events import Notice as EvNotice
from codeflow.spine.events import NoticeKind, ToolPhase
from codeflow.spine.events import Reasoning as EvReasoning
from codeflow.spine.events import StreamDelta as EvStreamDelta
from codeflow.spine.events import Text as EvText
from codeflow.spine.events import ToolEvent as EvToolEvent
from codeflow.spine.message import ChatType, Source
from codeflow.spine.turn import Origin, TurnRequest


class _ShortCircuitHook(AgentHook):
    def __init__(self, content: str, media: list[str] | None = None) -> None:
        self._result = (content, media or [])

    async def before_user_inbound(self, ctx: AgentHookContext) -> HookDecision:
        return HookDecision(short_circuit_result=self._result)


class _AppendHook(AgentHook):
    async def after_send(self, ctx: AgentHookContext) -> HookDecision:
        return HookDecision(modified_content=f"{ctx.outbound_content} [hook]")


class _FakeTool(Tool):
    """Minimal no-sandbox tool so a tool-call turn can dispatch + fire events."""

    @property
    def name(self) -> str:
        return "faketool"

    @property
    def description(self) -> str:
        return "characterization fake tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs) -> str:
        return "tool-ran"


class _FailingTool(_FakeTool):
    async def execute(self, **kwargs) -> str:
        return "Error: protected operation failed"


class _ConcurrentFakeTool(_FakeTool):
    capability = ToolCapability(effect=ToolEffect.READ, concurrency_safe=True)

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def execute(self, **kwargs) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.02)
            return "tool-ran"
        finally:
            self.active -= 1


class _FakeChatProvider:
    """Non-streaming path: ``chat_with_retry`` returns scripted LLMResponses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._i = 0

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        r = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return r

    def get_default_model(self) -> str:
        return "fake/model"


class _FakeStreamProvider:
    """Yields scripted stream chunks via ``chat_stream`` (the streaming path that
    ``run(req, emit)`` takes when emitting StreamDelta tokens)."""

    def __init__(self, chunks: list[StreamDelta]) -> None:
        self._chunks = chunks

    async def chat_stream(self, **kwargs):
        for chunk in self._chunks:
            yield chunk

    def get_default_model(self) -> str:
        return "fake/model"


class _FakeStreamToolProvider:
    """``chat_stream`` yields a fresh scripted chunk-list per call, so a
    tool-call iteration works under run() (call 1 -> tool_call_delta, call 2 ->
    final content). run() always wires on_token_delta, so every turn streams."""

    def __init__(self, scripts: list[list[StreamDelta]]) -> None:
        self._scripts = scripts
        self._i = 0

    async def chat_stream(self, **kwargs):
        script = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        for chunk in script:
            yield chunk

    def get_default_model(self) -> str:
        return "fake/model"


class _EmitCollector:
    """Records every RunnerEvent run() emits, in order."""

    def __init__(self) -> None:
        self.events: list = []

    async def __call__(self, ev) -> None:
        self.events.append(ev)


def _drain() -> list:
    return []


def _req(text: str, *, media=(), origin: Origin = Origin.USER) -> TurnRequest:
    return TurnRequest(
        origin=origin,
        source=Source(channel="cli", chat_id="c", sender_id="u", chat_type=ChatType.DM),
        text=text,
        media=media,
    )


def _stub_edges(loop: AgentLoop) -> None:
    """No-op the sandbox/MCP bring-up so a text-only turn runs without a VM."""

    async def _noop() -> None:
        return None

    loop._start_executor = _noop
    loop._connect_mcp = _noop


async def test_help_slash_returns_command_list_at_outbound_layer(tmp_path):

    loop = AgentLoop(provider=_FakeChatProvider([]), workspace=tmp_path)
    _stub_edges(loop)

    out = await loop._process_message(_req("/help"))

    assert out is not None
    content, _media = out
    assert "CodeFlow commands" in content


async def test_hook_short_circuit_preserves_media_at_outbound_layer(tmp_path):

    loop = AgentLoop(
        provider=_FakeChatProvider([]),
        workspace=tmp_path,
        hooks=CompositeHook([_ShortCircuitHook("short", ["/tmp/x.png"])]),
    )
    _stub_edges(loop)

    out = await loop._process_message(_req("hi"))

    assert out is not None
    content, media = out
    assert media == ["/tmp/x.png"]
    assert content == "short"


async def test_run_streams_then_dissolves_main_response(tmp_path):

    chunks = [
        StreamDelta(content="Hel"),
        StreamDelta(content="lo", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
    ]
    loop = AgentLoop(provider=_FakeStreamProvider(chunks), workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    assert [type(e).__name__ for e in sink.events] == ["StreamDelta", "StreamDelta"]
    assert [e.delta for e in sink.events] == ["Hel", "lo"]
    assert not any(isinstance(e, EvText) for e in sink.events)
    assert outcome.usage.total_tokens == 5
    assert outcome.explicit_reply is True


async def test_run_emits_reasoning_then_stream(tmp_path):
    chunks = [
        StreamDelta(content=None, reasoning_content="think"),
        StreamDelta(content="answer"),
    ]
    loop = AgentLoop(provider=_FakeStreamProvider(chunks), workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    assert isinstance(sink.events[0], EvReasoning) and sink.events[0].content == "think"
    assert any(isinstance(e, EvStreamDelta) and e.delta == "answer" for e in sink.events)
    assert not any(isinstance(e, EvText) for e in sink.events)


async def test_run_tool_call_emits_tool_events_and_notice(tmp_path):

    provider = _FakeStreamToolProvider(
        [
            [
                StreamDelta(
                    content=None,
                    tool_call_delta={
                        "tool_calls": [{"index": 0, "id": "t1", "function": {"name": "faketool", "arguments": "{}"}}]
                    },
                )
            ],
            [StreamDelta(content="done")],
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    loop.tools.register(_FakeTool())
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    tool_events = [e for e in sink.events if isinstance(e, EvToolEvent)]
    start = next(e for e in tool_events if e.phase == ToolPhase.START)
    assert start.tool_call_id == "t1" and start.name == "faketool" and start.arguments == {}
    complete = next(e for e in tool_events if e.phase == ToolPhase.COMPLETE)
    assert complete.tool_call_id == "t1" and complete.result_preview == "tool-ran"
    assert complete.truncated is False
    assert complete.failed is False
    assert outcome.tool_calls == 1
    assert outcome.tool_failures == 0

    assert any(isinstance(e, EvNotice) and e.kind is NoticeKind.TOOL_HINT for e in sink.events)
    assert any(isinstance(e, EvStreamDelta) and e.delta == "done" for e in sink.events)
    assert not any(isinstance(e, EvText) for e in sink.events)


async def test_run_marks_failed_tool_completion(tmp_path):
    provider = _FakeStreamToolProvider(
        [
            [
                StreamDelta(
                    content=None,
                    tool_call_delta={
                        "tool_calls": [{"index": 0, "id": "t1", "function": {"name": "faketool", "arguments": "{}"}}]
                    },
                )
            ],
            [StreamDelta(content="blocked")],
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    loop.tools.register(_FailingTool())
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    complete = next(
        event for event in sink.events if isinstance(event, EvToolEvent) and event.phase == ToolPhase.COMPLETE
    )
    assert complete.failed is True
    assert complete.result_preview.startswith("Error:")
    assert outcome.tool_calls == 1
    assert outcome.tool_failures == 1


async def test_run_executes_concurrency_safe_tool_calls_in_parallel(tmp_path):
    provider = _FakeStreamToolProvider(
        [
            [
                StreamDelta(
                    content=None,
                    tool_call_delta={
                        "tool_calls": [
                            {"index": 0, "id": "t1", "function": {"name": "faketool", "arguments": "{}"}},
                            {"index": 1, "id": "t2", "function": {"name": "faketool", "arguments": "{}"}},
                        ]
                    },
                )
            ],
            [StreamDelta(content="done")],
        ]
    )
    tool = _ConcurrentFakeTool()
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    loop.tools.register(tool)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    assert tool.peak == 2
    assert outcome.tool_calls == 2
    assert outcome.tool_failures == 0
    complete_ids = [
        event.tool_call_id
        for event in sink.events
        if isinstance(event, EvToolEvent) and event.phase == ToolPhase.COMPLETE
    ]
    assert sorted(complete_ids) == ["t1", "t2"]


async def test_run_observes_and_parallelizes_progressive_target_tools(tmp_path):
    provider = _FakeStreamToolProvider(
        [
            [
                StreamDelta(
                    content=None,
                    tool_call_delta={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "outer-1",
                                "function": {
                                    "name": "tool_call",
                                    "arguments": '{"name":"faketool","arguments":{}}',
                                },
                            },
                            {
                                "index": 1,
                                "id": "outer-2",
                                "function": {
                                    "name": "tool_call",
                                    "arguments": '{"name":"faketool","arguments":{}}',
                                },
                            },
                        ]
                    },
                )
            ],
            [StreamDelta(content="done")],
        ]
    )
    tool = _ConcurrentFakeTool()
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path,
        tool_search_config=ToolSearchConfig(enabled=True),
    )
    _stub_edges(loop)
    loop.tools.register(tool)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    assert tool.peak == 2
    assert outcome.tool_calls == 2
    events = [event for event in sink.events if isinstance(event, EvToolEvent)]
    assert {event.name for event in events} == {"tool_call"}
    assert {event.tool_call_id for event in events} == {
        "outer-1",
        "outer-2",
    }
    assert {event.target_name for event in events} == {"faketool"}
    assert {event.target_call_id for event in events} == {
        "outer-1:faketool",
        "outer-2:faketool",
    }


async def test_inject_message_merged_before_next_iteration(tmp_path):

    class _RecordingStreamToolProvider:
        def __init__(self, scripts):
            self._scripts = scripts
            self._i = 0
            self.calls: list[list[dict]] = []

        async def chat_stream(self, **kwargs):
            self.calls.append(list(kwargs.get("messages") or []))
            script = self._scripts[min(self._i, len(self._scripts) - 1)]
            self._i += 1
            for chunk in script:
                yield chunk

        def get_default_model(self) -> str:
            return "fake/model"

    provider = _RecordingStreamToolProvider(
        [
            [
                StreamDelta(
                    content=None,
                    tool_call_delta={
                        "tool_calls": [{"index": 0, "id": "t1", "function": {"name": "faketool", "arguments": "{}"}}]
                    },
                )
            ],
            [StreamDelta(content="done")],
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    loop.tools.register(_FakeTool())

    injects: list[list] = [[], [_req("also check the logs")]]
    n = 0

    def _drain_inject() -> list:
        nonlocal n
        out = injects[n] if n < len(injects) else []
        n += 1
        return out

    await loop.run_turn(_req("start"), _EmitCollector(), _drain_inject, stream=True)

    assert len(provider.calls) >= 2
    second = provider.calls[1]
    assert any(m.get("role") == "user" and "also check the logs" in str(m.get("content", "")) for m in second), (
        f"injected message not merged into the second iteration: {second}"
    )


async def test_run_slash_emits_text_not_streamed(tmp_path):

    loop = AgentLoop(provider=_FakeChatProvider([]), workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("/help"), sink, _drain)

    texts = [e for e in sink.events if isinstance(e, EvText)]
    assert len(texts) == 1 and "CodeFlow commands" in texts[0].content
    assert not any(isinstance(e, EvStreamDelta) for e in sink.events)
    assert outcome.explicit_reply is True


async def test_run_short_circuit_emits_media_before_text(tmp_path):

    loop = AgentLoop(
        provider=_FakeChatProvider([]),
        workspace=tmp_path,
        hooks=CompositeHook([_ShortCircuitHook("short", ["/tmp/x.png"])]),
    )
    _stub_edges(loop)
    sink = _EmitCollector()

    await loop.run_turn(_req("hi"), sink, _drain)

    kinds = [type(e).__name__ for e in sink.events]
    assert kinds == ["MediaOut", "Text"]
    assert sink.events[0].media[0].path == "/tmp/x.png"
    assert sink.events[1].content == "short"


async def test_run_propagates_sandbox_error_not_error_string(tmp_path):

    loop = AgentLoop(provider=_FakeChatProvider([]), workspace=tmp_path)

    async def _boom() -> None:
        raise SandboxInitError("test: sandbox down")

    loop._start_executor = _boom
    sink = _EmitCollector()

    with pytest.raises(SandboxInitError):
        await loop.run_turn(_req("hi"), sink, _drain)

    assert not any(isinstance(e, EvText) for e in sink.events)


async def test_run_propagates_mid_turn_error_not_sorry_text(tmp_path):

    class _BoomStreamProvider:
        async def chat_stream(self, **kwargs):
            raise RuntimeError("mid-turn boom")
            yield

        def get_default_model(self) -> str:
            return "fake/model"

    loop = AgentLoop(provider=_BoomStreamProvider(), workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    with pytest.raises(RuntimeError):
        await loop.run_turn(_req("hi"), sink, _drain)

    assert not any(isinstance(e, EvText) for e in sink.events)


async def test_run_propagates_provider_error_response_as_turn_failure(tmp_path):
    loop = AgentLoop(
        provider=_FakeChatProvider(
            [
                LLMResponse(
                    content="request rejected",
                    finish_reason="error",
                    error_classification=ErrorClassification("invalid_request"),
                )
            ]
        ),
        workspace=tmp_path,
    )
    _stub_edges(loop)
    sink = _EmitCollector()

    with pytest.raises(
        ProviderTurnError,
        match="provider_error:invalid_request",
    ):
        await loop.run_turn(_req("hi"), sink, _drain, stream=False)

    assert sink.events == []


async def test_run_propagates_chat_only_provider_error_in_streaming_path(tmp_path):
    class _ChatOnlyErrorProvider(LLMProvider):
        async def chat(self, **kwargs) -> LLMResponse:
            return LLMResponse(
                content="provider rejected request",
                finish_reason="error",
                error_classification=ErrorClassification("auth"),
            )

        def get_default_model(self) -> str:
            return "fake/model"

    loop = AgentLoop(provider=_ChatOnlyErrorProvider(), workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    with pytest.raises(ProviderTurnError, match="provider_error:auth"):
        await loop.run_turn(_req("hi"), sink, _drain)

    assert sink.events == []


def _message_tool_call(arguments: str) -> StreamDelta:
    return StreamDelta(
        content=None,
        tool_call_delta={
            "tool_calls": [{"index": 0, "id": "m1", "function": {"name": "message", "arguments": arguments}}]
        },
    )


async def test_run_message_tool_text_streams_and_dissolves(tmp_path):

    provider = _FakeStreamToolProvider(
        [
            [_message_tool_call('{"content": "hi via tool"}')],
            [StreamDelta(content="")],
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    assert any(isinstance(e, EvStreamDelta) and e.delta == "hi via tool" for e in sink.events)
    assert not any(isinstance(e, EvText) for e in sink.events)
    assert not any(isinstance(e, EvToolEvent) for e in sink.events)
    assert outcome.explicit_reply is True
    assert outcome.tool_calls == 1
    assert outcome.tool_failures == 0


async def test_run_failed_message_tool_emits_one_failed_completion(tmp_path):
    provider = _FakeStreamToolProvider(
        [
            [_message_tool_call("{}")],
            [StreamDelta(content="I could not send the message.")],
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    tool_events = [event for event in sink.events if isinstance(event, EvToolEvent)]
    assert len(tool_events) == 1
    assert tool_events[0].phase is ToolPhase.COMPLETE
    assert tool_events[0].name == "message"
    assert tool_events[0].failed is True
    assert outcome.tool_calls == 1
    assert outcome.tool_failures == 1


async def test_run_message_tool_media_is_not_dropped(tmp_path):

    provider = _FakeStreamToolProvider(
        [
            [_message_tool_call('{"content": "see this", "media": ["/tmp/pic.png"]}')],
            [StreamDelta(content="")],
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain)

    media = [e for e in sink.events if isinstance(e, EvMediaOut)]
    assert media and media[0].media[0].path == "/tmp/pic.png"
    assert any(isinstance(e, EvStreamDelta) and e.delta == "see this" for e in sink.events)
    assert outcome.explicit_reply is True


async def test_run_stream_false_main_reply_is_one_text(tmp_path):

    provider = _FakeChatProvider([LLMResponse(content="full reply", finish_reason="stop")])
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain, stream=False)

    texts = [e for e in sink.events if isinstance(e, EvText)]
    assert len(texts) == 1 and texts[0].content == "full reply"
    assert not any(isinstance(e, EvStreamDelta) for e in sink.events)
    assert outcome.explicit_reply is True


async def test_run_stream_false_message_tool_emits_text(tmp_path):

    provider = _FakeChatProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCallRequest(id="m1", name="message", arguments={"content": "hi via tool"})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(provider=provider, workspace=tmp_path)
    _stub_edges(loop)
    sink = _EmitCollector()

    outcome = await loop.run_turn(_req("hi"), sink, _drain, stream=False)

    assert any(isinstance(e, EvText) and e.content == "hi via tool" for e in sink.events)
    assert not any(isinstance(e, EvStreamDelta) for e in sink.events)
    assert not any(isinstance(e, EvToolEvent) for e in sink.events)
    assert outcome.explicit_reply is True
    assert outcome.tool_calls == 1
    assert outcome.tool_failures == 0


def _hook_loop(tmp_path):
    """An AgentLoop whose Agent Hook short-circuits with 'hook-fired' when
    the user-inbound hook runs; otherwise the (streaming) LLM reply 'llm' streams.
    So the hook firing vs being skipped is observable from the output."""

    loop = AgentLoop(
        provider=_FakeStreamProvider([StreamDelta(content="llm")]),
        workspace=tmp_path,
        hooks=CompositeHook([_ShortCircuitHook("hook-fired")]),
    )
    _stub_edges(loop)
    return loop


async def test_run_turn_user_origin_fires_user_inbound_hook(tmp_path):
    sink = _EmitCollector()
    await _hook_loop(tmp_path).run_turn(_req("hi", origin=Origin.USER), sink, _drain)

    assert any(isinstance(e, EvText) and e.content == "hook-fired" for e in sink.events)
    assert not any(isinstance(e, EvStreamDelta) for e in sink.events)


async def test_run_turn_cron_origin_skips_user_inbound_hook(tmp_path):
    sink = _EmitCollector()
    await _hook_loop(tmp_path).run_turn(_req("tick", origin=Origin.CRON), sink, _drain)
    assert any(isinstance(e, EvStreamDelta) and e.delta == "llm" for e in sink.events)
    assert not any(isinstance(e, EvText) and e.content == "hook-fired" for e in sink.events)


async def test_run_turn_subagent_origin_suppresses_user_inbound_hook(tmp_path):
    sink = _EmitCollector()
    await _hook_loop(tmp_path).run_turn(_req("result", origin=Origin.SUBAGENT), sink, _drain)
    assert any(isinstance(e, EvStreamDelta) and e.delta == "llm" for e in sink.events)
    assert not any(isinstance(e, EvText) and e.content == "hook-fired" for e in sink.events)


async def _process_via_chat(loop, msg):

    return await loop._process_message(msg)


async def test_process_message_origin_none_plain_fires_hook(tmp_path):

    loop = AgentLoop(
        provider=_FakeChatProvider([LLMResponse(content="llm", finish_reason="stop")]),
        workspace=tmp_path,
        hooks=CompositeHook([_ShortCircuitHook("hook-fired")]),
    )
    _stub_edges(loop)
    out = await _process_via_chat(loop, _req("hi"))
    assert out is not None
    content, _media = out
    assert content == "hook-fired"


async def test_cron_origin_runs_generic_after_send_hook(tmp_path):
    loop = AgentLoop(
        provider=_FakeChatProvider([LLMResponse(content="done", finish_reason="stop")]),
        workspace=tmp_path,
        hooks=CompositeHook([_AppendHook()]),
    )
    _stub_edges(loop)
    sink = _EmitCollector()

    await loop.run_turn(_req("tick", origin=Origin.CRON), sink, _drain, stream=False)

    assert any(isinstance(event, EvText) and event.content == "done [hook]" for event in sink.events)


async def test_subagent_after_send_hook_matches_persisted_summary(tmp_path):
    loop = AgentLoop(
        provider=_FakeChatProvider([LLMResponse(content="done", finish_reason="stop")]),
        workspace=tmp_path,
        hooks=CompositeHook([_AppendHook()]),
    )
    _stub_edges(loop)
    sink = _EmitCollector()

    await loop.run_turn(_req("result", origin=Origin.SUBAGENT), sink, _drain, stream=False)

    assert any(isinstance(event, EvText) and event.content == "done [hook]" for event in sink.events)
    messages = loop.sessions.get_or_create("cli:c").messages
    assert [(message["role"], message["content"]) for message in messages] == [("assistant", "done [hook]")]


def test_agent_loop_has_no_sentinel_callback_parameters():
    params = inspect.signature(AgentLoop).parameters
    assert {"response_modifier", "on_user_inbound", "decision_consumer"}.isdisjoint(params)


async def test_run_turn_reconstructs_metadata_from_source_extras(tmp_path):

    loop = AgentLoop(
        provider=_FakeChatProvider([LLMResponse(content="ok", finish_reason="stop", tool_calls=[])]),
        workspace=tmp_path,
    )
    _stub_edges(loop)
    seen: dict = {}
    real = loop._set_tool_context

    def _spy(channel, chat_id, message_id=None, session_key=None):
        seen["message_id"] = message_id
        return real(channel, chat_id, message_id, session_key=session_key)

    loop._set_tool_context = _spy

    req = TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="tg",
            chat_id="c",
            sender_id="u",
            chat_type=ChatType.DM,
            extras={"message_id": "m1"},
        ),
        text="hi",
    )
    await loop.run_turn(req, _EmitCollector(), _drain, stream=False)
    assert seen.get("message_id") == "m1"


async def test_run_turn_empty_extras_reconstructs_empty_metadata(tmp_path):

    loop = AgentLoop(
        provider=_FakeChatProvider([LLMResponse(content="ok", finish_reason="stop", tool_calls=[])]),
        workspace=tmp_path,
    )
    _stub_edges(loop)
    seen: dict = {"message_id": "original"}

    def _spy(channel, chat_id, message_id=None, session_key=None):
        seen["message_id"] = message_id

    loop._set_tool_context = _spy
    await loop.run_turn(_req("hi"), _EmitCollector(), _drain, stream=False)
    assert seen["message_id"] is None
