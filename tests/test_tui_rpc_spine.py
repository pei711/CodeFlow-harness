import asyncio
from dataclasses import replace

import pytest

from codeflow.agent.tools.message import MessageTool
from codeflow.spine import (
    ChatType,
    MediaOut,
    Notice,
    NoticeKind,
    Origin,
    Reasoning,
    Source,
    StreamDelta,
    Text,
    ToolEvent,
    ToolPhase,
    TurnOutcome,
    TurnRequest,
    TurnRunner,
    Usage,
)
from codeflow.spine.delivery import Outlet, SupportsStreaming
from codeflow.spine.message import Media
from codeflow.tui_rpc.spine import (
    TuiOutlet,
    TuiTurnRunner,
    build_tui,
)


def _src(channel="tui", chat_id="c1") -> Source:
    return Source(channel=channel, chat_id=chat_id, sender_id="user", chat_type=ChatType.DM)


class FakeEmitter:
    """Records (session_key, event) — stands in for SubscriptionEmitter."""

    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, session_key: str, event: dict) -> None:
        self.emitted.append((session_key, event))

    def types(self) -> list[str]:
        return [e["type"] for _k, e in self.emitted]


class _RunTurnLoop:
    """Fake AgentLoop whose run_turn emits scripted spine events and fills the
    caller's usage_sink — stands in for the native run_turn (stream=True). For a
    CRON turn the runner passes stream=False + text_sink; ``reply_text`` is written
    into text_sink so the read-back path can be exercised."""

    def __init__(self, events=(), usage=None, *, tools=None, reply_text=None) -> None:
        self._events = list(events)
        self._usage = usage
        self._reply_text = reply_text
        self.tools = tools if tools is not None else {}
        self.last_stream = None

    async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None) -> TurnOutcome:
        self.last_stream = stream
        for ev in self._events:
            await emit(ev)
        if usage_sink is not None and self._usage:
            usage_sink.update(self._usage)
        if text_sink is not None and self._reply_text is not None:
            text_sink["text"] = self._reply_text
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)


def _collect():
    events: list = []

    async def emit(e):
        events.append(e)

    return events, emit


def test_pieces_satisfy_their_spine_protocols():
    assert isinstance(TuiTurnRunner(object(), FakeEmitter(), {}, {}, {}), TurnRunner)
    outlet = TuiOutlet("tui", FakeEmitter())
    assert isinstance(outlet, Outlet)
    assert isinstance(outlet, SupportsStreaming)
    assert outlet.capabilities.streaming is True


async def test_runner_drives_run_turn_and_stashes_rich_usage():
    rich = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8, "cost_usd": 0.01, "context_used": 42}
    loop = _RunTurnLoop(events=[StreamDelta(delta="he"), StreamDelta(delta="llo")], usage=rich)
    usages: dict[int, dict] = {}
    req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
    runner = TuiTurnRunner(
        loop,
        FakeEmitter(),
        usages,
        {id(req): "T1"},
        {},
        submission_ids={id(req): "submission-1"},
    )
    events, emit = _collect()

    outcome = await runner.run(req, emit, lambda: [])

    assert [e.delta for e in events] == ["he", "llo"]

    assert usages[id(req)] == rich
    assert outcome.explicit_reply is True


async def test_runner_emits_eve22_synthetic_tool_complete_when_message_tool_fired():
    message_tool = MessageTool()
    loop = _RunTurnLoop(tools={"message": message_tool})

    async def _run_turn(req, emit, drain, *, stream, usage_sink=None):

        message_tool._turn.set(replace(message_tool._cur(), sent=True))
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    loop.run_turn = _run_turn
    req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
    runner = TuiTurnRunner(
        loop,
        FakeEmitter(),
        {},
        {id(req): "T7"},
        {},
        submission_ids={id(req): "submission-7"},
    )
    events, emit = _collect()

    await runner.run(req, emit, lambda: [])

    assert len(events) == 1 and isinstance(events[0], ToolEvent)
    assert events[0].phase is ToolPhase.COMPLETE and events[0].tool_call_id == "msg-T7"


async def test_runner_no_synthetic_when_message_tool_did_not_fire():
    loop = _RunTurnLoop(tools={"message": MessageTool()})
    req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
    runner = TuiTurnRunner(
        loop,
        FakeEmitter(),
        {},
        {id(req): "T7"},
        {},
        submission_ids={id(req): "submission-7"},
    )
    events, emit = _collect()
    await runner.run(req, emit, lambda: [])
    assert events == []


async def test_runner_cron_captures_reply_non_streaming():

    loop = _RunTurnLoop(reply_text="reminder fired")
    readback: dict[str, str] = {}
    runner = TuiTurnRunner(loop, FakeEmitter(), {}, {}, readback)
    req = TurnRequest(origin=Origin.CRON, source=_src(chat_id="direct"), text="[cron]", conversation="cron:job1")
    events, emit = _collect()

    await runner.run(req, emit, lambda: [])

    assert loop.last_stream is False
    assert readback["cron:job1"] == "reminder fired"


async def test_runner_delivers_subagent_reply_without_user_turn_correlation():
    loop = _RunTurnLoop(
        events=[StreamDelta(delta="must not enter the user stream")],
        reply_text="delegated result",
    )
    req = TurnRequest(
        origin=Origin.SUBAGENT,
        source=_src(),
        text="reinjection",
        conversation="tui:c1",
    )
    emitter = FakeEmitter()
    runner = TuiTurnRunner(
        loop,
        emitter,
        {},
        {id(req): "must-not-bind"},
        {},
        submission_ids={id(req): "must-not-bind"},
    )
    events, emit = _collect()

    await runner.run(req, emit, lambda: [])

    assert loop.last_stream is False
    assert events == []
    assert emitter.emitted == [
        (
            "tui:c1",
            {
                "type": "subagent.delivered",
                "payload": {"text": "delegated result"},
            },
        )
    ]


async def test_outlet_deliver_reasoning_to_thinking_delta():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.deliver(Reasoning(content="thinking", conversation_id="tui:c1"))
    assert emitter.emitted == [("tui:c1", {"type": "thinking.delta", "payload": {"text": "thinking"}})]


async def test_outlet_deliver_tool_event_to_tool_start_and_complete():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.deliver(
        ToolEvent(
            phase=ToolPhase.START, tool_call_id="t1", name="shell", arguments={"cmd": "ls"}, conversation_id="tui:c1"
        )
    )
    await outlet.deliver(
        ToolEvent(
            phase=ToolPhase.COMPLETE, tool_call_id="t1", result_preview="ok", truncated=False, conversation_id="tui:c1"
        )
    )
    assert emitter.emitted == [
        (
            "tui:c1",
            {"type": "tool.start", "payload": {"tool_call_id": "t1", "name": "shell", "arguments": {"cmd": "ls"}}},
        ),
        (
            "tui:c1",
            {"type": "tool.complete", "payload": {"tool_call_id": "t1", "result_preview": "ok", "truncated": False}},
        ),
    ]


async def test_outlet_marks_failed_tool_completion():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)

    await outlet.deliver(
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="t1",
            result_preview="Error: permission denied",
            failed=True,
            conversation_id="tui:c1",
        )
    )

    assert emitter.emitted == [
        (
            "tui:c1",
            {
                "type": "tool.complete",
                "payload": {
                    "tool_call_id": "t1",
                    "result_preview": "Error: permission denied",
                    "truncated": False,
                    "failed": True,
                },
            },
        )
    ]


async def test_outlet_deliver_text_to_token_delta():

    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.deliver(Text(content="please clarify", conversation_id="tui:c1"))
    assert emitter.emitted == [("tui:c1", {"type": "token.delta", "payload": {"text": "please clarify"}})]


async def test_outlet_deliver_eats_notice_and_media():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.deliver(Notice(kind=NoticeKind.PROGRESS, detail="working", conversation_id="tui:c1"))
    await outlet.deliver(
        MediaOut(media=(Media(path="/tmp/x.png", mime="image/png", kind="image"),), conversation_id="tui:c1")
    )
    assert emitter.emitted == []


async def test_outlet_emits_token_delta_on_a_chunk():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.send_stream_chunk("c1", "tui:c1", "hi", done=False)
    assert emitter.emitted == [("tui:c1", {"type": "token.delta", "payload": {"text": "hi"}})]


async def test_outlet_done_chunk_is_a_noop():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.send_stream_chunk("c1", "tui:c1", "", done=True)
    assert emitter.emitted == []


async def test_outlet_eats_empty_delta():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.send_stream_chunk("c1", "tui:c1", "", done=False)
    assert emitter.emitted == []


async def test_outlet_emit_complete_and_error_shapes():
    emitter = FakeEmitter()
    outlet = TuiOutlet("tui", emitter)
    await outlet.emit_complete("tui:c1", "submission-1", "t1", {"total_tokens": 7})
    await outlet.emit_error("tui:c1", "submission-1", "t1", -32099, "turn_failed", "internal")
    assert emitter.emitted == [
        (
            "tui:c1",
            {
                "type": "message.complete",
                "payload": {
                    "submission_id": "submission-1",
                    "turn_id": "t1",
                    "usage": {"total_tokens": 7},
                },
            },
        ),
        (
            "tui:c1",
            {
                "type": "error",
                "payload": {
                    "code": -32099,
                    "message": "turn_failed",
                    "reason": "internal",
                    "submission_id": "submission-1",
                    "turn_id": "t1",
                },
            },
        ),
    ]


async def test_build_tui_defaults_to_single_slot_pools():
    scheduler, _hub, _turn_ids, _submission_ids, teardown = build_tui(_RunTurnLoop(events=[]), FakeEmitter())
    try:
        assert scheduler._pools._user._value == 1
        assert scheduler._pools._system._value == 1
    finally:
        await teardown()


async def test_build_tui_honors_configured_pool_sizes():
    scheduler, _hub, _turn_ids, _submission_ids, teardown = build_tui(
        _RunTurnLoop(events=[]),
        FakeEmitter(),
        user_pool=6,
        system_pool=4,
    )
    try:
        assert scheduler._pools._user._value == 6
        assert scheduler._pools._system._value == 4
    finally:
        await teardown()


@pytest.mark.parametrize("system_state", ["pending", "running"])
async def test_turn_send_rejects_instead_of_queueing_behind_system_turn(system_state):
    from codeflow.tui_rpc.errors import TurnInProgressError
    from codeflow.tui_rpc.methods import turn as turn_module
    from codeflow.tui_rpc.methods.turn import turn_send

    system_started = asyncio.Event()
    release_system = asyncio.Event()
    user_started = asyncio.Event()

    class _BlockedSystemLoop:
        tools = {}

        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None) -> TurnOutcome:
            if req.origin is Origin.SUBAGENT:
                system_started.set()
                await release_system.wait()
            else:
                user_started.set()
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    session_key = f"tui:admission:{system_state}"
    emitter = FakeEmitter()
    turn_module.clear_active(session_key)
    scheduler, _hub, turn_ids, submission_ids, teardown = build_tui(_BlockedSystemLoop(), emitter)
    system_handle = scheduler.submit(
        TurnRequest(
            origin=Origin.SUBAGENT,
            source=_src(),
            text="long-running reinjection",
            conversation=session_key,
        )
    )
    try:
        if system_state == "running":
            await system_started.wait()

        with pytest.raises(TurnInProgressError) as excinfo:
            await turn_send(
                {
                    "session_key": session_key,
                    "content": "user prompt",
                    "submission_id": "submission-user",
                },
                emitter=emitter,
                scheduler=scheduler,
                turn_ids=turn_ids,
                submission_ids=submission_ids,
            )

        assert excinfo.value.CODE == -32003
        assert turn_module.is_turn_active(session_key) is False
        assert turn_ids == {}
        assert submission_ids == {}
        assert emitter.types() == []
        assert user_started.is_set() is False
    finally:
        release_system.set()
        await system_handle.result()
        await teardown()
        turn_module.clear_active(session_key)


async def test_turn_send_acks_before_runtime_ready_but_runner_waits():
    from codeflow.tui_rpc.methods import turn as turn_module
    from codeflow.tui_rpc.methods.turn import turn_send

    ready = asyncio.Event()
    readiness_entered = asyncio.Event()
    loop_started = asyncio.Event()

    class _Loop:
        tools = {}

        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None):
            loop_started.set()
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    async def _await_runtime_ready():
        readiness_entered.set()
        await ready.wait()

    session_key = "tui:runtime-ready"
    emitter = FakeEmitter()
    turn_module.clear_active(session_key)
    scheduler, _hub, turn_ids, submission_ids, teardown = build_tui(
        _Loop(),
        emitter,
        on_turn_end=turn_module.clear_active,
        await_runtime_ready=_await_runtime_ready,
    )
    try:
        result = await turn_send(
            {
                "session_key": session_key,
                "content": "wait for memory",
                "submission_id": "submission-ready",
            },
            emitter=emitter,
            scheduler=scheduler,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )
        handle = turn_module._active_turns[session_key]
        await readiness_entered.wait()

        assert result["accepted"] is True
        assert emitter.types() == ["message.start"]
        assert loop_started.is_set() is False

        ready.set()
        await handle.result()

        assert loop_started.is_set() is True
        assert emitter.types() == ["message.start", "message.complete"]
    finally:
        ready.set()
        await teardown()
        turn_module.clear_active(session_key)


async def test_runtime_start_failure_emits_typed_error_without_running_loop():
    from codeflow.tui_rpc.errors import InternalError
    from codeflow.tui_rpc.methods import turn as turn_module
    from codeflow.tui_rpc.methods.turn import turn_send

    class _Loop:
        tools = {}

        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None):
            raise AssertionError("failed runtime must not enter AgentLoop")

    async def _await_runtime_ready():
        raise InternalError(
            detail="memory backend failed to start",
            data={"reason": "memory_backend_start_failed"},
        )

    session_key = "tui:runtime-failed"
    emitter = FakeEmitter()
    turn_module.clear_active(session_key)
    scheduler, _hub, turn_ids, submission_ids, teardown = build_tui(
        _Loop(),
        emitter,
        on_turn_end=turn_module.clear_active,
        await_runtime_ready=_await_runtime_ready,
    )
    try:
        result = await turn_send(
            {
                "session_key": session_key,
                "content": "must fail closed",
                "submission_id": "submission-failed",
            },
            emitter=emitter,
            scheduler=scheduler,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )
        handle = turn_module._active_turns[session_key]
        await handle.result()

        assert result["accepted"] is True
        assert turn_module.is_turn_active(session_key) is False
        assert emitter.types() == ["message.start", "error"]
        assert emitter.emitted[-1][1]["payload"] == {
            "code": -32603,
            "message": "internal_error",
            "reason": "internal",
            "submission_id": "submission-failed",
            "turn_id": result["turn_id"],
        }
    finally:
        await teardown()
        turn_module.clear_active(session_key)


async def test_runtime_readiness_also_blocks_cron_turns():
    ready = asyncio.Event()
    loop_started = asyncio.Event()

    class _Loop:
        tools = {}

        async def run_turn(self, req, emit, drain, *, stream, text_sink=None):
            loop_started.set()
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    async def _await_runtime_ready():
        await ready.wait()

    scheduler, _hub, _turn_ids, _submission_ids, teardown = build_tui(
        _Loop(),
        FakeEmitter(),
        await_runtime_ready=_await_runtime_ready,
    )
    try:
        handle = scheduler.submit(
            TurnRequest(
                origin=Origin.CRON,
                source=_src(),
                text="scheduled work",
                conversation="cron:runtime-ready",
            )
        )
        await asyncio.sleep(0)
        assert loop_started.is_set() is False

        ready.set()
        await handle.result()
        assert loop_started.is_set() is True
    finally:
        ready.set()
        await teardown()


@pytest.mark.parametrize("system_fails", [False, True])
async def test_preceding_subagent_cannot_emit_settle_or_clear_queued_user_turn(system_fails):
    from codeflow.tui_rpc.methods import turn as turn_module
    from codeflow.tui_rpc.methods.turn import turn_send

    system_started = asyncio.Event()
    release_system = asyncio.Event()
    user_started = asyncio.Event()
    release_user = asyncio.Event()

    class _SequencedLoop:
        tools = {}

        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None) -> TurnOutcome:
            if req.origin is Origin.SUBAGENT:
                system_started.set()
                await release_system.wait()
                await emit(Reasoning(content="system reasoning"))
                await emit(ToolEvent(phase=ToolPhase.START, tool_call_id="system-tool", name="shell"))
                await emit(StreamDelta(delta="system answer"))
                await emit(ToolEvent(phase=ToolPhase.COMPLETE, tool_call_id="system-tool"))
                if text_sink is not None:
                    text_sink["text"] = "system answer"
                if system_fails:
                    raise RuntimeError("subagent reinjection failed")
            else:
                user_started.set()
                await release_user.wait()
                if usage_sink is not None:
                    usage_sink.update({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    session_key = f"tui:correlation:{system_fails}"
    emitter = FakeEmitter()
    turn_module.clear_active(session_key)
    scheduler, _hub, turn_ids, submission_ids, teardown = build_tui(
        _SequencedLoop(),
        emitter,
        on_turn_end=turn_module.clear_active,
    )

    class _UncheckedScheduler:
        def submit(self, req):
            return scheduler.submit(req)

        def has_pending_or_running(self, conversation_id):
            return False

    try:
        system_handle = scheduler.submit(
            TurnRequest(
                origin=Origin.SUBAGENT,
                source=_src(),
                text="reinjection",
                conversation=session_key,
            )
        )
        await system_started.wait()

        result = await turn_send(
            {
                "session_key": session_key,
                "content": "user prompt",
                "submission_id": "submission-user",
            },
            emitter=emitter,
            scheduler=_UncheckedScheduler(),
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )
        user_handle = turn_module._active_turns[session_key]
        assert emitter.types() == []

        release_system.set()
        await system_handle.result()
        await user_started.wait()

        assert turn_module.is_turn_active(session_key) is True
        assert result["turn_id"] in turn_ids.values()
        assert list(submission_ids.values()) == ["submission-user"]
        expected_before_complete = ["message.start"]
        if not system_fails:
            expected_before_complete.insert(0, "subagent.delivered")
            assert emitter.emitted[0][1]["payload"] == {"text": "system answer"}
        assert emitter.types() == expected_before_complete
        assert emitter.emitted[-1][1]["payload"] == {
            "submission_id": "submission-user",
            "turn_id": result["turn_id"],
        }

        release_user.set()
        await user_handle.result()

        assert turn_module.is_turn_active(session_key) is False
        assert emitter.types() == [*expected_before_complete, "message.complete"]
        assert emitter.emitted[-1][1]["payload"] == {
            "submission_id": "submission-user",
            "turn_id": result["turn_id"],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    finally:
        release_system.set()
        release_user.set()
        await teardown()
        turn_module.clear_active(session_key)


async def test_streaming_turn_emits_token_deltas_then_message_complete():
    emitter = FakeEmitter()
    loop = _RunTurnLoop(
        events=[StreamDelta(delta="a"), StreamDelta(delta="b")],
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    )
    scheduler, hub, turn_ids, submission_ids, teardown = build_tui(loop, emitter)
    try:
        req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
        turn_ids[id(req)] = "t1"
        submission_ids[id(req)] = "submission-1"
        handle = scheduler.submit(req)
        await handle.result()
    finally:
        await teardown()

    assert emitter.types() == ["message.start", "token.delta", "token.delta", "message.complete"]
    last_key, last = emitter.emitted[-1]
    assert last_key == "tui:c1"
    assert last["payload"] == {
        "submission_id": "submission-1",
        "turn_id": "t1",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


async def test_interleaved_events_keep_emit_order_through_one_queue():

    emitter = FakeEmitter()
    loop = _RunTurnLoop(
        events=[
            Reasoning(content="thinking"),
            ToolEvent(phase=ToolPhase.START, tool_call_id="t1", name="shell", arguments={}),
            StreamDelta(delta="answer"),
            ToolEvent(phase=ToolPhase.COMPLETE, tool_call_id="t1", result_preview="ok", truncated=False),
        ]
    )
    scheduler, hub, turn_ids, submission_ids, teardown = build_tui(loop, emitter)
    try:
        req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
        turn_ids[id(req)] = "t1"
        submission_ids[id(req)] = "submission-1"
        handle = scheduler.submit(req)
        await handle.result()
    finally:
        await teardown()

    assert emitter.types() == [
        "message.start",
        "thinking.delta",
        "tool.start",
        "token.delta",
        "tool.complete",
        "message.complete",
    ]


async def test_non_streamed_text_reaches_the_wire_as_a_token_delta():

    emitter = FakeEmitter()
    loop = _RunTurnLoop(events=[Text(content="which file?")])
    scheduler, hub, turn_ids, submission_ids, teardown = build_tui(loop, emitter)
    try:
        req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
        turn_ids[id(req)] = "t1"
        submission_ids[id(req)] = "submission-1"
        handle = scheduler.submit(req)
        await handle.result()
    finally:
        await teardown()

    assert emitter.types() == ["message.start", "token.delta", "message.complete"]
    assert emitter.emitted[1][1]["payload"]["text"] == "which file?"


async def test_empty_stream_turn_still_emits_message_complete():

    emitter = FakeEmitter()
    scheduler, hub, turn_ids, submission_ids, teardown = build_tui(_RunTurnLoop(events=[]), emitter)
    try:
        req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
        turn_ids[id(req)] = "t9"
        submission_ids[id(req)] = "submission-9"
        handle = scheduler.submit(req)
        await handle.result()
    finally:
        await teardown()

    assert emitter.types() == ["message.start", "message.complete"]
    assert emitter.emitted[-1][1]["payload"]["turn_id"] == "t9"


async def test_cron_turn_deliverables_key_to_dead_conversation_not_user_session():

    emitter = FakeEmitter()
    loop = _RunTurnLoop(events=[Text(content="reminder")], reply_text="reminder")
    readback: dict[str, str] = {}
    scheduler, hub, _turn_ids, _submission_ids, teardown = build_tui(loop, emitter, readback_texts=readback)
    try:
        handle = scheduler.submit(
            TurnRequest(
                origin=Origin.CRON,
                source=_src(chat_id="direct"),
                text="[cron]",
                conversation="cron:job1",
            )
        )
        await handle.result()
        await hub.wait_idle("tui")
    finally:
        await teardown()

    assert emitter.emitted == []
    assert readback["cron:job1"] == "reminder"


async def test_failed_turn_emits_error():
    class _BoomLoop:
        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None) -> TurnOutcome:
            raise RuntimeError("boom")

    emitter = FakeEmitter()
    scheduler, hub, turn_ids, submission_ids, teardown = build_tui(_BoomLoop(), emitter)
    try:
        req = TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1")
        turn_ids[id(req)] = "t1"
        submission_ids[id(req)] = "submission-1"
        handle = scheduler.submit(req)
        await handle.result()
    finally:
        await teardown()

    assert emitter.types() == ["message.start", "error"]
    payload = emitter.emitted[-1][1]["payload"]
    assert payload["reason"] == "internal"
    assert payload["submission_id"] == "submission-1"
    assert payload["turn_id"] == "t1"


async def test_cancelled_turn_does_not_emit_error():

    import asyncio

    started = asyncio.Event()

    class _HangLoop:
        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None) -> TurnOutcome:
            started.set()
            await asyncio.sleep(3600)
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)

    emitter = FakeEmitter()
    scheduler, hub, _turn_ids, _submission_ids, teardown = build_tui(_HangLoop(), emitter)
    try:
        handle = scheduler.submit(TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="tui:c1"))
        await started.wait()
        handle.cancel()
        await handle.result()
    finally:
        await teardown()

    assert emitter.emitted == []
