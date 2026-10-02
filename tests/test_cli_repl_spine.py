import asyncio
from contextlib import nullcontext

from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.cli._repl_spine import (
    CliOutlet,
    build_repl,
    make_hub_sink,
    run_repl_loop,
)
from codeflow.spine import (
    ChatType,
    Notice,
    NoticeKind,
    Origin,
    Source,
    Text,
    ToolEvent,
    ToolPhase,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    TurnRunner,
    TurnStarted,
    Usage,
)
from codeflow.spine.delivery import Outlet
from codeflow.spine.events import Reasoning


def _src(channel="cli", chat_id="c1") -> Source:
    return Source(channel=channel, chat_id=chat_id, sender_id="user", chat_type=ChatType.DM)


class FakeAgentLoop:
    def __init__(self, reply="hello") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def run_turn(self, req, emit, drain, *, stream) -> TurnOutcome:
        self.calls.append(
            {
                "text": req.text,
                "stream": stream,
                "conversation": req.conversation,
            }
        )

        await emit(Text(content=self.reply, source=req.source))
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)


def test_pieces_satisfy_their_spine_protocols():
    assert isinstance(CliOutlet("cli", lambda s: None), Outlet)
    assert isinstance(AgentTurnRunner(object(), stream=False), TurnRunner)


def _collect():
    events: list = []

    async def emit(e):
        events.append(e)

    return events, emit


async def test_runner_delegates_to_run_turn_with_stream_false():
    loop = FakeAgentLoop("hi there")
    runner = AgentTurnRunner(loop, stream=False)
    src = _src()
    req = TurnRequest(origin=Origin.USER, source=src, text="hi", conversation="cli:c1")
    events, emit = _collect()
    outcome = await runner.run(req, emit, lambda: [])

    assert loop.calls == [{"text": "hi", "stream": False, "conversation": "cli:c1"}]
    assert len(events) == 1 and isinstance(events[0], Text)
    assert events[0].content == "hi there" and events[0].source is src
    assert outcome.explicit_reply is True


async def test_runner_stream_flag_is_forwarded():
    loop = FakeAgentLoop()
    runner = AgentTurnRunner(loop, stream=True)
    events, emit = _collect()
    await runner.run(TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="cli:c1"), emit, lambda: [])
    assert loop.calls[0]["stream"] is True


async def test_cli_outlet_renders_text():
    rendered: list[str] = []
    outlet = CliOutlet("cli", rendered.append)
    await outlet.deliver(Text(content="rendered me"))
    assert rendered == ["rendered me"]


async def test_cli_outlet_eats_non_text():
    rendered: list[str] = []
    outlet = CliOutlet("cli", rendered.append)

    await outlet.deliver(Notice(kind=NoticeKind.PROGRESS))
    await outlet.deliver(Notice(kind=NoticeKind.TOOL_HINT))
    assert rendered == []


async def test_cli_outlet_surfaces_failed_tool_completion():
    errors: list[str] = []
    outlet = CliOutlet("cli", lambda _text: None, render_error=errors.append)

    await outlet.deliver(
        ToolEvent(
            phase=ToolPhase.COMPLETE,
            tool_call_id="tool-1",
            result_preview="Error: permission denied",
            failed=True,
        )
    )

    assert errors == ["Tool failed: Error: permission denied"]


def _notice_outlet(*, send_progress: bool, send_tool_hints: bool):
    notices: list[str] = []
    outlet = CliOutlet(
        "cli",
        lambda t: None,
        render_notice=notices.append,
        send_progress=send_progress,
        send_tool_hints=send_tool_hints,
    )
    return notices, outlet


async def test_cli_outlet_renders_progress_when_send_progress_on():
    notices, outlet = _notice_outlet(send_progress=True, send_tool_hints=False)
    await outlet.deliver(Notice(kind=NoticeKind.PROGRESS, detail="thinking"))
    assert notices == ["thinking"]


async def test_cli_outlet_default_config_does_not_leak_tool_hints():

    notices, outlet = _notice_outlet(send_progress=True, send_tool_hints=False)
    await outlet.deliver(Notice(kind=NoticeKind.PROGRESS, detail="thinking"))
    await outlet.deliver(Notice(kind=NoticeKind.TOOL_HINT, detail='read_file("x")'))
    assert notices == ["thinking"]


async def test_cli_outlet_renders_tool_hint_when_send_tool_hints_on():
    notices, outlet = _notice_outlet(send_progress=False, send_tool_hints=True)
    await outlet.deliver(Notice(kind=NoticeKind.PROGRESS, detail="thinking"))
    await outlet.deliver(Notice(kind=NoticeKind.TOOL_HINT, detail='read_file("x")'))
    assert notices == ['read_file("x")']


async def test_cli_outlet_renders_reasoning_as_progress():
    notices, outlet = _notice_outlet(send_progress=True, send_tool_hints=False)
    await outlet.deliver(Reasoning(content="searching the web..."))
    assert notices == ["searching the web..."]


async def test_cli_outlet_eats_reasoning_without_progress():

    rendered: list[str] = []
    await CliOutlet("cli", rendered.append).deliver(Reasoning(content="searching..."))
    assert rendered == []

    notices, outlet = _notice_outlet(send_progress=False, send_tool_hints=False)
    await outlet.deliver(Reasoning(content="searching..."))
    assert notices == []


class FakeHub:
    def __init__(self) -> None:
        self.dispatched: list = []

    async def dispatch(self, event) -> None:
        self.dispatched.append(event)


async def test_sink_routes_deliverables_and_drops_lifecycle():
    hub = FakeHub()
    sink = make_hub_sink(hub)
    await sink(Text(content="t", source=_src()))
    await sink(TurnStarted())
    await sink(TurnFailed(error="e", cancelled=False))
    await sink(TurnEnded(usage=Usage(0, 0, 0), latency_ms=1.0, explicit_reply=False))
    assert len(hub.dispatched) == 1
    assert isinstance(hub.dispatched[0], Text)


class _EchoLoop:
    async def run_turn(self, req, emit, drain, *, stream) -> TurnOutcome:

        await emit(Text(content=f"reply<{req.text}>", source=req.source))
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)


def _wire_real_spine():
    """The real assembly via build_repl (Scheduler + DeliveryHub + CliOutlet); only
    the agent loop is fake. The render callback appends to a shared event log
    alongside the prompt marker, so this exercises build_repl's wiring too."""
    events: list[str] = []
    scheduler, hub, teardown = build_repl(_EchoLoop(), "cli", lambda t: events.append(f"render:{t}"))
    return events, hub, scheduler, teardown


async def test_build_repl_defaults_to_single_slot_pools():
    scheduler, _hub, teardown = build_repl(_EchoLoop(), "cli", lambda t: None)
    try:
        assert scheduler._pools._user._value == 1
        assert scheduler._pools._system._value == 1
    finally:
        await teardown()


async def test_build_repl_honors_configured_pool_sizes():
    scheduler, _hub, teardown = build_repl(_EchoLoop(), "cli", lambda t: None, user_pool=5, system_pool=3)
    try:
        assert scheduler._pools._user._value == 5
        assert scheduler._pools._system._value == 3
    finally:
        await teardown()


async def test_build_repl_surfaces_turn_failure():
    class _FailingLoop:
        async def run_turn(self, req, emit, drain, *, stream) -> TurnOutcome:
            raise RuntimeError("provider offline")

    errors: list[str] = []
    scheduler, hub, teardown = build_repl(
        _FailingLoop(),
        "cli",
        lambda _text: None,
        render_error=errors.append,
    )
    try:
        outcome = await scheduler.submit(
            TurnRequest(
                origin=Origin.USER,
                source=_src(),
                text="hello",
                conversation="cli:c1",
            )
        ).result()
        await hub.wait_idle("cli")
    finally:
        await teardown()

    assert outcome is None
    assert errors == ["Turn failed: provider offline"]


async def test_repl_loop_renders_each_reply_before_the_next_prompt():
    events, hub, scheduler, teardown = _wire_real_spine()
    await run_repl_loop(
        read_input=_make_reader(events, ["a", "b", "exit"]),
        submit=scheduler.submit,
        wait_idle=hub.wait_idle,
        channel="cli",
        chat_id="c",
        is_exit=lambda c: c == "exit",
        handle_slash=lambda c: False,
        thinking=nullcontext,
        on_exit=lambda: events.append("exit"),
    )
    await teardown()

    assert events == [
        "prompt",
        "render:reply<a>",
        "prompt",
        "render:reply<b>",
        "prompt",
        "exit",
    ]


async def test_repl_loop_handles_empty_reply_without_hanging():
    events: list[str] = []

    class EmptyLoop:
        async def run_turn(self, req, emit, drain, *, stream) -> TurnOutcome:

            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    scheduler, hub, teardown = build_repl(EmptyLoop(), "cli", lambda t: events.append(f"render:{t!r}"))
    await run_repl_loop(
        read_input=_make_reader(events, ["hi", "exit"]),
        submit=scheduler.submit,
        wait_idle=hub.wait_idle,
        channel="cli",
        chat_id="c",
        is_exit=lambda c: c == "exit",
        handle_slash=lambda c: False,
        thinking=nullcontext,
        on_exit=lambda: events.append("exit"),
    )
    await teardown()
    assert events == ["prompt", "prompt", "exit"]


async def test_repl_loop_ctrl_c_mid_turn_exits_cleanly():
    events: list[str] = []

    class _BoomHandle:
        async def result(self):
            raise KeyboardInterrupt

    await run_repl_loop(
        read_input=_make_reader(events, ["hi", "exit"]),
        submit=lambda req: _BoomHandle(),
        wait_idle=_anoop,
        channel="cli",
        chat_id="c",
        is_exit=lambda c: c == "exit",
        handle_slash=lambda c: False,
        thinking=nullcontext,
        on_exit=lambda: events.append("clean-exit"),
    )
    assert events == ["prompt", "clean-exit"]


async def test_repl_loop_slash_command_does_not_submit():
    submitted: list = []
    await run_repl_loop(
        read_input=_make_reader([], ["/help", "exit"]),
        submit=lambda req: submitted.append(req),
        wait_idle=_anoop,
        channel="cli",
        chat_id="c",
        is_exit=lambda c: c == "exit",
        handle_slash=lambda c: True,
        thinking=nullcontext,
        on_exit=lambda: None,
    )
    assert submitted == []


async def test_build_repl_teardown_leaves_no_pending_tasks():

    baseline = asyncio.all_tasks()
    scheduler, hub, teardown = build_repl(_EchoLoop(), "cli", lambda t: None)
    handle = scheduler.submit(TurnRequest(origin=Origin.USER, source=_src(), text="hi", conversation="cli:c1"))
    await handle.result()
    await hub.wait_idle("cli")
    spawned = asyncio.all_tasks() - baseline - {asyncio.current_task()}
    assert any(not t.done() for t in spawned)
    await teardown()
    assert all(t.done() for t in spawned)


def _make_reader(events, inputs):
    queue = list(inputs)

    async def read_input() -> str:
        events.append("prompt")
        if not queue:
            raise EOFError
        return queue.pop(0)

    return read_input


async def _anoop(*a, **k):
    return None
