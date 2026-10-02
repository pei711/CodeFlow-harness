import asyncio
from types import SimpleNamespace

import pytest

from codeflow.channels.intake import Intake
from codeflow.channels.manager import ChannelManager
from codeflow.cli._gateway_spine import build_gateway
from codeflow.spine import (
    ChatType,
    MediaOut,
    Origin,
    Source,
    Text,
    ToolEvent,
    ToolPhase,
    TurnOutcome,
    TurnRequest,
    Usage,
)
from codeflow.spine import delivery as delivery_mod
from codeflow.spine.delivery import Capabilities, DeliveryHub
from codeflow.spine.message import Media


def _src(channel="telegram", chat_id="c1") -> Source:
    return Source(channel=channel, chat_id=chat_id, sender_id="user", chat_type=ChatType.DM)


def _req(text="ping", *, channel="telegram", chat_id="c1", conversation="cron:1") -> TurnRequest:
    return TurnRequest(
        origin=Origin.CRON,
        source=_src(channel, chat_id),
        text=text,
        conversation=conversation,
    )


class _FakeChannel:
    def __init__(self, name="telegram") -> None:
        self.name = name
        self.sent: list[tuple[str, str, list[str] | None]] = []

    async def send(self, chat_id: str, content: str, media: list[str] | None = None) -> None:
        self.sent.append((chat_id, content, media))


class _ReplyAgent:
    """run_turn emits scripted spine events (the gateway runner wires stream=False
    -> cron replies are Text / MediaOut) and fills ``text_sink`` with the
    first Text's content, mirroring run_turn's observation copy."""

    def __init__(self, events=()) -> None:
        self._events = list(events)
        self.notify_count = 0

    def _notify_turn_complete(self) -> None:
        self.notify_count += 1

    async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None) -> TurnOutcome:
        for ev in self._events:
            await emit(ev)
            if text_sink is not None and isinstance(ev, Text) and ev.content:
                text_sink["text"] = ev.content
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=True)


def test_build_gateway_requires_a_running_loop():

    with pytest.raises(RuntimeError):
        build_gateway(_ReplyAgent(), {})


async def test_build_gateway_registers_an_outlet_per_channel():
    channels = {"telegram": _FakeChannel("telegram"), "discord": _FakeChannel("discord")}
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(_ReplyAgent(), channels)
    try:
        assert {"telegram", "discord"} <= set(hub._outlets)
    finally:
        await teardown()


async def test_build_gateway_defaults_to_canonical_pool_and_retry_sizes():
    scheduler, _hub, _rb, _sources, teardown = build_gateway(_ReplyAgent(), {})
    try:
        assert scheduler._pools._user._value == 4
        assert scheduler._pools._system._value == 2
        assert _hub._send_max_retries == 3
    finally:
        await teardown()


async def test_build_gateway_honors_configured_pool_and_retry_sizes():
    scheduler, hub, _rb, _sources, teardown = build_gateway(
        _ReplyAgent(), {}, user_pool=7, system_pool=3, send_max_retries=5
    )
    try:
        assert scheduler._pools._user._value == 7
        assert scheduler._pools._system._value == 3
        assert hub._send_max_retries == 5
    finally:
        await teardown()


async def test_cron_reply_reaches_the_channel_via_outlet():

    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(
        _ReplyAgent([Text(content="reminder!")]), {"telegram": ch}
    )
    try:
        await scheduler.submit(_req(channel="telegram", chat_id="c9")).result()
        await hub.wait_idle("telegram")
    finally:
        await teardown()

    assert len(ch.sent) == 1
    assert ch.sent[0][0] == "c9"
    assert ch.sent[0][1] == "reminder!"


async def test_readback_captures_cron_reply_text():

    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(
        _ReplyAgent([Text(content="done at 17:05")]), {"telegram": ch}
    )
    try:
        await scheduler.submit(_req(channel="telegram", conversation="cron:42")).result()
        await hub.wait_idle("telegram")
        assert readback_texts["cron:42"] == "done at 17:05"
    finally:
        await teardown()


async def test_readback_skips_non_readback_origin():

    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(
        _ReplyAgent([Text(content="hello")]), {"telegram": ch}
    )
    user_req = TurnRequest(origin=Origin.USER, source=_src("telegram", "u1"), text="hi", conversation="telegram:u1")
    try:
        await scheduler.submit(user_req).result()
        await hub.wait_idle("telegram")
        assert "telegram:u1" not in readback_texts
        assert len(ch.sent) == 1
    finally:
        await teardown()


async def test_cron_media_reply_sends_local_paths():
    ch = _FakeChannel("telegram")
    media = (Media(path="/tmp/chart.png", mime="image/png", kind="image"),)
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(
        _ReplyAgent([MediaOut(media=media)]), {"telegram": ch}
    )
    try:
        await scheduler.submit(_req(channel="telegram")).result()
        await hub.wait_idle("telegram")
    finally:
        await teardown()

    assert len(ch.sent) == 1 and ch.sent[0][2] == ["/tmp/chart.png"]


async def test_reply_to_unregistered_channel_is_dropped_not_raised():

    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(
        _ReplyAgent([Text(content="hi")]), {"telegram": ch}
    )
    try:
        await scheduler.submit(_req(channel="discord", conversation="cron:2")).result()
        await hub.wait_idle("telegram")
    finally:
        await teardown()

    assert ch.sent == []


async def test_gateway_sink_notifies_on_turn_end():
    agent = _ReplyAgent([Text(content="hi")])
    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(agent, {"telegram": ch})
    try:
        await scheduler.submit(_req(channel="telegram")).result()
        await hub.wait_idle("telegram")
    finally:
        await teardown()
    assert agent.notify_count >= 1


async def test_gateway_logs_recovered_tool_failures_for_fire_and_forget(
    monkeypatch: pytest.MonkeyPatch,
):
    completed = asyncio.Event()
    warnings: list[tuple[tuple, dict]] = []

    class _RecoveredAgent(_ReplyAgent):
        def _notify_turn_complete(self) -> None:
            super()._notify_turn_complete()
            completed.set()

        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None):
            await emit(
                ToolEvent(
                    phase=ToolPhase.COMPLETE,
                    tool_call_id="tool-1",
                    name="read_file",
                    result_preview="Error: denied",
                    failed=True,
                )
            )
            await emit(Text(content="Recovered with a safe fallback."))
            return TurnOutcome(
                usage=Usage(1, 2, 3),
                explicit_reply=True,
                tool_calls=1,
                tool_failures=1,
            )

    monkeypatch.setattr(
        "codeflow.cli._gateway_spine.logger.warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )
    agent = _RecoveredAgent()
    channel = _FakeChannel("telegram")
    scheduler, hub, _readback, _sources, teardown = build_gateway(
        agent,
        {"telegram": channel},
    )
    try:
        scheduler.submit(_req(channel="telegram", chat_id="c9"))
        await asyncio.wait_for(completed.wait(), timeout=2)
        await hub.wait_idle("telegram")
    finally:
        await teardown()

    assert channel.sent == [("c9", "Recovered with a safe fallback.", None)]
    assert len(warnings) == 1
    args, kwargs = warnings[0]
    assert kwargs == {}
    assert args[1:] == ("cron:1", 1, 1, True)


async def test_gateway_sink_sends_error_reply_on_failure():
    class _BoomAgent(_ReplyAgent):
        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None):
            raise RuntimeError("boom")

    agent = _BoomAgent()
    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(agent, {"telegram": ch})
    try:
        try:
            await scheduler.submit(_req(channel="telegram", chat_id="c9")).result()
        except Exception:
            pass
        await hub.wait_idle("telegram")
    finally:
        await teardown()

    assert len(ch.sent) == 1 and ch.sent[0][1] == "Sorry, I encountered an error."
    assert ch.sent[0][0] == "c9"
    assert agent.notify_count >= 1


async def test_gateway_sink_no_error_reply_on_cancel():

    started = asyncio.Event()

    class _BlockingAgent(_ReplyAgent):
        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None):
            started.set()
            await asyncio.Event().wait()

    agent = _BlockingAgent()
    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(agent, {"telegram": ch})
    try:
        handle = scheduler.submit(_req(channel="telegram", conversation="telegram:u1"))
        await started.wait()
        scheduler.cancel_conversation("telegram:u1")
        try:
            await handle.result()
        except (asyncio.CancelledError, Exception):
            pass
        await hub.wait_idle("telegram")
    finally:
        await teardown()
    assert ch.sent == []
    assert agent.notify_count >= 1


async def test_build_gateway_teardown_leaves_no_pending_tasks():
    baseline = asyncio.all_tasks()
    ch = _FakeChannel("telegram")
    scheduler, hub, readback_texts, _sources, teardown = build_gateway(
        _ReplyAgent([Text(content="hi")]), {"telegram": ch}
    )
    await scheduler.submit(_req(channel="telegram")).result()
    await hub.wait_idle("telegram")
    spawned = asyncio.all_tasks() - baseline - {asyncio.current_task()}
    assert any(not t.done() for t in spawned)
    await teardown()
    assert all(t.done() for t in spawned)


async def test_gateway_teardown_closes_delivery_after_scheduler_cancellation():
    started = asyncio.Event()
    cancellation_swallowed = asyncio.Event()
    release_runner = asyncio.Event()

    class _DelayedCancellationAgent(_ReplyAgent):
        async def run_turn(self, req, emit, drain, *, stream, usage_sink=None, text_sink=None):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_swallowed.set()
                await release_runner.wait()
                raise

    scheduler, hub, _readback, _sources, teardown = build_gateway(
        _DelayedCancellationAgent(),
        {"telegram": _FakeChannel()},
    )
    await hub.dispatch(Text(content="resident worker", source=_src("telegram")))
    await hub.wait_idle("telegram")
    worker = hub._workers["telegram"]
    handle = scheduler.submit(_req(channel="telegram"))
    await started.wait()
    tearing_down = asyncio.create_task(teardown())
    await cancellation_swallowed.wait()
    tearing_down.cancel()
    await asyncio.sleep(0)

    try:
        assert not tearing_down.done()
        release_runner.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(tearing_down, timeout=0.2)
        assert await asyncio.wait_for(handle.result(), timeout=0.2) is None
        assert hub._closed
        assert worker.done()
        assert hub._workers == {}
    finally:
        release_runner.set()
        if not tearing_down.done():
            tearing_down.cancel()
        await asyncio.gather(tearing_down, return_exceptions=True)
        await hub.aclose()


async def test_gateway_teardown_attempts_delivery_and_prefers_cancellation(monkeypatch):
    scheduler, hub, _readback, _sources, teardown = build_gateway(_ReplyAgent(), {})
    original_shutdown = scheduler.shutdown
    original_aclose = hub.aclose
    events: list[str] = []

    async def fail_scheduler(*, grace: float) -> None:
        events.append("scheduler")
        raise RuntimeError("scheduler failed")

    async def cancel_delivery() -> None:
        events.append("delivery")
        raise asyncio.CancelledError

    monkeypatch.setattr(scheduler, "shutdown", fail_scheduler)
    monkeypatch.setattr(hub, "aclose", cancel_delivery)
    try:
        with pytest.raises(asyncio.CancelledError):
            await teardown()
    finally:
        await original_shutdown(grace=0.0)
        await original_aclose()

    assert events == ["scheduler", "delivery"]


async def test_intake_quiesce_is_bounded_when_delivery_queue_is_full(monkeypatch):
    monkeypatch.setattr(delivery_mod, "_OUTLET_QUEUE_MAXSIZE", 1)
    monkeypatch.setattr("codeflow.channels.manager._INTAKE_DRAIN_TIMEOUT_S", 0.01, raising=False)
    delivery_started = asyncio.Event()

    class _BlockedOutlet:
        name = "telegram"
        capabilities = Capabilities()

        async def deliver(self, out) -> None:
            delivery_started.set()
            await asyncio.Event().wait()

    hub = DeliveryHub()
    hub.register(_BlockedOutlet())
    intake = Intake("telegram", SimpleNamespace(allow_from=["*"]))

    async def submit(req) -> None:
        await hub.dispatch(Text(content="late", source=req.source))

    intake.set_submit(submit)
    manager = object.__new__(ChannelManager)
    manager.channels = {"telegram": SimpleNamespace(intake=intake)}
    publish = None
    try:
        await hub.dispatch(Text(content="in flight", source=_src("telegram")))
        await delivery_started.wait()
        await hub.dispatch(Text(content="queued", source=_src("telegram")))
        publish = asyncio.create_task(intake.publish("user", "c", "blocked enqueue"))
        await asyncio.sleep(0)
        assert not publish.done()

        with pytest.raises(TimeoutError, match="intake drain timed out"):
            await asyncio.wait_for(manager.quiesce_intake(), timeout=0.2)

        assert publish.cancelled()
        await intake.wait_idle()
    finally:
        if publish is not None and not publish.done():
            publish.cancel()
            await asyncio.gather(publish, return_exceptions=True)
        await hub.aclose()
