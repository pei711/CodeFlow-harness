import asyncio
import dataclasses
import json

import pytest
from loguru import logger

from codeflow.spine import (
    ChatType,
    Media,
    MediaOut,
    Notice,
    NoticeKind,
    Reasoning,
    Source,
    StreamDelta,
    Text,
    ToolEvent,
    ToolPhase,
)
from codeflow.spine import delivery as delivery_mod
from codeflow.spine.delivery import Capabilities, DeliveryHub, Outlet, SupportsStreaming
from codeflow.tracing import spans as _spans
from codeflow.tracing import trace


def test_capabilities_default_to_all_off():
    caps = Capabilities()
    assert caps.interactive_login is False
    assert caps.streaming is False


def test_capabilities_is_frozen():
    caps = Capabilities(streaming=True)
    assert caps.streaming is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.streaming = False


def test_supports_streaming_is_a_runtime_checkable_structural_protocol():
    class WithStreaming:
        async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False):
            pass

    class WithoutStreaming:
        pass

    assert isinstance(WithStreaming(), SupportsStreaming)
    assert not isinstance(WithoutStreaming(), SupportsStreaming)


def _src(channel: str) -> Source:
    return Source(channel=channel, chat_id="c", sender_id="u", chat_type=ChatType.DM)


class FakeOutlet:
    """Records what it delivers; raises on the first ``fail_times`` calls to
    exercise retry. A normal return is the eat / success path."""

    def __init__(self, name: str, *, fail_times: int = 0) -> None:
        self.name = name
        self.capabilities = Capabilities()
        self.received: list = []
        self._fail_times = fail_times
        self.calls = 0

    async def deliver(self, out) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("transport down")
        self.received.append(out)


def test_fake_outlet_satisfies_the_outlet_protocol():
    assert isinstance(FakeOutlet("tg"), Outlet)
    assert not isinstance(object(), Outlet)


async def test_register_rejects_a_duplicate_outlet_name(hub):
    first = FakeOutlet("tg")
    replacement = FakeOutlet("tg")
    hub.register(first)
    await hub.dispatch(Text(content="one", source=_src("tg")))
    await hub.wait_idle("tg")

    with pytest.raises(ValueError, match="outlet 'tg' is already registered"):
        hub.register(replacement)

    await hub.dispatch(Text(content="two", source=_src("tg")))
    await hub.wait_idle("tg")
    assert [item.content for item in first.received] == ["one", "two"]
    assert replacement.received == []


async def _settle(predicate, *, tries: int = 2000) -> None:

    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never reached")


@pytest.fixture
async def hub():
    h = DeliveryHub()
    yield h
    await h.aclose()


class GatedOutlet:
    """Blocks in deliver until its gate is set — for testing per-outlet isolation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.capabilities = Capabilities()
        self.received: list = []
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def deliver(self, out) -> None:
        self.entered.set()
        await self.gate.wait()
        self.received.append(out)


_DELIVERABLES = [
    Text(content="hi", source=_src("tg")),
    MediaOut(media=(Media(path="/tmp/a.jpg", mime="image/jpeg", kind="image"),), source=_src("tg")),
    ToolEvent(phase=ToolPhase.START, tool_call_id="t1", name="grep", source=_src("tg")),
    Reasoning(content="r", source=_src("tg")),
    Notice(kind=NoticeKind.PROGRESS, source=_src("tg")),
]


@pytest.mark.parametrize("out", _DELIVERABLES, ids=lambda o: type(o).__name__)
async def test_dispatch_routes_every_deliverable_to_its_channel_outlet(hub, out):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    await hub.dispatch(out)
    await _settle(lambda: outlet.received == [out])
    assert outlet.calls == 1


async def test_dispatch_keeps_same_channel_order(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    for n in ("a", "b", "c"):
        await hub.dispatch(Text(content=n, source=_src("tg")))
    await _settle(lambda: len(outlet.received) == 3)
    assert [m.content for m in outlet.received] == ["a", "b", "c"]


async def test_per_outlet_serial_holds_across_a_retry(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)

    class FailsFirstEvent:
        name = "tg"
        capabilities = Capabilities()

        def __init__(self) -> None:
            self.received: list = []
            self._fail_left = 2

        async def deliver(self, out) -> None:
            if out.content == "a" and self._fail_left > 0:
                self._fail_left -= 1
                raise RuntimeError("transport down")
            self.received.append(out)

    outlet = FailsFirstEvent()
    hub.register(outlet)
    await hub.dispatch(Text(content="a", source=_src("tg")))
    await hub.dispatch(Text(content="b", source=_src("tg")))
    await _settle(lambda: len(outlet.received) == 2)
    assert [m.content for m in outlet.received] == ["a", "b"]


async def test_send_max_retries_is_configurable(monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)

    class AlwaysFails:
        name = "tg"
        capabilities = Capabilities()

        def __init__(self) -> None:
            self.attempts = 0

        async def deliver(self, out) -> None:
            self.attempts += 1
            raise RuntimeError("transport down")

    hub = DeliveryHub(send_max_retries=1)
    outlet = AlwaysFails()
    hub.register(outlet)
    try:
        await hub.dispatch(Text(content="a", source=_src("tg")))
        await hub.wait_idle("tg")
        assert outlet.attempts == 2
    finally:
        await hub.aclose()


async def test_dispatch_routes_by_source_channel_across_outlets(hub):
    tg, wx = FakeOutlet("tg"), FakeOutlet("wx")
    hub.register(tg)
    hub.register(wx)
    await hub.dispatch(Text(content="a", source=_src("tg")))
    await hub.dispatch(Text(content="b", source=_src("wx")))
    await _settle(lambda: tg.received and wx.received)
    assert [m.content for m in tg.received] == ["a"]
    assert [m.content for m in wx.received] == ["b"]


async def test_a_slow_outlet_does_not_block_another(hub):
    slow = GatedOutlet("slow")
    fast = FakeOutlet("fast")
    hub.register(slow)
    hub.register(fast)
    await hub.dispatch(Text(content="s", source=_src("slow")))
    await hub.dispatch(Text(content="f", source=_src("fast")))
    await _settle(lambda: fast.received and fast.received[0].content == "f")
    assert not slow.received
    slow.gate.set()
    await _settle(lambda: slow.received and slow.received[0].content == "s")


async def test_per_outlet_backpressure_isolates_channels(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_OUTLET_QUEUE_MAXSIZE", 1)
    slow = GatedOutlet("slow")
    fast = FakeOutlet("fast")
    hub.register(slow)
    hub.register(fast)
    await hub.dispatch(Text(content="s1", source=_src("slow")))
    await slow.entered.wait()
    await hub.dispatch(Text(content="s2", source=_src("slow")))
    blocked = asyncio.ensure_future(hub.dispatch(Text(content="s3", source=_src("slow"))))
    await asyncio.sleep(0)
    assert not blocked.done()
    await hub.dispatch(Text(content="f", source=_src("fast")))
    await _settle(lambda: fast.received and fast.received[0].content == "f")
    slow.gate.set()
    await blocked


async def test_a_dead_worker_self_heals_on_next_enqueue(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    await hub.dispatch(Text(content="a", source=_src("tg")))
    await _settle(lambda: len(outlet.received) == 1)
    dead = hub._workers["tg"]
    dead.cancel()
    await asyncio.gather(dead, return_exceptions=True)
    await hub.dispatch(Text(content="b", source=_src("tg")))
    await _settle(lambda: len(outlet.received) == 2)
    assert [m.content for m in outlet.received] == ["a", "b"]
    assert hub._workers["tg"] is not dead


async def test_a_raising_deliver_is_retried_then_succeeds(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    outlet = FakeOutlet("tg", fail_times=2)
    hub.register(outlet)
    out = Text(content="hi", source=_src("tg"))
    await hub.dispatch(out)
    await _settle(lambda: outlet.received == [out])
    assert outlet.calls == 3


async def test_exhausted_retries_log_an_error_and_drop(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    outlet = FakeOutlet("tg", fail_times=99)
    hub.register(outlet)
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="ERROR", format="{message}")
    try:
        await hub.dispatch(Text(content="hi", source=_src("tg")))
        await _settle(lambda: any("delivery failed" in line for line in lines))
    finally:
        logger.remove(sink_id)
    assert outlet.calls == delivery_mod._SEND_MAX_RETRIES + 1
    assert not outlet.received
    err = next(line for line in lines if "delivery failed" in line)
    assert "tg" in err and "Text" in err and "transport down" in err


async def test_retry_backoff_doubles_from_the_base_delay(hub, monkeypatch):
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(delivery_mod.asyncio, "sleep", fake_sleep)
    outlet = FakeOutlet("tg", fail_times=99)
    hub.register(outlet)
    await hub.dispatch(Text(content="hi", source=_src("tg")))
    for _ in range(50):
        if len(delays) == 3:
            break
        await real_sleep(0)
    base = delivery_mod._RETRY_BASE_DELAY
    assert delays == [base, base * 2, base * 4]
    assert outlet.calls == delivery_mod._SEND_MAX_RETRIES + 1


async def test_an_unreachable_channel_warns_and_drops(hub):
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="WARNING", format="{message}")
    try:
        await hub.dispatch(Text(content="hi", source=_src("ghost")))
    finally:
        logger.remove(sink_id)
    warn = next(line for line in lines if "no outlet" in line)
    assert "ghost" in warn


async def test_post_routes_like_dispatch(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    out = Text(content="menu", source=_src("tg"))
    await hub.post(out)
    await _settle(lambda: outlet.received == [out])


async def test_a_sourceless_deliverable_fails_loud(hub):
    hub.register(FakeOutlet("tg"))
    with pytest.raises(ValueError, match="no source"):
        await hub.dispatch(Text(content="hi"))


class FakeStreamingOutlet:
    """A streaming-capable outlet: deliver() consumes the stream instead of
    delivering Text, send_stream_chunk records each chunk."""

    def __init__(self, name: str = "tg") -> None:
        self.name = name
        self.capabilities = Capabilities(streaming=True)
        self.received: list = []
        self.chunks: list[tuple] = []

    async def deliver(self, out) -> None:
        self.received.append(out)

    async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None:
        self.chunks.append((chat_id, stream_id, delta, done))


class GatedStreamingOutlet(FakeStreamingOutlet):
    def __init__(self, name: str = "tg") -> None:
        super().__init__(name)
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()

    async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None:
        self.entered.set()
        await self.gate.wait()
        await super().send_stream_chunk(chat_id, stream_id, delta, done=done)


async def test_streaming_outlet_gets_chunks_then_a_done_close(hub):
    outlet = FakeStreamingOutlet("tg")
    hub.register(outlet)
    src = _src("tg")
    await hub.dispatch(StreamDelta(delta="he", source=src, conversation_id="tg:c"))
    await hub.dispatch(StreamDelta(delta="llo", source=src, conversation_id="tg:c"))
    await hub.close_stream("tg:c")
    await hub.wait_idle("tg")
    assert outlet.chunks == [
        ("c", "tg:c", "he", False),
        ("c", "tg:c", "llo", False),
        ("c", "tg:c", "", True),
    ]


async def test_close_stream_is_a_noop_for_an_unopened_conversation(hub):
    outlet = FakeStreamingOutlet("tg")
    hub.register(outlet)
    await hub.close_stream("tg:never")
    assert outlet.chunks == []


async def test_non_streaming_outlet_eats_stream_delta(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    await hub.dispatch(StreamDelta(delta="x", source=_src("tg"), conversation_id="tg:c"))
    await hub.close_stream("tg:c")
    await hub.wait_idle("tg")
    assert outlet.received == []


async def test_a_stream_reopens_cleanly_for_the_next_turn(hub):

    outlet = FakeStreamingOutlet("tg")
    hub.register(outlet)
    src = _src("tg")
    for delta in ("a", "b"):
        await hub.dispatch(StreamDelta(delta=delta, source=src, conversation_id="tg:c"))
        await hub.close_stream("tg:c")
        await hub.wait_idle("tg")
    assert outlet.chunks == [
        ("c", "tg:c", "a", False),
        ("c", "tg:c", "", True),
        ("c", "tg:c", "b", False),
        ("c", "tg:c", "", True),
    ]
    assert "tg:c" not in hub._open_streams
    assert "tg:c" not in hub._stream_channel


async def test_aclose_cancels_workers(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    await hub.dispatch(Text(content="a", source=_src("tg")))
    await _settle(lambda: len(outlet.received) == 1)
    worker = hub._workers["tg"]
    await hub.aclose()
    assert worker.cancelled()
    assert hub._workers == {}


async def test_dispatch_after_aclose_is_rejected_without_restarting_worker(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    await hub.dispatch(Text(content="before", source=_src("tg")))
    await hub.wait_idle("tg")
    await hub.aclose()

    with pytest.raises(RuntimeError, match="delivery hub is closed"):
        await hub.dispatch(Text(content="after", source=_src("tg")))

    assert hub._workers == {}
    assert [item.content for item in outlet.received] == ["before"]


async def test_concurrent_aclose_calls_share_worker_shutdown_barrier(hub):
    cancellation_swallowed = asyncio.Event()
    release_delivery = asyncio.Event()

    class _SwallowOnceOutlet(FakeOutlet):
        async def deliver(self, out) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_swallowed.set()
                await release_delivery.wait()

    hub.register(_SwallowOnceOutlet("tg"))
    await hub.dispatch(Text(content="in flight", source=_src("tg")))
    worker = hub._workers["tg"]
    first_close = asyncio.create_task(hub.aclose())
    await cancellation_swallowed.wait()
    second_close = asyncio.create_task(hub.aclose())
    await asyncio.sleep(0)

    try:
        assert not second_close.done()
        release_delivery.set()
        await asyncio.wait_for(asyncio.gather(first_close, second_close), timeout=0.2)
    finally:
        release_delivery.set()
        for task in (first_close, second_close, worker):
            if not task.done():
                task.cancel()
        await asyncio.gather(first_close, second_close, worker, return_exceptions=True)

    assert worker.done()
    assert hub._workers == {}


async def test_aclose_delays_caller_cancellation_until_worker_shutdown(hub):
    cancellation_swallowed = asyncio.Event()
    release_delivery = asyncio.Event()

    class _DelayedCancellationOutlet(FakeOutlet):
        async def deliver(self, out) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_swallowed.set()
                await release_delivery.wait()

    hub.register(_DelayedCancellationOutlet("tg"))
    await hub.dispatch(Text(content="in flight", source=_src("tg")))
    worker = hub._workers["tg"]
    closing = asyncio.create_task(hub.aclose())
    await cancellation_swallowed.wait()
    closing.cancel()
    await asyncio.sleep(0)

    try:
        assert not closing.done()
        release_delivery.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, timeout=0.2)
    finally:
        release_delivery.set()
        for task in (closing, worker):
            if not task.done():
                task.cancel()
        await asyncio.gather(closing, worker, return_exceptions=True)

    assert closing.cancelled()
    assert worker.done()
    assert hub._workers == {}


async def test_aclose_preserves_caller_cancellation_over_barrier_failure(monkeypatch):
    cancellation_swallowed = asyncio.Event()
    release_delivery = asyncio.Event()

    class _DelayedCancellationOutlet(FakeOutlet):
        async def deliver(self, out) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_swallowed.set()
                await release_delivery.wait()

    hub = DeliveryHub()
    hub.register(_DelayedCancellationOutlet("tg"))
    await hub.dispatch(Text(content="in flight", source=_src("tg")))
    worker = hub._workers["tg"]

    def fail_drain() -> int:
        raise RuntimeError("drain failed")

    monkeypatch.setattr(hub, "drain", fail_drain)
    closing = asyncio.create_task(hub.aclose())
    await cancellation_swallowed.wait()
    closing.cancel()
    await asyncio.sleep(0)

    try:
        assert not closing.done()
        release_delivery.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(closing, timeout=0.2)
    finally:
        release_delivery.set()
        if not closing.done():
            closing.cancel()
        await asyncio.gather(closing, worker, return_exceptions=True)

    assert closing.cancelled()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "drain failed"
    assert worker.done()
    assert hub._workers == {}


async def test_aclose_rejects_dispatch_already_blocked_on_full_queue(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_OUTLET_QUEUE_MAXSIZE", 1)
    outlet = GatedOutlet("slow")
    hub.register(outlet)
    await hub.dispatch(Text(content="in flight", source=_src("slow")))
    await outlet.entered.wait()
    await hub.dispatch(Text(content="queued", source=_src("slow")))
    blocked = asyncio.create_task(hub.dispatch(Text(content="blocked", source=_src("slow"))))
    await asyncio.sleep(0)
    assert not blocked.done()

    await hub.aclose()

    with pytest.raises(RuntimeError, match="delivery hub is closed"):
        await blocked
    queue = hub._queues["slow"]
    assert queue.qsize() == 0
    assert queue._unfinished_tasks == 0
    assert hub._workers == {}


async def test_aclose_rejects_close_marker_already_blocked_on_full_queue(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_OUTLET_QUEUE_MAXSIZE", 1)
    outlet = GatedStreamingOutlet("tg")
    hub.register(outlet)
    conversation_id = "tg:c"
    await hub.dispatch(StreamDelta(delta="a", source=_src("tg"), conversation_id=conversation_id))
    await outlet.entered.wait()
    await hub.dispatch(Text(content="queued", source=_src("tg")))
    closing = asyncio.create_task(hub.close_stream(conversation_id))
    await asyncio.sleep(0)
    assert not closing.done()

    await hub.aclose()

    with pytest.raises(RuntimeError, match="delivery hub is closed"):
        await closing
    queue = hub._queues["tg"]
    assert queue.qsize() == 0
    assert queue._unfinished_tasks == 0
    assert conversation_id not in hub._stream_channel
    assert conversation_id not in hub._open_streams
    assert hub._workers == {}


async def test_cancelled_close_stream_restores_route_for_retry(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_OUTLET_QUEUE_MAXSIZE", 1)
    outlet = GatedStreamingOutlet("tg")
    hub.register(outlet)
    conversation_id = "tg:c"
    await hub.dispatch(StreamDelta(delta="a", source=_src("tg"), conversation_id=conversation_id))
    await outlet.entered.wait()
    await hub.dispatch(Text(content="queued", source=_src("tg")))
    closing = asyncio.create_task(hub.close_stream(conversation_id))
    await asyncio.sleep(0)
    assert not closing.done()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert hub._stream_channel[conversation_id] == "tg"

    outlet.gate.set()
    await hub.wait_idle("tg")
    await hub.close_stream(conversation_id)
    await hub.wait_idle("tg")
    assert outlet.chunks == [
        ("c", conversation_id, "a", False),
        ("c", conversation_id, "", True),
    ]


async def test_drain_drops_queued_events_and_counts_them(hub):
    slow = GatedOutlet("slow")
    hub.register(slow)
    await hub.dispatch(Text(content="s1", source=_src("slow")))
    await slow.entered.wait()
    await hub.dispatch(Text(content="s2", source=_src("slow")))
    await hub.dispatch(Text(content="s3", source=_src("slow")))
    assert hub.drain() == 2
    assert hub.drain() == 0


async def test_wait_idle_returns_at_once_when_nothing_queued(hub):
    hub.register(FakeOutlet("tg"))
    await hub.wait_idle("tg")
    await hub.wait_idle("ghost")


async def test_wait_idle_blocks_until_in_flight_delivery_completes(hub):
    slow = GatedOutlet("slow")
    hub.register(slow)
    await hub.dispatch(Text(content="s", source=_src("slow")))
    await slow.entered.wait()
    waiter = asyncio.ensure_future(hub.wait_idle("slow"))
    await asyncio.sleep(0)
    assert not waiter.done()
    slow.gate.set()
    await waiter


async def test_wait_idle_returns_only_after_all_queued_delivered(hub):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    for n in ("a", "b", "c"):
        await hub.dispatch(Text(content=n, source=_src("tg")))
    await hub.wait_idle("tg")
    assert [m.content for m in outlet.received] == ["a", "b", "c"]


async def test_wait_idle_returns_even_when_delivery_is_dropped(hub, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    outlet = FakeOutlet("tg", fail_times=99)
    hub.register(outlet)
    await hub.dispatch(Text(content="x", source=_src("tg")))
    await hub.wait_idle("tg")
    assert outlet.calls == delivery_mod._SEND_MAX_RETRIES + 1


async def test_drain_keeps_wait_idle_consistent(hub):
    slow = GatedOutlet("slow")
    hub.register(slow)
    await hub.dispatch(Text(content="s1", source=_src("slow")))
    await slow.entered.wait()
    await hub.dispatch(Text(content="s2", source=_src("slow")))
    assert hub.drain() == 1
    slow.gate.set()
    await hub.wait_idle("slow")


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "1")
    monkeypatch.setenv("CODEFLOW_TRACING_DIR", str(tmp_path))
    _spans._store = None
    yield tmp_path
    _spans._store = None


def _delivery_spans(trace_dir):
    log = trace_dir / "logs" / "audit-spans.log"
    if not log.exists():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["name"] == "channel.deliver"]


async def test_a_delivered_text_records_a_channel_span(hub, trace_dir):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    await hub.dispatch(Text(content="hi", source=_src("tg"), conversation_id="tg:c"))
    await hub.wait_idle("tg")
    await _settle(lambda: len(_delivery_spans(trace_dir)) == 1)
    span = _delivery_spans(trace_dir)[0]
    assert span["attributes"]["span.type"] == "channel"
    assert span["attributes"]["channel.name"] == "tg"
    assert span["attributes"]["channel.event"] == "Text"
    assert span["attributes"]["channel.conversation_id"] == "tg:c"
    assert span["attributes"]["channel.outcome"] == "delivered"
    assert span["attributes"]["channel.retries"] == 0
    assert span["status"]["code"] == "OK"


async def test_media_out_records_a_channel_span(hub, trace_dir):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    media = (Media(path="/tmp/a.jpg", mime="image/jpeg", kind="image"),)
    await hub.dispatch(MediaOut(media=media, source=_src("tg")))
    await hub.wait_idle("tg")
    await _settle(lambda: len(_delivery_spans(trace_dir)) == 1)
    assert _delivery_spans(trace_dir)[0]["attributes"]["channel.event"] == "MediaOut"


@pytest.mark.parametrize(
    "out",
    [
        StreamDelta(delta="x", source=_src("tg"), conversation_id="tg:c"),
        Reasoning(content="r", source=_src("tg")),
        Notice(kind=NoticeKind.PROGRESS, source=_src("tg")),
        ToolEvent(phase=ToolPhase.START, tool_call_id="t1", name="grep", source=_src("tg")),
    ],
    ids=lambda o: type(o).__name__,
)
async def test_progress_surface_gets_no_delivery_span(hub, trace_dir, out):
    hub.register(FakeOutlet("tg"))
    await hub.dispatch(out)
    await hub.wait_idle("tg")
    assert _delivery_spans(trace_dir) == []


async def test_retries_are_counted_on_the_delivery_span(hub, trace_dir, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    hub.register(FakeOutlet("tg", fail_times=2))
    await hub.dispatch(Text(content="hi", source=_src("tg")))
    await hub.wait_idle("tg")
    await _settle(lambda: len(_delivery_spans(trace_dir)) == 1)
    attrs = _delivery_spans(trace_dir)[0]["attributes"]
    assert attrs["channel.outcome"] == "delivered"
    assert attrs["channel.attempts"] == 3
    assert attrs["channel.retries"] == 2


async def test_exhausted_retries_record_dropped_and_notify_out_of_band(trace_dir, monkeypatch):
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    notices: list[Notice] = []

    async def sink(notice: Notice) -> None:
        notices.append(notice)

    hub = DeliveryHub(on_delivery_failure=sink)
    outlet = FakeOutlet("tg", fail_times=99)
    hub.register(outlet)
    try:
        await hub.dispatch(Text(content="hi", source=_src("tg"), conversation_id="tg:c"))
        await hub.wait_idle("tg")
        await _settle(lambda: notices and _delivery_spans(trace_dir))
    finally:
        await hub.aclose()
    attrs = _delivery_spans(trace_dir)[0]["attributes"]
    assert attrs["channel.outcome"] == "dropped"
    assert attrs["channel.error"] == "RuntimeError"
    assert _delivery_spans(trace_dir)[0]["status"]["code"] == "ERROR"
    assert [n.kind for n in notices] == [NoticeKind.DELIVERY_FAILED]
    assert notices[0].conversation_id == "tg:c"
    assert not outlet.received


async def test_a_delivery_failure_notice_is_never_redispatched(trace_dir, monkeypatch):
    """The failing channel must not be asked to carry its own failure report."""
    monkeypatch.setattr(delivery_mod, "_RETRY_BASE_DELAY", 0)
    hub = DeliveryHub()
    outlet = FakeOutlet("tg", fail_times=99)
    hub.register(outlet)
    try:
        await hub.dispatch(Text(content="hi", source=_src("tg")))
        await hub.wait_idle("tg")
        await _settle(lambda: _delivery_spans(trace_dir))
    finally:
        await hub.aclose()
    assert outlet.calls == delivery_mod._SEND_MAX_RETRIES + 1
    assert _delivery_spans(trace_dir)[0]["attributes"]["channel.outcome"] == "dropped"


async def test_an_unreachable_channel_records_no_outlet(hub, trace_dir):
    await hub.dispatch(Text(content="hi", source=_src("ghost")))
    spans = _delivery_spans(trace_dir)
    assert [s["attributes"]["channel.outcome"] for s in spans] == ["no_outlet"]
    assert spans[0]["attributes"]["channel.attempts"] == 0


async def test_a_delivery_span_joins_the_enclosing_turn_trace(hub, trace_dir):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    with trace.span("spine.turn", kind="session") as root:
        turn_trace, turn_span = root.trace_id, root.span_id
        await hub.dispatch(Text(content="hi", source=_src("tg")))
    await hub.wait_idle("tg")
    await _settle(lambda: len(_delivery_spans(trace_dir)) == 1)
    span = _delivery_spans(trace_dir)[0]
    assert span["traceId"] == turn_trace
    assert span["parentSpanId"] == turn_span


async def test_a_resident_worker_does_not_reuse_a_stale_turn_trace(hub, trace_dir):
    outlet = FakeOutlet("tg")
    hub.register(outlet)
    with trace.span("spine.turn", kind="session") as first:
        await hub.dispatch(Text(content="a", source=_src("tg")))
    await hub.wait_idle("tg")
    with trace.span("spine.turn", kind="session") as second:
        await hub.dispatch(Text(content="b", source=_src("tg")))
    await hub.wait_idle("tg")
    await _settle(lambda: len(_delivery_spans(trace_dir)) == 2)
    traces = [s["traceId"] for s in _delivery_spans(trace_dir)]
    assert traces == [first.trace_id, second.trace_id]
    assert first.trace_id != second.trace_id


async def test_a_raising_stream_chunk_does_not_kill_the_worker(hub):
    class BrokenStreaming:
        name = "tg"
        capabilities = Capabilities(streaming=True)

        def __init__(self) -> None:
            self.received: list = []

        async def deliver(self, out) -> None:
            self.received.append(out)

        async def send_stream_chunk(self, chat_id, stream_id, delta, *, done=False) -> None:
            raise RuntimeError("render blew up")

    outlet = BrokenStreaming()
    hub.register(outlet)
    await hub.dispatch(StreamDelta(delta="x", source=_src("tg"), conversation_id="tg:c"))
    await hub.wait_idle("tg")
    worker = hub._workers["tg"]
    await hub.dispatch(Text(content="after", source=_src("tg")))
    await _settle(lambda: outlet.received and outlet.received[0].content == "after")
    assert hub._workers["tg"] is worker


async def test_delivery_evidence_is_absent_when_tracing_is_disabled(hub, trace_dir, monkeypatch):
    monkeypatch.setenv("CODEFLOW_TRACING", "0")
    hub.register(FakeOutlet("tg"))
    await hub.dispatch(Text(content="hi", source=_src("tg")))
    await hub.wait_idle("tg")
    assert _delivery_spans(trace_dir) == []
