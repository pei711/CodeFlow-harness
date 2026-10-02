import asyncio

import pytest

from codeflow.spine import (
    ChatType,
    Notice,
    NoticeKind,
    Origin,
    Source,
    Text,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    TurnStarted,
    Usage,
)
from codeflow.spine.scheduler import Lane, OriginPools


def _req(text: str = "hi") -> TurnRequest:
    src = Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    return TurnRequest(origin=Origin.USER, source=src, text=text)


def _collector():
    events: list = []

    async def sink(event) -> None:
        events.append(event)

    return events, sink


def _lane(runner) -> Lane:
    return Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=_collector()[1], conversation_id="c")


class SuccessRunner:
    def __init__(self, outcome: TurnOutcome | None = None):
        self.outcome = outcome or TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)

    async def run(self, req, emit, drain) -> TurnOutcome:
        await emit(Text(content="reply"))
        return self.outcome


class OrderRunner:
    def __init__(self):
        self.order: list[str] = []
        self.live = 0
        self.max_live = 0

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        self.order.append(req.text)
        await asyncio.sleep(0)
        self.live -= 1
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class HangingRunner:
    def __init__(self):
        self.started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.started.set()
        await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class FailingRunner:
    async def run(self, req, emit, drain) -> TurnOutcome:
        raise ValueError("boom")


class RecordingHangRunner:
    def __init__(self):
        self.ran: list[str] = []
        self.first_started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.ran.append(req.text)
        if len(self.ran) == 1:
            self.first_started.set()
            await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class LifecycleEmittingRunner:
    def __init__(self):
        self.rejected: bool | None = None

    async def run(self, req, emit, drain) -> TurnOutcome:
        try:
            await emit(TurnStarted())
            self.rejected = False
        except TypeError:
            self.rejected = True
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class StampProbeRunner:
    def __init__(self, source=None):
        self._source = source

    async def run(self, req, emit, drain) -> TurnOutcome:
        await emit(Text(content="x", source=self._source))
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


async def test_lane_runs_fifo_one_at_a_time():
    runner = OrderRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=5, system=5), sink=sink, conversation_id="c")
    futs = [lane.submit(_req(t)) for t in ("a", "b", "c")]
    await asyncio.gather(*futs)
    assert runner.order == ["a", "b", "c"]
    assert runner.max_live == 1


async def test_success_emits_started_then_deliverables_then_ended():
    runner = SuccessRunner(TurnOutcome(usage=Usage(5, 7, 12), explicit_reply=True))
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    await lane.submit(_req())
    assert isinstance(events[0], TurnStarted)
    assert isinstance(events[1], Text)
    assert isinstance(events[-1], TurnEnded)
    assert events[-1].usage == Usage(5, 7, 12)
    assert events[-1].explicit_reply is True


async def test_emit_guard_rejects_lifecycle_events_at_runtime():

    runner = LifecycleEmittingRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    await lane.submit(_req())
    assert runner.rejected is True


async def test_emit_stamps_source_when_absent_and_preserves_explicit():
    src = Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    other = Source(channel="x", chat_id="y", sender_id="z", chat_type=ChatType.GROUP)

    events1, sink1 = _collector()
    lane1 = Lane(
        runner=StampProbeRunner(source=None), pools=OriginPools(user=1, system=1), sink=sink1, conversation_id="c"
    )
    await lane1.submit(TurnRequest(origin=Origin.USER, source=src, text="x"))
    text1 = next(e for e in events1 if isinstance(e, Text))
    assert text1.source == src

    events2, sink2 = _collector()
    lane2 = Lane(
        runner=StampProbeRunner(source=other), pools=OriginPools(user=1, system=1), sink=sink2, conversation_id="c"
    )
    await lane2.submit(TurnRequest(origin=Origin.USER, source=src, text="x"))
    text2 = next(e for e in events2 if isinstance(e, Text))
    assert text2.source == other


async def test_emit_stamps_source_on_a_non_text_deliverable():

    src = Source(channel="tg", chat_id="1", sender_id="u", chat_type=ChatType.DM)
    events, sink = _collector()

    class R:
        async def run(self, req, emit, drain) -> TurnOutcome:
            await emit(Notice(kind=NoticeKind.PROGRESS))
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    lane = Lane(runner=R(), pools=OriginPools(user=1, system=1), sink=sink, conversation_id="tg:1")
    await lane.submit(TurnRequest(origin=Origin.USER, source=src, text="x"))
    notice = next(e for e in events if isinstance(e, Notice))
    assert notice.source == src


async def test_emit_stamps_conversation_id_and_preserves_explicit():
    src = Source(channel="tg", chat_id="1", sender_id="u", chat_type=ChatType.DM)

    class EmitCid:
        def __init__(self, cid):
            self._cid = cid

        async def run(self, req, emit, drain) -> TurnOutcome:
            await emit(Text(content="x", conversation_id=self._cid))
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    events1, sink1 = _collector()
    lane1 = Lane(runner=EmitCid(None), pools=OriginPools(user=1, system=1), sink=sink1, conversation_id="tg:1")
    await lane1.submit(TurnRequest(origin=Origin.USER, source=src, text="x"))
    text1 = next(e for e in events1 if isinstance(e, Text))
    assert text1.conversation_id == "tg:1"

    events2, sink2 = _collector()
    lane2 = Lane(runner=EmitCid("explicit"), pools=OriginPools(user=1, system=1), sink=sink2, conversation_id="tg:1")
    await lane2.submit(TurnRequest(origin=Origin.USER, source=src, text="x"))
    text2 = next(e for e in events2 if isinstance(e, Text))
    assert text2.conversation_id == "explicit"


async def test_worker_stamps_conversation_id_on_lifecycle_events():

    runner = SuccessRunner(TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False))
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="tg:7")
    await lane.submit(_req())
    started = next(e for e in events if isinstance(e, TurnStarted))
    ended = next(e for e in events if isinstance(e, TurnEnded))
    assert started.conversation_id == "tg:7"
    assert ended.conversation_id == "tg:7"


async def test_worker_copies_tool_counts_to_turn_ended():
    runner = SuccessRunner(
        TurnOutcome(
            usage=Usage(1, 2, 3),
            explicit_reply=True,
            tool_calls=4,
            tool_failures=2,
        )
    )
    events, sink = _collector()
    lane = Lane(
        runner=runner,
        pools=OriginPools(user=1, system=1),
        sink=sink,
        conversation_id="tg:tools",
    )

    await lane.submit(_req())

    ended = next(event for event in events if isinstance(event, TurnEnded))
    assert ended.tool_calls == 4
    assert ended.tool_failures == 2


async def test_turn_failed_carries_conversation_id():
    runner = FailingRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="tg:9")
    await lane.submit(_req())
    failed = next(e for e in events if isinstance(e, TurnFailed))
    assert failed.conversation_id == "tg:9"


async def test_run_exception_yields_turn_failed_and_resolves_future():
    runner = FailingRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    result = await lane.submit(_req())
    assert result is None
    failed = next(e for e in events if isinstance(e, TurnFailed))
    assert failed.cancelled is False
    assert "boom" in failed.error


async def test_cancel_resolves_future_with_cancelled_terminal():
    runner = HangingRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    fut = lane.submit(_req())
    await runner.started.wait()
    lane.cancel()
    result = await asyncio.wait_for(fut, timeout=1.0)
    assert result is None
    assert any(isinstance(e, TurnFailed) and e.cancelled for e in events)


async def test_cancel_after_completion_does_not_double_resolve():

    runner = SuccessRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    fut = lane.submit(_req())
    result = await fut
    lane.cancel()
    await asyncio.sleep(0)
    assert isinstance(result, TurnOutcome)
    assert fut.done() and fut.exception() is None


async def test_worker_clears_running_state_when_future_is_already_cancelled():
    runner = HangingRunner()
    lane = _lane(runner)
    fut = lane.submit(_req())
    await runner.started.wait()

    fut.cancel()
    assert lane.cancel_running() == 1
    await asyncio.wait_for(lane._worker, timeout=1.0)

    assert lane.running_future() is None
    assert lane.has_pending_or_running() is False


async def test_cancel_drains_queue_resolving_pending_as_cancelled():

    runner = RecordingHangRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    f1 = lane.submit(_req("a"))
    f2 = lane.submit(_req("b"))
    f3 = lane.submit(_req("c"))
    await runner.first_started.wait()
    stopped = lane.cancel()
    assert stopped == 3
    for fut in (f1, f2, f3):
        assert await asyncio.wait_for(fut, timeout=1.0) is None
    assert runner.ran == ["a"]


async def test_cancel_during_setup_window_stops_the_turn():

    ran: list[str] = []
    at_started = asyncio.Event()
    release = asyncio.Event()

    async def sink(event) -> None:
        if isinstance(event, TurnStarted):
            at_started.set()
            await release.wait()

    class R:
        async def run(self, req, emit, drain) -> TurnOutcome:
            ran.append(req.text)
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    lane = Lane(runner=R(), pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    fut = lane.submit(_req("X"))
    await at_started.wait()
    stopped = lane.cancel()
    release.set()
    assert await asyncio.wait_for(fut, timeout=1.0) is None
    assert stopped == 1
    assert ran == []


async def test_cancel_before_turnstarted_emits_no_lifecycle():

    ran: list[str] = []
    events, sink = _collector()
    pools = OriginPools(user=1, system=1)
    await pools.for_origin(Origin.USER).acquire()

    class R:
        async def run(self, req, emit, drain) -> TurnOutcome:
            ran.append(req.text)
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    lane = Lane(runner=R(), pools=pools, sink=sink, conversation_id="c")
    fut = lane.submit(_req("X"))
    await asyncio.sleep(0.02)
    stopped = lane.cancel()
    assert await asyncio.wait_for(fut, timeout=1.0) is None
    assert stopped == 1
    assert ran == []
    assert events == []


async def test_worker_exits_when_queue_drains():
    runner = SuccessRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    await lane.submit(_req())
    await asyncio.sleep(0)
    assert lane._worker is None or lane._worker.done()


async def test_cancel_is_scoped_to_its_lane():
    r1, r2 = HangingRunner(), SuccessRunner()
    _, sink1 = _collector()
    _, sink2 = _collector()
    lane1 = Lane(runner=r1, pools=OriginPools(user=1, system=1), sink=sink1, conversation_id="c")
    lane2 = Lane(runner=r2, pools=OriginPools(user=1, system=1), sink=sink2, conversation_id="c")
    fut1 = lane1.submit(_req())
    fut2 = lane2.submit(_req())
    await r1.started.wait()
    lane1.cancel()
    await asyncio.wait_for(fut1, timeout=1.0)
    assert isinstance(await asyncio.wait_for(fut2, timeout=1.0), TurnOutcome)


async def test_worker_self_cancellation_drains_the_payload_leaving_no_zombie():

    runner = HangingRunner()
    events, sink = _collector()
    lane = Lane(runner=runner, pools=OriginPools(user=1, system=1), sink=sink, conversation_id="c")
    fut = lane.submit(_req())
    await runner.started.wait()
    run_task = lane._run_task
    lane._worker.cancel()
    result = await asyncio.wait_for(fut, timeout=1.0)
    assert result is None
    assert run_task.cancelled() or run_task.cancelling() > 0
    assert any(isinstance(e, TurnFailed) and e.cancelled for e in events)


async def test_scheduler_cancel_conversation_stops_running_and_returns_count():

    from codeflow.spine.scheduler import Scheduler

    runner = HangingRunner()
    _events, sink = _collector()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    handle = sched.submit(_req())
    await runner.started.wait()

    assert sched.cancel_conversation("t:c") == 1
    assert sched.cancel_conversation("nope") == 0

    try:
        await handle.result()
    except (asyncio.CancelledError, Exception):
        pass
    await sched.shutdown(grace=0.0)


async def test_scheduler_has_inflight_tracks_running_turn():

    from codeflow.spine.scheduler import Scheduler

    runner = HangingRunner()
    _events, sink = _collector()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    assert sched.has_inflight("t:c") is False
    handle = sched.submit(_req())
    await runner.started.wait()
    assert sched.has_inflight("t:c") is True
    assert sched.has_inflight("nope") is False

    sched.cancel_conversation("t:c")
    try:
        await handle.result()
    except (asyncio.CancelledError, Exception):
        pass
    await sched.shutdown(grace=0.0)


async def test_scheduler_has_pending_or_running_tracks_lane_work():
    from codeflow.spine.scheduler import Scheduler

    runner = HangingRunner()
    _events, sink = _collector()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)

    assert sched.has_pending_or_running("t:c") is False
    handle = sched.submit(_req())
    assert sched.has_pending_or_running("t:c") is True

    await runner.started.wait()
    assert sched.has_pending_or_running("t:c") is True
    assert sched.has_pending_or_running("nope") is False

    handle.cancel()
    await handle.result()
    assert sched.has_pending_or_running("t:c") is False
    await sched.shutdown(grace=0.0)


async def test_cancelled_result_waiter_does_not_cancel_shared_turn_future():
    from codeflow.spine.scheduler import Scheduler

    runner = HangingRunner()
    events, sink = _collector()
    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    handle = sched.submit(_req())
    await runner.started.wait()

    waiter = asyncio.create_task(handle.result())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert sched.has_pending_or_running("t:c") is True
    handle.cancel()
    assert await asyncio.wait_for(handle.result(), timeout=1.0) is None
    assert any(isinstance(event, TurnFailed) and event.cancelled for event in events)
    assert sched.has_pending_or_running("t:c") is False
    await sched.shutdown(grace=0.0)
