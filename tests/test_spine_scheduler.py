import asyncio

import pytest
from loguru import logger

from codeflow.spine import (
    BusyPolicy,
    ChatType,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    TurnEnded,
    TurnFailed,
    TurnOutcome,
    TurnRequest,
    Usage,
)
from codeflow.spine.scheduler import _DEFAULT_IDLE_TTL, SchedulerDrainingError


def _req(
    *,
    conversation=None,
    channel="t",
    chat_id="c",
    origin=Origin.USER,
    busy=BusyPolicy.APPEND,
    text="hi",
) -> TurnRequest:
    src = Source(channel=channel, chat_id=chat_id, sender_id="u", chat_type=ChatType.DM)
    return TurnRequest(origin=origin, source=src, text=text, conversation=conversation, busy=busy)


async def _sink(event) -> None:
    pass


class SuccessRunner:
    def __init__(self, outcome=None):
        self.outcome = outcome or TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)

    async def run(self, req, emit, drain) -> TurnOutcome:
        return self.outcome


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


class ConcurrentProbe:
    def __init__(self, target: int):
        self.running = 0
        self.reached = asyncio.Event()
        self._target = target

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.running += 1
        if self.running >= self._target:
            self.reached.set()
        await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


def _scheduler(runner) -> Scheduler:
    return Scheduler(runner, OriginPools(user=1, system=1), _sink)


async def test_submit_result_returns_the_outcome():
    sched = _scheduler(SuccessRunner(TurnOutcome(usage=Usage(5, 7, 12), explicit_reply=True)))
    handle = sched.submit(_req())
    outcome = await handle.result()
    assert outcome == TurnOutcome(usage=Usage(5, 7, 12), explicit_reply=True)


async def test_failed_turn_result_is_none():
    sched = _scheduler(FailingRunner())
    assert await sched.submit(_req()).result() is None


async def test_terminal_sink_failure_does_not_kill_lane_worker():
    terminal_calls = 0

    async def failing_terminal_sink(event) -> None:
        nonlocal terminal_calls
        if isinstance(event, TurnEnded):
            terminal_calls += 1
            if terminal_calls == 1:
                raise RuntimeError("terminal transport failed")

    sched = Scheduler(SuccessRunner(), OriginPools(user=1, system=1), failing_terminal_sink)
    first = sched.submit(_req(text="first"))
    second = sched.submit(_req(text="second"))

    assert isinstance(await asyncio.wait_for(first.result(), timeout=1.0), TurnOutcome)
    assert isinstance(await asyncio.wait_for(second.result(), timeout=1.0), TurnOutcome)
    assert terminal_calls == 2
    assert not sched._lanes["t:c"].has_pending_or_running()


def test_submit_with_no_running_loop_fails_fast():
    loop = asyncio.new_event_loop()
    try:
        sched = loop.run_until_complete(_aconstruct())
        with pytest.raises(RuntimeError):
            sched.submit(_req())
    finally:
        loop.close()


def test_submit_from_a_different_loop_fails_fast():

    home = asyncio.new_event_loop()
    other = asyncio.new_event_loop()
    try:
        sched = home.run_until_complete(_aconstruct())

        async def call():
            sched.submit(_req())

        with pytest.raises(RuntimeError):
            other.run_until_complete(call())
    finally:
        home.close()
        other.close()


async def _aconstruct() -> Scheduler:
    return Scheduler(SuccessRunner(), OriginPools(user=1, system=1), _sink)


async def test_conversation_id_explicit_then_default():
    sched = _scheduler(SuccessRunner())
    assert sched._conversation_id(_req(conversation="cron:7")) == "cron:7"
    assert sched._conversation_id(_req(channel="tg", chat_id="42")) == "tg:42"


async def test_different_conversations_run_concurrently():

    runner = ConcurrentProbe(target=2)
    sched = Scheduler(runner, OriginPools(user=2, system=1), _sink)
    sched.submit(_req(channel="tg", chat_id="1"))
    sched.submit(_req(channel="tg", chat_id="2"))
    await asyncio.wait_for(runner.reached.wait(), timeout=1.0)


async def test_same_conversation_reuses_one_lane():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await sched.submit(_req(channel="tg", chat_id="2")).result()
    assert set(sched._lanes) == {"tg:1", "tg:2"}


async def test_handle_cancel_running_turn():
    runner = HangingRunner()
    sched = _scheduler(runner)
    handle = sched.submit(_req())
    await runner.started.wait()
    handle.cancel()
    assert await asyncio.wait_for(handle.result(), timeout=1.0) is None


async def test_handle_cancel_queued_turn_without_running_it():
    runner = HangingRunner()
    sched = _scheduler(runner)
    first = sched.submit(_req(channel="tg", chat_id="1"))
    await runner.started.wait()
    queued = sched.submit(_req(channel="tg", chat_id="1"))
    queued.cancel()
    assert await asyncio.wait_for(queued.result(), timeout=1.0) is None
    assert not first._fut.done()


async def test_handle_cancel_is_idempotent_after_completion():
    sched = _scheduler(SuccessRunner())
    handle = sched.submit(_req())
    await handle.result()
    handle.cancel()


async def test_reap_then_resubmit_rebuilds_lane_without_losing_the_request():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    assert lane._idle_since is not None
    assert sched._sweep(now=lane._idle_since + _DEFAULT_IDLE_TTL + 1) == 1
    assert "tg:1" not in sched._lanes
    handle = sched.submit(_req(channel="tg", chat_id="1"))
    assert isinstance(await handle.result(), TurnOutcome)
    assert "tg:1" in sched._lanes and sched._lanes["tg:1"] is not lane


async def test_sweep_does_not_reap_a_lane_with_an_inflight_turn():
    runner = HangingRunner()
    sched = _scheduler(runner)
    sched.submit(_req(channel="tg", chat_id="1"))
    await runner.started.wait()
    assert sched._sweep(now=1e9) == 0
    assert "tg:1" in sched._lanes


async def test_sweep_keeps_a_lane_still_within_its_idle_ttl():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    assert sched._sweep(now=lane._idle_since + _DEFAULT_IDLE_TTL / 2) == 0
    assert "tg:1" in sched._lanes


async def test_cancel_on_a_reaped_lanes_handle_is_a_noop():
    sched = _scheduler(SuccessRunner())
    handle = sched.submit(_req(channel="tg", chat_id="1"))
    await handle.result()
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    sched._sweep(now=lane._idle_since + _DEFAULT_IDLE_TTL + 1)
    assert "tg:1" not in sched._lanes
    handle.cancel()
    assert handle._fut.done()


async def test_resubmit_clears_the_idle_clock_so_a_reused_lane_is_not_reaped():
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    await asyncio.sleep(0)
    lane = sched._lanes["tg:1"]
    assert lane._idle_since is not None
    handle = sched.submit(_req(channel="tg", chat_id="1"))
    assert lane._idle_since is None
    assert sched._sweep(now=1e9) == 0
    assert "tg:1" in sched._lanes
    await handle.result()


class RecordingHangRunner:
    """Records the order turns run in; any turn whose text starts 'hang' blocks
    until cancelled (so an interrupter can preempt it)."""

    def __init__(self):
        self.ran: list[str] = []
        self.hanging_started = asyncio.Event()

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.ran.append(req.text)
        if req.text.startswith("hang"):
            self.hanging_started.set()
            await asyncio.Event().wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


def _capture_logs() -> tuple[list[str], int]:
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="INFO", format="{message}")
    return lines, sink_id


async def test_non_user_interrupt_is_demoted_to_append_with_a_log():
    sched = _scheduler(SuccessRunner())
    lines, sink_id = _capture_logs()
    try:
        assert sched._effective_busy(_req(origin=Origin.CRON, busy=BusyPolicy.INTERRUPT)) is BusyPolicy.APPEND
    finally:
        logger.remove(sink_id)
    demote = next(line for line in lines if "demoted" in line)
    assert "cron" in demote and "interrupt" in demote and "append" in demote


async def test_user_busy_policy_is_not_demoted():
    sched = _scheduler(SuccessRunner())
    lines, sink_id = _capture_logs()
    try:
        assert sched._effective_busy(_req(origin=Origin.USER, busy=BusyPolicy.INTERRUPT)) is BusyPolicy.INTERRUPT
        assert sched._effective_busy(_req(origin=Origin.USER, busy=BusyPolicy.INJECT)) is BusyPolicy.INJECT
        assert sched._effective_busy(_req(origin=Origin.CRON, busy=BusyPolicy.APPEND)) is BusyPolicy.APPEND
    finally:
        logger.remove(sink_id)
    assert not any("demoted" in line for line in lines)


async def test_non_user_interrupt_does_not_preempt_the_running_turn():

    runner = RecordingHangRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    user = sched.submit(_req(channel="tg", chat_id="1", text="hang"))
    await runner.hanging_started.wait()
    sched.submit(_req(channel="tg", chat_id="1", origin=Origin.CRON, busy=BusyPolicy.INTERRUPT, text="cron"))
    await asyncio.sleep(0)
    assert not user._fut.done()


async def test_user_interrupt_preempts_then_runs_before_appended_backlog():
    runner = RecordingHangRunner()
    events: list = []

    async def sink(event) -> None:
        events.append(event)

    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    hang = sched.submit(_req(channel="tg", chat_id="1", text="hang"))
    await runner.hanging_started.wait()
    appended = sched.submit(_req(channel="tg", chat_id="1", text="append"))
    interrupter = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="interrupt"))
    assert await asyncio.wait_for(hang.result(), timeout=1.0) is None
    await asyncio.wait_for(interrupter.result(), timeout=1.0)
    await asyncio.wait_for(appended.result(), timeout=1.0)
    assert runner.ran == ["hang", "interrupt", "append"]
    assert any(isinstance(e, TurnFailed) and e.cancelled for e in events)


async def test_two_interrupts_run_latest_first():
    runner = RecordingHangRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    sched.submit(_req(channel="tg", chat_id="1", text="hang"))
    await runner.hanging_started.wait()
    i1 = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="i1"))
    i2 = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="i2"))
    await asyncio.wait_for(i1.result(), timeout=1.0)
    await asyncio.wait_for(i2.result(), timeout=1.0)
    assert runner.ran == ["hang", "i2", "i1"]


async def test_interrupt_on_an_idle_lane_runs_like_append():
    runner = RecordingHangRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    handle = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="x"))
    assert isinstance(await asyncio.wait_for(handle.result(), timeout=1.0), TurnOutcome)
    assert runner.ran == ["x"]


class MergingRunner:
    """Drains the inject mailbox once at a simulated tool-loop gap and records the
    merged content; the host turn then completes with its outcome."""

    def __init__(self, gate, outcome):
        self.started = asyncio.Event()
        self.merged: list[str] = []
        self._gate = gate
        self._outcome = outcome

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.started.set()
        await self._gate.wait()
        self.merged.extend(r.text for r in drain())
        return self._outcome


class NonDrainingRunner:
    """The minimal-legal runner: never drains, so any inject must fall back."""

    def __init__(self, gate):
        self.started = asyncio.Event()
        self.ran: list[str] = []
        self._gate = gate

    async def run(self, req, emit, drain) -> TurnOutcome:
        self.ran.append(req.text)
        self.started.set()
        await self._gate.wait()
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


async def test_inject_merges_into_host_and_chains_its_outcome():
    gate = asyncio.Event()
    outcome = TurnOutcome(usage=Usage(3, 4, 7), explicit_reply=True)
    runner = MergingRunner(gate, outcome)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    gate.set()
    assert await asyncio.wait_for(host.result(), timeout=1.0) == outcome
    assert await asyncio.wait_for(inject.result(), timeout=1.0) == outcome
    assert runner.merged == ["inject"]


async def test_cancel_on_a_merged_inject_is_noop_and_host_survives():
    drain_gate, finish_gate = asyncio.Event(), asyncio.Event()
    outcome = TurnOutcome(usage=Usage(1, 1, 2), explicit_reply=True)

    class DrainThenWaitRunner:
        def __init__(self):
            self.started = asyncio.Event()
            self.drained = asyncio.Event()

        async def run(self, req, emit, drain) -> TurnOutcome:
            self.started.set()
            await drain_gate.wait()
            drain()
            self.drained.set()
            await finish_gate.wait()
            return outcome

    runner = DrainThenWaitRunner()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    drain_gate.set()
    await runner.drained.wait()
    inject.cancel()
    finish_gate.set()
    assert await asyncio.wait_for(host.result(), timeout=1.0) == outcome
    assert await asyncio.wait_for(inject.result(), timeout=1.0) == outcome


async def test_undrained_inject_falls_back_to_append_with_a_log():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    lines, sink_id = _capture_logs()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    gate.set()
    try:
        assert isinstance(await asyncio.wait_for(inject.result(), timeout=1.0), TurnOutcome)
    finally:
        logger.remove(sink_id)
    assert runner.ran == ["host", "inject"]
    assert any("fell back" in line for line in lines)


async def test_stop_clears_the_inject_mailbox_as_cancelled():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    stopped = sched._lanes["tg:1"].cancel()
    assert stopped == 2
    assert await asyncio.wait_for(inject.result(), timeout=1.0) is None
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None
    assert runner.ran == ["host"]


async def test_cancel_on_a_mailboxed_inject_removes_it_before_merge():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    inject.cancel()
    assert await asyncio.wait_for(inject.result(), timeout=1.0) is None
    assert len(sched._lanes["tg:1"]._inject_mailbox) == 0
    gate.set()
    await asyncio.wait_for(host.result(), timeout=1.0)
    assert runner.ran == ["host"]


async def test_interrupt_runs_before_a_hosts_fallen_back_inject():

    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    interrupter = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INTERRUPT, text="interrupt"))
    gate.set()
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None
    await asyncio.wait_for(interrupter.result(), timeout=1.0)
    await asyncio.wait_for(inject.result(), timeout=1.0)
    assert runner.ran == ["host", "interrupt", "inject"]


async def test_stop_after_drain_resolves_a_chained_inject_exactly_once():

    drain_gate = asyncio.Event()

    class R:
        def __init__(self):
            self.started = asyncio.Event()
            self.drained = asyncio.Event()

        async def run(self, req, emit, drain) -> TurnOutcome:
            self.started.set()
            await drain_gate.wait()
            drain()
            self.drained.set()
            await asyncio.Event().wait()
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    runner = R()
    sched = Scheduler(runner, OriginPools(user=1, system=1), _sink)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inject"))
    drain_gate.set()
    await runner.drained.wait()
    stopped = sched._lanes["tg:1"].cancel()
    assert stopped == 1
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None
    assert await asyncio.wait_for(inject.result(), timeout=1.0) is None


async def test_shutdown_seals_the_scheduler_then_submit_raises():
    sched = _scheduler(SuccessRunner())
    lines, sink_id = _capture_logs()
    await sched.shutdown(grace=1.0)
    try:
        with pytest.raises(SchedulerDrainingError):
            sched.submit(_req())
    finally:
        logger.remove(sink_id)
    assert any("draining" in line for line in lines)


async def test_shutdown_drains_unstarted_work_and_lets_the_running_turn_finish():
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = _scheduler(runner)
    host = sched.submit(_req(channel="tg", chat_id="1", text="host"))
    await runner.started.wait()
    queued = sched.submit(_req(channel="tg", chat_id="1", text="queued"))
    inject = sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text="inj"))
    gate.set()
    await sched.shutdown(grace=1.0)
    assert isinstance(await host.result(), TurnOutcome)
    assert await queued.result() is None
    assert await inject.result() is None
    assert runner.ran == ["host"]


async def test_shutdown_cancels_a_running_turn_that_outlasts_grace():
    runner = HangingRunner()
    events: list = []

    async def sink(event) -> None:
        events.append(event)

    sched = Scheduler(runner, OriginPools(user=1, system=1), sink)
    host = sched.submit(_req(channel="tg", chat_id="1"))
    await runner.started.wait()
    await sched.shutdown(grace=0.05)
    assert await asyncio.wait_for(host.result(), timeout=1.0) is None
    assert any(isinstance(e, TurnFailed) and e.cancelled for e in events)


async def test_shutdown_delays_caller_cancellation_until_cascade_finishes():
    started = asyncio.Event()
    cancellation_swallowed = asyncio.Event()
    release_runner = asyncio.Event()

    class _DelayedCancellationRunner:
        async def run(self, req, emit, drain) -> TurnOutcome:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_swallowed.set()
                await release_runner.wait()
                raise

    sched = _scheduler(_DelayedCancellationRunner())
    host = sched.submit(_req())
    await started.wait()
    shutting_down = asyncio.create_task(sched.shutdown(grace=0.0))
    await cancellation_swallowed.wait()
    shutting_down.cancel()
    await asyncio.sleep(0)

    try:
        assert not shutting_down.done()
        release_runner.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(shutting_down, timeout=0.2)
    finally:
        release_runner.set()
        if not shutting_down.done():
            shutting_down.cancel()
        await asyncio.gather(shutting_down, return_exceptions=True)

    assert shutting_down.cancelled()
    assert await asyncio.wait_for(host.result(), timeout=0.2) is None
    assert not sched.has_pending_or_running("t:c")


async def test_shutdown_preserves_caller_cancellation_over_barrier_failure(monkeypatch):
    barrier_started = asyncio.Event()
    release_barrier = asyncio.Event()
    sched = _scheduler(SuccessRunner())

    async def fail_shutdown(grace: float) -> None:
        barrier_started.set()
        await release_barrier.wait()
        raise RuntimeError("scheduler shutdown failed")

    monkeypatch.setattr(sched, "_finish_shutdown", fail_shutdown)
    shutting_down = asyncio.create_task(sched.shutdown(grace=0.0))
    await barrier_started.wait()
    shutting_down.cancel()
    await asyncio.sleep(0)

    try:
        assert not shutting_down.done()
        release_barrier.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.wait_for(shutting_down, timeout=0.2)
    finally:
        release_barrier.set()
        if not shutting_down.done():
            shutting_down.cancel()
        await asyncio.gather(shutting_down, return_exceptions=True)

    assert shutting_down.cancelled()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "scheduler shutdown failed"


async def test_reaper_lazy_starts_on_first_submit_and_shutdown_cancels_it():
    sched = _scheduler(SuccessRunner())
    assert sched._reaper is None
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    reaper = sched._reaper
    assert reaper is not None and not reaper.done()
    await sched.shutdown(grace=1.0)
    assert reaper.done()


async def test_reap_loop_self_terminates_when_no_lanes():
    sched = _scheduler(SuccessRunner())
    await asyncio.wait_for(sched._reap_loop(), timeout=1.0)


async def test_running_reaper_actually_sweeps_and_reaps_an_idle_lane(monkeypatch):

    monkeypatch.setattr("codeflow.spine.scheduler._SWEEP_INTERVAL", 0.01)
    monkeypatch.setattr("codeflow.spine.scheduler._DEFAULT_IDLE_TTL", 0.0)
    sched = _scheduler(SuccessRunner())
    await sched.submit(_req(channel="tg", chat_id="1")).result()
    for _ in range(100):
        if "tg:1" not in sched._lanes:
            break
        await asyncio.sleep(0.01)
    assert "tg:1" not in sched._lanes


async def test_shutdown_on_an_idle_scheduler_is_a_clean_noop():
    sched = _scheduler(SuccessRunner())
    await sched.shutdown(grace=1.0)
    with pytest.raises(SchedulerDrainingError):
        sched.submit(_req())


async def test_submit_backlog_warns_when_pending_reaches_the_threshold(monkeypatch):
    monkeypatch.setattr("codeflow.spine.scheduler._DEPTH_WARN_THRESHOLD", 3)
    runner = HangingRunner()
    sched = _scheduler(runner)
    lines, sink_id = _capture_logs()
    try:
        sched.submit(_req(channel="tg", chat_id="1", text="host"))
        await runner.started.wait()
        sched.submit(_req(channel="tg", chat_id="1", text="q1"))
        sched.submit(_req(channel="tg", chat_id="1", text="q2"))
        assert not any("depth" in line for line in lines)
        sched.submit(_req(channel="tg", chat_id="1", text="q3"))
    finally:
        logger.remove(sink_id)
    warn = next(line for line in lines if "depth" in line)
    assert "tg:1" in warn and "3" in warn
    await sched.shutdown(grace=0.01)


async def test_inject_fallback_re_enqueue_also_triggers_the_depth_warning(monkeypatch):

    monkeypatch.setattr("codeflow.spine.scheduler._DEPTH_WARN_THRESHOLD", 3)
    gate = asyncio.Event()
    runner = NonDrainingRunner(gate)
    sched = _scheduler(runner)
    lines, sink_id = _capture_logs()
    try:
        sched.submit(_req(channel="tg", chat_id="1", text="host"))
        await runner.started.wait()
        injects = [
            sched.submit(_req(channel="tg", chat_id="1", busy=BusyPolicy.INJECT, text=f"inj{i}")) for i in range(3)
        ]
        gate.set()
        for handle in injects:
            await asyncio.wait_for(handle.result(), timeout=1.0)
    finally:
        logger.remove(sink_id)
    assert any("depth" in line and "3" in line for line in lines)
    await sched.shutdown(grace=0.01)
