from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field

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
    TurnStarted,
    Usage,
)
from codeflow.spine.scheduler import SchedulerDrainingError

from .models import LatencySummary, R0RuntimeResult, RequestFate

R0_ACCEPTED_REQUESTS = 2_000
R0_REJECTION_PROBES = 64
R0_CONVERSATIONS = 64
R0_USER_SLOTS = 16
R0_SYSTEM_SLOTS = 4

_DISPATCH_PROBES = 16
_USER_SATURATION = 16
_SYSTEM_SATURATION = 4
_SPECIAL_CONVERSATIONS = 16
_SPECIAL_REQUESTS = _SPECIAL_CONVERSATIONS * 8
_SHUTDOWN_REQUESTS = 64
_BULK_REQUESTS = (
    R0_ACCEPTED_REQUESTS
    - _DISPATCH_PROBES
    - _USER_SATURATION
    - _SYSTEM_SATURATION
    - _SPECIAL_REQUESTS
    - _SHUTDOWN_REQUESTS
)


@dataclass
class _Control:
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event | None = None
    drain_ready: asyncio.Event | None = None
    latency_kind: str = "queue"


class _R0Recorder:
    def __init__(self) -> None:
        self.submitted_ns: dict[str, int] = {}
        self.requests: dict[str, TurnRequest] = {}
        self.injected_while_running: set[str] = set()
        self.fates: dict[str, RequestFate] = {}
        self.duplicate_fates = 0
        self.invocations: Counter[str] = Counter()
        self.dispatch_overhead_ms: list[float] = []
        self.queue_wait_ms: list[float] = []
        self.execution_latency_ms: list[float] = []
        self.active_user = 0
        self.active_system = 0
        self.peak_user = 0
        self.peak_system = 0
        self.pool_limit_violations = 0
        self.user_saturated = asyncio.Event()
        self.system_saturated = asyncio.Event()
        self.lifecycle_active: set[str] = set()
        self.lifecycle_contradictions = 0
        self.outstanding_by_conversation: Counter[str] = Counter()
        self.max_pending_depth = 0

    def accept(self, req: TurnRequest, *, injected_while_running: bool = False) -> None:
        message_id = _message_id(req)
        self.submitted_ns[message_id] = time.perf_counter_ns()
        self.requests[message_id] = req
        if injected_while_running:
            self.injected_while_running.add(message_id)
        conversation = _conversation(req)
        self.outstanding_by_conversation[conversation] += 1
        pending_depth = max(0, self.outstanding_by_conversation[conversation] - 1)
        self.max_pending_depth = max(self.max_pending_depth, pending_depth)

    def enter(self, req: TurnRequest, latency_kind: str) -> int:
        message_id = _message_id(req)
        self.invocations[message_id] += 1
        elapsed_ms = (time.perf_counter_ns() - self.submitted_ns[message_id]) / 1_000_000
        if latency_kind == "dispatch":
            self.dispatch_overhead_ms.append(elapsed_ms)
        else:
            self.queue_wait_ms.append(elapsed_ms)
        if req.origin is Origin.USER:
            self.active_user += 1
            self.peak_user = max(self.peak_user, self.active_user)
            if self.active_user == R0_USER_SLOTS:
                self.user_saturated.set()
            if self.active_user > R0_USER_SLOTS:
                self.pool_limit_violations += 1
        else:
            self.active_system += 1
            self.peak_system = max(self.peak_system, self.active_system)
            if self.active_system == R0_SYSTEM_SLOTS:
                self.system_saturated.set()
            if self.active_system > R0_SYSTEM_SLOTS:
                self.pool_limit_violations += 1
        return time.perf_counter_ns()

    def leave(self, req: TurnRequest, started_ns: int) -> None:
        self.execution_latency_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)
        if req.origin is Origin.USER:
            self.active_user -= 1
        else:
            self.active_system -= 1

    def fate(self, message_id: str, fate: RequestFate) -> None:
        if message_id in self.fates:
            self.duplicate_fates += 1
            return
        self.fates[message_id] = fate
        conversation = _conversation(self.requests[message_id])
        self.outstanding_by_conversation[conversation] -= 1

    async def lifecycle(self, event: object) -> None:
        conversation = getattr(event, "conversation_id", None)
        if conversation is None:
            return
        if isinstance(event, TurnStarted):
            if conversation in self.lifecycle_active:
                self.lifecycle_contradictions += 1
            self.lifecycle_active.add(conversation)
        elif isinstance(event, (TurnEnded, TurnFailed)):
            if conversation not in self.lifecycle_active:
                self.lifecycle_contradictions += 1
            else:
                self.lifecycle_active.remove(conversation)


class _ScriptedTurnRunner:
    def __init__(self, recorder: _R0Recorder, controls: dict[str, _Control]) -> None:
        self._recorder = recorder
        self._controls = controls

    async def run(self, req: TurnRequest, emit, drain) -> TurnOutcome:
        message_id = _message_id(req)
        control = self._controls.get(message_id, _Control())
        started_ns = self._recorder.enter(req, control.latency_kind)
        control.entered.set()
        try:
            if control.drain_ready is not None:
                await control.drain_ready.wait()
                for injected in drain():
                    self._recorder.fate(
                        _message_id(injected),
                        RequestFate.MERGED_INTO_RUNNING_TURN,
                    )
            if control.release is not None:
                await control.release.wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._recorder.fate(message_id, RequestFate.CANCELLED_WHILE_RUNNING)
            raise
        else:
            fate = (
                RequestFate.FALLBACK_EXECUTED
                if message_id in self._recorder.injected_while_running
                else RequestFate.EXECUTED
            )
            self._recorder.fate(message_id, fate)
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)
        finally:
            self._recorder.leave(req, started_ns)


async def run_r0_scheduler_track() -> R0RuntimeResult:
    recorder = _R0Recorder()
    controls: dict[str, _Control] = {}
    runner = _ScriptedTurnRunner(recorder, controls)
    scheduler = Scheduler(
        runner,
        OriginPools(user=R0_USER_SLOTS, system=R0_SYSTEM_SLOTS),
        recorder.lifecycle,
    )
    handles: dict[str, object] = {}
    results: dict[str, TurnOutcome | None] = {}
    next_id = 0

    def submit(
        *,
        conversation_index: int,
        origin: Origin = Origin.USER,
        busy: BusyPolicy = BusyPolicy.APPEND,
        control: _Control | None = None,
        injected_while_running: bool = False,
    ):
        nonlocal next_id
        message_id = f"r0-{next_id:04d}"
        next_id += 1
        req = _request(
            message_id,
            conversation_index=conversation_index,
            origin=origin,
            busy=busy,
        )
        if control is not None:
            controls[message_id] = control
        recorder.accept(req, injected_while_running=injected_while_running)
        handle = scheduler.submit(req)
        handles[message_id] = handle
        return message_id, handle

    for index in range(_DISPATCH_PROBES):
        control = _Control(latency_kind="dispatch")
        message_id, handle = submit(conversation_index=index, control=control)
        results[message_id] = await handle.result()

    user_release = asyncio.Event()
    user_batch = [
        submit(
            conversation_index=index,
            control=_Control(release=user_release),
        )
        for index in range(_USER_SATURATION)
    ]
    await asyncio.wait_for(recorder.user_saturated.wait(), timeout=5)
    user_release.set()
    await _collect(user_batch, results, recorder)

    system_release = asyncio.Event()
    system_batch = [
        submit(
            conversation_index=index,
            origin=Origin.CRON if index % 2 == 0 else Origin.SUBAGENT,
            control=_Control(release=system_release),
        )
        for index in range(_SYSTEM_SATURATION)
    ]
    await asyncio.wait_for(recorder.system_saturated.wait(), timeout=5)
    system_release.set()
    await _collect(system_batch, results, recorder)

    for index in range(_SPECIAL_CONVERSATIONS):
        merge_control = _Control(drain_ready=asyncio.Event())
        merge_id, merge_handle = submit(conversation_index=index, control=merge_control)
        await merge_control.entered.wait()
        inject_id, inject_handle = submit(
            conversation_index=index,
            busy=BusyPolicy.INJECT,
            injected_while_running=True,
        )
        merge_control.drain_ready.set()
        await _collect(
            [(merge_id, merge_handle), (inject_id, inject_handle)],
            results,
            recorder,
        )

        fallback_release = asyncio.Event()
        fallback_control = _Control(release=fallback_release)
        host_id, host_handle = submit(conversation_index=index, control=fallback_control)
        await fallback_control.entered.wait()
        fallback_id, fallback_handle = submit(
            conversation_index=index,
            busy=BusyPolicy.INJECT,
            injected_while_running=True,
        )
        fallback_release.set()
        await _collect(
            [(host_id, host_handle), (fallback_id, fallback_handle)],
            results,
            recorder,
        )

        interrupt_release = asyncio.Event()
        interrupt_control = _Control(release=interrupt_release)
        interrupted_id, interrupted_handle = submit(
            conversation_index=index,
            control=interrupt_control,
        )
        await interrupt_control.entered.wait()
        interrupter_id, interrupter_handle = submit(
            conversation_index=index,
            busy=BusyPolicy.INTERRUPT,
        )
        await _collect(
            [
                (interrupted_id, interrupted_handle),
                (interrupter_id, interrupter_handle),
            ],
            results,
            recorder,
        )

        queue_release = asyncio.Event()
        queue_control = _Control(release=queue_release)
        running_id, running_handle = submit(
            conversation_index=index,
            control=queue_control,
        )
        await queue_control.entered.wait()
        cancelled_id, cancelled_handle = submit(conversation_index=index)
        cancelled_handle.cancel()
        queue_release.set()
        await _collect(
            [(running_id, running_handle), (cancelled_id, cancelled_handle)],
            results,
            recorder,
        )

    bulk = [
        submit(
            conversation_index=index % R0_CONVERSATIONS,
            origin=(Origin.USER if index % 5 else (Origin.CRON if index % 2 == 0 else Origin.SUBAGENT)),
        )
        for index in range(_BULK_REQUESTS)
    ]
    await _collect(bulk, results, recorder)

    for index in range(_SHUTDOWN_REQUESTS // 2):
        control = _Control(release=asyncio.Event())
        submit(conversation_index=index, control=control)
    for index in range(_SHUTDOWN_REQUESTS // 2):
        submit(conversation_index=index)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await scheduler.shutdown(grace=0)
    await _collect(
        list(handles.items())[R0_ACCEPTED_REQUESTS - _SHUTDOWN_REQUESTS :],
        results,
        recorder,
    )

    for message_id, outcome in results.items():
        if message_id not in recorder.fates:
            if outcome is None:
                recorder.fate(message_id, RequestFate.CANCELLED_BEFORE_START)

    rejected_without_handle = 0
    for index in range(R0_REJECTION_PROBES):
        req = _request(
            f"r0-rejected-{index:02d}",
            conversation_index=index,
        )
        try:
            scheduler.submit(req)
        except SchedulerDrainingError:
            rejected_without_handle += 1

    unresolved = len(handles) - len(results)
    lost = len(set(handles) - set(recorder.fates))
    duplicate_executions = sum(max(0, count - 1) for count in recorder.invocations.values()) + recorder.duplicate_fates
    lifecycle_contradictions = recorder.lifecycle_contradictions + len(recorder.lifecycle_active)
    fate_counts = Counter(fate.value for fate in recorder.fates.values())
    shutdown_ids = list(handles)[-_SHUTDOWN_REQUESTS:]
    shutdown_cancelled = sum(
        recorder.fates.get(message_id)
        in {
            RequestFate.CANCELLED_BEFORE_START,
            RequestFate.CANCELLED_WHILE_RUNNING,
        }
        for message_id in shutdown_ids
    )

    return R0RuntimeResult(
        accepted_requests=len(handles),
        rejected_requests=R0_REJECTION_PROBES,
        rejected_without_handle=rejected_without_handle,
        conversation_count=len({_conversation(req) for req in recorder.requests.values()}),
        fate_counts={fate.value: fate_counts.get(fate.value, 0) for fate in RequestFate},
        runner_invocations=sum(recorder.invocations.values()),
        lost_requests=lost,
        unresolved_handles=unresolved,
        unexpected_duplicate_executions=duplicate_executions,
        lifecycle_contradictions=lifecycle_contradictions,
        pool_limit_violations=recorder.pool_limit_violations,
        peak_user_concurrency=recorder.peak_user,
        peak_system_concurrency=recorder.peak_system,
        shutdown_cancelled_requests=shutdown_cancelled,
        max_observed_pending_depth=recorder.max_pending_depth,
        dispatch_overhead_ms=LatencySummary.from_values(
            recorder.dispatch_overhead_ms,
        ),
        queue_wait_ms=LatencySummary.from_values(recorder.queue_wait_ms),
        execution_latency_ms=LatencySummary.from_values(
            recorder.execution_latency_ms,
        ),
    )


async def _collect(
    batch,
    results: dict[str, TurnOutcome | None],
    recorder: _R0Recorder,
) -> None:
    if not batch:
        return
    values = await asyncio.wait_for(
        asyncio.gather(*(handle.result() for _message_id, handle in batch)),
        timeout=30,
    )
    for (message_id, _handle), result in zip(batch, values, strict=True):
        results[message_id] = result
        if result is None and message_id not in recorder.fates:
            recorder.fate(message_id, RequestFate.CANCELLED_BEFORE_START)


def _request(
    message_id: str,
    *,
    conversation_index: int,
    origin: Origin = Origin.USER,
    busy: BusyPolicy = BusyPolicy.APPEND,
) -> TurnRequest:
    return TurnRequest(
        origin=origin,
        source=Source(
            channel="codeflowbench-r0",
            chat_id=f"conversation-{conversation_index:02d}",
            sender_id="codeflowbench",
            chat_type=ChatType.DM,
        ),
        text=message_id,
        message_id=message_id,
        conversation=f"codeflowbench-r0:conversation-{conversation_index:02d}",
        busy=busy,
    )


def _message_id(req: TurnRequest) -> str:
    if req.message_id is None:
        raise ValueError("R0 requests require message_id")
    return req.message_id


def _conversation(req: TurnRequest) -> str:
    return req.conversation or f"{req.source.channel}:{req.source.chat_id}"
