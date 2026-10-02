from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from loguru import logger

from codeflow.spine import (
    ChatType,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    TurnOutcome,
    TurnRequest,
    Usage,
)

from ...artifacts import ArtifactStore
from ...canonical import canonical_digest, to_primitive
from ...environment import capture_environment_identity
from ...schema import ExperimentRef
from .models import LatencySummary
from .r0 import run_r0_scheduler_track

SCHEDULER_EXPERIMENT_SCHEMA = "codeflow.codeflowbench.runtime-scheduler-experiments.v1"


@dataclass(frozen=True)
class SchedulerExperimentConfig:
    repetitions: int = 7
    fate_repetitions: int = 5
    worker_slots: int = 16
    system_slots: int = 4
    hol_cycles: int = 4
    hol_regular_sessions: int = 63
    hol_hot_turns_per_cycle: int = 8
    foreground_delay_ms: float = 2.0
    hot_delay_ms: float = 10.0
    bulkhead_user_sessions: int = 64
    bulkhead_turns_per_session: int = 4
    background_tasks: int = 20
    background_delay_ms: float = 50.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.repetitions,
            self.fate_repetitions,
            self.worker_slots,
            self.system_slots,
            self.hol_cycles,
            self.hol_regular_sessions,
            self.hol_hot_turns_per_cycle,
            self.bulkhead_user_sessions,
            self.bulkhead_turns_per_session,
            self.background_tasks,
        )
        if any(value < 1 for value in integer_fields):
            raise ValueError("scheduler experiment counts must be positive")
        if self.background_tasks < self.system_slots:
            raise ValueError("background_tasks must saturate the system pool")
        if min(self.foreground_delay_ms, self.hot_delay_ms, self.background_delay_ms) <= 0:
            raise ValueError("scheduler experiment delays must be positive")


@dataclass(frozen=True)
class _Work:
    request: TurnRequest
    delay_ms: float
    measured: bool
    sequence: int


class _Recorder:
    def __init__(self, work: list[_Work]) -> None:
        self._work = {_message_id(item.request): item for item in work}
        self._accepted_ns: dict[str, int] = {}
        self._active_conversations: set[str] = set()
        self._next_sequence: Counter[str] = Counter()
        self._active_by_origin: Counter[Origin] = Counter()
        self._started_system = 0
        self._system_events: dict[int, asyncio.Event] = {}
        self.invocations: Counter[str] = Counter()
        self.queue_wait_ms: list[float] = []
        self.order_violations = 0
        self.concurrent_session_violations = 0
        self.peak_by_origin: Counter[Origin] = Counter()
        self.peak_system_concurrency = 0

    def accept(self, request: TurnRequest) -> None:
        self._accepted_ns[_message_id(request)] = time.perf_counter_ns()

    def wait_for_system_starts(self, count: int) -> asyncio.Event:
        event = self._system_events.setdefault(count, asyncio.Event())
        if self._started_system >= count:
            event.set()
        return event

    async def run(self, request: TurnRequest) -> TurnOutcome:
        message_id = _message_id(request)
        item = self._work[message_id]
        conversation = _conversation(request)
        self.invocations[message_id] += 1
        if item.measured:
            elapsed = (time.perf_counter_ns() - self._accepted_ns[message_id]) / 1_000_000
            self.queue_wait_ms.append(elapsed)
        if conversation in self._active_conversations:
            self.concurrent_session_violations += 1
        if item.sequence != self._next_sequence[conversation]:
            self.order_violations += 1
        self._next_sequence[conversation] += 1
        self._active_conversations.add(conversation)
        self._active_by_origin[request.origin] += 1
        if request.origin is not Origin.USER:
            self._started_system += 1
        self.peak_by_origin[request.origin] = max(
            self.peak_by_origin[request.origin],
            self._active_by_origin[request.origin],
        )
        self.peak_system_concurrency = max(
            self.peak_system_concurrency,
            sum(self._active_by_origin[origin] for origin in (Origin.CRON, Origin.SUBAGENT)),
        )
        for count, event in self._system_events.items():
            if self._started_system >= count:
                event.set()
        try:
            await asyncio.sleep(item.delay_ms / 1_000)
        finally:
            self._active_by_origin[request.origin] -= 1
            self._active_conversations.remove(conversation)
        return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


class _TimedRunner:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    async def run(self, req, emit, drain) -> TurnOutcome:
        return await self._recorder.run(req)


class _SharedOriginPools:
    def __init__(self, slots: int) -> None:
        self._shared = asyncio.Semaphore(slots)

    def for_origin(self, origin: Origin) -> asyncio.Semaphore:
        return self._shared


class _GlobalHandle:
    def __init__(self, future: asyncio.Future[TurnOutcome]) -> None:
        self._future = future

    async def result(self) -> TurnOutcome:
        return await asyncio.shield(self._future)


class _GlobalFifoScheduler:
    def __init__(self, runner: _TimedRunner, slots: int) -> None:
        self._runner = runner
        self._slots = slots
        self._queue: deque[tuple[TurnRequest, asyncio.Future[TurnOutcome]]] = deque()
        self._active_conversations: set[str] = set()
        self._running = 0
        self._changed = asyncio.Event()
        self._dispatcher: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, request: TurnRequest) -> _GlobalHandle:
        future = asyncio.get_running_loop().create_future()
        self._queue.append((request, future))
        self._changed.set()
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._dispatch())
        return _GlobalHandle(future)

    async def close(self) -> None:
        if self._dispatcher is not None:
            await self._dispatcher
        if self._tasks:
            await asyncio.gather(*self._tasks)

    async def _dispatch(self) -> None:
        while self._queue or self._running:
            if self._can_dispatch_head():
                request, future = self._queue.popleft()
                conversation = _conversation(request)
                self._active_conversations.add(conversation)
                self._running += 1
                task = asyncio.create_task(self._run(request, future))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                continue
            self._changed.clear()
            if self._can_dispatch_head() or (not self._queue and not self._running):
                continue
            await self._changed.wait()

    def _can_dispatch_head(self) -> bool:
        if not self._queue or self._running >= self._slots:
            return False
        request, _future = self._queue[0]
        return _conversation(request) not in self._active_conversations

    async def _run(
        self,
        request: TurnRequest,
        future: asyncio.Future[TurnOutcome],
    ) -> None:
        conversation = _conversation(request)
        try:
            outcome = await self._runner.run(request, _noop_emit, lambda: [])
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        else:
            if not future.done():
                future.set_result(outcome)
        finally:
            self._active_conversations.remove(conversation)
            self._running -= 1
            self._changed.set()


async def _noop_emit(event: object) -> None:
    return None


async def run_head_of_line_experiment(config: SchedulerExperimentConfig) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for repetition in range(config.repetitions):
        arm_order = ("global_fifo", "session_lanes") if repetition % 2 == 0 else ("session_lanes", "global_fifo")
        arms: dict[str, dict[str, Any]] = {}
        for arm in arm_order:
            work = _head_of_line_work(config, repetition)
            arms[arm] = await _run_arm(
                work,
                scheduler_kind=arm,
                user_slots=config.worker_slots,
                system_slots=config.system_slots,
            )
        global_p95 = _required_p95(arms["global_fifo"])
        lane_p95 = _required_p95(arms["session_lanes"])
        pairs.append(
            {
                "repetition": repetition,
                "arm_order": list(arm_order),
                "global_fifo": arms["global_fifo"],
                "session_lanes": arms["session_lanes"],
                "p95_reduction_percent": (global_p95 - lane_p95) / global_p95 * 100,
            }
        )
    reduction = median(pair["p95_reduction_percent"] for pair in pairs)
    return {
        "experiment": "head_of_line_isolation",
        "control": "strict_global_fifo_with_session_serialization",
        "treatment": "per_session_lanes_with_cross_session_parallelism",
        "primary_metric": "foreground_queue_wait_p95_ms",
        "aggregation": "median_of_paired_repetition_reductions",
        "repetitions": pairs,
        "summary": {
            "foreground_p95_reduction_percent": reduction,
            "control_p95_ms": median(_required_p95(pair["global_fifo"]) for pair in pairs),
            "treatment_p95_ms": median(_required_p95(pair["session_lanes"]) for pair in pairs),
            "correctness_passed": all(
                _arm_correct(pair[arm]) for pair in pairs for arm in ("global_fifo", "session_lanes")
            ),
        },
    }


async def run_bulkhead_experiment(config: SchedulerExperimentConfig) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for repetition in range(config.repetitions):
        policy_order = ("shared_pool", "isolated_pools") if repetition % 2 == 0 else ("isolated_pools", "shared_pool")
        policies: dict[str, dict[str, Any]] = {}
        for policy in policy_order:
            cases: dict[str, dict[str, Any]] = {}
            case_order = ("idle", "loaded") if repetition % 2 == 0 else ("loaded", "idle")
            for load in case_order:
                work = _bulkhead_work(config, repetition, loaded=load == "loaded")
                cases[load] = await _run_arm(
                    work,
                    scheduler_kind=policy,
                    user_slots=config.worker_slots,
                    system_slots=config.system_slots,
                    wait_for_background=(
                        config.background_tasks
                        if policy == "shared_pool" and load == "loaded"
                        else config.system_slots
                        if load == "loaded"
                        else 0
                    ),
                )
            idle_p95 = _required_p95(cases["idle"])
            loaded_p95 = _required_p95(cases["loaded"])
            policies[policy] = {
                "idle": cases["idle"],
                "loaded": cases["loaded"],
                "slowdown": loaded_p95 / idle_p95,
            }
        pairs.append(
            {
                "repetition": repetition,
                "policy_order": list(policy_order),
                **policies,
            }
        )
    shared_slowdown = median(pair["shared_pool"]["slowdown"] for pair in pairs)
    isolated_slowdown = median(pair["isolated_pools"]["slowdown"] for pair in pairs)
    return {
        "experiment": "foreground_background_bulkhead",
        "control": "one_shared_pool_with_equal_total_capacity",
        "treatment": "independent_user_and_runtime_origin_pools",
        "primary_metric": "loaded_to_idle_foreground_queue_wait_p95_ratio",
        "aggregation": "median_of_paired_repetition_slowdown_ratios",
        "repetitions": pairs,
        "summary": {
            "shared_pool_slowdown": shared_slowdown,
            "isolated_pools_slowdown": isolated_slowdown,
            "correctness_passed": all(
                _arm_correct(pair[policy][load])
                for pair in pairs
                for policy in ("shared_pool", "isolated_pools")
                for load in ("idle", "loaded")
            ),
        },
    }


async def run_request_fate_experiment(config: SchedulerExperimentConfig) -> dict[str, Any]:
    repetitions: list[dict[str, Any]] = []
    for repetition in range(config.fate_repetitions):
        result = await run_r0_scheduler_track()
        repetitions.append(
            {
                "repetition": repetition,
                "passed": result.passed,
                "metrics": to_primitive(result),
            }
        )
    summed = {
        "accepted_requests": sum(item["metrics"]["accepted_requests"] for item in repetitions),
        "lost_requests": sum(item["metrics"]["lost_requests"] for item in repetitions),
        "unexpected_duplicate_executions": sum(
            item["metrics"]["unexpected_duplicate_executions"] for item in repetitions
        ),
        "unresolved_handles": sum(item["metrics"]["unresolved_handles"] for item in repetitions),
        "lifecycle_contradictions": sum(item["metrics"]["lifecycle_contradictions"] for item in repetitions),
        "pool_limit_violations": sum(item["metrics"]["pool_limit_violations"] for item in repetitions),
    }
    return {
        "experiment": "accepted_request_fate_accounting",
        "scope": "in_process_scheduler_acceptance_to_terminal_handle_resolution",
        "repetitions": repetitions,
        "summary": {
            **summed,
            "correctness_passed": all(item["passed"] for item in repetitions)
            and all(value == 0 for key, value in summed.items() if key != "accepted_requests"),
        },
    }


async def run_scheduler_experiment_suite(config: SchedulerExperimentConfig) -> dict[str, Any]:
    head_of_line = await run_head_of_line_experiment(config)
    bulkhead = await run_bulkhead_experiment(config)
    request_fate = await run_request_fate_experiment(config)
    performance_direction_passed = (
        head_of_line["summary"]["foreground_p95_reduction_percent"] > 0
        and bulkhead["summary"]["shared_pool_slowdown"] > bulkhead["summary"]["isolated_pools_slowdown"]
    )
    gates = {
        "head_of_line_correctness": head_of_line["summary"]["correctness_passed"],
        "bulkhead_correctness": bulkhead["summary"]["correctness_passed"],
        "request_fate_correctness": request_fate["summary"]["correctness_passed"],
        "performance_direction": performance_direction_passed,
    }
    return {
        "schema": SCHEDULER_EXPERIMENT_SCHEMA,
        "config": to_primitive(config),
        "experiments": {
            "head_of_line": head_of_line,
            "bulkhead": bulkhead,
            "request_fate": request_fate,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "resume_metrics": {
            "foreground_p95_reduction_percent": head_of_line["summary"]["foreground_p95_reduction_percent"],
            "shared_pool_slowdown": bulkhead["summary"]["shared_pool_slowdown"],
            "isolated_pools_slowdown": bulkhead["summary"]["isolated_pools_slowdown"],
            "accepted_requests": request_fate["summary"]["accepted_requests"],
            "lost_requests": request_fate["summary"]["lost_requests"],
            "unexpected_duplicate_executions": request_fate["summary"]["unexpected_duplicate_executions"],
            "unresolved_handles": request_fate["summary"]["unresolved_handles"],
        },
    }


async def run_and_write_scheduler_experiments(
    config: SchedulerExperimentConfig,
    *,
    repository_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    source_commit = _git(repository_root, "rev-parse", "HEAD")
    clean_before = _worktree_clean(repository_root)
    environment = capture_environment_identity(repository_root)
    plan = {
        "schema": SCHEDULER_EXPERIMENT_SCHEMA,
        "source_commit": source_commit,
        "environment": environment,
        "config": to_primitive(config),
        "aggregation": {
            "percentiles": "nearest_rank",
            "paired_summary": "median",
        },
        "claim_gates": {
            "foreground_p95_reduction_percent_greater_than": 0,
            "shared_pool_slowdown_greater_than_isolated": True,
            "correctness_counters_equal_zero": True,
        },
    }
    plan_digest = canonical_digest(plan)
    suite = await run_scheduler_experiment_suite(config)
    final_commit = _git(repository_root, "rev-parse", "HEAD")
    clean_after = _worktree_clean(repository_root)
    final_environment = capture_environment_identity(repository_root)
    stable_checkout = source_commit == final_commit and clean_before == clean_after
    stable_environment = environment == final_environment
    evidence_payload = {
        **suite,
        "plan_digest": plan_digest,
        "source_commit": source_commit,
        "worktree_clean": clean_before and clean_after,
        "stable_checkout": stable_checkout,
        "stable_environment": stable_environment,
        "environment": environment,
        "evidence_scope": "deterministic_local_scheduler_comparison",
        "claim_eligible": (suite["passed"] and clean_before and clean_after and stable_checkout and stable_environment),
    }
    evidence_digest = canonical_digest(evidence_payload)
    evidence = {**evidence_payload, "evidence_digest": evidence_digest}
    root = output_root / source_commit / plan_digest
    store = ArtifactStore(ExperimentRef(experiment_id=plan_digest, root=root))
    with store.exclusive_run_lock():
        store.freeze_manifest(plan)
        evidence_path = root / "runs" / f"scheduler-experiments.{evidence_digest}.json"
        store.append_immutable(evidence_path, evidence)
    return evidence, evidence_path


async def _run_arm(
    work: list[_Work],
    *,
    scheduler_kind: str,
    user_slots: int,
    system_slots: int,
    wait_for_background: int = 0,
) -> dict[str, Any]:
    recorder = _Recorder(work)
    runner = _TimedRunner(recorder)
    if scheduler_kind == "global_fifo":
        scheduler: Any = _GlobalFifoScheduler(runner, user_slots)
    elif scheduler_kind == "shared_pool":
        scheduler = Scheduler(runner, _SharedOriginPools(user_slots + system_slots), _noop_emit)
    elif scheduler_kind in {"session_lanes", "isolated_pools"}:
        scheduler = Scheduler(runner, OriginPools(user=user_slots, system=system_slots), _noop_emit)
    else:
        raise ValueError(f"unknown scheduler kind: {scheduler_kind}")

    started_ns = time.perf_counter_ns()
    background = [item for item in work if not item.measured]
    foreground = [item for item in work if item.measured]
    handles: list[Any] = []
    if wait_for_background:
        for item in background:
            recorder.accept(item.request)
            handles.append(scheduler.submit(item.request))
        event = recorder.wait_for_system_starts(wait_for_background)
        await asyncio.wait_for(event.wait(), timeout=5)
        submission_order = foreground
    else:
        submission_order = work
    for item in submission_order:
        recorder.accept(item.request)
        handles.append(scheduler.submit(item.request))
    outcomes = await asyncio.wait_for(
        asyncio.gather(*(handle.result() for handle in handles)),
        timeout=30,
    )
    if isinstance(scheduler, _GlobalFifoScheduler):
        await scheduler.close()
    else:
        await scheduler.shutdown(grace=0)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    duplicate_executions = sum(max(0, count - 1) for count in recorder.invocations.values())
    missing_executions = len(work) - len(recorder.invocations)
    latency = LatencySummary.from_values(recorder.queue_wait_ms)
    return {
        "accepted_requests": len(work),
        "measured_foreground_requests": len(foreground),
        "elapsed_ms": elapsed_ms,
        "throughput_requests_per_second": len(work) / (elapsed_ms / 1_000),
        "queue_wait_ms": to_primitive(latency),
        "missing_executions": missing_executions,
        "unexpected_duplicate_executions": duplicate_executions,
        "unresolved_handles": sum(not handle._future.done() for handle in handles)
        if scheduler_kind == "global_fifo"
        else 0,
        "none_outcomes": sum(outcome is None for outcome in outcomes),
        "order_violations": recorder.order_violations,
        "concurrent_session_violations": recorder.concurrent_session_violations,
        "peak_user_concurrency": recorder.peak_by_origin[Origin.USER],
        "peak_system_concurrency": recorder.peak_system_concurrency,
    }


def _head_of_line_work(config: SchedulerExperimentConfig, repetition: int) -> list[_Work]:
    work: list[_Work] = []
    sequences: Counter[str] = Counter()
    for cycle in range(config.hol_cycles):
        for hot_index in range(config.hol_hot_turns_per_cycle):
            work.append(
                _make_work(
                    prefix=f"hol-{repetition}",
                    conversation="hot",
                    item=f"{cycle}-hot-{hot_index}",
                    delay_ms=config.hot_delay_ms,
                    measured=False,
                    origin=Origin.USER,
                    sequences=sequences,
                )
            )
        for session in range(config.hol_regular_sessions):
            work.append(
                _make_work(
                    prefix=f"hol-{repetition}",
                    conversation=f"foreground-{session:02d}",
                    item=f"{cycle}-foreground",
                    delay_ms=config.foreground_delay_ms,
                    measured=True,
                    origin=Origin.USER,
                    sequences=sequences,
                )
            )
    return work


def _bulkhead_work(
    config: SchedulerExperimentConfig,
    repetition: int,
    *,
    loaded: bool,
) -> list[_Work]:
    work: list[_Work] = []
    sequences: Counter[str] = Counter()
    if loaded:
        for index in range(config.background_tasks):
            work.append(
                _make_work(
                    prefix=f"bulk-{repetition}",
                    conversation=f"background-{index:02d}",
                    item="runtime",
                    delay_ms=config.background_delay_ms,
                    measured=False,
                    origin=Origin.CRON if index % 2 == 0 else Origin.SUBAGENT,
                    sequences=sequences,
                )
            )
    for turn in range(config.bulkhead_turns_per_session):
        for session in range(config.bulkhead_user_sessions):
            work.append(
                _make_work(
                    prefix=f"bulk-{repetition}",
                    conversation=f"user-{session:02d}",
                    item=f"turn-{turn}",
                    delay_ms=config.foreground_delay_ms,
                    measured=True,
                    origin=Origin.USER,
                    sequences=sequences,
                )
            )
    return work


def _make_work(
    *,
    prefix: str,
    conversation: str,
    item: str,
    delay_ms: float,
    measured: bool,
    origin: Origin,
    sequences: Counter[str],
) -> _Work:
    conversation_id = f"codeflowbench-scheduler:{conversation}"
    message_id = f"{prefix}:{conversation}:{item}"
    sequence = sequences[conversation_id]
    sequences[conversation_id] += 1
    return _Work(
        request=TurnRequest(
            origin=origin,
            source=Source(
                channel="codeflowbench-scheduler",
                chat_id=conversation,
                sender_id="codeflowbench",
                chat_type=ChatType.DM,
            ),
            text=message_id,
            message_id=message_id,
            conversation=conversation_id,
        ),
        delay_ms=delay_ms,
        measured=measured,
        sequence=sequence,
    )


def _arm_correct(arm: dict[str, Any]) -> bool:
    return all(
        arm[key] == 0
        for key in (
            "missing_executions",
            "unexpected_duplicate_executions",
            "unresolved_handles",
            "none_outcomes",
            "order_violations",
            "concurrent_session_violations",
        )
    )


def _required_p95(arm: dict[str, Any]) -> float:
    value = arm["queue_wait_ms"]["p95"]
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError("experiment arm did not produce a positive P95")
    return float(value)


def _message_id(request: TurnRequest) -> str:
    if request.message_id is None:
        raise ValueError("scheduler experiment requests require message_id")
    return request.message_id


def _conversation(request: TurnRequest) -> str:
    return request.conversation or f"{request.source.channel}:{request.source.chat_id}"


def _git(repository_root: Path, *args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to bind scheduler experiment evidence")
    completed = subprocess.run(
        [git, *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _worktree_clean(repository_root: Path) -> bool:
    return not _git(repository_root, "status", "--short")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Runtime scheduler A/B experiments")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".codeflow/evidence/codeflowbench-runtime-scheduler"),
    )
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--fate-repetitions", type=int, default=5)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[4]
    config = SchedulerExperimentConfig(
        repetitions=args.repetitions,
        fate_repetitions=args.fate_repetitions,
    )
    logger.disable("codeflow.spine.scheduler")
    try:
        evidence, path = await run_and_write_scheduler_experiments(
            config,
            repository_root=repository_root,
            output_root=args.output_root,
        )
    finally:
        logger.enable("codeflow.spine.scheduler")
    metrics = evidence["resume_metrics"]
    print(f"evidence: {path}")
    print(f"claim_eligible: {str(evidence['claim_eligible']).lower()}")
    print(f"foreground_p95_reduction_percent: {metrics['foreground_p95_reduction_percent']:.2f}")
    print(f"shared_pool_slowdown: {metrics['shared_pool_slowdown']:.2f}")
    print(f"isolated_pools_slowdown: {metrics['isolated_pools_slowdown']:.2f}")
    print(f"accepted_requests: {metrics['accepted_requests']}")
    print(f"lost_requests: {metrics['lost_requests']}")
    print(f"unexpected_duplicate_executions: {metrics['unexpected_duplicate_executions']}")
    print(f"unresolved_handles: {metrics['unresolved_handles']}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
