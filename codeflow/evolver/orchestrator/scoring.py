"""Bench-neutral scoring contract — the one currency the funnel scores in.

Every consumer in the seven-step funnel (screen, paired gate, sealed ledger)
reads only :attr:`TaskEval.pass_rate`, so a *scorer's* sole job is to turn one
evaluation of a candidate into ``{task_id: TaskEval}``. That makes the whole
benchmark surface collapse to a single value — :class:`EvalBackend` — bundling
the four bench-specific things the orchestrator needs: how to score a task set
(``eval``), the vanilla cold-start baseline over train (``cold_start``), the
screen anchor drawn from train (``anchor``), and the trajectories that feed
diagnosis (``trajectories``). SWE-bench, AppWorld, and the no-benchmark LLM
judge are each one :class:`EvalBackend` instance; nothing above this module
imports a concrete bench.

Train vs. sealed test is first-class here (SOP's iron law, as a mechanism not a
rule): ``EvalBackend`` carries both ``train_task_ids`` (the evolvable pool —
diagnosis, screen, confirm, gate, parent selection all live here) and
``test_task_ids`` (held-out; blind-scored only, never consulted for any
decision). ``eval`` takes an explicit ``split`` so one scorer serves both: the
funnel passes ``split="train"``, the sealed runner ``split="test"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from codeflow.evolver.analysis.stability_bucket import TaskStability
from codeflow.evolver.scheduler.anchor_selection import AnchorSelection

if TYPE_CHECKING:
    from codeflow.evolver.tree.node import HarnessNode


class MeasurementFailure(str, Enum):
    provider = "provider"
    infrastructure = "infrastructure"
    inconclusive = "inconclusive"


class MeasurementStatus(str, Enum):
    measured = "measured"
    failed = "failed"
    inconclusive = "inconclusive"


class EvaluationVerdict(str, Enum):
    accepted = "accepted"
    rejected = "rejected"
    failed = "failed"
    inconclusive = "inconclusive"


@dataclass(frozen=True)
class TaskEval:
    """Per-task K-trial outcome — the universal currency the funnel scores in.

    ``infra_attempts`` is a subset of ``attempts``. A surviving infrastructure
    trial invalidates promotion rather than becoming a low score. ``failure``
    preserves Provider, infrastructure, and inconclusive measurement outcomes.
    """

    task_id: str
    passes: int
    attempts: int
    infra_attempts: int = 0
    failure: MeasurementFailure | None = None

    def __post_init__(self) -> None:
        if isinstance(self.failure, str):
            object.__setattr__(self, "failure", MeasurementFailure(self.failure))
        if self.passes < 0 or self.attempts < 0 or self.infra_attempts < 0:
            raise ValueError("TaskEval counts must be non-negative")
        if self.passes > self.attempts:
            raise ValueError("TaskEval passes cannot exceed attempts")
        if self.infra_attempts > self.attempts:
            raise ValueError("TaskEval infra_attempts cannot exceed attempts")

    @property
    def pass_rate(self) -> float:
        """Fraction of trials that passed; 0.0 when no trial was observed."""
        if self.attempts == 0:
            return 0.0
        return self.passes / self.attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passes": self.passes,
            "attempts": self.attempts,
            "infra_attempts": self.infra_attempts,
            "failure": self.failure.value if self.failure is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskEval":
        return cls(
            task_id=str(data["task_id"]),
            passes=int(data["passes"]),
            attempts=int(data["attempts"]),
            infra_attempts=int(data.get("infra_attempts", 0)),
            failure=data.get("failure"),
        )


@dataclass(frozen=True)
class MeasurementValidity:
    status: MeasurementStatus
    missing: tuple[str, ...] = field(default_factory=tuple)
    provider_failures: tuple[str, ...] = field(default_factory=tuple)
    infrastructure_failures: tuple[str, ...] = field(default_factory=tuple)
    inconclusive: tuple[str, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return self.status is MeasurementStatus.measured

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "missing": list(self.missing),
            "provider_failures": list(self.provider_failures),
            "infrastructure_failures": list(self.infrastructure_failures),
            "inconclusive": list(self.inconclusive),
        }


def measurement_validity(
    evals: dict[str, TaskEval],
    task_ids: list[str],
    *,
    expected_attempts: int,
) -> MeasurementValidity:
    """Classify evidence only after every requested task reaches the expected K."""

    if expected_attempts < 1:
        raise ValueError("expected_attempts must be >= 1")
    missing: list[str] = []
    provider_failures: list[str] = []
    infrastructure_failures: list[str] = []
    inconclusive: list[str] = []
    for task_id in task_ids:
        ev = evals.get(task_id)
        if ev is None:
            missing.append(task_id)
            continue
        if ev.failure is MeasurementFailure.provider:
            provider_failures.append(task_id)
        if ev.failure is MeasurementFailure.infrastructure or ev.infra_attempts > 0:
            infrastructure_failures.append(task_id)
        if ev.failure is MeasurementFailure.inconclusive or ev.attempts < expected_attempts:
            inconclusive.append(task_id)

    if provider_failures or infrastructure_failures:
        status = MeasurementStatus.failed
    elif missing or inconclusive:
        status = MeasurementStatus.inconclusive
    else:
        status = MeasurementStatus.measured
    return MeasurementValidity(
        status=status,
        missing=tuple(missing),
        provider_failures=tuple(provider_failures),
        infrastructure_failures=tuple(infrastructure_failures),
        inconclusive=tuple(inconclusive),
    )


def prefer_rerun_measurement(
    current: TaskEval | None,
    candidate: TaskEval | None,
    *,
    expected_attempts: int,
) -> bool:
    """Return whether one rerun measurement should replace the kept value."""

    if candidate is None:
        return False
    if current is None:
        return True

    def exact_valid(value: TaskEval) -> bool:
        return (
            value.attempts == expected_attempts
            and measurement_validity(
                {value.task_id: value},
                [value.task_id],
                expected_attempts=expected_attempts,
            ).valid
        )

    candidate_valid = exact_valid(candidate)
    current_valid = exact_valid(current)
    if candidate_valid and not current_valid:
        return True

    candidate_infra = candidate.infra_attempts > 0 or candidate.failure is MeasurementFailure.infrastructure
    current_infra = current.infra_attempts > 0 or current.failure is MeasurementFailure.infrastructure
    return candidate_infra and current_infra and candidate.infra_attempts < current.infra_attempts


class EvalFn(Protocol):
    """Score ``node`` on ``task_ids`` at K trials; return per-task outcomes.

    ``split`` selects the dataset partition — ``"train"`` for the funnel's
    screen/confirm, ``"test"`` for the sealed held-out runner. A task that never
    launched is simply absent from the returned mapping (the caller decides
    whether that is an infra failure per Gate-f).
    """

    def __call__(
        self,
        node: "HarnessNode",
        task_ids: list[str],
        k: int,
        job_name: str,
        *,
        split: str = "train",
    ) -> dict[str, TaskEval]: ...


# 诊断使用的轮次级轨迹输入：
# （轮次索引，父节点）-> [(轨迹 ID，任务描述，轨迹文本), ...]
TrajectorySource = Callable[[int, "HarnessNode"], list[tuple[str, str, str]]]

# 评分前的环境健康预检（SOP 第 0 节门控 0，在任何评分之前）。环境脏污（沙箱停机、网络不可
# 路由、验证器无法输出结果）时抛错，避免在损坏环境上评分。该检查因基准而异；每个基准验证
# 自身所需条件，未接入检查时保留为 None。
PrecheckFn = Callable[[], None]


def eval_with_infra_rerun(
    eval_fn: "EvalFn",
    node: "HarnessNode",
    task_ids: list[str],
    k: int,
    job_name: str,
    *,
    split: str = "train",
    max_reruns: int = 2,
) -> dict[str, TaskEval]:
    """Wrap an ``eval_fn`` with the SOP §0 infra rerun ladder (detect -> rerun).

    After the first eval, any task that still carries an infra trial — or that
    was requested but came back with NO measurement at all (never launched /
    result never written: infra by definition) — is re-scored (fresh, at K) up
    to ``max_reruns`` times — re-run once the environment is healthy, topping
    the task back up to K attempts. A complete, clean expected-K measurement
    replaces an invalid one; while both measurements remain infrastructure
    failures, the one with fewer infra trials wins. This salvages recoverable
    transient infra so the task is measured for real; genuinely persistent
    infra survives all reruns with its failure provenance intact, making the
    gate fail closed.
    A bench with no infra signal that returns every requested task triggers no
    rerun, so this is a transparent no-op there.
    """
    evals = dict(eval_fn(node, task_ids, k, job_name, split=split))
    for i in range(1, max_reruns + 1):
        infra_tasks = [
            t
            for t in task_ids
            if (ev := evals.get(t)) is None
            or ev.infra_attempts > 0
            or ev.failure in {MeasurementFailure.infrastructure, MeasurementFailure.inconclusive}
            or (ev.failure is not MeasurementFailure.provider and ev.attempts < k)
        ]
        if not infra_tasks:
            break
        redo = eval_fn(node, infra_tasks, k, f"{job_name}_infra_rerun{i}", split=split)
        for t in infra_tasks:
            new = redo.get(t)
            old = evals.get(t)
            if prefer_rerun_measurement(
                old,
                new,
                expected_attempts=k,
            ):
                evals[t] = new
    return evals


def with_infra_rerun(inner: "EvalFn", max_reruns: int) -> "EvalFn":
    """Wrap an ``EvalFn`` with the SOP §0 infra rerun ladder.

    A no-op when ``max_reruns <= 0`` or the bench emits no infra signal (every
    requested task comes back with ``infra_attempts == 0``), so a bench with no
    infra concept is unaffected. Benches compose this around their raw scorer
    when building an :class:`EvalBackend`.
    """
    if max_reruns <= 0:
        return inner

    def wrapped(node, task_ids, k, job_name, *, split="train"):
        return eval_with_infra_rerun(inner, node, task_ids, k, job_name, split=split, max_reruns=max_reruns)

    return wrapped


def flip_summary(
    candidate_evals: dict[str, TaskEval],
    control_evals: dict[str, TaskEval],
    task_ids: list[str],
    *,
    max_ids: int = 12,
) -> dict:
    """Per-task flip table between a candidate and its control (SOP §2 ①).

    ``rescued`` = tasks whose pass rate ROSE under the candidate (including
    partial, e.g. 1/3 -> 2/3), ``regressed`` = tasks whose rate fell,
    ``still_failing`` = tasks below a perfect rate under the candidate. The
    id lists are capped at ``max_ids`` (counts are always exact) — this feeds
    prompts and ledgers, not statistics; the paired gate owns the stats. A
    task absent from an arm scores 0.0 for that arm, same as the gate.
    """
    rescued: list[str] = []
    regressed: list[str] = []
    still_failing: list[str] = []
    for t in task_ids:
        c_ev, v_ev = candidate_evals.get(t), control_evals.get(t)
        c = c_ev.pass_rate if c_ev is not None else 0.0
        v = v_ev.pass_rate if v_ev is not None else 0.0
        if c > v:
            rescued.append(t)
        elif c < v:
            regressed.append(t)
        if c < 1.0:
            still_failing.append(t)
    return {
        "n_rescued": len(rescued),
        "n_regressed": len(regressed),
        "n_still_failing": len(still_failing),
        "rescued": rescued[:max_ids],
        "regressed": regressed[:max_ids],
        "still_failing": still_failing[:max_ids],
    }


def anchor_mean_pass_rate(evals: dict[str, TaskEval], anchor_task_ids: list[str]) -> float:
    """Mean per-task pass rate over the anchor subset.

    The quantity the screen compares against vanilla's anchor mean. Anchor tasks
    with no observed trial contribute 0.0 (a candidate that failed to even launch
    on an anchor task is not rewarded for the gap).
    """
    if not anchor_task_ids:
        raise ValueError("anchor_mean_pass_rate requires a non-empty anchor list")
    total = 0.0
    for task_id in anchor_task_ids:
        ev = evals.get(task_id)
        total += ev.pass_rate if ev is not None else 0.0
    return total / len(anchor_task_ids)


@dataclass(frozen=True)
class EvalBackend:
    """Everything bench-specific the funnel needs, in one value.

    One instance per benchmark (SWE-bench, AppWorld) or per no-benchmark LLM
    judge. The FSM stays bench-agnostic: it screens/confirms via ``eval`` on
    ``train_task_ids``, seeds the baseline from ``cold_start``, draws the screen
    anchor from ``anchor``, and feeds ``trajectories`` to diagnosis.
    ``test_task_ids`` is scored blind only, by the sealed runner.
    """

    train_task_ids: list[str]
    test_task_ids: list[str]
    eval: EvalFn
    cold_start: Callable[[], dict[str, TaskStability]]
    anchor: Callable[..., AnchorSelection]
    trajectories: Optional[TrajectorySource] = None
    precheck: Optional[PrecheckFn] = None


__all__ = [
    "TaskEval",
    "MeasurementFailure",
    "MeasurementStatus",
    "MeasurementValidity",
    "EvaluationVerdict",
    "EvalFn",
    "TrajectorySource",
    "PrecheckFn",
    "EvalBackend",
    "anchor_mean_pass_rate",
    "eval_with_infra_rerun",
    "flip_summary",
    "measurement_validity",
    "prefer_rerun_measurement",
    "with_infra_rerun",
]
