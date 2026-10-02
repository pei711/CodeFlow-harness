"""为 supported Candidate Label 绑定可执行 fixture 与 evaluator。

Candidate manifest 只声明 label/fixture/evaluator 名称，本模块把这些声明连接到实际验证代码。
Runtime label 的 fixture 检查 mutable surface、before/after content change 与 manifest digest；
evaluator 从 candidate/control per-task measurement 重算 three-shield gate、eligible attribution、
full-train lift 与 score。accepted 决策不能由 caller 布尔值直接声明，必须由 canonical
``AcceptedRuntimeEvidence`` 重现。

fixture 通过只证明 patch snapshot 合规；gate 重算 accepted 才能证明当前 train evidence 满足
promotion policy；两者都不等于 candidate 已 activated、sealed generalisation 为正或业务任务
已交付。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeflow.evolver.activation.artifacts import EvidenceDecision, EvidenceOutcome
from codeflow.evolver.candidate_manifest import LABEL_POLICIES, CandidateLabel, CandidateManifest
from codeflow.evolver.orchestrator.gates.fisher import train_mean
from codeflow.evolver.orchestrator.gates.pipeline import GateResult, run_gates
from codeflow.evolver.orchestrator.gates.policy import CandidateOutcome
from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementFailure,
    MeasurementStatus,
    TaskEval,
)
from codeflow.evolver.tree.node import NodeStatus

if TYPE_CHECKING:
    from codeflow.evolver.orchestrator.scoring import MeasurementValidity


class CandidateEvidenceError(ValueError):
    """declared evidence 无法由 bound evaluator 重现时抛出的异常。

    该异常表示 evidence contract 失败，不应降级为普通 candidate rejection 后仍允许 accepted。
    """


Snapshot = Mapping[str, bytes | None]
FixtureValidator = Callable[[CandidateManifest, Snapshot, Snapshot], None]
OutcomeEvaluator = Callable[..., Any]

_ACCEPTED_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "evaluator",
        "task_ids",
        "expected_attempts",
        "eligible_tasks",
        "candidate_evals",
        "control_evals",
    }
)
_TASK_EVAL_KEYS = frozenset(
    {
        "task_id",
        "passes",
        "attempts",
        "infra_attempts",
        "failure",
    }
)


@dataclass(frozen=True)
class FixtureBinding:
    """fixture 名称与 snapshot validator 的不可变绑定。"""

    name: str
    validate: FixtureValidator


@dataclass(frozen=True)
class EvaluatorBinding:
    """evaluator 名称与 outcome recomputation callable 的不可变绑定。"""

    name: str
    evaluate: OutcomeEvaluator


@dataclass(frozen=True)
class AcceptedRuntimeEvidence:
    """可重建 accepted Runtime verdict 的 canonical measurements。

    对象保存 schema/evaluator、canonical train task order、expected attempts、Gate-b eligible
    subset，以及 candidate/control 的完整 ``TaskEval`` sequence。它是 measurement payload，
    不直接存 ``accepted=True``；decision 必须通过 ``recompute_accepted_runtime_evidence`` 重算。
    """

    schema_version: int
    evaluator: str
    task_ids: tuple[str, ...]
    expected_attempts: int
    eligible_tasks: tuple[str, ...]
    candidate_evals: tuple[TaskEval, ...]
    control_evals: tuple[TaskEval, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator": self.evaluator,
            "task_ids": list(self.task_ids),
            "expected_attempts": self.expected_attempts,
            "eligible_tasks": list(self.eligible_tasks),
            "candidate_evals": [value.to_dict() for value in self.candidate_evals],
            "control_evals": [value.to_dict() for value in self.control_evals],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcceptedRuntimeEvidence":
        """从 exact-field mapping 解析 accepted Runtime evidence。

        字段集合必须精确匹配 schema；schema_version/expected_attempts 必须是 int，task/eligible
        必须为 string list，candidate/control 必须为 TaskEval list。shape 失败抛出
        ``CandidateEvidenceError``；更深的 gate 语义由 recomputation 验证。
        """
        raw = dict(value)
        if set(raw) != _ACCEPTED_RUNTIME_KEYS:
            raise CandidateEvidenceError(
                f"accepted runtime evidence fields must be exactly {sorted(_ACCEPTED_RUNTIME_KEYS)}"
            )
        if type(raw["schema_version"]) is not int:
            raise CandidateEvidenceError("accepted runtime evidence schema_version must be an integer")
        if not isinstance(raw["evaluator"], str):
            raise CandidateEvidenceError("accepted runtime evidence evaluator must be a string")
        if type(raw["expected_attempts"]) is not int:
            raise CandidateEvidenceError("accepted runtime evidence expected_attempts must be an integer")
        for field in ("task_ids", "eligible_tasks"):
            if not isinstance(raw[field], list) or not all(isinstance(item, str) for item in raw[field]):
                raise CandidateEvidenceError(f"accepted runtime evidence {field} must be a list of strings")
        for field in ("candidate_evals", "control_evals"):
            if not isinstance(raw[field], list):
                raise CandidateEvidenceError(f"accepted runtime evidence {field} must be a list")
        return cls(
            schema_version=raw["schema_version"],
            evaluator=raw["evaluator"],
            task_ids=tuple(raw["task_ids"]),
            expected_attempts=raw["expected_attempts"],
            eligible_tasks=tuple(raw["eligible_tasks"]),
            candidate_evals=tuple(_task_eval_from_dict(item) for item in raw["candidate_evals"]),
            control_evals=tuple(_task_eval_from_dict(item) for item in raw["control_evals"]),
        )


def _task_eval_from_dict(value: Any) -> TaskEval:
    if not isinstance(value, dict) or set(value) != _TASK_EVAL_KEYS:
        raise CandidateEvidenceError("accepted runtime evidence contains an invalid TaskEval")
    if not isinstance(value["task_id"], str):
        raise CandidateEvidenceError("accepted runtime TaskEval task_id must be a string")
    for field in ("passes", "attempts", "infra_attempts"):
        if type(value[field]) is not int:
            raise CandidateEvidenceError(f"accepted runtime TaskEval {field} must be an integer")
    failure = value["failure"]
    if failure is not None:
        if not isinstance(failure, str):
            raise CandidateEvidenceError("accepted runtime TaskEval failure must be a string or null")
        try:
            MeasurementFailure(failure)
        except ValueError as exc:
            raise CandidateEvidenceError("accepted runtime TaskEval failure is invalid") from exc
    try:
        return TaskEval(
            task_id=value["task_id"],
            passes=value["passes"],
            attempts=value["attempts"],
            infra_attempts=value["infra_attempts"],
            failure=failure,
        )
    except ValueError as exc:
        raise CandidateEvidenceError("accepted runtime evidence contains an invalid TaskEval") from exc


def _canonical_task_evals(
    evals: Mapping[str, TaskEval],
    task_ids: tuple[str, ...],
    *,
    expected_attempts: int,
    arm: str,
) -> tuple[TaskEval, ...]:
    if set(evals) != set(task_ids):
        raise CandidateEvidenceError(f"accepted runtime {arm} measurements must cover exactly the declared train tasks")
    ordered: list[TaskEval] = []
    for task_id in task_ids:
        value = evals[task_id]
        if not isinstance(value, TaskEval) or value.task_id != task_id:
            raise CandidateEvidenceError(f"accepted runtime {arm} measurement identity is invalid")
        if value.attempts != expected_attempts:
            raise CandidateEvidenceError(
                f"accepted runtime {arm} measurement for {task_id!r} must contain exactly {expected_attempts} attempts"
            )
        ordered.append(value)
    return tuple(ordered)


def _validated_runtime_measurements(
    manifest: CandidateManifest,
    evidence: AcceptedRuntimeEvidence,
) -> tuple[dict[str, TaskEval], dict[str, TaskEval], GateResult, float]:
    if type(evidence.schema_version) is not int or evidence.schema_version != 1:
        raise CandidateEvidenceError("accepted runtime evidence has an unsupported schema")
    if manifest.label is not CandidateLabel.runtime:
        raise CandidateEvidenceError("accepted runtime evidence requires the Runtime Candidate Label")
    if not isinstance(evidence.evaluator, str) or evidence.evaluator != manifest.evaluator:
        raise CandidateEvidenceError("accepted runtime evidence evaluator does not match the manifest")
    task_ids = evidence.task_ids
    if not task_ids or not all(isinstance(task_id, str) for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise CandidateEvidenceError("accepted runtime evidence requires unique train task ids")
    if type(evidence.expected_attempts) is not int or evidence.expected_attempts < 1:
        raise CandidateEvidenceError("accepted runtime evidence expected_attempts must be at least one")
    eligible = evidence.eligible_tasks
    if not eligible or not all(isinstance(task_id, str) for task_id in eligible) or len(eligible) != len(set(eligible)):
        raise CandidateEvidenceError("accepted runtime evidence requires unique eligible task ids")
    eligible_set = set(eligible)
    if any(task_id not in task_ids for task_id in eligible):
        raise CandidateEvidenceError("accepted runtime eligible tasks must be train tasks")
    if eligible != tuple(task_id for task_id in task_ids if task_id in eligible_set):
        raise CandidateEvidenceError("accepted runtime eligible task order is not canonical")
    if tuple(value.task_id for value in evidence.candidate_evals) != task_ids:
        raise CandidateEvidenceError("accepted runtime candidate measurement order is not canonical")
    if tuple(value.task_id for value in evidence.control_evals) != task_ids:
        raise CandidateEvidenceError("accepted runtime control measurement order is not canonical")

    candidate = _canonical_task_evals(
        {value.task_id: value for value in evidence.candidate_evals},
        task_ids,
        expected_attempts=evidence.expected_attempts,
        arm="candidate",
    )
    control = _canonical_task_evals(
        {value.task_id: value for value in evidence.control_evals},
        task_ids,
        expected_attempts=evidence.expected_attempts,
        arm="control",
    )
    if len(candidate) != len(evidence.candidate_evals) or len(control) != len(evidence.control_evals):
        raise CandidateEvidenceError("accepted runtime evidence contains duplicate task measurements")
    candidate_map = {value.task_id: value for value in candidate}
    control_map = {value.task_id: value for value in control}
    fired_tasks = None if eligible == task_ids else eligible_set
    gate = run_gates(
        candidate_evals=candidate_map,
        control_evals=control_map,
        task_ids=list(task_ids),
        expected_attempts=evidence.expected_attempts,
        fired_tasks=fired_tasks,
    )
    if (
        not gate.promoted
        or gate.verdict is not EvaluationVerdict.accepted
        or tuple(gate.eligible_tasks) != eligible
        or not _validity_is_measured(gate.candidate_validity)
        or not _validity_is_measured(gate.control_validity)
    ):
        raise CandidateEvidenceError("accepted runtime measurements do not reproduce a promoted three-shield gate")
    full_lift = train_mean(candidate_map, list(task_ids)) - train_mean(control_map, list(task_ids))
    if not math.isfinite(full_lift) or full_lift <= 0:
        raise CandidateEvidenceError("accepted runtime measurements require a positive finite full-train lift")
    return candidate_map, control_map, gate, full_lift


def recompute_accepted_runtime_evidence(
    manifest: CandidateManifest,
    evidence: AcceptedRuntimeEvidence,
) -> EvidenceDecision:
    """只根据 canonical task measurements 重算 accepted verdict。

    验证 Runtime label、evaluator identity、unique/canonical task order、exact attempt count、
    eligible subset、candidate/control coverage，并重新运行 three-shield gate。gate 必须 promoted
    且两臂 measurement valid，full-train lift 必须 finite positive。成功返回
    ``EvidenceDecision(accepted, gate_passed=True)``；任一条件失败抛出
    ``CandidateEvidenceError``。
    """

    _validated_runtime_measurements(manifest, evidence)
    return EvidenceDecision(
        outcome=EvidenceOutcome.accepted,
        gate_passed=True,
        reason="complete Focused-Fisher train evidence passed",
    )


def _runtime_fixture(
    manifest: CandidateManifest,
    before: Snapshot,
    after: Snapshot,
) -> None:
    policy = LABEL_POLICIES[CandidateLabel.runtime]
    targets = tuple(manifest.target_files)
    if not targets:
        raise CandidateEvidenceError("runtime fixture requires at least one target")
    if set(before) != set(targets) or set(after) != set(targets):
        raise CandidateEvidenceError("runtime fixture snapshots do not cover manifest targets")
    outside = [path for path in targets if path not in policy.mutable_paths]
    if outside:
        raise CandidateEvidenceError(f"runtime fixture targets are outside its mutable surface: {outside}")
    if all(before[path] == after[path] for path in targets):
        raise CandidateEvidenceError("runtime fixture contains no content change")
    before_sha256 = tuple(
        hashlib.sha256(before[path]).hexdigest() if before[path] is not None else None for path in targets
    )
    after_sha256 = tuple(
        hashlib.sha256(after[path]).hexdigest() if after[path] is not None else None for path in targets
    )
    if before_sha256 != manifest.before_sha256 or after_sha256 != manifest.after_sha256:
        raise CandidateEvidenceError("runtime fixture snapshots do not match manifest content digests")


def _validity_is_measured(validity: "MeasurementValidity") -> bool:
    return validity.status is MeasurementStatus.measured and validity.valid


def _failure_class(outcome: CandidateOutcome) -> str | None:
    stats = outcome.stats or {}
    validities = [
        stats.get("candidate_validity"),
        stats.get("control_validity"),
    ]
    if any(isinstance(validity, Mapping) and validity.get("provider_failures") for validity in validities):
        return "provider"
    if any(isinstance(validity, Mapping) and validity.get("infrastructure_failures") for validity in validities):
        return "infrastructure"
    return None


def _runtime_outcome(
    manifest: CandidateManifest,
    outcome: CandidateOutcome,
    *,
    control_evals: Mapping[str, TaskEval] | None,
    task_ids: tuple[str, ...] | None,
    expected_attempts: int | None,
) -> EvidenceDecision | AcceptedRuntimeEvidence:
    verdict = EvaluationVerdict(outcome.verdict)
    regression = bool((outcome.stats or {}).get("sentinel_regression", False))
    reason = str((outcome.stats or {}).get("error") or outcome.status.value)
    if verdict is not EvaluationVerdict.accepted:
        return EvidenceDecision(
            outcome=EvidenceOutcome(verdict.value),
            gate_passed=False,
            reason=reason,
            regression=regression,
            failure_class=_failure_class(outcome),
        )

    gate = outcome.gate
    if not isinstance(gate, GateResult):
        raise CandidateEvidenceError("accepted runtime evidence requires the bound three-shield gate result")
    if (
        not gate.promoted
        or gate.verdict is not EvaluationVerdict.accepted
        or not _validity_is_measured(gate.candidate_validity)
        or not _validity_is_measured(gate.control_validity)
    ):
        raise CandidateEvidenceError("accepted runtime evidence is not fully measured and promoted")
    if outcome.status is not NodeStatus.promoted_to_baseline or not outcome.promoted:
        raise CandidateEvidenceError("accepted runtime evidence does not match candidate promotion state")
    if control_evals is None or task_ids is None or expected_attempts is None:
        raise CandidateEvidenceError(
            "accepted runtime evidence requires complete candidate and control train measurements"
        )
    evidence = AcceptedRuntimeEvidence(
        schema_version=1,
        evaluator=str(manifest.evaluator),
        task_ids=task_ids,
        expected_attempts=expected_attempts,
        eligible_tasks=tuple(gate.eligible_tasks),
        candidate_evals=_canonical_task_evals(
            outcome.confirm_evals,
            task_ids,
            expected_attempts=expected_attempts,
            arm="candidate",
        ),
        control_evals=_canonical_task_evals(
            control_evals,
            task_ids,
            expected_attempts=expected_attempts,
            arm="control",
        ),
    )
    candidate_map, _, recomputed_gate, full_lift = _validated_runtime_measurements(manifest, evidence)
    if tuple(gate.eligible_tasks) != tuple(recomputed_gate.eligible_tasks):
        raise CandidateEvidenceError("accepted runtime evidence does not match the recomputed attribution set")
    reported_lift = (outcome.stats or {}).get("full_lift")
    if (
        not isinstance(reported_lift, (int, float))
        or not math.isfinite(float(reported_lift))
        or not math.isclose(float(reported_lift), full_lift, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise CandidateEvidenceError("accepted runtime full-train lift does not match canonical measurements")
    candidate_mean = train_mean(candidate_map, list(task_ids))
    if not math.isclose(float(outcome.score), candidate_mean, rel_tol=0.0, abs_tol=1e-12):
        raise CandidateEvidenceError("accepted runtime score does not match canonical measurements")
    if regression:
        raise CandidateEvidenceError("regressing runtime evidence cannot be accepted")
    return evidence


FIXTURE_BINDINGS: dict[str, FixtureBinding] = {
    "appworld_runtime_v1": FixtureBinding(
        name="appworld_runtime_v1",
        validate=_runtime_fixture,
    ),
}

EVALUATOR_BINDINGS: dict[str, EvaluatorBinding] = {
    "appworld_focused_fisher_v1": EvaluatorBinding(
        name="appworld_focused_fisher_v1",
        evaluate=_runtime_outcome,
    ),
}


def evaluate_candidate_evidence(
    manifest: CandidateManifest,
    outcome: CandidateOutcome,
    *,
    before: Snapshot,
    after: Snapshot,
    control_evals: Mapping[str, TaskEval] | None = None,
    task_ids: list[str] | tuple[str, ...] | None = None,
    expected_attempts: int | None = None,
) -> EvidenceDecision | AcceptedRuntimeEvidence:
    """执行 supported manifest 声明的 fixture 与 evaluator。

    首先确认 label policy supported，manifest fixture/evaluator 与 canonical policy 完全一致，且
    两者都有 executable binding；随后验证 before/after snapshot，再评估 ``CandidateOutcome``。
    non-accepted outcome 返回 ``EvidenceDecision``；accepted outcome 必须提供 complete
    candidate/control train measurements、task_ids 与 expected_attempts，返回可重算的
    ``AcceptedRuntimeEvidence``。

    missing binding、policy drift、fixture mismatch 或 accepted measurement 不完整都会抛出
    ``CandidateEvidenceError``，不能把无法测量误报为正向结论。
    """

    policy = LABEL_POLICIES[manifest.label]
    if not policy.supported:
        raise CandidateEvidenceError(
            policy.unsupported_reason or f"Candidate Label {manifest.label.value!r} is unsupported"
        )
    if manifest.fixture != policy.fixture or manifest.evaluator != policy.evaluator:
        raise CandidateEvidenceError("manifest fixture or evaluator differs from the canonical label policy")
    fixture = FIXTURE_BINDINGS.get(str(manifest.fixture))
    evaluator = EVALUATOR_BINDINGS.get(str(manifest.evaluator))
    if fixture is None or evaluator is None:
        raise CandidateEvidenceError("manifest fixture or evaluator has no executable binding")
    fixture.validate(manifest, before, after)
    return evaluator.evaluate(
        manifest,
        outcome,
        control_evals=control_evals,
        task_ids=tuple(task_ids) if task_ids is not None else None,
        expected_attempts=expected_attempts,
    )


__all__ = [
    "AcceptedRuntimeEvidence",
    "CandidateEvidenceError",
    "EVALUATOR_BINDINGS",
    "FIXTURE_BINDINGS",
    "evaluate_candidate_evidence",
    "recompute_accepted_runtime_evidence",
]
