"""把 three-shield gate 实现为 composable pipeline（SOP §2 ⑥）。

Order matters and each shield narrows the task set the next one judges:

1. **Gate-f (measurement validity).** Fail closed when either arm has a
   Provider failure, infrastructure failure, inconclusive result, or missing
   task. Recoverable infra is salvaged upstream by the rerun ladder
   (:func:`codeflow.evolver.orchestrator.scoring.eval_with_infra_rerun`); by the
   time run_gates sees the evals, any surviving failure is invalid evidence,
   not a low score.
2. **Gate-b (attribution).** Only credit the candidate on tasks where its
   mechanism actually fired — a patch can't get credit for a task it never
   touched. The set of fired tasks comes from an injectable source (the
   activation ledger / beacon); when none is given this shield is a no-op.
3. **Gate2 (significance).** Paired lift on the surviving tasks: navigator
   promotion (mean > vanilla) plus a separate credited-2σ label.

pure eval-map function 使它 bench-agnostic 且 unit-testable；Loop 在 confirm 后每 candidate 调用
一次。任何 shield 的通过都只是 promotion evidence 的一部分，不等于 activation 或 sealed success。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codeflow.evolver.orchestrator.gates.paired import PairedResult, paired_lift
from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementFailure,
    MeasurementStatus,
    MeasurementValidity,
    TaskEval,
    measurement_validity,
)


@dataclass
class GateResult:
    """一个 candidate 通过 three-shield pipeline 后的 combined verdict。

    ``infra_contaminated`` 保留供 audit；failed/inconclusive measurement 在 Gate-b 缩小 task set 前
    就阻止 promotion。``eligible_tasks`` 是 attribution 后集合，``unfired_excluded`` 记录排除项。
    """

    promoted: bool
    verdict: EvaluationVerdict
    paired: PairedResult | None
    eligible_tasks: list[str]
    candidate_validity: MeasurementValidity
    control_validity: MeasurementValidity
    infra_contaminated: list[str] = field(default_factory=list)
    unfired_excluded: list[str] = field(default_factory=list)


def _is_infrastructure_failure(ev: TaskEval | None, threshold: int) -> bool:
    return ev is not None and (ev.failure is MeasurementFailure.infrastructure or ev.infra_attempts >= threshold)


def run_gates(
    *,
    candidate_evals: dict[str, TaskEval],
    control_evals: dict[str, TaskEval],
    task_ids: list[str],
    expected_attempts: int,
    z_threshold: float = 2.0,
    fired_tasks: set[str] | None = None,
    infra_threshold: int = 1,
) -> GateResult:
    """依次运行 Gate-f -> Gate-b -> Gate2，返回 combined verdict。

    ``expected_attempts`` 是每个 requested task 的 K。``fired_tasks`` 非 None 时限制 attribution，
    None 跳过 Gate-b；``infra_threshold`` 决定 audit 中 contamination。surviving failure 永远不会
    转成 low score。Gate-f failed/inconclusive 立即返回；Gate-b 若无 eligible task，则 rejected
    且 paired=None；否则运行 paired lift。返回 promoted 只表示 train gate navigator condition。
    """
    infra_contaminated = [
        t
        for t in task_ids
        if _is_infrastructure_failure(candidate_evals.get(t), infra_threshold)
        or _is_infrastructure_failure(control_evals.get(t), infra_threshold)
    ]
    candidate_validity = measurement_validity(
        candidate_evals,
        task_ids,
        expected_attempts=expected_attempts,
    )
    control_validity = measurement_validity(
        control_evals,
        task_ids,
        expected_attempts=expected_attempts,
    )
    validity_statuses = {candidate_validity.status, control_validity.status}
    if MeasurementStatus.failed in validity_statuses:
        return GateResult(
            promoted=False,
            verdict=EvaluationVerdict.failed,
            paired=None,
            eligible_tasks=list(task_ids),
            candidate_validity=candidate_validity,
            control_validity=control_validity,
            infra_contaminated=infra_contaminated,
        )
    if MeasurementStatus.inconclusive in validity_statuses:
        return GateResult(
            promoted=False,
            verdict=EvaluationVerdict.inconclusive,
            paired=None,
            eligible_tasks=list(task_ids),
            candidate_validity=candidate_validity,
            control_validity=control_validity,
            infra_contaminated=infra_contaminated,
        )
    eligible = list(task_ids)
    unfired_excluded: list[str] = []
    if fired_tasks is not None:
        unfired_excluded = [t for t in eligible if t not in fired_tasks]
        eligible = [t for t in eligible if t in fired_tasks]

    if not eligible:
        return GateResult(
            promoted=False,
            verdict=EvaluationVerdict.rejected,
            paired=None,
            eligible_tasks=[],
            candidate_validity=candidate_validity,
            control_validity=control_validity,
            infra_contaminated=infra_contaminated,
            unfired_excluded=unfired_excluded,
        )

    paired = paired_lift(
        candidate_evals=candidate_evals,
        control_evals=control_evals,
        task_ids=eligible,
        expected_attempts=expected_attempts,
        z_threshold=z_threshold,
    )
    return GateResult(
        promoted=paired.promoted,
        verdict=paired.verdict,
        paired=paired,
        eligible_tasks=eligible,
        candidate_validity=candidate_validity,
        control_validity=control_validity,
        infra_contaminated=infra_contaminated,
        unfired_excluded=unfired_excluded,
    )


__all__ = ["GateResult", "run_gates"]
