"""Gate2：paired lift + 2σ significance，推广自 ``round7_paired``。

The scratchpad ``round7_paired.py`` hard-coded its anchor/explore task lists and
its two arms. This is the same statistics with the task set and the two arms
passed in, so it works for any round and any domain.

Pairing is what makes the test sensitive: comparing candidate and vanilla on the
*same* tasks removes between-task difficulty, so the standard error is the
spread of the per-task differences, not the spread of raw pass rates. For each
task we take the per-task pass-rate difference ``d_i = rate_candidate,i -
rate_vanilla,i`` (K=3 rates), and test whether the mean difference is far enough
from zero:

    mean_lift = mean(d_i)
    se        = stdev(d_i) / sqrt(n)          # paired standard error
    z         = mean_lift / se

Promotion to bank (SOP §2 ⑥ Gate2) is the *navigator* condition alone: the
candidate's mean pass@1 beats vanilla on the shared train set. The paired 2σ
test is a separate *credited-significance* label (``credited_2sigma``) reported
alongside — it says whether the lift is statistically significant, not whether
the candidate banks. This matches the manual round-3 decision, where
budgetnudge banked on +6.4pp over vanilla even though its paired z (1.71) fell
short of 2σ. A candidate that improves every shared task identically has
``se == 0``; that deterministic win is reported as ``z = inf``.

banked candidate 是否成为 next parent，或因 anchor/full-set sign-flip 等 qualitative reason 被
prune，属于 gate 之上的 semantic step ⑦，不是 paired test 职责。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev

from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementStatus,
    MeasurementValidity,
    TaskEval,
    measurement_validity,
)


@dataclass(frozen=True)
class PairedResult:
    """candidate/control paired lift test 的完整结果。

    同时保存 navigator ``promoted`` 与独立 ``credited_2sigma``，不能将二者混用；还保留两臂
    measurement validity，便于 fail/inconclusive 审计。
    """

    n_tasks: int
    candidate_mean: float
    control_mean: float
    mean_lift: float
    se: float
    z: float
    z_threshold: float
    promoted: bool
    credited_2sigma: bool
    verdict: EvaluationVerdict
    candidate_validity: MeasurementValidity
    control_validity: MeasurementValidity


def _rate(evals: dict[str, TaskEval], task_id: str) -> float:
    ev = evals.get(task_id)
    return ev.pass_rate if ev is not None else 0.0


def paired_lift(
    *,
    candidate_evals: dict[str, TaskEval],
    control_evals: dict[str, TaskEval],
    task_ids: list[str],
    expected_attempts: int,
    z_threshold: float = 2.0,
) -> PairedResult:
    """在 shared ``task_ids`` 上执行 candidate/control paired lift + 2σ test。

    task_ids 通常是 confirm full train shared set；``expected_attempts`` 是两臂每个 task 必须达到
    的 K。先计算 measurement validity，再计算 per-task diff、mean、paired SE 与 z。failed 优先于
    inconclusive；只有两臂 measured 且 candidate mean > control mean 才 accepted/promoted；
    ``credited_2sigma`` 还要求 z >= threshold。空 task list 抛 ``ValueError``。
    """
    if not task_ids:
        raise ValueError("paired_lift requires a non-empty task list")

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
    diffs = [_rate(candidate_evals, t) - _rate(control_evals, t) for t in task_ids]
    candidate_mean = mean(_rate(candidate_evals, t) for t in task_ids)
    control_mean = mean(_rate(control_evals, t) for t in task_ids)
    mean_lift = mean(diffs)

    n = len(task_ids)
    se = stdev(diffs) / math.sqrt(n) if n > 1 else 0.0
    if se == 0.0:
        z = 0.0 if mean_lift == 0.0 else math.copysign(math.inf, mean_lift)
    else:
        z = mean_lift / se

    statuses = {candidate_validity.status, control_validity.status}
    if MeasurementStatus.failed in statuses:
        verdict = EvaluationVerdict.failed
    elif MeasurementStatus.inconclusive in statuses:
        verdict = EvaluationVerdict.inconclusive
    elif candidate_mean > control_mean:
        verdict = EvaluationVerdict.accepted
    else:
        verdict = EvaluationVerdict.rejected
    promoted = verdict is EvaluationVerdict.accepted
    credited_2sigma = promoted and z >= z_threshold

    return PairedResult(
        n_tasks=n,
        candidate_mean=candidate_mean,
        control_mean=control_mean,
        mean_lift=mean_lift,
        se=se,
        z=z,
        z_threshold=z_threshold,
        promoted=promoted,
        credited_2sigma=credited_2sigma,
        verdict=verdict,
        candidate_validity=candidate_validity,
        control_validity=control_validity,
    )


__all__ = ["PairedResult", "paired_lift"]
