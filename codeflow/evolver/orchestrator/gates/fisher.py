"""实现 two-stage Fisher gate 的 focused-subset statistics（SOP §2 ⑤/⑥）。

Ported from the AppWorld evolution driver so the two-stage gate is in-package
and unit-testable. The stage-1 test asks a sharp, cheap question on a candidate's
WHY subset: *is the candidate's pass-rate significantly above the baseline's on
exactly the tasks the pathology occurs?* A one-sided Fisher exact on the 2x2
trial table answers it without a full-train run.

Denominator discipline (SOP §0 hard rule): infra-contaminated trials are NOT
dropped from the denominator. Recoverable infra is salvaged upstream by the ≤2
rerun ladder (:func:`codeflow.evolver.orchestrator.scoring.eval_with_infra_rerun`);
只有 persistent infra 到达这里，并作为 non-pass 留在 denominator。删除 infra task 会缩小分母、
高估 pass@1，形成 fair-subset extrapolation trap。统计显著只针对 focused subset，不等于 full
train promotion 或 sealed generalisation。
"""

from __future__ import annotations

import math

from codeflow.evolver.orchestrator.scoring import TaskEval


def focused_counts(evals: dict[str, TaskEval], focused_ids: list[str]) -> tuple[int, int]:
    """返回 focused subset 的 ``(passes, fails)`` trial counts。

    persistent infra trial 计 fail、不排除，符合 SOP §0。``evals`` 缺 task 表示 never launched，
    这里无 trial 可计而跳过；full-train fixed denominator 由 :func:`train_mean` 保证。
    """
    passes = fails = 0
    for tid in focused_ids:
        ev = evals.get(tid)
        if ev is None:
            continue
        passes += ev.passes
        fails += ev.attempts - ev.passes
    return passes, fails


def train_mean(evals: dict[str, TaskEval], task_ids: list[str]) -> float:
    """用 FIXED denominator 计算 ``task_ids`` 的 mean per-task pass@1（SOP §0）。

    denominator 永远是 task count ``len(task_ids)``。missing/all-infra task 贡献 0.0，绝不 drop；
    infra 通过 ``TaskEval.pass_rate = passes / attempts`` 计 non-pass。空 task list 返回 0.0。
    """
    if not task_ids:
        return 0.0
    total = 0.0
    for t in task_ids:
        ev = evals.get(t)
        total += ev.pass_rate if ev is not None else 0.0
    return total / len(task_ids)


def fisher_one_sided(cp: int, cn: int, vp: int, vn: int) -> float:
    """在 2x2 table 上计算 one-sided Fisher exact P(candidate rate > vanilla)。

    ``[[cp, cn], [vp, vn]]`` 表示 candidate pass/fail 与 vanilla pass/fail trial count。degenerate
    margin 返回 1.0（not significant）；否则用 log-gamma 计算 upper tail，结果上限 1.0。
    """
    row1, row2 = cp + cn, vp + vn
    col1, tot = cp + vp, cp + cn + vp + vn
    if row1 == 0 or row2 == 0 or col1 == 0 or col1 == tot:
        return 1.0

    def _p(a: int) -> float:
        b, c, d = row1 - a, col1 - a, tot - col1 - (row1 - a)
        if b < 0 or c < 0 or d < 0:
            return 0.0
        return math.exp(
            math.lgamma(row1 + 1)
            + math.lgamma(row2 + 1)
            + math.lgamma(col1 + 1)
            + math.lgamma(tot - col1 + 1)
            - math.lgamma(tot + 1)
            - sum(math.lgamma(x + 1) for x in (a, b, c, d))
        )

    hi = min(row1, col1)
    return min(1.0, sum(_p(a) for a in range(cp, hi + 1)))


__all__ = [
    "focused_counts",
    "train_mean",
    "fisher_one_sided",
]
