"""实现 SWE paired-2sigma 与 AppWorld focused-Fisher 两个 concrete GatePolicy。

Both implement :class:`GatePolicy.decide` over a :class:`DecisionContext`,
owning their own ``eval`` calls. Neither knows how its control arm was produced
(frozen vs same-session) — that is the :class:`BaselineProvider`'s concern.

Both stages follow the SOP's two disciplines:

- **Wide-pass screen (SOP §2 ⑤a).** A candidate advances to the full-train
  confirm unless it is *clearly worse* than the baseline on its probe set. A
  slightly-low or noise-band probe is NOT evidence — small K=1/K=3 probes have
  huge variance and the probe mean does not predict the full-set mean.
- **Two-threshold verdict (SOP §0).** Promotion (banking, parent selection) is
  the loose *navigator* condition: candidate full-train mean beats the control.
  The paired-2σ significance is a separate *credited* label reported alongside
  (``CandidateOutcome.paired.credited_2sigma``), never the promotion bar.

confirm job name 由 :func:`confirm_job_name` 统一定义，因为它同时是 diagnosis 回读 trajectory 的
on-disk out-dir；该 naming 是 cross-module contract。screen 通过只允许进入 confirm，confirm
accepted 也仍需与 parent selection/semantic verdict 分开。
"""

from __future__ import annotations

from codeflow.evolver.orchestrator.gates.fisher import (
    fisher_one_sided,
    focused_counts,
    train_mean,
)
from codeflow.evolver.orchestrator.gates.pipeline import run_gates
from codeflow.evolver.orchestrator.gates.policy import CandidateOutcome, DecisionContext
from codeflow.evolver.orchestrator.nodes.screen import screen_candidate
from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementStatus,
    measurement_validity,
)
from codeflow.evolver.tree.node import NodeStatus

CONFIRM_JOB_SUFFIX = "_confirm"


def confirm_job_name(node_id: str) -> str:
    """返回 node full-train confirm 的 job/out-dir name。

    AppWorld 等回读 promoted parent confirm artifact 的 bench wiring 必须调用本函数，不能手写
    f-string，防止 gate writer 与 diagnosis reader drift。
    """
    return f"{node_id}{CONFIRM_JOB_SUFFIX}"


def _measurement_verdict(
    candidate,
    control,
    task_ids,
    *,
    expected_attempts: int,
) -> tuple[EvaluationVerdict | None, dict]:
    candidate_validity = measurement_validity(
        candidate,
        task_ids,
        expected_attempts=expected_attempts,
    )
    control_validity = measurement_validity(
        control,
        task_ids,
        expected_attempts=expected_attempts,
    )
    statuses = {candidate_validity.status, control_validity.status}
    verdict = None
    if MeasurementStatus.failed in statuses:
        verdict = EvaluationVerdict.failed
    elif MeasurementStatus.inconclusive in statuses:
        verdict = EvaluationVerdict.inconclusive
    return verdict, {
        "candidate_validity": candidate_validity.to_dict(),
        "control_validity": control_validity.to_dict(),
    }


class PairedTwoSigmaGate:
    """SWE line：K=1 anchor wide-pass screen -> K=3 full-train three-shield gate。

    promotion bar 是 candidate mean beat vanilla 的 navigator condition；2sigma 仅是 alongside
    credited label，不是门槛。full-train score 始终使用 fixed denominator，即使 Gate-b paired
    attribution 缩到 fired subset。
    """

    def __init__(self, *, k_screen: int = 1, k_confirm: int = 3, z_threshold: float = 2.0):
        self.k_screen = k_screen
        self.k_confirm = k_confirm
        self.z_threshold = z_threshold

    def decide(self, ctx: DecisionContext) -> CandidateOutcome:
        if ctx.anchor is None:
            raise ValueError("PairedTwoSigmaGate requires an anchor in the context")
        node_id = ctx.node.node_id
        screen_evals = ctx.eval(ctx.node, ctx.anchor.task_ids, self.k_screen, f"{node_id}_screen")
        invalid, validity_stats = _measurement_verdict(
            screen_evals,
            ctx.baseline.evals,
            ctx.anchor.task_ids,
            expected_attempts=self.k_screen,
        )
        if invalid is not None:
            return CandidateOutcome(
                node_id,
                NodeStatus.errored if invalid is EvaluationVerdict.failed else NodeStatus.pruned_at_screen,
                stats={"phase": "screen", "verdict": invalid.value, **validity_stats},
                verdict=invalid,
            )
        screen = screen_candidate(candidate_evals=screen_evals, anchor=ctx.anchor, vanilla_evals=ctx.baseline.evals)
        if not screen.passes_to_confirm:
            return CandidateOutcome(
                node_id,
                NodeStatus.pruned_at_screen,
                screen=screen,
                verdict=EvaluationVerdict.rejected,
            )

        confirm = ctx.eval(ctx.node, ctx.train_task_ids, self.k_confirm, confirm_job_name(node_id))
        fired = ctx.fired_source(ctx.node, ctx.train_task_ids) if ctx.fired_source else None
        gate = run_gates(
            candidate_evals=confirm,
            control_evals=ctx.baseline.evals,
            task_ids=ctx.train_task_ids,
            expected_attempts=self.k_confirm,
            fired_tasks=fired,
            z_threshold=self.z_threshold,
        )
        # 报告分数始终是固定分母上的完整训练均值；配对统计可收窄到门控 b 的触发子集，但若把
        # 子集均值泄漏为“分数”，就会用不同分母的数字污染 beat_vanilla、父节点选择和日志曲线
        # （评审第 2 轮 P0-4）。晋升需同时满足：可能基于子集的配对结论成立，且完整训练均值
        # 不低于对照组。触发子集可以获得功劳，但绝不能以整个集合为代价。
        full_mean = train_mean(confirm, ctx.train_task_ids)
        control_full_mean = train_mean(ctx.baseline.evals, ctx.train_task_ids)
        promoted = gate.promoted and full_mean >= control_full_mean
        if gate.verdict is EvaluationVerdict.failed:
            status = NodeStatus.errored
        else:
            status = NodeStatus.promoted_to_baseline if promoted else NodeStatus.pruned_at_confirm
        return CandidateOutcome(
            node_id,
            status,
            score=full_mean,
            confirm_evals=confirm,
            screen=screen,
            paired=gate.paired,
            gate=gate,
            stats={
                "full_mean": full_mean,
                "control_full_mean": control_full_mean,
                "verdict": gate.verdict.value,
                "candidate_validity": gate.candidate_validity.to_dict(),
                "control_validity": gate.control_validity.to_dict(),
            },
            verdict=(
                EvaluationVerdict.accepted
                if promoted
                else EvaluationVerdict.rejected
                if gate.verdict is EvaluationVerdict.accepted
                else gate.verdict
            ),
        )


class FocusedFisherGate:
    """AppWorld line：focused-subset wide-pass probe -> full-train three-shield gate。

    Stage 1 runs the candidate only on its WHY's focused subset (plus the
    sentinel controls) and culls it ONLY when the probe shows it clearly worse
    than the baseline — a one-sided Fisher test in the *worse* direction at
    ``alpha``, or a sentinel regression beyond one flaky trial. Everything else
    (better, slightly low, indistinguishable) advances to the full confirm:
    wide-pass, SOP §2 ⑤a. The improvement-direction Fisher p is still reported
    (``fisher_p``) as evidence, but it is not an advancement bar.

    Stage 2 confirms on the full train set through the same three-shield
    pipeline as the SWE line: Gate-f infra report, Gate-b attribution when the
    bench wires a ``fired_source``, then the paired gate — navigator promotion
    (mean beats the control) with the credited-2σ label reported alongside.
    ``min_confirm_lift`` default 0，可在 navigator 上额外要求 minimum full-train lift。stage 1
    improvement Fisher p 只报告 evidence，不作为 advancement bar；sentinel regression 可直接
    prune。最终 promotion 仍要求 Gate-f/Gate-b/paired 与 full lift 同时满足。
    """

    def __init__(
        self,
        *,
        k: int = 3,
        alpha: float = 0.05,
        min_confirm_lift: float = 0.0,
        z_threshold: float = 2.0,
    ):
        self.k = k
        self.alpha = alpha
        self.min_confirm_lift = min_confirm_lift
        self.z_threshold = z_threshold

    def decide(self, ctx: DecisionContext) -> CandidateOutcome:
        node_id = ctx.node.node_id
        focused = ctx.focused_task_ids
        sentinels = ctx.sentinel_task_ids
        # 对聚焦集合（WHY 子集）和哨兵（稳定通过对照）只做一次评测，因此退化防护无需额外运行。
        probe_ids = list(dict.fromkeys(list(focused) + list(sentinels)))
        cand_probe = ctx.eval(ctx.node, probe_ids, self.k, f"{node_id}_focused") if probe_ids else {}
        stats: dict = {}
        invalid, validity_stats = _measurement_verdict(
            cand_probe,
            ctx.baseline.evals,
            probe_ids,
            expected_attempts=self.k,
        )
        if invalid is not None:
            return CandidateOutcome(
                node_id,
                NodeStatus.errored if invalid is EvaluationVerdict.failed else NodeStatus.pruned_at_screen,
                stats={"phase": "screen", "verdict": invalid.value, **validity_stats},
                verdict=invalid,
            )
        if focused:
            cp, cn = focused_counts(cand_probe, focused)
            vp, vn = focused_counts(ctx.baseline.evals, focused)
            foc_c = cp / (cp + cn) if (cp + cn) else 0.0
            foc_v = vp / (vp + vn) if (vp + vn) else 0.0
            stats.update(
                fisher_p=fisher_one_sided(cp, cn, vp, vn),
                fisher_p_worse=fisher_one_sided(vp, vn, cp, cn),
                foc_c=foc_c,
                foc_v=foc_v,
            )

        # 分层的哨兵退化防护（SOP 第 2 节第 5a 项）：稳定通过对照在基线下不会波动，因此超过
        # 一次波动试验的下降就是信号；边界对照天然会翻转，故其结论使用试验级 Fisher 检验
        # （变差方向）而非均值防护，否则均值防护会在波动任务的噪声上触发。
        if sentinels:
            base = ctx.baseline.evals

            def _bmean(tid: str) -> float | None:
                ev = base.get(tid)
                good = (ev.attempts - ev.infra_attempts) if ev else 0
                return (ev.passes / good) if ev and good else None

            stable = [t for t in sentinels if _bmean(t) == 1.0]
            fragile = [t for t in sentinels if t not in stable and _bmean(t)]
            stats.update(
                sent_c=train_mean(cand_probe, sentinels),
                sent_v=train_mean(base, sentinels),
            )
            if stable:
                st_c = train_mean(cand_probe, stable)
                st_v = train_mean(base, stable)
                guard = 1.5 / (len(stable) * self.k)
                stats.update(sentinel_guard=guard)
                if st_c < st_v - guard:
                    stats["sentinel_regression"] = True
                    return CandidateOutcome(
                        node_id,
                        NodeStatus.pruned_at_screen,
                        stats=stats,
                        verdict=EvaluationVerdict.rejected,
                    )
            if fragile:
                fc_p, fc_n = focused_counts(cand_probe, fragile)
                fv_p, fv_n = focused_counts(base, fragile)
                p_worse = fisher_one_sided(fv_p, fv_n, fc_p, fc_n)
                stats.update(sent_fragile_p_worse=p_worse)
                frag_c = fc_p / (fc_p + fc_n) if (fc_p + fc_n) else 0.0
                frag_v = fv_p / (fv_p + fv_n) if (fv_p + fv_n) else 0.0
                if frag_c < frag_v and p_worse < self.alpha:
                    stats["sentinel_regression"] = True
                    return CandidateOutcome(
                        node_id,
                        NodeStatus.pruned_at_screen,
                        stats=stats,
                        verdict=EvaluationVerdict.rejected,
                    )

        # 宽松通过裁剪：只有在 WHY 子集上显著差于基线的探针才会不经完整运行直接裁掉。
        # 略低或无法区分的探针继续前进；SOP 规定略低不能淘汰。
        if focused and stats["foc_c"] < stats["foc_v"] and stats["fisher_p_worse"] < self.alpha:
            stats["pruned_significantly_worse"] = True
            return CandidateOutcome(
                node_id,
                NodeStatus.pruned_at_screen,
                stats=stats,
                verdict=EvaluationVerdict.rejected,
            )

        confirm = ctx.eval(ctx.node, ctx.train_task_ids, self.k, confirm_job_name(node_id))
        fired = ctx.fired_source(ctx.node, ctx.train_task_ids) if ctx.fired_source else None
        gate = run_gates(
            candidate_evals=confirm,
            control_evals=ctx.baseline.evals,
            task_ids=ctx.train_task_ids,
            expected_attempts=self.k,
            fired_tasks=fired,
            z_threshold=self.z_threshold,
        )
        cand_mean = train_mean(confirm, ctx.train_task_ids)
        lift = cand_mean - ctx.baseline.mean
        stats["full_lift"] = lift
        stats.update(
            verdict=gate.verdict.value,
            candidate_validity=gate.candidate_validity.to_dict(),
            control_validity=gate.control_validity.to_dict(),
        )
        promoted = gate.promoted and lift >= self.min_confirm_lift
        if gate.verdict is EvaluationVerdict.failed:
            status = NodeStatus.errored
        else:
            status = NodeStatus.promoted_to_baseline if promoted else NodeStatus.pruned_at_confirm
        return CandidateOutcome(
            node_id,
            status,
            score=cand_mean,
            confirm_evals=confirm,
            paired=gate.paired,
            gate=gate,
            stats=stats,
            verdict=(
                EvaluationVerdict.accepted
                if promoted
                else EvaluationVerdict.rejected
                if gate.verdict is EvaluationVerdict.accepted
                else gate.verdict
            ),
        )


__all__ = [
    "PairedTwoSigmaGate",
    "FocusedFisherGate",
    "confirm_job_name",
    "CONFIRM_JOB_SUFFIX",
]
