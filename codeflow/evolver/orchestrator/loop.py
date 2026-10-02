"""The seven-step funnel as a finite-state machine.

This is the layer the SOP used to delegate to a long, high-compliance Claude
session. Here the control flow is code: the round loop, the per-candidate fork,
parent selection, and the stop decision. Everything bench-specific is bundled in
an injected :class:`~codeflow.evolver.orchestrator.scoring.EvalBackend`; the
per-candidate decision (screen -> confirm -> promote) and the control arm are
injected as a :class:`GatePolicy` and a :class:`BaselineProvider`. So a weaker
driver model (Qwen / Kimi) can run the loop without remembering the funnel's
shape, and SWE-bench / AppWorld / a no-benchmark LLM judge all share one loop —
they differ only in which backend + policy + baseline get wired in.

Two per-round signals feed termination, and they are NOT the same thing:

- ``promoted`` — the parent changed: some candidate passed the gate against its
  round baseline AND beat the incumbent parent's train score (the Alg.1 L135
  argmax). A gate-passer that loses the argmax banks but does not take over.
- ``beat_vanilla`` — a candidate's full-train confirm beat the FIXED vanilla
  cold-start mean. This is the SOP's patience signal (no candidate's train mean
  beats vanilla for N consecutive rounds), measured against vanilla for every
  benchmark regardless of which
  baseline provider gates promotion. A round that erred out entirely sets
  ``errored`` instead and burns neither counter (it has its own stop).

The semantic steps are still injected callables:

- ``diagnose_fn`` (①): read the last child's trajectories, return a failure map.
- ``design_fn``   (②): pick WHYs and design candidates (:class:`AppliedPatch`).
- ``preflight_fn`` (③, optional): drop inert candidates; default keeps all.
- ``apply_fn``    (④): apply a patch on the parent, persist a child node.
- ``verdict_fn``  (⑦, optional): draft a per-round verdict for the findings log.

⑤/⑥ (screen/confirm/gate) are the ``gate_policy``'s job; the control arm each
round comes from the ``baseline_provider`` (frozen cold-start by default, or the
methodology-correct same-session provider). ``focused_source`` supplies a
candidate's WHY subset (AppWorld's Fisher gate); ``outcome_hook`` lets a bench
learn across rounds (AppWorld's attempt history), and ``inert_hook`` feeds it
the preflight-pruned candidates that never reach a DecisionContext. All
default off.

On-disk state under ``config.work_dir`` (all best-effort, all resume-safe):
``failure_map.json`` (the cross-round live map), ``nodes/<node_id>.json`` (the
node ledger: identity + git anchor + final status + gate stats, one file per
candidate), and ``findings.md`` (a human-readable per-round log with the
driver's verdict). The round journal the caller passes to :meth:`run` is the
loop-progress record those three complement.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

if TYPE_CHECKING:
    from codeflow.evolver.orchestrator.state.journal import RoundJournal

from codeflow.evolver.candidate_manifest import ManifestGateError
from codeflow.evolver.orchestrator.archive import (
    CellElite,
    GsmeArchive,
    describe_candidate,
)
from codeflow.evolver.orchestrator.config import OrchestratorConfig
from codeflow.evolver.orchestrator.gates.fisher import train_mean
from codeflow.evolver.orchestrator.gates.policy import (
    BaselineProvider,
    CandidateOutcome,
    DecisionContext,
    FiredSourceFn,
    FocusedSourceFn,
    FrozenColdStartBaseline,
    GatePolicy,
)
from codeflow.evolver.orchestrator.gates.strategies import PairedTwoSigmaGate
from codeflow.evolver.orchestrator.nodes.diagnose import merge_failure_maps
from codeflow.evolver.orchestrator.scoring import (
    EvalBackend,
    EvaluationVerdict,
    TaskEval,
    flip_summary,
)
from codeflow.evolver.orchestrator.sealed.runner import assert_no_test_leak
from codeflow.evolver.orchestrator.termination import TerminationTracker
from codeflow.evolver.scheduler.anchor_selection import AnchorSelection
from codeflow.evolver.tree.node import AppliedPatch, HarnessNode, NodeStatus


class DiagnoseFn(Protocol):
    def __call__(self, round_index: int, parent: HarnessNode) -> dict: ...


class DesignFn(Protocol):
    def __call__(self, round_index: int, failure_map: dict, parent: HarnessNode) -> list[AppliedPatch]: ...


class ApplyFn(Protocol):
    def __call__(self, parent_id: str, patch: AppliedPatch, round_index: int) -> HarnessNode: ...


# （候选项，父节点）-> 是否保留？父节点向预检提供历史语料，即触发谓词要检查的轨迹。
PreflightFn = Callable[[AppliedPatch, HarnessNode], bool]
VerdictFn = Callable[["RoundResult"], str]
OutcomeHook = Callable[[DecisionContext, CandidateOutcome], None]

# 对预检裁剪候选项使用（候选项，结果）：它从未应用，因此没有节点或 DecisionContext，
# 只有原始候选项。
InertHook = Callable[[Any, CandidateOutcome], None]
# 将单元精英的编辑实体化到父节点，作为基准候选项；None 表示无法构建组合
# （提交丢失或没有内容可叠加），应跳过。
RecombineFn = Callable[[HarnessNode, CellElite], Optional[object]]


@dataclass
class RoundResult:
    round_index: int
    parent_id: str
    next_parent_id: str
    promoted: bool
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    verdict: Optional[str] = None
    # SOP 耐心信号：本轮某候选项的完整训练确认超过固定的原始均值，与门控自身基线无关。
    beat_vanilla: bool = False
    # 所有候选项最终均失败或无结论，因此没有作出真实决策。
    errored: bool = False
    # 为事后密封解封（C3，方案 B）记录：可交付评测框架的提交及其训练 pass@1，
    # 使演化结束后无需决策期测试评分也能重建测试曲线。
    next_parent_sha: Optional[str] = None
    next_parent_train: Optional[float] = None


@dataclass
class RunResult:
    rounds: list[RoundResult] = field(default_factory=list)
    stop_reason: Optional[str] = None
    final_parent_id: Optional[str] = None
    resumed_rounds: int = 0  # 从日志重放而非重新运行的轮次数


def summarize_round(rr: RoundResult) -> str:
    """Factual one-round summary for the verdict draft / findings log."""
    lines = [f"round {rr.round_index}: parent={rr.parent_id} promoted={rr.promoted}"]
    for o in rr.outcomes:
        parts = [f"  {o.node_id}: {o.status.value}"]
        if o.screen is not None:
            parts.append(f"screen={o.screen.candidate_mean:.3f} vs van {o.screen.vanilla_mean:.3f} ({o.screen.bucket})")
        if o.paired is not None:
            parts.append(
                f"confirm={o.paired.candidate_mean:.3f} vs van "
                f"{o.paired.control_mean:.3f} z={o.paired.z:.2f} "
                f"credited={o.paired.credited_2sigma}"
            )
        if o.stats:
            parts.append(" ".join(f"{k}={v}" for k, v in o.stats.items()))
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _sha_or_none(sha: Optional[str]) -> Optional[str]:
    """A journal-safe commit SHA: the root shim's ``"unknown"`` placeholder is
    recorded as None so the post-hoc unseal never tries to check it out."""
    return None if sha in (None, "", "unknown") else sha


def _vanilla_control(vanilla_stability) -> dict[str, TaskEval]:
    """The cold-start baseline as an eval map to serve as the control arm."""
    return {tid: TaskEval(task_id=tid, passes=st.passes, attempts=st.attempts) for tid, st in vanilla_stability.items()}


class EvolutionOrchestrator:
    """Drives the seven-step funnel across rounds until a stop condition fires."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        backend: EvalBackend,
        diagnose_fn: DiagnoseFn,
        design_fn: DesignFn,
        apply_fn: ApplyFn,
        gate_policy: Optional[GatePolicy] = None,
        baseline_provider: Optional[BaselineProvider] = None,
        preflight_fn: Optional[PreflightFn] = None,
        verdict_fn: Optional[VerdictFn] = None,
        fired_source: Optional[FiredSourceFn] = None,
        focused_source: Optional[FocusedSourceFn] = None,
        outcome_hook: Optional[OutcomeHook] = None,
        evidence_hook: Optional[OutcomeHook] = None,
        inert_hook: Optional[InertHook] = None,
        seed_failure_map: Optional[dict] = None,
        archive: Optional[GsmeArchive] = None,
        recombine_fn: Optional[RecombineFn] = None,
    ) -> None:
        self._cfg = config
        self._backend = backend
        self._diagnose = diagnose_fn
        self._design = design_fn
        self._apply = apply_fn
        self._eval = backend.eval
        self._preflight = preflight_fn or (lambda _patch, _parent: True)
        self._verdict = verdict_fn
        self._fired_source = fired_source
        self._focused_source = focused_source
        self._outcome_hook = outcome_hook
        self._evidence_hook = evidence_hook
        self._inert_hook = inert_hook
        # GSME：逐单元精英库与跨单元重组器。归档会自行加载持久化状态，因此恢复运行仍保留精英。
        self._archive = archive
        self._recombine = recombine_fn

        self._vanilla_stability = backend.cold_start()
        if not self._vanilla_stability:
            raise ValueError("backend.cold_start() returned an empty baseline")
        self._train_task_ids = list(backend.train_task_ids) or sorted(self._vanilla_stability)
        self._sentinel_task_ids = self._sample_sentinels(config.anchor.n_sentinel)
        # 耐心信号使用固定比较锚点（SOP：候选项训练均值与原始版本比较，绝不与上一轮父节点
        # 比较）；无论哪个基线提供器负责晋升门控，所有基准都使用同一锚点。
        self._vanilla_train_mean = train_mean(_vanilla_control(self._vanilla_stability), self._train_task_ids)

        # 默认策略为 SWE 配对 2σ 线；默认基线为冻结冷启动（受成本约束，跨时间无效，见
        # gates.policy）。AppWorld 或同会话运行会注入自己的策略和基线。
        self._gate: GatePolicy = gate_policy or PairedTwoSigmaGate(k_screen=config.k_screen, k_confirm=config.k_confirm)
        self._baselines: BaselineProvider = baseline_provider or FrozenColdStartBaseline(
            _vanilla_control(self._vanilla_stability)
        )

        # 以机制落实密封测试铁律：若留出 ID 混入锚点或训练集，立即明确失败（SOP 第 0 节）。
        if backend.test_task_ids:
            assert_no_test_leak(
                anchor_task_ids=self.select_anchor().task_ids,
                train_task_ids=self._train_task_ids,
                sealed_test_ids=list(backend.test_task_ids),
            )

        # 跨轮实时失败图（SOP 第 2 节第 1 项）：持续累积，而非冻结。``seed_failure_map``
        # 会预填充它（分类归纳的种子，并把根加入 ``_diagnosed_parents``，使第 1 轮不再评判
        # 归纳阶段已判断的同一批轨迹）；从日志恢复时则以磁盘状态覆盖。
        self._failure_map: dict = dict(seed_failure_map) if seed_failure_map else {}

        # 按 ID 保存真实已应用节点（run() 收到根节点时以它初始化）。晋升后的父节点必须解析为
        # 含提交 SHA 和补丁的真实节点，而不是垫片；否则同会话基线或工作树评测会检出错误
        # 提交，甚至 "unknown" 提交。
        self._node_registry: dict[str, HarnessNode] = {}
        # 非 AppliedPatch 候选项（接入的基准线路）的账本元数据。此时 node.patch 为 None，
        # 若不另存，WHERE/WHY/文件/激活信息就永远不会进入节点记录。
        self._cand_meta: dict[str, dict] = {}

    @property
    def vanilla_train_mean(self) -> float:
        """The fixed vanilla train anchor (benches read it at unseal time)."""
        return self._vanilla_train_mean

    def _sample_sentinels(self, n: int) -> list[str]:
        """Deterministic default sentinel set (used when no node id is at hand)."""
        stable, fragile = self._sentinel_pools()
        return stable[: n - n // 2] + fragile[: n // 2]

    def _sentinel_pools(self) -> tuple[list[str], list[str]]:
        from codeflow.evolver.analysis.stability_bucket import StabilityBucket

        train = set(self._train_task_ids)
        stable, fragile = [], []
        for tid, st in self._vanilla_stability.items():
            if tid not in train:
                continue
            if st.bucket == StabilityBucket.STABLE_PASS:
                stable.append(tid)
            elif st.bucket in (StabilityBucket.BORDERLINE_2_3, StabilityBucket.BORDERLINE_1_3):
                fragile.append(tid)
        return sorted(stable), sorted(fragile)

    def _sentinels_for(self, node_id: str, n: int) -> list[str]:
        """Per-candidate regression controls: half stable-pass, half borderline.

        Stratified because over-trigger regressions concentrate on BORDERLINE
        tasks (fragile passes flip first) — a stable-only sentinel set is
        systematically blind to them (observed live: a candidate with a 58%
        passing-task regression rate sailed through 3 stable sentinels).
        Rotated per candidate (hash of node_id) so no fixed control set can be
        sailed past twice."""
        import hashlib

        stable, fragile = self._sentinel_pools()

        def pick(pool: list[str], m: int, salt: str) -> list[str]:
            if not pool or m <= 0:
                return []
            h = int(hashlib.sha256(f"{salt}:{node_id}".encode()).hexdigest(), 16)
            start = h % len(pool)
            return [pool[(start + i) % len(pool)] for i in range(min(m, len(pool)))]

        return pick(stable, n - n // 2, "stable") + pick(fragile, n // 2, "fragile")

    def select_anchor(self, affinity: dict[str, float] | None = None) -> AnchorSelection:
        return self._backend.anchor(affinity)

    def run(
        self,
        root_node_id: str,
        journal: Optional["RoundJournal"] = None,
        *,
        root_node: Optional[HarnessNode] = None,
    ) -> RunResult:
        """Run rounds from ``root_node_id`` (vanilla) until termination.

        If ``journal`` is given, previously-completed rounds are replayed to seed
        the termination counters, current parent, and round index, and the loop
        continues from the next round — a killed run resumes without re-running
        the evals it already did. Replay also re-registers each promoted
        parent's recorded commit SHA, so a resumed round's design / worktree
        eval / baseline fallback sees the real commit, not an "unknown" shim;
        and the accumulated failure map is re-read from disk so the cross-round
        live map (SOP §2 ①) is not truncated back to empty.

        ``root_node`` (optional) registers the real vanilla node so a round whose
        parent is the root can resolve its commit SHA (needed by same-session
        baselines / worktree evals); without it the root falls back to a shim.
        """
        term = TerminationTracker(
            patience=self._cfg.termination.patience,
            max_rounds=self._cfg.termination.max_rounds,
            max_consecutive_errors=self._cfg.termination.max_consecutive_errors,
        )
        result = RunResult()
        parent_id = root_node_id
        # 现任节点的训练分数——挑战者必须超过该最大值门槛才能接任父节点（算法 1 第 135 行）。
        # 根节点分数取原始版本均值。
        parent_score = self._vanilla_train_mean
        round_index = 0
        if root_node is not None:
            self._node_registry[root_node.node_id] = root_node

        if journal is not None:
            records = journal.load()
            for rec in records:
                term.record_round(
                    promoted=rec.get("beat_vanilla", rec["promoted"]),
                    errored=rec.get("errored", False),
                )
                round_index = rec["round_index"]
                parent_id = rec["next_parent_id"]
                if rec.get("next_parent_train") is not None:
                    parent_score = rec["next_parent_train"]
                sha = rec.get("next_parent_sha")
                if (
                    sha
                    and rec["next_parent_id"] != rec["parent_id"]
                    and rec["next_parent_id"] not in self._node_registry
                ):
                    self._node_registry[rec["next_parent_id"]] = HarnessNode(
                        node_id=rec["next_parent_id"],
                        parent_id=rec["parent_id"],
                        git_commit_sha=sha,
                        git_branch="journal-resume",
                        created_at=HarnessNode.utc_now(),
                        created_at_iter=rec["round_index"],
                    )
                result.resumed_rounds += 1
            if records:
                self._reload_failure_map()
            stop, reason = term.should_stop()
            if stop:
                result.stop_reason = reason
                result.final_parent_id = parent_id
                return result

        while True:
            round_index += 1
            round_result = self._run_round(round_index, parent_id, parent_score)
            result.rounds.append(round_result)
            if journal is not None:
                journal.append(round_result)
            self._persist_node_records(round_result)
            self._append_findings(round_result)
            if self._archive is not None:
                self._archive.save()
            parent_id = round_result.next_parent_id
            if round_result.promoted and round_result.next_parent_train is not None:
                parent_score = round_result.next_parent_train

            term.record_round(promoted=round_result.beat_vanilla, errored=round_result.errored)
            stop, reason = term.should_stop()
            if stop:
                result.stop_reason = reason
                break

        result.final_parent_id = parent_id
        return result

    def _run_round(self, round_index: int, parent_id: str, parent_score: float) -> RoundResult:
        # 门控 0（SOP 第 0 节，所有评分之前）：本轮评分前验证环境。脏环境（沙箱停机、
        # 网络不可路由、验证器无法输出结果）会使所有分数失效，因此允许抛错；修复环境后
        # 再恢复。未接入预检的基准跳过此步。
        if self._backend.precheck is not None:
            self._backend.precheck()
        parent = self._load_parent(parent_id)
        anchor = self.select_anchor()
        # 对照组本身可能是评测（同会话配对）或磁盘重建；此处的暂时失败和其他评测失败
        # 一样仅影响本轮：记录错误轮次并交由错误计数器决定，而不是中止无人值守运行且
        # 不留下日志记录。
        try:
            baseline = self._baselines.for_round(
                round_index,
                parent,
                eval=self._eval,
                train_task_ids=self._train_task_ids,
                anchor=anchor,
            )
        except Exception as exc:  # noqa: BLE001 — 记录后继续，不中止
            outcome = CandidateOutcome(
                f"r{round_index}-baseline",
                NodeStatus.errored,
                stats={"phase": "baseline", "error": repr(exc)},
            )
            return RoundResult(
                round_index=round_index,
                parent_id=parent_id,
                next_parent_id=parent_id,
                promoted=False,
                outcomes=[outcome],
                errored=True,
                next_parent_sha=_sha_or_none(parent.git_commit_sha),
            )

            # ① 诊断并合入跨轮实时失败图；② 设计；③ 预检裁剪。若父节点在早先轮次已诊断且
            # 此后未晋升，则不重复诊断：其轨迹未变化，重判只会重复消耗驱动器，并在累积图中
            # 重复计算同一失败。诊断/设计失败不得中止整个运行（与下方逐候选项捕获原则一致）：
            # 在错误结果上记录原因，以无候选项结束本轮；持续故障由追踪器的错误计数器带着
            # 真实原因停止，而不是消耗耐心额度。
        outcomes: list[CandidateOutcome] = []
        try:
            diagnosed = set(self._failure_map.get("_diagnosed_parents") or [])
            if parent_id not in diagnosed:
                round_map = self._diagnose(round_index, parent)
                self._failure_map = merge_failure_maps(self._failure_map, round_map)
                self._failure_map["_diagnosed_parents"] = sorted(diagnosed | {parent_id})
                self._persist_failure_map()
            candidates = []
            for i, c in enumerate(self._design(round_index, self._failure_map, parent)):
                if self._preflight(c, parent):
                    candidates.append(c)
                else:
                    # ③ 零推理裁剪，必须记录（绝不静默丢弃）：裁掉无作用候选项也是本轮真实决策。
                    outcome = CandidateOutcome(
                        f"r{round_index}-preflight{i}",
                        NodeStatus.pruned_inert,
                        stats={
                            "phase": "preflight",
                            "why": str(getattr(c, "why", "")),
                            "reason": "trigger has zero historical hits",
                        },
                    )
                    outcomes.append(outcome)
                    if self._inert_hook is not None:
                        try:
                            self._inert_hook(c, outcome)
                        except Exception:  # noqa: BLE001 — 建议性学习钩子，失败不致命
                            pass
        except Exception as exc:  # noqa: BLE001 — 记录后继续，不中止
            outcomes.append(
                CandidateOutcome(
                    f"r{round_index}-design",
                    NodeStatus.errored,
                    stats={"phase": "diagnose_design", "error": repr(exc)},
                )
            )
            candidates = []

            # GSME 跨单元重组：把父节点谱系尚未覆盖单元的精英叠加到父节点，作为普通候选项
            # 通过相同的应用 -> 门控流水线。刻意放在设计阶段的 try/except 之外；驱动器故障
            # 不应阻止确定性重组。
        if self._archive is not None and self._recombine is not None:
            try:
                elites = self._archive.eligible_elites(parent_id, limit=self._cfg.budget.recombinations_per_round)
                for elite in elites:
                    recomb = self._recombine(parent, elite)
                    if recomb is None:
                        self._archive.record_pairing(parent_id, elite.node_id, "recombine_failed")
                        continue
                    candidates.append(recomb)
            except Exception as exc:  # noqa: BLE001 — 记录后继续
                outcomes.append(
                    CandidateOutcome(
                        f"r{round_index}-recombine",
                        NodeStatus.errored,
                        stats={"phase": "recombine", "error": repr(exc)},
                    )
                )

        best_node_id: Optional[str] = None
        best_score = -1.0

        for idx, patch in enumerate(candidates):
            # 单个候选项的应用/评测崩溃不得拖垮整轮：捕获异常，在 ``errored`` 结果上记录
            # 原因，然后继续（C）。
            elite_id = getattr(patch, "elite_node_id", None)
            try:
                node = self._apply(parent_id, patch, round_index)  # ④
            except ManifestGateError as exc:
                if elite_id and self._archive is not None:
                    self._archive.record_pairing(parent_id, elite_id, "rejected_at_manifest")
                outcome = CandidateOutcome(
                    str(getattr(patch, "candidate_id", "") or f"r{round_index}-cand{idx}"),
                    NodeStatus.rejected_at_manifest,
                    stats={"phase": "G5", "error": str(exc)},
                    verdict=EvaluationVerdict.rejected,
                )
                outcomes.append(outcome)
                if self._inert_hook is not None:
                    try:
                        self._inert_hook(patch, outcome)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            except Exception as exc:  # noqa: BLE001 — 记录后跳过，不中止
                if elite_id and self._archive is not None:
                    self._archive.record_pairing(parent_id, elite_id, "errored")
                outcome = CandidateOutcome(
                    str(getattr(patch, "candidate_id", "") or f"r{round_index}-cand{idx}"),
                    NodeStatus.errored,
                    stats={"phase": "apply", "error": repr(exc)},
                )
                outcomes.append(outcome)
                if self._inert_hook is not None:
                    try:
                        self._inert_hook(patch, outcome)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            self._node_registry[node.node_id] = node
            meta = describe_candidate(patch)
            if meta:
                self._cand_meta[node.node_id] = meta
            ctx = DecisionContext(
                node=node,
                parent_id=parent_id,
                round_index=round_index,
                eval=self._eval,
                baseline=baseline,
                train_task_ids=self._train_task_ids,
                anchor=anchor,
                focused_task_ids=(self._focused_source(node) if self._focused_source else []),
                sentinel_task_ids=self._sentinels_for(node.node_id, self._cfg.anchor.n_sentinel),
                fired_source=self._fired_source,
            )
            try:
                outcome = self._gate.decide(ctx)  # ⑤⑥ 委托给策略
            except Exception as exc:  # noqa: BLE001 — 记录后跳过，不中止
                if elite_id and self._archive is not None:
                    self._archive.record_pairing(parent_id, elite_id, "errored")
                outcome = CandidateOutcome(
                    node.node_id,
                    NodeStatus.errored,
                    stats={"phase": "decide", "error": repr(exc)},
                )
                outcomes.append(outcome)
                if self._evidence_hook is not None:
                    try:
                        self._evidence_hook(ctx, outcome)
                    except Exception as evidence_exc:  # noqa: BLE001
                        outcome.stats["evidence_error"] = repr(evidence_exc)
                if self._outcome_hook is not None:
                    try:
                        self._outcome_hook(ctx, outcome)
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if elite_id:
                outcome.stats["recombination_of"] = elite_id
                # 翻转表（SOP 第 2 节第 1 项：哪些任务发生翻转）：相对本轮对照组获救/退化，
                # 同时记录到账本和实时失败图，使下一轮诊断/设计看到因果反馈，而非只有静态失败集。
            if outcome.confirm_evals:
                flips = flip_summary(outcome.confirm_evals, baseline.evals, self._train_task_ids)
                outcome.stats["flips"] = flips
                self._failure_map.setdefault("_flips", {})[node.node_id] = {
                    "round": round_index,
                    "vs": baseline.label,
                    **flips,
                }
            if self._evidence_hook is not None:
                try:
                    self._evidence_hook(ctx, outcome)
                except Exception as exc:  # noqa: BLE001
                    outcome.status = NodeStatus.errored
                    outcome.verdict = EvaluationVerdict.failed
                    outcome.stats.update(
                        phase="candidate_evidence",
                        error=repr(exc),
                    )
            outcomes.append(outcome)
            if self._archive is not None:
                try:
                    self._archive.consider(
                        parent_id=parent_id,
                        node=node,
                        cand=patch,
                        outcome=outcome,
                        round_index=round_index,
                        vanilla_train_mean=self._vanilla_train_mean,
                    )
                except Exception:  # noqa: BLE001 — 维护候选库不能拖垮本轮
                    pass
            if self._outcome_hook is not None:
                try:
                    self._outcome_hook(ctx, outcome)
                except Exception:  # noqa: BLE001 — 建议性学习钩子，失败不致命
                    pass
            if outcome.promoted:
                self._baselines.on_promote(node, outcome, train_task_ids=self._train_task_ids)
                if outcome.score > best_score:
                    best_score = outcome.score
                    best_node_id = node.node_id

        if any(o.confirm_evals for o in outcomes):
            self._persist_failure_map()  # 本轮翻转已合入实时失败图

            # ⑦ 选择父节点——算法 1 第 135 行的最大值：本轮最佳门控通过者只有超过现任节点的
            # 训练分数才接任；平局保留现任节点（无改进则不抖动）。未赢得最大值的门控通过者
            # 仍会入库（状态/归档），只是不会成为父节点。棘轮基线下门控已隐含此规则；冻结
            # 原始对照下，该规则保持冠军链单调。
        promoted = best_node_id is not None and best_score > parent_score
        next_parent_id = best_node_id if promoted else parent_id
        next_parent_node = self._node_registry.get(next_parent_id) or parent
        beat_vanilla = any(
            o.verdict is EvaluationVerdict.accepted and o.confirm_evals and o.score > self._vanilla_train_mean
            for o in outcomes
        )
        errored = bool(outcomes) and all(
            o.verdict in {EvaluationVerdict.failed, EvaluationVerdict.inconclusive} for o in outcomes
        )

        round_result = RoundResult(
            round_index=round_index,
            parent_id=parent_id,
            next_parent_id=next_parent_id,
            promoted=promoted,
            outcomes=outcomes,
            beat_vanilla=beat_vanilla,
            errored=errored,
            next_parent_sha=_sha_or_none(next_parent_node.git_commit_sha),
            next_parent_train=(best_score if promoted else None),
        )
        if self._verdict is not None:
            round_result.verdict = self._verdict(round_result)
        return round_result

    def _persist_failure_map(self) -> None:
        """Write the accumulated failure map for audit/resume (best-effort)."""
        try:
            path = self._cfg.failure_map_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._failure_map, indent=2))
        except OSError:
            pass

    def _reload_failure_map(self) -> None:
        """Re-read the accumulated failure map from disk (resume path)."""
        try:
            path = self._cfg.failure_map_path
            if path.exists():
                self._failure_map = json.loads(path.read_text())
        except (OSError, ValueError):
            pass

    def _persist_node_records(self, rr: RoundResult) -> None:
        """Write the node ledger (SOP §3.1): one JSON per candidate under
        ``work_dir/nodes/``, carrying identity + git anchor + final status +
        gate stats. Best-effort — the ledger is the audit trail, not control
        state (resume runs off the journal)."""
        try:
            ndir = self._cfg.nodes_dir
            ndir.mkdir(parents=True, exist_ok=True)
            for o in rr.outcomes:
                node = self._node_registry.get(o.node_id)
                if node is None:  # 出错的伪候选项从未得到节点
                    continue
                patch = node.patch
                patch_to_dict = getattr(patch, "to_dict", None)
                rec = {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "git_commit_sha": node.git_commit_sha,
                    "git_branch": node.git_branch,
                    "created_at": node.created_at,
                    "created_at_iter": node.created_at_iter,
                    "patch": (
                        patch_to_dict() if callable(patch_to_dict) else (repr(patch) if patch is not None else None)
                    ),
                    "status": o.status.value,
                    "verdict": o.verdict.value,
                    "round_index": rr.round_index,
                    "score": o.score,
                }
                if o.node_id in self._cand_meta:
                    rec["candidate"] = self._cand_meta[o.node_id]
                if o.screen is not None:
                    rec["screen"] = dataclasses.asdict(o.screen)
                if o.paired is not None:
                    rec["paired"] = dataclasses.asdict(o.paired)
                if o.stats:
                    rec["stats"] = o.stats
                (ndir / f"{o.node_id}.json").write_text(json.dumps(rec, indent=2, default=str))
        except OSError:
            pass

    def _append_findings(self, rr: RoundResult) -> None:
        """Append the per-round findings-log entry (SOP §3 layer 1) to
        ``work_dir/findings.md`` — the human-readable record of what each round
        tried, what the gates said, and the driver's verdict. Best-effort."""
        try:
            path = self._cfg.findings_path
            path.parent.mkdir(parents=True, exist_ok=True)
            block = [f"\n## round {rr.round_index}\n", "```", summarize_round(rr), "```"]
            if rr.verdict:
                block.append(f"\nverdict: {rr.verdict}")
            with path.open("a") as f:
                f.write("\n".join(block) + "\n")
        except OSError:
            pass

    def _load_parent(self, parent_id: str) -> HarnessNode:
        # 晋升父节点已在早先轮次应用（或恢复时从日志重新注册），因此注册表中有带真实提交 SHA
        # 的节点，直接返回。只有从未应用的根节点回退为垫片；向 run() 传入 ``root_node``
        # 也能为它提供真实 SHA。
        node = self._node_registry.get(parent_id)
        if node is not None:
            return node
        return HarnessNode(
            node_id=parent_id,
            parent_id=None,
            git_commit_sha="unknown",
            git_branch="unknown",
            created_at=HarnessNode.utc_now(),
            created_at_iter=0,
        )


__all__ = [
    "EvolutionOrchestrator",
    "CandidateOutcome",
    "RoundResult",
    "RunResult",
    "DiagnoseFn",
    "DesignFn",
    "ApplyFn",
    "PreflightFn",
    "VerdictFn",
    "InertHook",
    "OutcomeHook",
    "RecombineFn",
    "summarize_round",
]
