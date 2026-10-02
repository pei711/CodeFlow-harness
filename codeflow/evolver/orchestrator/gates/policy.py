"""定义两个 benchmark line 分化位置的 pluggable decision policy seam。

The seven-step funnel's outer loop is identical across benchmarks; only the
per-candidate *decision* (screen -> confirm -> promote) and the *control arm* it
compares against differ. Those two concerns are captured here as two protocols
so ``loop._run_round`` stays a thin driver:

- :class:`GatePolicy` — given a :class:`DecisionContext`, run whatever
  screen/confirm/significance logic the bench uses and return one
  :class:`CandidateOutcome`. The policy owns its own ``eval`` calls because the
  two lines eval different task sets at different stages (K=1 anchor vs K=3
  focused subset). The loop never scores for itself anymore.
- :class:`BaselineProvider` — supply the control arm (a :class:`Baseline`) for
  the round and absorb the ratchet on promotion. Frozen cold-start, per-parent
  frozen, and same-session paired are the three implementations; the gate never
  knows which produced its control, which is why the two seams are orthogonal.

empirical regime-shift 表明 whole run 可移动约 12pp，跨时间 frozen baseline 不可靠，因此
:class:`SameSessionPairedBaseline` 是 methodology-correct choice；frozen variant 只作为明确标注的
cost-bound fallback。policy decision 只基于 train evidence，不读取 sealed test。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from codeflow.evolver.orchestrator.gates.fisher import train_mean
from codeflow.evolver.orchestrator.scoring import EvalFn, EvaluationVerdict, TaskEval
from codeflow.evolver.scheduler.anchor_selection import AnchorSelection
from codeflow.evolver.tree.node import HarnessNode, NodeStatus

if TYPE_CHECKING:
    from codeflow.evolver.orchestrator.gates.paired import PairedResult
    from codeflow.evolver.orchestrator.gates.pipeline import GateResult
    from codeflow.evolver.orchestrator.nodes.screen import ScreenResult


# （节点，任务 ID）-> 已触发任务集合；节点无归因数据（候选项未插桩或未接入收集）时为 None。
# None 让门控 b 开放通过（跳过），空集合则如实表示“从未触发”。
FiredSourceFn = Callable[[HarnessNode, list[str]], Optional[set]]
FocusedSourceFn = Callable[[HarnessNode], list[str]]


@dataclass
class CandidateOutcome:
    """一个 candidate 经过 funnel 后的 policy-agnostic outcome。

    ``score``/``confirm_evals``/``stats`` 让 parent selection 与 baseline ratchet 留在 Loop；
    optional screen/paired/gate 保存 paired-line detail，stats 保存 Fisher p、focused rate、lift 等。
    ``promoted`` 只有 status 与 verdict 同时 accepted 才为真。
    """

    node_id: str
    status: NodeStatus
    score: float = 0.0
    confirm_evals: dict[str, TaskEval] = field(default_factory=dict)
    screen: Optional["ScreenResult"] = None
    paired: Optional["PairedResult"] = None
    gate: Optional["GateResult"] = None
    stats: dict = field(default_factory=dict)
    verdict: EvaluationVerdict | None = None

    def __post_init__(self) -> None:
        if isinstance(self.verdict, str):
            self.verdict = EvaluationVerdict(self.verdict)
        if self.verdict is not None:
            return
        if self.status is NodeStatus.promoted_to_baseline:
            self.verdict = EvaluationVerdict.accepted
        elif self.status is NodeStatus.errored:
            self.verdict = EvaluationVerdict.failed
        else:
            self.verdict = EvaluationVerdict.rejected

    @property
    def promoted(self) -> bool:
        return self.status == NodeStatus.promoted_to_baseline and self.verdict is EvaluationVerdict.accepted


@dataclass(frozen=True)
class Baseline:
    """一个 round 的 control arm：per-task eval、fixed-denominator train mean 与 label。"""

    evals: dict[str, TaskEval]
    mean: float
    label: str


@dataclass
class DecisionContext:
    """Loop 交给 :class:`GatePolicy` 判断单个 candidate 的完整 context。

    包含 node/parent/round、EvalFn、baseline、train/anchor/focused/sentinel task 与 optional fired
    source。context 不包含 sealed test result。
    """

    node: HarnessNode
    parent_id: str
    round_index: int
    eval: EvalFn
    baseline: Baseline
    train_task_ids: list[str]
    anchor: Optional[AnchorSelection] = None
    focused_task_ids: list[str] = field(default_factory=list)
    # 作为退化防护带入聚焦评测的稳定通过对照任务（SOP 第 2 节第 5a 项，“2 到 3 个全通过
    # 哨兵”）：帮助 WHY 子集的修复不得破坏原本通过的任务。空集合表示不防护。
    sentinel_task_ids: list[str] = field(default_factory=list)
    fired_source: Optional[FiredSourceFn] = None


class GatePolicy(Protocol):
    def decide(self, ctx: DecisionContext) -> CandidateOutcome: ...


class BaselineProvider(Protocol):
    def for_round(
        self,
        round_index: int,
        parent: HarnessNode,
        *,
        eval: EvalFn,
        train_task_ids: list[str],
        anchor: Optional[AnchorSelection],
    ) -> Baseline: ...

    def on_promote(self, node: HarnessNode, outcome: CandidateOutcome, *, train_task_ids: list[str]) -> None: ...


class FrozenColdStartBaseline:
    """每轮复用同一 vanilla cold-start control 的 baseline provider。

    它 cross-time-invalid，因为 later round 可能处于不同 regime；这是 SWE default 的 cost-bound
    fallback，不是 methodology-correct choice。promotion 不移动 baseline。
    """

    def __init__(self, control_evals: dict[str, TaskEval], *, label: str = "vanilla"):
        self._evals = dict(control_evals)
        self._label = label

    def for_round(self, round_index, parent, *, eval, train_task_ids, anchor) -> Baseline:
        return Baseline(self._evals, train_mean(self._evals, train_task_ids), self._label)

    def on_promote(self, node, outcome, *, train_task_ids) -> None:  # 冻结基线永不移动
        return None


class PerParentFrozenBaseline:
    """per-parent frozen control，promoted node confirm eval 成为 child baseline。

    这是 AppWorld ratchet，也 cross-time-invalid。``fallback`` 可在 resume 时从 durable confirm
    artifact 重建 missing parent；artifact 缺失返回 None，``for_round`` 随后抛错，绝不静默用空
    baseline。promotion 用 fixed train denominator 存入新 Baseline。
    """

    def __init__(
        self,
        seed: dict[str, Baseline],
        *,
        fallback: Optional[Callable[[HarnessNode, list[str]], Optional[Baseline]]] = None,
    ):
        self._by_parent: dict[str, Baseline] = dict(seed)
        self._fallback = fallback

    def for_round(self, round_index, parent, *, eval, train_task_ids, anchor) -> Baseline:
        baseline = self._by_parent.get(parent.node_id)
        if baseline is None and self._fallback is not None:
            baseline = self._fallback(parent, list(train_task_ids))
            if baseline is not None:
                self._by_parent[parent.node_id] = baseline
        if baseline is None:
            raise KeyError(
                f"no baseline for parent {parent.node_id!r} "
                f"(resumed without a fallback, or its confirm artifacts are gone)"
            )
        return baseline

    def on_promote(self, node, outcome, *, train_task_ids) -> None:
        # 完整训练集均值（SOP 第 0 节固定分母）：确认运行遗漏的任务仍必须按 0 计，使该基线与
        # 下一轮候选组使用相同分母基础。
        evals = outcome.confirm_evals
        mean = train_mean(evals, train_task_ids)
        self._by_parent[node.node_id] = Baseline(evals, mean, f"{node.node_id}_confirm")


class SameSessionPairedBaseline:
    """每轮重跑 parent Harness 作为 same-session control，C0 使用 vanilla。

    candidate/control 总在同一 session/window 测量，因此 regime shift 下 methodology-correct，
    代价约 2x eval。backend 的 EvalFn 必须能重现 parent harness。baseline 每轮重测、不 ratchet cache。
    """

    def __init__(self, k: int, *, label: str = "control"):
        self._k = k
        self._label = label

    def for_round(self, round_index, parent, *, eval, train_task_ids, anchor) -> Baseline:
        evals = eval(parent, train_task_ids, self._k, f"{self._label}_r{round_index}")
        return Baseline(evals, train_mean(evals, train_task_ids), f"{parent.node_id}_{self._label}_r{round_index}")

    def on_promote(self, node, outcome, *, train_task_ids) -> None:  # 每轮重新测量
        return None


def make_frozen_baseline(
    *,
    root_node_id: str,
    vanilla_dir: "Path",
    kept_reader: Callable[["Path"], dict[str, TaskEval]],
    confirm_dir_of: Callable[[HarnessNode], "Path"],
    train_task_ids: list[str],
    seed_label: str,
) -> PerParentFrozenBaseline:
    """从 vanilla ledger 装配 :class:`PerParentFrozenBaseline`。

    seed 经 ``kept_reader`` 读取 infra-rerun KEPT overlay，使 control 与 candidate 使用同一 SOP §0
    salvage rule；resume fallback 同样回读 promoted parent confirm dir。bench-specific 只有 reader
    format 与 confirm locator。必须在 vanilla cold start 已 materialize ledger 后调用。
    """
    van_evals = kept_reader(vanilla_dir)

    def fallback(parent: HarnessNode, train_ids: list[str]) -> Optional[Baseline]:
        d = vanilla_dir if parent.node_id == root_node_id else confirm_dir_of(parent)
        try:
            evals = kept_reader(d)
        except FileNotFoundError:
            return None
        return Baseline(evals, train_mean(evals, train_ids), f"{d.name}(resumed)")

    return PerParentFrozenBaseline(
        seed={root_node_id: Baseline(van_evals, train_mean(van_evals, list(train_task_ids)), seed_label)},
        fallback=fallback,
    )


__all__ = [
    "CandidateOutcome",
    "Baseline",
    "DecisionContext",
    "GatePolicy",
    "BaselineProvider",
    "FrozenColdStartBaseline",
    "PerParentFrozenBaseline",
    "SameSessionPairedBaseline",
    "make_frozen_baseline",
    "FiredSourceFn",
    "FocusedSourceFn",
]
