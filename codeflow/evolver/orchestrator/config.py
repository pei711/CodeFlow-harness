"""用一组对象装配完整 seven-step funnel 的 Orchestrator config。

Everything the FSM needs that is *not* code: where the scorer lives
(``framework``), where the vanilla thick ledger sits (``cold_start_ledger_dir``,
the fixed comparison baseline — the funnel always compares against vanilla, not
the previous parent), the anchor/screen knobs, the per-round design budget, the
termination thresholds, and the on-disk state roots.

driver model 不在这里构造；``driver_llm_spec`` 交给
``codeflow.evolver.judge.llm_client.build_judge_llm``，使 config 可序列化为 YAML，test 也能不经
factory 注入 ``MockBackend``。self-hosted Qwen、Claude、OpenRouter 等支持来自现有 backend。

config 只定义 funnel 与 state roots，不证明 scorer/model 可用，也不承载 run progress。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnchorParams:
    """K=1 anchor screen 的 slot budget 与 cull width，见 ``select_anchor``。

    default 12 sentinel 由 stable/borderline 分层轮换；n_icebreaker/n_borderline 控制组成，
    cull_sigma_mult 控制 pruning 宽度。它们是策略参数，不是 measurement result。
    """

    # 每个候选项使用 12 个对照：6 个稳定任务和 6 个边界任务，在 loop._sentinels_for 中分层并
    # 轮换。退化集中在边界任务；观察到仅 3 个稳定哨兵会漏掉退化 58% 的候选项。
    n_sentinel: int = 12
    n_icebreaker: int = 5
    n_borderline: int = 7
    cull_sigma_mult: float = 1.5


@dataclass(frozen=True)
class Budget:
    """per-round design budget（SOP §2 ②：1-2 WHY x 2-3 candidates）。

    ``driver_token_budget`` 可限制 driver；``recombinations_per_round`` 限制 designed candidate
    后追加的 deterministic cross-cell GSME recombination，0 表示关闭。
    """

    max_why_per_round: int = 2
    candidates_per_why: int = 3
    driver_token_budget: int | None = None
    recombinations_per_round: int = 1


@dataclass(frozen=True)
class Termination:
    """Loop stop conditions；只与 vanilla train 比较，绝不 consult test。

    ``patience`` 是连续无 candidate beat vanilla 的 primary exhaustion signal，``max_rounds``
    是 hard cap；``max_consecutive_errors`` 限制 driver/infra/incomplete measurement 造成的
    no-decision run，以 honest ``errors_exhausted`` 停止，不让错误 burn patience 并伪装成探索耗尽。
    """

    patience: int = 10
    max_rounds: int = 20
    max_consecutive_errors: int = 5


@dataclass(frozen=True)
class OrchestratorConfig:
    """bench-neutral top-level Orchestrator config。

    scorer、dataset split、cold-start baseline 与 anchor 都隐藏在 injected
    :class:`~codeflow.evolver.orchestrator.scoring.EvalBackend` 后；本对象只保存 driver spec、funnel
    numeric knob 与 on-disk roots。``anchor`` 是 backend factory 构建 trial-ledger anchor 时可用
    的 neutral tuning。post-init 规范化 Path，并要求 K >= 1。
    """

    repo_root: Path
    work_dir: Path
    driver_llm_spec: dict[str, Any]

    k_screen: int = 1
    k_confirm: int = 3
    anchor: AnchorParams = field(default_factory=AnchorParams)
    budget: Budget = field(default_factory=Budget)
    termination: Termination = field(default_factory=Termination)

    # 密封测试：由脚本评分并写入驱动器永不读取的目录，因此密封规则由隔离保证，而非依赖驱动器自律。
    sealed_test_split: str = "test"
    sealed_output_dir: Path | None = None

    def __post_init__(self) -> None:
        for name in ("repo_root", "work_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.sealed_output_dir is not None:
            object.__setattr__(self, "sealed_output_dir", Path(self.sealed_output_dir))
        if self.k_screen < 1 or self.k_confirm < 1:
            raise ValueError("k_screen and k_confirm must be >= 1")

    # work_dir 下的约定磁盘状态布局。
    @property
    def nodes_dir(self) -> Path:
        return self.work_dir / "nodes"

    @property
    def failure_map_path(self) -> Path:
        return self.work_dir / "failure_map.json"

    @property
    def archive_path(self) -> Path:
        return self.work_dir / "archive.json"

    @property
    def findings_path(self) -> Path:
        return self.work_dir / "findings.md"

    @property
    def journal_dir(self) -> Path:
        return self.work_dir / "journal"


__all__ = [
    "AnchorParams",
    "Budget",
    "Termination",
    "OrchestratorConfig",
]
