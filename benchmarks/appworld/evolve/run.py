"""Entrypoint: assemble the in-package AppWorld evolution orchestrator.

Everything generic (round loop, focused-Fisher gate, per-parent frozen baseline,
edit-then-commit apply, termination, journal/resume) comes from
``codeflow.evolver.orchestrator``; only the AppWorld brain is wired here:

- diagnose_fn = W1-W7 judge over the parent's failing trajectories
- design_fn   = bash-editor producing candidate file edits off the parent commit
- apply_fn    = commit those edits as a real child commit (edit-then-commit)
- eval        = check the candidate commit out into a worktree, run batch.py there
                (cwd=worktree, zero live-repo mutation)
- baseline    = per-parent frozen, seeded with the vanilla van0 out-dir; on
                resume a missing parent's baseline is rebuilt from its confirm
                out-dir on disk
- focused_source / outcome_hook = the WHY's evidence subset / cross-round history
- verdict_fn  = the driver drafts each round's findings-log narrative

The scorer subprocess + driver endpoints are external, so an end-to-end run is
validated only in the real AppWorld environment; this module is the wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from benchmarks.appworld.evolve import adapter as aw_adapter
from benchmarks.appworld.evolve.adapter import make_appworld_backend
from benchmarks.appworld.evolve.diagnose import (
    DEFAULT_APPWORLD_TAXONOMY,
    make_appworld_diagnose_fn,
)
from benchmarks.appworld.evolve.editor import make_bash_editor_design_fn
from benchmarks.appworld.evolve.eval import (
    deletions_of,
    files_of,
    make_appworld_eval_fn,
    prepare_candidate_manifest,
)
from benchmarks.appworld.evolve.precheck import make_appworld_precheck
from benchmarks.appworld.evolve.trajectories import (
    build_failed_attempt_renderer,
    build_out_dir_trajectory_source,
    build_passing_ids_source,
    render_candidate_failure,
)
from codeflow.evolver.orchestrator.config import Budget, OrchestratorConfig
from codeflow.evolver.orchestrator.gates.policy import make_frozen_baseline
from codeflow.evolver.orchestrator.gates.strategies import (
    FocusedFisherGate,
    confirm_job_name,
)
from codeflow.evolver.orchestrator.loop import EvolutionOrchestrator
from codeflow.evolver.orchestrator.nodes.taxonomy import resolve_taxonomy
from codeflow.evolver.orchestrator.production import build_evolution_orchestrator
from codeflow.evolver.tree.node import HarnessNode


def build_appworld_orchestrator(
    *,
    config: OrchestratorConfig,
    aw_cfg: "aw_adapter.AppWorldConfig",
    repo_root: str | Path,
    base_sha: str,
    driver_call_fn: Callable[[list], str],
    design_call_fn: Callable[[list], str],
    verdict_call_fn: Optional[Callable[[list], str]] = None,
    vanilla_out_dir: str | Path,
    train_task_ids: list[str],
    runs_root: str | Path,
    ws_root: str | Path,
    worktree_root: str | Path,
    root_node_id: str = "C0",
    test_task_ids: list[str] = (),
    budget: Optional[Budget] = None,
    min_confirm_lift: float = 0.0,
    exp_of: Optional[Callable[[HarnessNode], str]] = None,
    render_failed: Optional[Callable[[str], str]] = None,
    taxonomy_mode: str = "hardcoded",
    taxonomy_path: Optional[str | Path] = None,
    precheck: Optional[Callable[[], None]] = None,
    require_beacon: bool = True,
    zero_hit_preflight: bool = False,
    whitelist_prefixes: Optional[tuple[str, ...]] = None,
    why_selection: str = "driver",
    analysis_mode: str = "mapreduce",
    agentic_model: str = "claude-opus-4-8",
    baseline_mode: str = "frozen",
) -> EvolutionOrchestrator:
    """Wire the full AppWorld evolution into one :class:`EvolutionOrchestrator`.

    ``taxonomy_mode`` selects the WHY/WHERE taxonomy: ``"hardcoded"`` (default,
    the hand-derived W1-W7) or ``"induce"`` (discover it once from vanilla failures and cache
    to ``taxonomy_path`` / ``work_dir/taxonomy.json``).

    Promotion is the SOP navigator condition (full-train mean beats the parent
    baseline; the credited paired-2σ label is reported alongside on the
    outcome). ``min_confirm_lift`` optionally demands a minimum lift on top.

    ``precheck`` is the per-round Gate0 env health check; default =
    :func:`make_appworld_precheck` over ``aw_cfg`` (appworld install present, no
    orphan env servers on the batch ports, subject endpoint answering). Pass
    ``lambda: None`` to disable.

    ``zero_hit_preflight`` (default off) enables the SOP §2 ③ free prune: a
    candidate whose self-declared TRIGGER_REGEX matches none of the parent's
    failing trajectories is culled as ``pruned_inert`` before any eval spend.
    Off by default because Gate-b already denies credit to never-fired
    mechanisms; the preflight only saves budget, at a small false-prune risk.
    """
    from dataclasses import replace as _dc_replace

    if analysis_mode not in ("mapreduce", "agentic"):
        raise ValueError(f"unknown analysis_mode {analysis_mode!r}")
    if analysis_mode == "agentic":
        # 构建时快速失败：智能体分析只通过已安装且已登录的 Claude CLI 在 Claude 模型上
        # 运行，不能运行到中途才失败，也不适用于使用 mapreduce 分支的其他驱动。
        from codeflow.evolver.orchestrator.providers.claude_agentic import (
            require_claude_for_agentic,
        )

        require_claude_for_agentic(agentic_model)

    budget = budget or config.budget
    vanilla_out_dir = Path(vanilla_out_dir)
    runs_root = Path(runs_root)
    ws_root = Path(ws_root)
    if runs_root != aw_cfg.out_dir_root:
        # 诊断从 runs_root 下读取已晋升父节点的确认输出目录，而门禁写入
        # aw_cfg.out_dir_root；根目录分裂会使第二轮及之后的诊断静默为空。
        raise ValueError(f"runs_root ({runs_root}) must equal aw_cfg.out_dir_root ({aw_cfg.out_dir_root})")
    # 会话侧遵循同一约定：诊断在 ws_root 下查找会话 JSONL，因此批量执行器必须实际
    # 写入该位置。
    if aw_cfg.workspace is None:
        aw_cfg = _dc_replace(aw_cfg, workspace=ws_root)
    elif Path(aw_cfg.workspace) != ws_root:
        raise ValueError(f"ws_root ({ws_root}) must equal aw_cfg.workspace ({aw_cfg.workspace})")

    # ① 针对父节点基线失败轨迹执行 W1-W7 诊断：根节点按名称诊断原版输出目录；
    # 已晋升父节点诊断自身确认输出目录。confirm_job_name 是与门禁策略共享的命名约定。
    exp_of = exp_of or (
        lambda parent: vanilla_out_dir.name if parent.node_id == root_node_id else confirm_job_name(parent.node_id)
    )
    trajectory_source = build_out_dir_trajectory_source(
        runs_root=runs_root, ws_root=ws_root, exp_of=exp_of, k=config.k_confirm
    )

    # ⑤ 评测：在候选提交的工作树检出中，以 cwd=worktree 运行 batch.py。
    # vanilla_node 允许 cold_start 在缺少原版账本时运行该账本（SOP §1）。
    backend = make_appworld_backend(
        aw_cfg,
        vanilla_out_dir=vanilla_out_dir,
        train_task_ids=train_task_ids,
        test_task_ids=list(test_task_ids),
        eval_fn=make_appworld_eval_fn(aw_cfg, repo_root),
        vanilla_node=HarnessNode(
            node_id=root_node_id,
            parent_id=None,
            git_commit_sha=base_sha,
            git_branch="evolver/orchestrator",
            created_at=HarnessNode.utc_now(),
            created_at_iter=0,
        ),
        cold_start_k=config.k_confirm,
        precheck=precheck or make_appworld_precheck(aw_cfg),
    )

    # diagnose_of 捕获这些值，供判决的 next_target 约束和驱动的 WHY 选择使用；启用
    # 归纳时，键与定义只有在首次诊断解析后才存在。
    taxonomy_keys: list[str] = []
    taxonomy_why_defs: dict[str, str] = {}

    def diagnose_of(vanilla_node):
        taxonomy, seed = resolve_taxonomy(
            driver_call_fn,
            trajectory_source,
            vanilla_node,
            mode=taxonomy_mode,
            work_dir=config.work_dir,
            hardcoded=DEFAULT_APPWORLD_TAXONOMY,
            taxonomy_path=taxonomy_path,
        )
        taxonomy_keys[:] = list(taxonomy.why_classes)
        taxonomy_why_defs.clear()
        taxonomy_why_defs.update(taxonomy.why_classes)
        if analysis_mode == "agentic":
            from benchmarks.appworld.evolve.agentic import (
                make_agentic_diagnose_fn,
            )

            return (
                make_agentic_diagnose_fn(
                    repo_root=repo_root,
                    runs_root=runs_root,
                    ws_root=ws_root,
                    exp_of=exp_of,
                    work_dir=config.work_dir,
                    taxonomy=taxonomy,
                    k=config.k_confirm,
                    model=agentic_model,
                ),
                None,
            )
        return (
            make_appworld_diagnose_fn(driver_call_fn, trajectory_source, taxonomy=taxonomy),
            seed,
        )

    # ② 从父提交开始设计（Bash 编辑器）。设计器的 read_trajectory 默认渲染当前
    # 父节点失败尝试；由装配器拥有的 sha_of 解析父提交。
    def design_of(sha_of, history, archive_summary_of):
        return make_bash_editor_design_fn(
            design_call_fn,
            repo_root=repo_root,
            worktree_root=worktree_root,
            sha_of=sha_of,
            budget=budget,
            history=history,
            archive_summary_of=archive_summary_of,
            require_beacon=require_beacon,
            whitelist_prefixes=whitelist_prefixes,
            import_smoke_python=aw_cfg.python_exe,
            render_failed=render_failed,
            render_failed_of=(
                None
                if render_failed is not None
                else build_failed_attempt_renderer(
                    runs_root=runs_root, ws_root=ws_root, exp_of=exp_of, k=config.k_confirm
                )
            ),
            passing_ids_of=build_passing_ids_source(
                runs_root=runs_root, ws_root=ws_root, exp_of=exp_of, k=config.k_confirm
            ),
            why_selection=why_selection,
            why_defs_of=lambda: dict(taxonomy_why_defs) or None,
        )

    # 基线：默认 "frozen" 表示按父节点冻结，并通过基础设施重跑 KEPT 覆盖以原版账本
    # 初始化；控制组使用与候选评测相同的挽救规则（SOP §0）。恢复回退会重新读取父
    # 节点确认输出目录。冻结模式受成本约束但无法感知跨时间漂移，参见 gates.policy；
    # "same_session" 每轮重新测量父节点，评测成本约两倍，但不受漂移影响。
    if baseline_mode not in ("frozen", "same_session"):
        raise ValueError(f"baseline_mode must be 'frozen' or 'same_session', got {baseline_mode!r}")

    def baseline_of():
        if baseline_mode == "same_session":
            from codeflow.evolver.orchestrator.gates.policy import (
                SameSessionPairedBaseline,
            )

            return SameSessionPairedBaseline(k=config.k_confirm)
        return make_frozen_baseline(
            root_node_id=root_node_id,
            vanilla_dir=vanilla_out_dir,
            kept_reader=lambda path: aw_adapter.read_kept_out_dir(
                path,
                expected_attempts=config.k_confirm,
            ),
            confirm_dir_of=lambda p: runs_root / confirm_job_name(p.node_id),
            train_task_ids=train_task_ids,
            seed_label="van0",
        )

    # Gate-b 回读：携带信标的候选实际在哪些训练任务上触发，结果取确认输出目录与基础
    # 设施重跑阶梯兄弟目录的并集。
    from codeflow.evolver.activation.ledger import read_fired_tasks

    def fired_source_of(node: HarnessNode, task_ids: list[str]):
        dirs = aw_adapter.ladder_out_dirs(runs_root / confirm_job_name(node.node_id))
        return read_fired_tasks(dirs, task_ids)

    preflight_fn = None
    if zero_hit_preflight:
        from codeflow.evolver.orchestrator.production import make_zero_hit_preflight

        preflight_fn = make_zero_hit_preflight(trajectory_source)

    def prepare_candidate(node_id: str, _parent_id: str, parent_sha: str, candidate) -> None:
        prepare_candidate_manifest(
            node_id,
            parent_sha,
            candidate,
            repo_root=repo_root,
        )

    return build_evolution_orchestrator(
        config,
        repo_root=repo_root,
        base_sha=base_sha,
        root_node_id=root_node_id,
        backend=backend,
        gate_policy=FocusedFisherGate(k=config.k_confirm, min_confirm_lift=min_confirm_lift),
        diagnose_of=diagnose_of,
        design_of=design_of,
        baseline_of=baseline_of,
        files_of=files_of,
        deletions_of=deletions_of,
        prepare_candidate=prepare_candidate,
        applied_patch_of=lambda candidate: candidate.applied_patch,
        driver_call_fn=driver_call_fn,
        verdict_call_fn=verdict_call_fn,
        verdict_why_keys_of=lambda: taxonomy_keys or None,
        harm_excerpt_of=lambda node_id, tid: render_candidate_failure(
            runs_root, ws_root, confirm_job_name(node_id), tid, k=config.k_confirm
        ),
        preflight_fn=preflight_fn,
        fired_source_of=fired_source_of,
    )


def build_appworld_sealed_runner(
    *,
    aw_cfg: "aw_adapter.AppWorldConfig",
    repo_root: str | Path,
    test_task_ids: list[str],
    sealed_dir: str | Path,
    k: int = 3,
    infra_max_reruns: int = 2,
):
    """The C3 sealed test runner for AppWorld (approach B, post-hoc).

    Reuses the same worktree-checkout eval as the loop (a candidate commit is
    checked out and ``batch.py`` runs against it), invoked with ``split="test"``
    and the infra rerun ladder, so test is scored exactly like train. Never
    called during evolution — feed the journal records to
    :func:`codeflow.evolver.orchestrator.sealed.runner.unseal_retention` after the
    loop finishes.
    """
    from codeflow.evolver.orchestrator.scoring import eval_with_infra_rerun
    from codeflow.evolver.orchestrator.sealed.runner import SealedTestRunner

    raw = make_appworld_eval_fn(aw_cfg, repo_root)

    def sealed_eval(node, task_ids, k_, job_name, *, split="test"):
        return eval_with_infra_rerun(raw, node, task_ids, k_, job_name, split=split, max_reruns=infra_max_reruns)

    return SealedTestRunner(
        eval_fn=sealed_eval,
        test_task_ids=list(test_task_ids),
        sealed_dir=Path(sealed_dir),
        k=k,
    )


__all__ = ["build_appworld_orchestrator", "build_appworld_sealed_runner"]
