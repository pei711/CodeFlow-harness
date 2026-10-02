"""Small-real bench plugin: ``build(ctx) -> BenchBundle`` for this subject.

``bench_config`` schema (owned by this entry)::

    bench_config:
      train_task_ids: [...]      # optional subset of the built-in train split
      test_task_ids: [...]       # optional subset of the built-in sealed split
      timeout: 120               # grading subprocess timeout, seconds
      vanilla_experiment: vanilla
      precheck: true

The gate policy is ``FocusedFisherGate``: the accepted-runtime evidence
evaluator (``appworld_focused_fisher_v1``) rebuilds the verdict from
``stats["full_lift"]``, which only that gate records, so a manifested candidate
promoted under any other policy would fail evidence creation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks.appworld.evolve import adapter
from benchmarks.appworld.evolve import tasks as task_defs
from codeflow.evolver.launch.contract import BenchBundle, LaunchContext, validate_whitelist
from codeflow.evolver.orchestrator.gates.policy import make_frozen_baseline
from codeflow.evolver.orchestrator.gates.strategies import FocusedFisherGate, confirm_job_name
from codeflow.evolver.orchestrator.scoring import eval_with_infra_rerun, with_infra_rerun
from codeflow.evolver.tree.node import HarnessNode

_KNOWN_KEYS = {"train_task_ids", "test_task_ids", "timeout", "vanilla_experiment", "precheck"}

WHITELIST = ("benchmarks/appworld/agent_cli.py",)


def _task_subset(bc: dict, key: str, default: list[str]) -> list[str]:
    raw = bc.get(key)
    if not raw:
        return list(default)
    ids = [str(tid) for tid in raw]
    unknown = [tid for tid in ids if tid not in task_defs.ALL_TASKS]
    if unknown:
        raise ValueError(f"bench_config.{key}: unknown task ids {unknown}; known ids: {sorted(task_defs.ALL_TASKS)}")
    return ids


def _make_precheck(repo_root: Path, base_sha: str, timeout: float):
    def precheck() -> None:
        try:
            from codeflow.evolver.tree import git_ops

            source = git_ops.read_file_at(repo_root, base_sha, adapter.MODULE_PATH)
        except Exception as exc:  # noqa: BLE001 - 任意读取失败都表示目标不可用
            raise RuntimeError(f"cannot read {adapter.MODULE_PATH} at {base_sha[:12]} in {repo_root}: {exc}") from exc
        probe = task_defs.TRAIN_TASKS[0]
        with tempfile.TemporaryDirectory(prefix="small-real-precheck-") as tmp:
            module_path = Path(tmp) / "agent_cli.py"
            module_path.write_bytes(source)
            out_dir = Path(tmp) / "out"
            job = {
                "module_path": str(module_path),
                "out_dir": str(out_dir),
                "trials": [{"task_id": probe.task_id, "k": 0, "cases": probe.to_payload()["cases"]}],
            }
            try:
                proc = subprocess.run(
                    [sys.executable, str(adapter.GRADE_SCRIPT)],
                    input=json.dumps(job),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"grading probe timed out after {timeout:.0f}s") from exc
            if proc.returncode != 0:
                raise RuntimeError(f"grading probe exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
            result = out_dir / f"{probe.task_id}_k0.json"
            if not result.is_file():
                raise RuntimeError(f"grading probe wrote no result file at {result}")
            record = json.loads(result.read_text())
            if record.get("infra_error"):
                raise RuntimeError(f"grading probe reported an infra error: {record['infra_error']}")

    return precheck


def build(ctx: LaunchContext) -> BenchBundle:
    spec = ctx.spec
    bc = dict(spec.bench_config)
    unknown = set(bc) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"bench_config: unknown keys {sorted(unknown)}")
    timeout = float(bc.get("timeout") or adapter.DEFAULT_TIMEOUT)
    if timeout <= 0:
        raise ValueError(f"bench_config.timeout must be > 0, got {timeout}")

    train_ids = _task_subset(bc, "train_task_ids", task_defs.train_task_ids())
    test_ids = _task_subset(bc, "test_task_ids", task_defs.test_task_ids())
    if spec.smoke:
        train_ids = train_ids[:3]
        test_ids = test_ids[:1]
    if not train_ids:
        raise ValueError("bench_config.train_task_ids must not be empty")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"train/test task sets overlap: {sorted(overlap)}")

    validate_whitelist(spec.repo_root, spec.base_sha, WHITELIST)

    work = Path(spec.work_dir)
    runs_root = work / "runs"
    van_exp = str(bc.get("vanilla_experiment") or "vanilla")
    vanilla_out_dir = runs_root / van_exp
    k_confirm = spec.funnel.k_confirm

    root_node = HarnessNode(
        node_id="C0",
        parent_id=None,
        git_commit_sha=spec.base_sha,
        git_branch="",
        created_at=HarnessNode.utc_now(),
        created_at_iter=0,
    )

    raw_eval = adapter.make_eval_fn(spec.repo_root, runs_root, timeout=timeout)
    precheck = _make_precheck(Path(spec.repo_root), spec.base_sha, timeout) if bc.get("precheck", True) else None

    def cold_start_done() -> int:
        if not vanilla_out_dir.is_dir():
            return 0
        return sum(1 for tid in train_ids for k in range(k_confirm) if (vanilla_out_dir / f"{tid}_k{k}.json").is_file())

    def run_cold_start() -> None:
        runs_root.mkdir(parents=True, exist_ok=True)
        eval_with_infra_rerun(raw_eval, root_node, train_ids, k_confirm, van_exp)

    def read_stability():
        from codeflow.evolver.analysis.stability_bucket import TaskStability, _bucket_for

        kept = adapter.read_kept_out_dir(vanilla_out_dir, expected_attempts=k_confirm)
        return {
            tid: TaskStability(
                task_id=tid,
                attempts=ev.attempts,
                passes=ev.passes,
                bucket=_bucket_for(ev.passes, ev.attempts),
            )
            for tid, ev in kept.items()
            if tid in set(train_ids)
        }

    def cold_start():
        if cold_start_done() < len(train_ids) * k_confirm:
            run_cold_start()
        return read_stability()

    def anchor(affinity=None):
        from codeflow.evolver.scheduler.anchor_selection import simple_anchor

        return simple_anchor(cold_start(), cull_sigma_mult=spec.funnel.anchor.cull_sigma_mult)

    from codeflow.evolver.orchestrator.scoring import EvalBackend

    backend = EvalBackend(
        train_task_ids=list(train_ids),
        test_task_ids=list(test_ids),
        eval=with_infra_rerun(raw_eval, 2),
        cold_start=cold_start,
        anchor=anchor,
        precheck=precheck,
    )

    def ledger_dir_of(parent: HarnessNode) -> Path:
        if parent.node_id == "C0":
            return vanilla_out_dir
        return runs_root / confirm_job_name(parent.node_id)

    def evidence_of(parent: HarnessNode) -> dict[str, dict[str, list[dict]]]:
        try:
            failures = adapter.read_case_failures(ledger_dir_of(parent), train_ids)
        except OSError:
            return {}
        grouped: dict[str, dict[str, list[dict]]] = {}
        for tid, cases in failures.items():
            grouped.setdefault(task_defs.task_for(tid).why, {})[tid] = cases
        return grouped

    def build_orchestrator():
        from benchmarks.appworld.evolve.candidate import files_of, prepare_candidate_manifest
        from benchmarks.appworld.evolve.designer import make_design_fn
        from codeflow.evolver.orchestrator.production import build_evolution_orchestrator

        design_call_fn = ctx.models.get("design") or ctx.models.get("driver")

        def diagnose_of(vanilla_node):
            def diagnose_fn(round_index: int, parent: HarnessNode) -> dict:
                evidence = evidence_of(parent)
                cells = {
                    f"loop_override::{why}": {
                        "n_candidates": 0,
                        "trajectory_ids": sorted(failing),
                        "candidates": [],
                    }
                    for why, failing in evidence.items()
                }
                return {
                    "why_distribution": {why: len(failing) for why, failing in evidence.items()},
                    "cells": cells,
                    "covered_why_classes": sorted(evidence),
                    "n_total_judged": sum(len(failing) for failing in evidence.values()),
                }

            return diagnose_fn, None

        def design_of(sha_of, history, archive_summary_of):
            def missing_design_call(_messages):
                raise RuntimeError("small-real bench requires a design (or driver) model call_fn to run a round")

            def impact_ranked_evidence(parent: HarnessNode) -> dict:
                evidence = evidence_of(parent)
                ranked = sorted(evidence.items(), key=lambda item: (-len(item[1]), item[0]))
                return dict(ranked)

            return make_design_fn(
                design_call_fn or missing_design_call,
                repo_root=spec.repo_root,
                sha_of=sha_of,
                budget=spec.funnel.budget,
                evidence_of=impact_ranked_evidence,
            )

        def baseline_of():
            return make_frozen_baseline(
                root_node_id="C0",
                vanilla_dir=vanilla_out_dir,
                kept_reader=lambda path: adapter.read_kept_out_dir(path, expected_attempts=k_confirm),
                confirm_dir_of=lambda parent: runs_root / confirm_job_name(parent.node_id),
                train_task_ids=train_ids,
                seed_label="van0",
            )

        def prepare_candidate(node_id: str, _parent_id: str, parent_sha: str, candidate) -> None:
            prepare_candidate_manifest(node_id, parent_sha, candidate, repo_root=spec.repo_root)

        return build_evolution_orchestrator(
            spec.funnel,
            repo_root=spec.repo_root,
            base_sha=spec.base_sha,
            root_node_id="C0",
            backend=backend,
            gate_policy=FocusedFisherGate(k=k_confirm),
            diagnose_of=diagnose_of,
            design_of=design_of,
            baseline_of=baseline_of,
            files_of=files_of,
            prepare_candidate=prepare_candidate,
            applied_patch_of=lambda candidate: candidate.applied_patch,
        )

    unseal = None
    if test_ids:

        def unseal(records: list[dict], orch) -> dict:
            import dataclasses

            from codeflow.evolver.orchestrator.sealed.runner import SealedTestRunner, unseal_retention

            def sealed_eval(node, task_ids, k, job_name, *, split="test"):
                return eval_with_infra_rerun(raw_eval, node, task_ids, k, job_name, split=split)

            runner = SealedTestRunner(
                eval_fn=sealed_eval,
                test_task_ids=list(test_ids),
                sealed_dir=spec.funnel.sealed_output_dir or work / "sealed",
                k=k_confirm,
            )
            report = unseal_retention(
                runner,
                records,
                vanilla_node=root_node,
                vanilla_train=orch.vanilla_train_mean,
            )
            return dataclasses.asdict(report) if dataclasses.is_dataclass(report) else dict(report)

    return BenchBundle(
        root_node_id="C0",
        root_node=root_node,
        journal_path=work / "journal" / "rounds.jsonl",
        cold_start_total=len(train_ids) * k_confirm,
        cold_start_done=cold_start_done,
        run_cold_start=run_cold_start,
        build_orchestrator=build_orchestrator,
        unseal=unseal,
        precheck=precheck,
    )


__all__ = ["build"]
