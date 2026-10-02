from __future__ import annotations

import os
from pathlib import Path

from codeflow.evolver.analysis.stability_bucket import StabilityBucket, TaskStability
from codeflow.evolver.launch.contract import BenchBundle
from codeflow.evolver.orchestrator.loop import EvolutionOrchestrator
from codeflow.evolver.orchestrator.scoring import EvalBackend, TaskEval
from codeflow.evolver.scheduler.anchor_selection import simple_anchor
from codeflow.evolver.tree.node import HarnessNode

_INTERRUPT_ROUND_ENV = "EVOLVER_FIXTURE_INTERRUPT_ROUND"


def build(ctx) -> BenchBundle:
    work_dir = Path(ctx.spec.work_dir)
    cold_start_marker = work_dir / "runs" / "vanilla" / "fixture-task_k0.json"
    stability = {
        "fixture-task": TaskStability(
            task_id="fixture-task",
            passes=0,
            attempts=1,
            bucket=StabilityBucket.STABLE_FAIL,
        )
    }

    def cold_start_done() -> int:
        return int(cold_start_marker.is_file())

    def run_cold_start() -> None:
        cold_start_marker.parent.mkdir(parents=True, exist_ok=True)
        cold_start_marker.write_text("{}")

    backend = EvalBackend(
        train_task_ids=["fixture-task"],
        test_task_ids=[],
        eval=lambda node, ids, k, job, **kwargs: {
            task_id: TaskEval(task_id=task_id, passes=0, attempts=k) for task_id in ids
        },
        cold_start=lambda: stability,
        anchor=lambda affinity=None: simple_anchor(stability),
    )

    class Candidate:
        why = "fixture"
        files = {"src/subject.py": b"fixture"}
        summary = "deterministic fixture candidate"

    def design(round_index, failure_map, parent):
        interrupt_round = int(os.environ.get(_INTERRUPT_ROUND_ENV, "0"))
        if round_index == interrupt_round:
            raise KeyboardInterrupt
        return [Candidate()]

    def build_orchestrator() -> EvolutionOrchestrator:
        return EvolutionOrchestrator(
            ctx.spec.funnel,
            backend=backend,
            diagnose_fn=lambda round_index, parent: {"why_distribution": {"fixture": 1.0}},
            design_fn=design,
            apply_fn=lambda parent_id, patch, round_index: (_ for _ in ()).throw(
                AssertionError("preflight must reject the fixture candidate")
            ),
            preflight_fn=lambda candidate, parent: False,
        )

    root = HarnessNode(
        node_id="C0",
        parent_id=None,
        git_commit_sha=ctx.spec.base_sha,
        git_branch="",
        created_at=HarnessNode.utc_now(),
        created_at_iter=0,
    )

    return BenchBundle(
        root_node_id="C0",
        root_node=root,
        journal_path=work_dir / "journal" / "rounds.jsonl",
        cold_start_total=1,
        cold_start_done=cold_start_done,
        run_cold_start=run_cold_start,
        build_orchestrator=build_orchestrator,
        unseal=lambda records, orchestrator: {
            "best_round": len(records),
            "retention": 1.0,
        },
        precheck=lambda: None,
    )


__all__ = ["build"]
