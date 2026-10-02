from __future__ import annotations

import pytest

from codeflow.evolver.analysis.stability_bucket import StabilityBucket, TaskStability
from codeflow.evolver.orchestrator.config import OrchestratorConfig, Termination
from codeflow.evolver.orchestrator.gates.policy import (
    CandidateOutcome,
    FrozenColdStartBaseline,
)
from codeflow.evolver.orchestrator.loop import EvolutionOrchestrator
from codeflow.evolver.orchestrator.scoring import (
    EvalBackend,
    EvaluationVerdict,
    TaskEval,
)
from codeflow.evolver.scheduler.anchor_selection import simple_anchor
from codeflow.evolver.tree.node import HarnessNode, NodeStatus


class _NoDecisionGate:
    def __init__(self, verdict, status):
        self.verdict = verdict
        self.status = status

    def decide(self, ctx):
        return CandidateOutcome(
            node_id=ctx.node.node_id,
            status=self.status,
            confirm_evals={task_id: TaskEval(task_id=task_id, passes=1, attempts=1) for task_id in ctx.train_task_ids},
            verdict=self.verdict,
        )


@pytest.mark.parametrize(
    ("verdict", "status"),
    [
        (EvaluationVerdict.inconclusive, NodeStatus.pruned_at_confirm),
        (EvaluationVerdict.failed, NodeStatus.errored),
    ],
)
def test_no_decision_rounds_do_not_burn_patience(tmp_path, verdict, status):
    stability = {
        "task-1": TaskStability(
            task_id="task-1",
            passes=0,
            attempts=3,
            bucket=StabilityBucket.STABLE_FAIL,
        )
    }
    backend = EvalBackend(
        train_task_ids=["task-1"],
        test_task_ids=[],
        eval=lambda node, task_ids, k, job_name, **kwargs: {},
        cold_start=lambda: stability,
        anchor=lambda affinity=None: simple_anchor(stability),
    )
    config = OrchestratorConfig(
        repo_root=tmp_path,
        work_dir=tmp_path / "work",
        driver_llm_spec={},
        termination=Termination(
            patience=1,
            max_rounds=5,
            max_consecutive_errors=2,
        ),
    )

    def apply(parent_id, candidate, round_index):
        return HarnessNode(
            node_id=f"candidate-{round_index}",
            parent_id=parent_id,
            git_commit_sha=str(round_index) * 40,
            git_branch="fixture",
            created_at=HarnessNode.utc_now(),
            created_at_iter=round_index,
        )

    orchestrator = EvolutionOrchestrator(
        config,
        backend=backend,
        diagnose_fn=lambda round_index, parent: {},
        design_fn=lambda round_index, failure_map, parent: [object()],
        apply_fn=apply,
        gate_policy=_NoDecisionGate(verdict, status),
        baseline_provider=FrozenColdStartBaseline({"task-1": TaskEval(task_id="task-1", passes=0, attempts=3)}),
    )

    result = orchestrator.run("C0")

    assert result.stop_reason == "errors_exhausted"
    assert len(result.rounds) == 2
    assert all(round_result.errored for round_result in result.rounds)
