from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

from benchmarks.appworld.evolve.eval import (
    Candidate,
    deletions_of,
    files_of,
    prepare_candidate_manifest,
)
from codeflow.evolver.activation import load_activation_record
from codeflow.evolver.analysis.stability_bucket import StabilityBucket, TaskStability
from codeflow.evolver.orchestrator.config import OrchestratorConfig, Termination
from codeflow.evolver.orchestrator.gates.policy import (
    CandidateOutcome,
    FrozenColdStartBaseline,
)
from codeflow.evolver.orchestrator.gates.strategies import FocusedFisherGate
from codeflow.evolver.orchestrator.loop import EvolutionOrchestrator
from codeflow.evolver.orchestrator.production import build_evolution_orchestrator
from codeflow.evolver.orchestrator.scoring import (
    EvalBackend,
    EvaluationVerdict,
    TaskEval,
)
from codeflow.evolver.scheduler.anchor_selection import simple_anchor
from codeflow.evolver.tree.node import HarnessNode, NodeStatus


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _subject_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "subject"
    target = repo / "benchmarks/appworld/agent_cli.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")
    _git(repo, "init", "-q")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "CodeFlow Test",
        "GIT_AUTHOR_EMAIL": "codeflow-test@example.invalid",
        "GIT_COMMITTER_NAME": "CodeFlow Test",
        "GIT_COMMITTER_EMAIL": "codeflow-test@example.invalid",
    }
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True, env=env)
    return repo, _git(repo, "rev-parse", "HEAD")


class _AcceptingGate:
    def decide(self, ctx):
        measured = {task_id: TaskEval(task_id=task_id, passes=1, attempts=1) for task_id in ctx.train_task_ids}
        return CandidateOutcome(
            node_id=ctx.node.node_id,
            status=NodeStatus.promoted_to_baseline,
            score=1.0,
            confirm_evals=measured,
            verdict=EvaluationVerdict.accepted,
        )


def test_manifested_runtime_candidate_produces_manual_rollback_bundle(tmp_path: Path) -> None:
    repo, base_sha = _subject_repo(tmp_path)
    work_dir = tmp_path / "evolution"
    stability = {
        "task-1": TaskStability(
            task_id="task-1",
            passes=0,
            attempts=1,
            bucket=StabilityBucket.STABLE_FAIL,
        )
    }
    backend = EvalBackend(
        train_task_ids=["task-1"],
        test_task_ids=[],
        eval=lambda node, task_ids, k, job_name, **kwargs: {
            task_id: TaskEval(task_id=task_id, passes=k, attempts=k) for task_id in task_ids
        },
        cold_start=lambda: stability,
        anchor=lambda affinity=None: simple_anchor(stability),
    )
    candidate = Candidate(
        files={"benchmarks/appworld/agent_cli.py": b"VALUE = 2\n"},
        why="runtime_fixture",
        summary="change the fixture runtime",
    )

    def prepare(node_id: str, _parent_id: str, parent_sha: str, value: Candidate) -> None:
        prepare_candidate_manifest(node_id, parent_sha, value, repo_root=repo)

    config = OrchestratorConfig(
        repo_root=repo,
        work_dir=work_dir,
        driver_llm_spec={},
        k_confirm=1,
        termination=Termination(patience=2, max_rounds=1),
    )
    baseline = {"task-1": TaskEval(task_id="task-1", passes=0, attempts=1)}
    orchestrator = build_evolution_orchestrator(
        config,
        repo_root=repo,
        base_sha=base_sha,
        root_node_id="C0",
        backend=backend,
        gate_policy=FocusedFisherGate(k=1),
        diagnose_of=lambda root: (lambda round_index, parent: {}, None),
        design_of=lambda sha_of, history, archive: lambda round_index, failure_map, parent: [candidate],
        baseline_of=lambda: FrozenColdStartBaseline(baseline),
        files_of=files_of,
        deletions_of=deletions_of,
        prepare_candidate=prepare,
        applied_patch_of=lambda value: value.applied_patch,
        run_gate0=False,
    )

    result = orchestrator.run("C0")

    assert result.rounds[0].outcomes[0].verdict is EvaluationVerdict.accepted
    assert candidate.manifest is not None
    artifact_dir = work_dir / "activation" / candidate.candidate_id
    record = load_activation_record(artifact_dir)
    assert record["state"] == "pending_human"
    assert record["requires_human"] is True
    assert record["parent_sha"] == base_sha
    assert (artifact_dir / "before.json").read_bytes() == (artifact_dir / "rollback.json").read_bytes()
    before = json.loads((artifact_dir / "before.json").read_text())
    after = json.loads((artifact_dir / "after.json").read_text())
    assert before["files"][0]["sha256"] != after["files"][0]["sha256"]
    assert base64.b64decode(before["files"][0]["content_base64"]) == b"VALUE = 1\n"
    assert base64.b64decode(after["files"][0]["content_base64"]) == b"VALUE = 2\n"
    node_record = json.loads((work_dir / "nodes" / f"{candidate.candidate_id}.json").read_text())
    assert record["candidate_sha"] == node_record["git_commit_sha"]
    assert node_record["candidate"]["manifest"]["label"] == "runtime"
    assert node_record["verdict"] == "accepted"
    assert _git(repo, "rev-parse", "HEAD") == base_sha
    assert (repo / "benchmarks/appworld/agent_cli.py").read_text() == "VALUE = 1\n"


def test_missing_required_candidate_evidence_prevents_promotion(tmp_path: Path) -> None:
    stability = {
        "task-1": TaskStability(
            task_id="task-1",
            passes=0,
            attempts=1,
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
        work_dir=tmp_path / "evolution",
        driver_llm_spec={},
        termination=Termination(patience=1, max_rounds=1),
    )

    def apply(parent_id, candidate, round_index):
        return HarnessNode(
            node_id="candidate-1",
            parent_id=parent_id,
            git_commit_sha="a" * 40,
            git_branch="fixture",
            created_at=HarnessNode.utc_now(),
            created_at_iter=round_index,
        )

    def fail_evidence(ctx, outcome):
        raise OSError("evidence volume unavailable")

    orchestrator = EvolutionOrchestrator(
        config,
        backend=backend,
        diagnose_fn=lambda round_index, parent: {},
        design_fn=lambda round_index, failure_map, parent: [object()],
        apply_fn=apply,
        gate_policy=_AcceptingGate(),
        baseline_provider=FrozenColdStartBaseline({"task-1": TaskEval(task_id="task-1", passes=0, attempts=1)}),
        evidence_hook=fail_evidence,
    )

    result = orchestrator.run("C0")
    outcome = result.rounds[0].outcomes[0]

    assert outcome.verdict is EvaluationVerdict.failed
    assert outcome.status is NodeStatus.errored
    assert outcome.stats["phase"] == "candidate_evidence"
    assert result.rounds[0].promoted is False
    assert result.rounds[0].beat_vanilla is False
