from __future__ import annotations

import pytest

from benchmarks.codeflowbench.packs.trajectory import (
    FORMAL_TRAJECTORY_TASK_COUNT,
    SMOKE_TRAJECTORY_TASK_COUNT,
    DeterministicJudge,
    DeterministicTrajectoryRunner,
    JudgeScore,
    TrajectoryCategory,
    TrajectoryMetrics,
    TrajectoryPack,
    TrajectoryTask,
    TrajectoryTrack,
    TrajectoryVerifier,
    load_trajectory_tasks,
    trajectory_task_set_digest,
)
from benchmarks.codeflowbench.plan import compile_plan
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import (
    TrialKey,
    TrialStatus,
    VerificationState,
)
from benchmarks.codeflowbench.schema import ExperimentSpec, VariantSpec


def test_trajectory_tasks_loading() -> None:
    formal = load_trajectory_tasks(TrajectoryTrack.FORMAL)
    assert len(formal) == FORMAL_TRAJECTORY_TASK_COUNT == 6

    smoke = load_trajectory_tasks(TrajectoryTrack.SMOKE)
    assert len(smoke) == SMOKE_TRAJECTORY_TASK_COUNT == 3

    categories = {t.category for t in formal}
    assert categories == {
        TrajectoryCategory.LONG_SESSION,
        TrajectoryCategory.CROSS_TURN_DEPENDENCY,
        TrajectoryCategory.COMPRESSION_DEGRADATION,
    }


def test_trajectory_task_set_digest_deterministic() -> None:
    d1 = trajectory_task_set_digest(TrajectoryTrack.FORMAL)
    d2 = trajectory_task_set_digest(TrajectoryTrack.FORMAL)
    assert d1 == d2
    assert len(d1) == 64


@pytest.mark.asyncio
async def test_deterministic_judge_scoring() -> None:
    judge = DeterministicJudge()
    tasks = load_trajectory_tasks(TrajectoryTrack.SMOKE)
    task = tasks[0]

    score = await judge.evaluate_trajectory(task, task.turns)
    assert isinstance(score, JudgeScore)
    assert 1.0 <= score.tool_rationality <= 5.0
    assert 1.0 <= score.error_recovery <= 5.0
    assert 1.0 <= score.constraint_preservation <= 5.0
    assert 1.0 <= score.composite_score <= 5.0
    assert score.is_synthetic is True
    assert "is_synthetic=True" in score.rationale


def test_trajectory_verifier() -> None:
    verifier = TrajectoryVerifier()
    tasks = load_trajectory_tasks(TrajectoryTrack.SMOKE)
    task = tasks[0]
    runner = DeterministicTrajectoryRunner()

    context_res = runner.evaluate_context_quality(task, "curator")
    score_ok = JudgeScore(
        tool_rationality=4.5,
        error_recovery=4.5,
        constraint_preservation=4.8,
        composite_score=4.6,
        rationale="test",
        is_synthetic=True,
    )

    res_ok = verifier.verify_evaluation(task, score_ok, context_res)
    assert res_ok.state is VerificationState.PASSED

    # Out of bounds score
    score_bad = JudgeScore(
        tool_rationality=6.0,  # Invalid
        error_recovery=4.5,
        constraint_preservation=4.8,
        composite_score=6.0,
        rationale="bad",
        is_synthetic=True,
    )
    res_bad = verifier.verify_evaluation(task, score_bad, context_res)
    assert res_bad.state is VerificationState.FAILED


def test_trajectory_pack_definition() -> None:
    pack = TrajectoryPack(TrajectoryTrack.FORMAL)
    defn = pack.definition()

    assert defn.pack_id == "trajectory"
    assert len(defn.tasks) == 6
    assert len(defn.variants) == 2
    assert {v.variant_id for v in defn.variants} == {"trajectory-fifo", "trajectory-curator"}
    assert len(defn.pairs) == 1
    assert defn.pairs[0].treatment_axis == "history_manager"


def test_trajectory_plan_compilation(tmp_path) -> None:
    pack = TrajectoryPack(TrajectoryTrack.FORMAL)
    exp = ExperimentSpec(
        suite="trajectory-plan-suite",
        repetitions=1,
        pack_ids=("trajectory",),
        output_root=tmp_path,
        identity={"model": "offline-deterministic"},
    )
    plan = compile_plan(exp, (pack,))
    assert len(plan.comparison_blocks) == 6
    assert len(plan.pack_definitions) == 1


@pytest.mark.asyncio
async def test_trajectory_pack_smoke_trial_execution(tmp_path) -> None:
    pack = TrajectoryPack(TrajectoryTrack.SMOKE)
    tasks = load_trajectory_tasks(TrajectoryTrack.SMOKE)
    task = tasks[0]

    experiment = ExperimentSpec(
        suite="trajectory-smoke-suite",
        repetitions=1,
        pack_ids=("trajectory-smoke",),
        output_root=tmp_path,
        identity={"model": "offline-deterministic"},
    )

    # FIFO control
    context_fifo = TrialContext(
        experiment_id="test-traj-01",
        plan_digest="plan-123",
        key=TrialKey(
            experiment_id="test-traj-01",
            pack_id="trajectory-smoke",
            task_id=task.task_id,
            variant_id="trajectory-fifo",
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task.to_task_spec(),
        variant=VariantSpec(variant_id="trajectory-fifo", settings={"history_manager": "fifo_tail"}),
    )

    exec_fifo = await pack.run_trial(context_fifo)
    assert exec_fifo.status is TrialStatus.PASSED
    assert exec_fifo.verification.state is VerificationState.PASSED
    assert exec_fifo.metrics["context_strategy"] == "fifo_tail"

    # Curator treatment
    context_curator = TrialContext(
        experiment_id="test-traj-01",
        plan_digest="plan-123",
        key=TrialKey(
            experiment_id="test-traj-01",
            pack_id="trajectory-smoke",
            task_id=task.task_id,
            variant_id="trajectory-curator",
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task.to_task_spec(),
        variant=VariantSpec(variant_id="trajectory-curator", settings={"history_manager": "curator"}),
    )

    exec_curator = await pack.run_trial(context_curator)
    assert exec_curator.status is TrialStatus.PASSED
    assert exec_curator.verification.state is VerificationState.PASSED
    assert exec_curator.metrics["context_strategy"] == "curator"

    # Treatment (curator) retains 100% constraints and consumes fewer tokens
    assert (
        exec_curator.metrics["key_constraint_retention_rate"]
        > exec_fifo.metrics["key_constraint_retention_rate"]
    )
    assert exec_curator.metrics["input_tokens"] < exec_fifo.metrics["input_tokens"]


def test_trajectory_context_quality_full_comparison() -> None:
    runner = DeterministicTrajectoryRunner()
    tasks = load_trajectory_tasks(TrajectoryTrack.FORMAL)
    comparison = runner.run_full_comparison(tasks)

    summary = comparison["summary"]
    assert summary["fifo_avg_retention_rate"] < 70.0  # FIFO drops early constraints
    assert summary["curator_avg_retention_rate"] == 100.0  # Curator retains all
    assert summary["retention_gain_pct"] > 30.0  # +33.3% gain
    assert summary["curator_token_saving_pct"] >= 25.0  # ~26% token saving
