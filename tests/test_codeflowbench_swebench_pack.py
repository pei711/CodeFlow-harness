from __future__ import annotations

import pytest

from benchmarks.codeflowbench.packs.swebench import (
    FORMAL_SWEBENCH_TASK_COUNT,
    SMOKE_SWEBENCH_TASK_COUNT,
    DeterministicSwebenchRunner,
    SwebenchMetrics,
    SwebenchPack,
    SwebenchTask,
    SwebenchTrack,
    SwebenchVerifier,
    load_swebench_manifest,
    load_swebench_tasks,
    swebench_task_set_digest,
)
from benchmarks.codeflowbench.plan import compile_plan
from benchmarks.codeflowbench.protocol import TrialContext
from benchmarks.codeflowbench.records import (
    TrialKey,
    TrialStatus,
    VerificationState,
)
from benchmarks.codeflowbench.schema import ExperimentSpec, TaskSpec, VariantSpec


def test_swebench_manifest_frozen_50_tasks() -> None:
    manifest = load_swebench_manifest()
    assert manifest["schema"] == "codeflow.codeflowbench.swebench-manifest.v1"
    tasks = manifest["tasks"]
    assert len(tasks) == FORMAL_SWEBENCH_TASK_COUNT == 50

    repos = {t["repo"] for t in tasks}
    assert len(repos) == 12  # All 12 repositories in SWE-bench Verified represented
    assert "django/django" in repos
    assert "sympy/sympy" in repos
    assert "sphinx-doc/sphinx" in repos
    assert "matplotlib/matplotlib" in repos
    assert "scikit-learn/scikit-learn" in repos


def test_swebench_tasks_loading() -> None:
    formal_tasks = load_swebench_tasks(SwebenchTrack.FORMAL)
    assert len(formal_tasks) == 50

    smoke_tasks = load_swebench_tasks(SwebenchTrack.SMOKE)
    assert len(smoke_tasks) == 2

    first_task = formal_tasks[0]
    assert isinstance(first_task, SwebenchTask)
    assert first_task.instance_id
    assert first_task.base_commit
    assert first_task.fail_to_pass
    assert first_task.problem_statement


def test_swebench_task_set_digest_deterministic() -> None:
    digest1 = swebench_task_set_digest(SwebenchTrack.FORMAL)
    digest2 = swebench_task_set_digest(SwebenchTrack.FORMAL)
    assert digest1 == digest2
    assert len(digest1) == 64


def test_swebench_verifier_unified_diff() -> None:
    verifier = SwebenchVerifier()
    valid_diff = (
        "diff --git a/test.py b/test.py\n"
        "--- a/test.py\n"
        "+++ b/test.py\n"
        "@@ -1,2 +1,3 @@\n"
        " a = 1\n"
        "+b = 2\n"
        "-c = 3\n"
    )
    assert verifier.is_valid_unified_diff(valid_diff) is True
    added, removed = verifier.parse_patch_stats(valid_diff)
    assert added == 1
    assert removed == 1

    invalid_diff = "not a valid diff header"
    assert verifier.is_valid_unified_diff(invalid_diff) is False


def test_swebench_verifier_test_spec_evaluation() -> None:
    eval_result = SwebenchVerifier.evaluate_test_spec(
        fail_to_pass=("test_issue_fix",),
        pass_to_pass=("test_existing_feature",),
        test_outcomes={
            "test_issue_fix": "PASSED",
            "test_existing_feature": "PASSED",
        },
    )
    assert eval_result.resolved is True
    assert eval_result.fail_to_pass_passed == 1
    assert eval_result.pass_to_pass_passed == 1

    # Regression case
    regression_result = SwebenchVerifier.evaluate_test_spec(
        fail_to_pass=("test_issue_fix",),
        pass_to_pass=("test_existing_feature",),
        test_outcomes={
            "test_issue_fix": "PASSED",
            "test_existing_feature": "FAILED",
        },
    )
    assert regression_result.resolved is False
    assert len(regression_result.details) == 1


def test_swebench_pack_definition() -> None:
    pack = SwebenchPack(SwebenchTrack.FORMAL)
    defn = pack.definition()

    assert defn.pack_id == "swebench"
    assert len(defn.tasks) == 50
    assert len(defn.variants) == 2
    assert {v.variant_id for v in defn.variants} == {"swebench-fifo", "swebench-curator"}
    assert len(defn.pairs) == 1
    assert defn.pairs[0].treatment_axis == "history_manager"
    assert defn.pairs[0].control_variant_id == "swebench-fifo"
    assert defn.pairs[0].treatment_variant_id == "swebench-curator"


def test_swebench_plan_compilation(tmp_path) -> None:
    pack = SwebenchPack(SwebenchTrack.FORMAL)
    exp = ExperimentSpec(
        suite="swebench-plan-suite",
        repetitions=1,
        pack_ids=("swebench",),
        output_root=tmp_path,
        identity={"model": "offline-dry-run"},
    )
    plan = compile_plan(exp, (pack,))
    assert len(plan.comparison_blocks) == 50
    assert len(plan.pack_definitions) == 1


@pytest.mark.asyncio
async def test_swebench_pack_smoke_trial_execution(tmp_path) -> None:
    pack = SwebenchPack(SwebenchTrack.SMOKE)
    tasks = load_swebench_tasks(SwebenchTrack.SMOKE)
    task = tasks[0]

    experiment = ExperimentSpec(
        suite="swebench-smoke-suite",
        repetitions=1,
        pack_ids=("swebench-smoke",),
        output_root=tmp_path,
        identity={"model": "offline-deterministic"},
    )

    # Test Control: swebench-fifo
    context_fifo = TrialContext(
        experiment_id="test-exp-01",
        plan_digest="abc",
        key=TrialKey(
            experiment_id="test-exp-01",
            pack_id="swebench-smoke",
            task_id=task.task_id,
            variant_id="swebench-fifo",
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task.to_task_spec(),
        variant=VariantSpec(variant_id="swebench-fifo", settings={"history_manager": "fifo_tail"}),
    )

    execution_fifo = await pack.run_trial(context_fifo)
    assert execution_fifo.status is TrialStatus.PASSED
    assert execution_fifo.verification.state is VerificationState.PASSED
    assert execution_fifo.metrics["resolved"] == 1
    assert execution_fifo.metrics["context_strategy"] == "fifo_tail"
    tokens_fifo = execution_fifo.metrics["input_tokens"]

    # Test Treatment: swebench-curator
    context_curator = TrialContext(
        experiment_id="test-exp-01",
        plan_digest="abc",
        key=TrialKey(
            experiment_id="test-exp-01",
            pack_id="swebench-smoke",
            task_id=task.task_id,
            variant_id="swebench-curator",
            repetition=0,
        ),
        block_attempt=1,
        experiment=experiment,
        task=task.to_task_spec(),
        variant=VariantSpec(variant_id="swebench-curator", settings={"history_manager": "curator"}),
    )

    execution_curator = await pack.run_trial(context_curator)
    assert execution_curator.status is TrialStatus.PASSED
    assert execution_curator.verification.state is VerificationState.PASSED
    assert execution_curator.metrics["context_strategy"] == "curator"
    tokens_curator = execution_curator.metrics["input_tokens"]

    # Layered governance (curator) saves input tokens compared to uncurated FIFO tail
    assert tokens_curator < tokens_fifo
