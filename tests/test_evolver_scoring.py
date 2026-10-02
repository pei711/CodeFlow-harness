"""Unit tests for the scoring currency and the SOP 0 infra-rerun ladder.

The ladder decides which measurements are salvaged and which score 0 in a
fixed denominator; a bug here hands every candidate a free lift (or silently
throws away good trials), so the rerun/keep rules are pinned down exactly.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.appworld.evolve.adapter import read_kept_out_dir
from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementFailure,
    MeasurementStatus,
    TaskEval,
    anchor_mean_pass_rate,
    eval_with_infra_rerun,
    flip_summary,
    measurement_validity,
    with_infra_rerun,
)
from codeflow.evolver.orchestrator.sealed.runner import (
    SealedTestRunner,
    unseal_retention,
)
from codeflow.evolver.tree.node import HarnessNode


def _te(tid, passes, attempts, infra=0, failure=None):
    return TaskEval(
        task_id=tid,
        passes=passes,
        attempts=attempts,
        infra_attempts=infra,
        failure=failure,
    )


class _ScriptedEval:
    """EvalFn replaying a scripted list of result maps; records every call."""

    def __init__(self, results: list[dict[str, TaskEval]]):
        self.results = list(results)
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, node, task_ids, k, job_name, *, split="train"):
        self.calls.append((list(task_ids), job_name))
        return self.results.pop(0)


class TestTaskEval:
    def test_pass_rate(self):
        assert _te("t", 2, 3).pass_rate == pytest.approx(2 / 3)
        assert _te("t", 0, 0).pass_rate == 0.0

    def test_failure_round_trip_is_deterministic(self):
        ev = _te("t", 0, 3, failure=MeasurementFailure.provider)
        assert TaskEval.from_dict(ev.to_dict()) == ev

    def test_validity_separates_failed_and_inconclusive(self):
        failed = measurement_validity(
            {
                "provider": _te("provider", 0, 3, failure=MeasurementFailure.provider),
                "infra": _te("infra", 0, 3, infra=1),
            },
            ["provider", "infra"],
            expected_attempts=3,
        )
        assert failed.status is MeasurementStatus.failed
        assert failed.provider_failures == ("provider",)
        assert failed.infrastructure_failures == ("infra",)

        inconclusive = measurement_validity(
            {"empty": _te("empty", 0, 0)},
            ["empty", "missing"],
            expected_attempts=3,
        )
        assert inconclusive.status is MeasurementStatus.inconclusive
        assert inconclusive.inconclusive == ("empty",)
        assert inconclusive.missing == ("missing",)

    def test_partial_positive_measurement_is_inconclusive(self):
        validity = measurement_validity(
            {"partial": _te("partial", 1, 1)},
            ["partial"],
            expected_attempts=3,
        )

        assert validity.status is MeasurementStatus.inconclusive
        assert validity.inconclusive == ("partial",)

    def test_anchor_mean_missing_scores_zero(self):
        assert anchor_mean_pass_rate({"t1": _te("t1", 3, 3)}, ["t1", "t2"]) == 0.5
        with pytest.raises(ValueError, match="non-empty"):
            anchor_mean_pass_rate({}, [])


class TestInfraRerunLadder:
    def test_clean_eval_triggers_no_rerun(self):
        fake = _ScriptedEval([{"t1": _te("t1", 1, 3), "t2": _te("t2", 3, 3)}])
        out = eval_with_infra_rerun(fake, None, ["t1", "t2"], 3, "job")
        assert len(fake.calls) == 1
        assert out["t1"].passes == 1

    def test_infra_task_rerun_and_salvaged(self):
        fake = _ScriptedEval(
            [
                {"t1": _te("t1", 0, 3, infra=2), "t2": _te("t2", 3, 3)},
                {"t1": _te("t1", 2, 3, infra=0)},
            ]
        )
        out = eval_with_infra_rerun(fake, None, ["t1", "t2"], 3, "job")
        assert out["t1"].passes == 2 and out["t1"].infra_attempts == 0

        assert fake.calls[1] == (["t1"], "job_infra_rerun1")

    def test_missing_task_is_infra_by_definition(self):
        fake = _ScriptedEval(
            [
                {"t1": _te("t1", 3, 3)},
                {"t2": _te("t2", 1, 3)},
            ]
        )
        out = eval_with_infra_rerun(fake, None, ["t1", "t2"], 3, "job")
        assert fake.calls[1][0] == ["t2"]
        assert out["t2"].passes == 1

    def test_keeps_measurement_with_fewest_infra(self):

        first = _te("t1", 1, 3, infra=1)
        fake = _ScriptedEval(
            [
                {"t1": first},
                {"t1": _te("t1", 0, 3, infra=1)},
                {"t1": _te("t1", 0, 3, infra=2)},
            ]
        )
        out = eval_with_infra_rerun(fake, None, ["t1"], 3, "job")
        assert out["t1"] is first
        assert len(fake.calls) == 3

    def test_persistent_infra_survives_and_scores_low(self):
        results = [{"t1": _te("t1", 0, 3, infra=3)} for _ in range(3)]
        fake = _ScriptedEval(results)
        out = eval_with_infra_rerun(fake, None, ["t1"], 3, "job", max_reruns=2)
        assert len(fake.calls) == 3
        assert out["t1"].infra_attempts == 3

    def test_wrapper_is_identity_at_zero_reruns(self):
        inner = _ScriptedEval([])
        assert with_infra_rerun(inner, 0) is inner

    def test_wrapper_applies_ladder(self):
        fake = _ScriptedEval(
            [
                {"t1": _te("t1", 0, 3, infra=1)},
                {"t1": _te("t1", 2, 3)},
            ]
        )
        wrapped = with_infra_rerun(fake, 1)
        out = wrapped(None, ["t1"], 3, "job")
        assert out["t1"].passes == 2 and len(fake.calls) == 2

    def test_provider_failure_is_not_reclassified_as_infrastructure(self):
        failure = _te("t1", 0, 3, failure=MeasurementFailure.provider)
        fake = _ScriptedEval([{"t1": failure}])
        out = eval_with_infra_rerun(fake, None, ["t1"], 3, "job")
        assert out["t1"] is failure
        assert len(fake.calls) == 1

    def test_partial_measurement_is_retried_to_expected_k(self):
        fake = _ScriptedEval(
            [
                {"t1": _te("t1", 1, 1)},
                {"t1": _te("t1", 2, 3)},
            ]
        )

        out = eval_with_infra_rerun(fake, None, ["t1"], 3, "job")

        assert out["t1"] == _te("t1", 2, 3)
        assert len(fake.calls) == 2

    def test_durable_replay_matches_three_rung_partial_k_selection(
        self,
        tmp_path,
    ):
        base = {"t1": _te("t1", 1, 3, infra=1)}
        partial_clean = {"t1": _te("t1", 1, 2)}
        complete_clean = {"t1": _te("t1", 2, 3)}
        fake = _ScriptedEval([base, partial_clean, complete_clean])

        live = eval_with_infra_rerun(
            fake,
            None,
            ["t1"],
            3,
            "job",
        )

        for dirname, measurement in (
            ("job", base["t1"]),
            ("job_infra_rerun1", partial_clean["t1"]),
            ("job_infra_rerun2", complete_clean["t1"]),
        ):
            out_dir = tmp_path / dirname
            out_dir.mkdir()
            for index in range(measurement.attempts):
                record = {
                    "task_id": measurement.task_id,
                    "success": index < measurement.passes,
                }
                if index < measurement.infra_attempts:
                    record["infra_error"] = "fixture"
                (out_dir / f"{measurement.task_id}_k{index}.json").write_text(json.dumps(record))

        replayed = read_kept_out_dir(
            tmp_path / "job",
            expected_attempts=3,
        )

        assert live == complete_clean
        assert replayed == live


class TestFlipSummary:
    def test_partial_rescue_and_regression_accounting(self):
        cand = {"t1": _te("t1", 2, 3), "t2": _te("t2", 0, 3), "t3": _te("t3", 3, 3)}
        ctrl = {"t1": _te("t1", 1, 3), "t2": _te("t2", 1, 3), "t3": _te("t3", 3, 3)}
        s = flip_summary(cand, ctrl, ["t1", "t2", "t3"])
        assert s["rescued"] == ["t1"]
        assert s["regressed"] == ["t2"]
        assert s["still_failing"] == ["t1", "t2"]

    def test_missing_arm_scores_zero(self):
        cand = {"t1": _te("t1", 2, 3)}
        s = flip_summary(cand, {}, ["t1", "t2"])
        assert s["rescued"] == ["t1"]  # 0.0 -> 2/3
        assert s["n_regressed"] == 0

    def test_id_lists_capped_but_counts_exact(self):
        ids = [f"t{i}" for i in range(15)]
        cand = {t: _te(t, 3, 3) for t in ids}
        s = flip_summary(cand, {}, ids, max_ids=12)
        assert s["n_rescued"] == 15
        assert len(s["rescued"]) == 12


def _node(node_id):
    return HarnessNode(
        node_id=node_id,
        parent_id=None,
        git_commit_sha="0" * 40,
        git_branch="sealed",
        created_at=HarnessNode.utc_now(),
        created_at_iter=0,
    )


class TestSealedScoring:
    @pytest.mark.parametrize(
        ("candidate_evals", "expected_verdict"),
        [
            (
                {
                    "t1": _te("t1", 3, 3),
                    "t2": _te("t2", 0, 3, failure=MeasurementFailure.provider),
                },
                EvaluationVerdict.failed,
            ),
            ({"t1": _te("t1", 3, 3)}, EvaluationVerdict.inconclusive),
            (
                {"t1": _te("t1", 1, 1), "t2": _te("t2", 0, 3)},
                EvaluationVerdict.inconclusive,
            ),
        ],
    )
    def test_invalid_measurement_has_no_score_or_retention(
        self,
        tmp_path,
        candidate_evals,
        expected_verdict,
    ):
        def eval_fn(node, task_ids, k, job_name, *, split="train"):
            if node.node_id == "vanilla":
                return {"t1": _te("t1", 0, 3), "t2": _te("t2", 0, 3)}
            return candidate_evals

        runner = SealedTestRunner(eval_fn, ["t1", "t2"], tmp_path / expected_verdict.value)
        report = unseal_retention(
            runner,
            [
                {
                    "round_index": 1,
                    "next_parent_id": "candidate",
                    "next_parent_sha": "1" * 40,
                    "next_parent_train": 1.0,
                }
            ],
            vanilla_node=_node("vanilla"),
            vanilla_train=0.0,
        )
        candidate_record = next(r for r in runner.unseal() if r["node_id"] == "candidate")
        assert candidate_record["pass_at_1"] is None
        assert report.best_node_id == "candidate"
        assert report.best_test is None
        assert report.retention is None
        assert report.verdict is expected_verdict
        assert report.sealed_z is None
        assert report.sealed_credited_2sigma is False

    def test_invalid_vanilla_measurement_is_not_rejected_as_if_measured(self, tmp_path):
        def eval_fn(node, task_ids, k, job_name, *, split="train"):
            return {t: _te(t, 0, 3, failure=MeasurementFailure.provider) for t in task_ids}

        runner = SealedTestRunner(eval_fn, ["t1"], tmp_path / "vanilla_failure")
        report = unseal_retention(
            runner,
            [],
            vanilla_node=_node("vanilla"),
            vanilla_train=0.0,
        )
        assert report.verdict is EvaluationVerdict.failed
        assert report.vanilla_test is None
        assert report.retention is None

    def test_regression_is_rejected_with_negative_retention(self, tmp_path):
        def eval_fn(node, task_ids, k, job_name, *, split="train"):
            passes = 3 if node.node_id == "vanilla" else 0
            return {t: _te(t, passes, 3) for t in task_ids}

        runner = SealedTestRunner(eval_fn, ["t1", "t2"], tmp_path / "regression")
        report = unseal_retention(
            runner,
            [
                {
                    "round_index": 1,
                    "next_parent_id": "candidate",
                    "next_parent_sha": "1" * 40,
                    "next_parent_train": 1.0,
                }
            ],
            vanilla_node=_node("vanilla"),
            vanilla_train=0.0,
        )
        assert report.verdict is EvaluationVerdict.rejected
        assert report.retention == pytest.approx(-1.0)
        assert report.sealed_credited_2sigma is False
