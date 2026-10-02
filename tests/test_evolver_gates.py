"""Unit tests for the gate arithmetic (codeflow.evolver.orchestrator.gates).

These protect the promotion decision itself: paired z statistics, the Fisher
exact test, the three-shield pipeline's narrowing rules, and the two concrete
gate policies. Wrong math here does not crash — it silently promotes or prunes
the wrong candidate — so the expected values below are hand-computed.
"""

from __future__ import annotations

import json
import math

import pytest

from codeflow.evolver.orchestrator.archive import GsmeArchive
from codeflow.evolver.orchestrator.gates.fisher import (
    fisher_one_sided,
    focused_counts,
    train_mean,
)
from codeflow.evolver.orchestrator.gates.paired import paired_lift
from codeflow.evolver.orchestrator.gates.pipeline import run_gates
from codeflow.evolver.orchestrator.gates.policy import Baseline, CandidateOutcome, DecisionContext
from codeflow.evolver.orchestrator.gates.strategies import (
    FocusedFisherGate,
    PairedTwoSigmaGate,
    confirm_job_name,
)
from codeflow.evolver.orchestrator.loop import RoundResult
from codeflow.evolver.orchestrator.scoring import (
    EvaluationVerdict,
    MeasurementFailure,
    TaskEval,
)
from codeflow.evolver.orchestrator.state.journal import RoundJournal
from codeflow.evolver.scheduler.anchor_selection import AnchorSelection
from codeflow.evolver.tree.node import HarnessNode, NodeStatus


def _te(
    tid: str,
    passes: int,
    attempts: int,
    infra: int = 0,
    failure: MeasurementFailure | None = None,
) -> TaskEval:
    return TaskEval(
        task_id=tid,
        passes=passes,
        attempts=attempts,
        infra_attempts=infra,
        failure=failure,
    )


def _evals(spec: dict[str, tuple[int, int]]) -> dict[str, TaskEval]:
    return {tid: _te(tid, p, a) for tid, (p, a) in spec.items()}


class TestPairedLift:
    def test_deterministic_win_is_inf_z(self):
        ids = ["t1", "t2", "t3", "t4"]
        r = paired_lift(
            candidate_evals=_evals({t: (3, 3) for t in ids}),
            control_evals=_evals({t: (0, 3) for t in ids}),
            task_ids=ids,
            expected_attempts=3,
        )
        assert r.mean_lift == 1.0
        assert r.se == 0.0
        assert r.z == math.inf
        assert r.promoted and r.credited_2sigma
        assert r.verdict is EvaluationVerdict.accepted

    def test_hand_computed_z(self):

        cand = _evals({"t1": (3, 3), "t2": (3, 3), "t3": (0, 3), "t4": (0, 3)})
        ctrl = _evals({t: (0, 3) for t in ("t1", "t2", "t3", "t4")})
        r = paired_lift(
            candidate_evals=cand,
            control_evals=ctrl,
            task_ids=["t1", "t2", "t3", "t4"],
            expected_attempts=3,
        )
        assert r.mean_lift == pytest.approx(0.5)
        assert r.z == pytest.approx(math.sqrt(3))
        assert r.promoted
        assert not r.credited_2sigma

    def test_zero_lift_is_not_promoted(self):
        ids = ["t1", "t2"]
        same = _evals({t: (1, 3) for t in ids})
        r = paired_lift(
            candidate_evals=same,
            control_evals=dict(same),
            task_ids=ids,
            expected_attempts=3,
        )
        assert r.z == 0.0
        assert not r.promoted and not r.credited_2sigma
        assert r.verdict is EvaluationVerdict.rejected

    def test_deterministic_regression_is_minus_inf(self):
        ids = ["t1", "t2"]
        r = paired_lift(
            candidate_evals=_evals({t: (0, 3) for t in ids}),
            control_evals=_evals({t: (3, 3) for t in ids}),
            task_ids=ids,
            expected_attempts=3,
        )
        assert r.z == -math.inf
        assert not r.promoted and not r.credited_2sigma

    def test_missing_task_scores_zero_for_that_arm(self):

        cand = _evals({"t1": (3, 3)})
        ctrl = _evals({"t1": (0, 3), "t2": (3, 3)})
        r = paired_lift(
            candidate_evals=cand,
            control_evals=ctrl,
            task_ids=["t1", "t2"],
            expected_attempts=3,
        )
        assert r.candidate_mean == pytest.approx(0.5)
        assert r.control_mean == pytest.approx(0.5)
        assert not r.promoted
        assert r.verdict is EvaluationVerdict.inconclusive

    def test_missing_task_cannot_hide_behind_positive_mean_lift(self):
        r = paired_lift(
            candidate_evals=_evals({"t1": (3, 3)}),
            control_evals=_evals({"t1": (0, 3), "t2": (0, 3)}),
            task_ids=["t1", "t2"],
            expected_attempts=3,
        )
        assert r.candidate_mean > r.control_mean
        assert not r.promoted
        assert r.verdict is EvaluationVerdict.inconclusive

    @pytest.mark.parametrize("partial_arm", ["candidate", "control"])
    def test_partial_arm_cannot_produce_a_decision(self, partial_arm):
        candidate = _evals({"t1": (3, 3)})
        control = _evals({"t1": (0, 3)})
        if partial_arm == "candidate":
            candidate["t1"] = _te("t1", 1, 1)
        else:
            control["t1"] = _te("t1", 0, 1)

        result = paired_lift(
            candidate_evals=candidate,
            control_evals=control,
            task_ids=["t1"],
            expected_attempts=3,
        )

        assert result.verdict is EvaluationVerdict.inconclusive
        assert not result.promoted
        assert not result.credited_2sigma

    def test_single_task_has_zero_se(self):
        r = paired_lift(
            candidate_evals=_evals({"t1": (3, 3)}),
            control_evals=_evals({"t1": (0, 3)}),
            task_ids=["t1"],
            expected_attempts=3,
        )
        assert r.se == 0.0 and r.z == math.inf

    def test_empty_task_list_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            paired_lift(
                candidate_evals={},
                control_evals={},
                task_ids=[],
                expected_attempts=3,
            )


class TestFisher:
    def test_hand_computed_extreme(self):

        assert fisher_one_sided(3, 0, 0, 3) == pytest.approx(0.05)

    def test_hand_computed_moderate(self):

        # = (45*45 + 10*10 + 1) / 184756 = 2126/184756.
        assert fisher_one_sided(8, 2, 2, 8) == pytest.approx(2126 / 184756)

    def test_candidate_worst_is_one(self):
        assert fisher_one_sided(0, 3, 3, 0) == pytest.approx(1.0)

    def test_degenerate_margins_return_one(self):
        assert fisher_one_sided(0, 0, 2, 1) == 1.0
        assert fisher_one_sided(2, 1, 0, 0) == 1.0
        assert fisher_one_sided(0, 3, 0, 3) == 1.0
        assert fisher_one_sided(3, 0, 3, 0) == 1.0

    def test_focused_counts_keeps_infra_as_fails(self):
        evals = {"t1": _te("t1", 1, 3, infra=2), "t2": _te("t2", 3, 3)}

        assert focused_counts(evals, ["t1", "t2", "t3"]) == (4, 2)

    def test_train_mean_fixed_denominator(self):
        evals = _evals({"t1": (3, 3)})

        assert train_mean(evals, ["t1", "t2"]) == pytest.approx(0.5)
        assert train_mean(evals, []) == 0.0


class TestRunGates:
    def test_infra_failure_is_reported_and_refuses_promotion(self):
        ids = ["t1", "t2"]
        cand = {"t1": _te("t1", 3, 3, infra=1), "t2": _te("t2", 3, 3)}
        ctrl = _evals({t: (0, 3) for t in ids})
        g = run_gates(
            candidate_evals=cand,
            control_evals=ctrl,
            task_ids=ids,
            expected_attempts=3,
        )
        assert g.infra_contaminated == ["t1"]
        assert g.eligible_tasks == ids
        assert not g.promoted
        assert g.verdict is EvaluationVerdict.failed
        assert g.candidate_validity.infrastructure_failures == ("t1",)

    def test_provider_failure_refuses_promotion(self):
        cand = {
            "t1": _te("t1", 3, 3),
            "t2": _te("t2", 0, 3, failure=MeasurementFailure.provider),
        }
        ctrl = _evals({"t1": (0, 3), "t2": (0, 3)})
        g = run_gates(
            candidate_evals=cand,
            control_evals=ctrl,
            task_ids=["t1", "t2"],
            expected_attempts=3,
        )
        assert not g.promoted
        assert g.verdict is EvaluationVerdict.failed
        assert g.candidate_validity.provider_failures == ("t2",)

    def test_zero_attempt_measurement_is_inconclusive(self):
        cand = {"t1": _te("t1", 3, 3), "t2": _te("t2", 0, 0)}
        ctrl = _evals({"t1": (0, 3), "t2": (0, 3)})
        g = run_gates(
            candidate_evals=cand,
            control_evals=ctrl,
            task_ids=["t1", "t2"],
            expected_attempts=3,
        )
        assert not g.promoted
        assert g.verdict is EvaluationVerdict.inconclusive
        assert g.candidate_validity.inconclusive == ("t2",)

    def test_partial_measurement_stops_before_attribution(self):
        gate = run_gates(
            candidate_evals={"t1": _te("t1", 1, 1)},
            control_evals={"t1": _te("t1", 0, 3)},
            task_ids=["t1"],
            expected_attempts=3,
            fired_tasks={"t1"},
        )

        assert gate.verdict is EvaluationVerdict.inconclusive
        assert not gate.promoted
        assert gate.paired is None
        assert gate.candidate_validity.inconclusive == ("t1",)

    def test_gate_b_none_fails_open(self):
        ids = ["t1"]
        g = run_gates(
            candidate_evals=_evals({"t1": (3, 3)}),
            control_evals=_evals({"t1": (0, 3)}),
            task_ids=ids,
            expected_attempts=3,
            fired_tasks=None,
        )
        assert g.unfired_excluded == [] and g.eligible_tasks == ids

    def test_gate_b_empty_set_leaves_nothing_and_refuses(self):
        ids = ["t1", "t2"]
        g = run_gates(
            candidate_evals=_evals({t: (3, 3) for t in ids}),
            control_evals=_evals({t: (0, 3) for t in ids}),
            task_ids=ids,
            expected_attempts=3,
            fired_tasks=set(),
        )
        assert not g.promoted
        assert g.paired is None
        assert g.unfired_excluded == ids

    def test_gate_b_narrows_paired_to_fired_subset(self):
        cand = _evals({"t1": (3, 3), "t2": (0, 3)})
        ctrl = _evals({"t1": (0, 3), "t2": (0, 3)})
        g = run_gates(
            candidate_evals=cand,
            control_evals=ctrl,
            task_ids=["t1", "t2"],
            expected_attempts=3,
            fired_tasks={"t1"},
        )
        assert g.eligible_tasks == ["t1"]
        assert g.unfired_excluded == ["t2"]
        assert g.paired.n_tasks == 1 and g.promoted


class _FakeEval:
    """EvalFn returning canned evals keyed by job-name suffix; records calls."""

    def __init__(self, by_suffix: dict[str, dict[str, TaskEval]]):
        self.by_suffix = by_suffix
        self.calls: list[tuple[list[str], int, str]] = []

    def __call__(self, node, task_ids, k, job_name, *, split="train"):
        self.calls.append((list(task_ids), k, job_name))
        for suffix, evals in self.by_suffix.items():
            if job_name.endswith(suffix):
                return {t: ev for t, ev in evals.items() if t in task_ids}
        raise AssertionError(f"unexpected eval job {job_name!r}")


def _node(nid: str = "cand") -> HarnessNode:
    return HarnessNode(
        node_id=nid,
        parent_id="C0",
        git_commit_sha="0" * 40,
        git_branch="",
        created_at=HarnessNode.utc_now(),
        created_at_iter=1,
    )


def _ctx(eval_fn, baseline_evals, train_ids, **kw) -> DecisionContext:
    return DecisionContext(
        node=_node(),
        parent_id="C0",
        round_index=1,
        eval=eval_fn,
        baseline=Baseline(baseline_evals, train_mean(baseline_evals, train_ids), "vanilla"),
        train_task_ids=train_ids,
        **kw,
    )


class TestFocusedFisherGate:
    def test_partial_probe_is_inconclusive(self):
        base = _evals({"f1": (0, 3)})
        fake = _FakeEval({"_focused": _evals({"f1": (1, 1)})})

        outcome = FocusedFisherGate(k=3).decide(_ctx(fake, base, ["f1"], focused_task_ids=["f1"]))

        assert outcome.verdict is EvaluationVerdict.inconclusive
        assert outcome.status is NodeStatus.pruned_at_screen
        assert len(fake.calls) == 1

    def test_promotes_on_full_train_lift(self):
        train = ["f1", "t2", "t3"]
        base = _evals({"f1": (0, 3), "t2": (3, 3), "t3": (0, 3)})
        fake = _FakeEval(
            {
                "_focused": _evals({"f1": (2, 3)}),
                "_confirm": _evals({"f1": (2, 3), "t2": (3, 3), "t3": (0, 3)}),
            }
        )
        out = FocusedFisherGate(k=3).decide(_ctx(fake, base, train, focused_task_ids=["f1"]))
        assert out.status == NodeStatus.promoted_to_baseline
        assert out.score == pytest.approx(5 / 9)
        assert out.stats["full_lift"] == pytest.approx(5 / 9 - 1 / 3)
        probe_ids, _, probe_job = fake.calls[0]
        assert probe_ids == ["f1"] and probe_job == "cand_focused"
        confirm_ids, _, confirm_job = fake.calls[1]
        assert confirm_ids == train and confirm_job == confirm_job_name("cand")

    def test_min_confirm_lift_prunes(self):
        train = ["f1", "t2", "t3"]
        base = _evals({"f1": (0, 3), "t2": (3, 3), "t3": (0, 3)})
        fake = _FakeEval(
            {
                "_focused": _evals({"f1": (2, 3)}),
                "_confirm": _evals({"f1": (2, 3), "t2": (3, 3), "t3": (0, 3)}),
            }
        )
        out = FocusedFisherGate(k=3, min_confirm_lift=0.5).decide(_ctx(fake, base, train, focused_task_ids=["f1"]))
        assert out.status == NodeStatus.pruned_at_confirm

    def test_stable_sentinel_regression_prunes_at_screen(self):
        train = ["s1", "s2", "t3"]
        base = _evals({"s1": (3, 3), "s2": (3, 3), "t3": (0, 3)})
        fake = _FakeEval({"_focused": _evals({"s1": (1, 3), "s2": (3, 3)})})

        out = FocusedFisherGate(k=3).decide(_ctx(fake, base, train, sentinel_task_ids=["s1", "s2"]))
        assert out.status == NodeStatus.pruned_at_screen
        assert out.stats["sentinel_regression"] is True
        assert len(fake.calls) == 1

    def test_fragile_sentinel_noise_is_tolerated(self):

        train = ["s1", "t2"]
        base = _evals({"s1": (1, 3), "t2": (0, 3)})
        fake = _FakeEval(
            {
                "_focused": _evals({"s1": (0, 3)}),
                "_confirm": _evals({"s1": (1, 3), "t2": (2, 3)}),
            }
        )
        out = FocusedFisherGate(k=3).decide(_ctx(fake, base, train, sentinel_task_ids=["s1"]))
        assert out.status == NodeStatus.promoted_to_baseline
        assert out.stats["sent_fragile_p_worse"] == pytest.approx(0.5)

    def test_significantly_worse_probe_prunes_without_confirm(self):
        train = ["f1", "f2", "f3"]
        base = _evals({t: (3, 3) for t in train})
        fake = _FakeEval({"_focused": _evals({t: (0, 3) for t in train})})
        out = FocusedFisherGate(k=3).decide(_ctx(fake, base, train, focused_task_ids=train))
        assert out.status == NodeStatus.pruned_at_screen
        assert out.stats["pruned_significantly_worse"] is True

        assert out.stats["fisher_p_worse"] == pytest.approx(1 / 48620)
        assert len(fake.calls) == 1


class TestPairedTwoSigmaGate:
    def _anchor(self, ids, cull=0.1):
        return AnchorSelection(task_ids=ids, sigma_screen=cull, cull_threshold=cull, tasks=[], shortfalls={})

    def test_requires_anchor(self):
        fake = _FakeEval({})
        with pytest.raises(ValueError, match="anchor"):
            PairedTwoSigmaGate().decide(_ctx(fake, {}, ["t1"]))

    def test_clear_screen_loss_prunes_before_confirm(self):
        train = ["a1", "a2", "t3"]
        base = _evals({"a1": (3, 3), "a2": (3, 3), "t3": (0, 3)})
        fake = _FakeEval({"_screen": _evals({"a1": (0, 1), "a2": (0, 1)})})
        out = PairedTwoSigmaGate().decide(_ctx(fake, base, train, anchor=self._anchor(["a1", "a2"])))
        assert out.status == NodeStatus.pruned_at_screen
        assert out.screen.bucket == "cull"
        assert len(fake.calls) == 1

    def test_partial_control_at_confirm_is_inconclusive(self):
        base = _evals({"t1": (0, 1)})
        fake = _FakeEval(
            {
                "_screen": _evals({"t1": (1, 1)}),
                "_confirm": _evals({"t1": (3, 3)}),
            }
        )

        outcome = PairedTwoSigmaGate(k_screen=1, k_confirm=3).decide(
            _ctx(fake, base, ["t1"], anchor=self._anchor(["t1"]))
        )

        assert outcome.verdict is EvaluationVerdict.inconclusive
        assert outcome.status is NodeStatus.pruned_at_confirm
        assert not outcome.promoted

    def test_fired_subset_cannot_promote_a_full_train_regression(self):

        train = ["t1", "t2", "t3"]
        base = _evals({"t1": (0, 3), "t2": (3, 3), "t3": (3, 3)})
        fake = _FakeEval(
            {
                "_screen": _evals({"t2": (1, 1)}),
                "_confirm": _evals({"t1": (3, 3), "t2": (0, 3), "t3": (0, 3)}),
            }
        )
        out = PairedTwoSigmaGate().decide(
            _ctx(
                fake,
                base,
                train,
                anchor=self._anchor(["t2"]),
                fired_source=lambda node, ids: {"t1"},
            )
        )
        assert out.gate.paired.promoted
        assert out.status == NodeStatus.pruned_at_confirm
        assert out.verdict is EvaluationVerdict.rejected
        assert out.score == pytest.approx(1 / 3)
        assert out.gate.unfired_excluded == ["t2", "t3"]

    def test_provider_failure_at_confirm_is_failed_not_promoted(self):
        train = ["t1", "t2"]
        base = _evals({"t1": (0, 3), "t2": (0, 3)})
        fake = _FakeEval(
            {
                "_screen": _evals({"t1": (1, 1)}),
                "_confirm": {
                    "t1": _te("t1", 3, 3),
                    "t2": _te("t2", 0, 3, failure=MeasurementFailure.provider),
                },
            }
        )
        out = PairedTwoSigmaGate().decide(_ctx(fake, base, train, anchor=self._anchor(["t1"])))
        assert out.status is NodeStatus.errored
        assert out.verdict is EvaluationVerdict.failed
        assert not out.promoted
        assert out.stats["candidate_validity"]["provider_failures"] == ["t2"]


class TestVerdictPersistence:
    class _Candidate:
        why = "failure"
        files = {"codeflow/example.py": b"x"}
        deletions: list[str] = []
        summary = ""

    def test_archive_only_banks_accepted_outcomes(self, tmp_path):
        node = _node("candidate")
        evals = _evals({"t1": (3, 3)})
        for verdict in (
            EvaluationVerdict.rejected,
            EvaluationVerdict.failed,
            EvaluationVerdict.inconclusive,
        ):
            archive = GsmeArchive(tmp_path / f"{verdict.value}.json")
            archive.consider(
                parent_id="C0",
                node=node,
                cand=self._Candidate(),
                outcome=CandidateOutcome(
                    node.node_id,
                    NodeStatus.pruned_at_confirm,
                    score=1.0,
                    confirm_evals=evals,
                    verdict=verdict,
                ),
                vanilla_train_mean=0.0,
                round_index=1,
            )
            assert archive.cells == {}

        archive = GsmeArchive(tmp_path / "accepted.json")
        archive.consider(
            parent_id="C0",
            node=node,
            cand=self._Candidate(),
            outcome=CandidateOutcome(
                node.node_id,
                NodeStatus.promoted_to_baseline,
                score=1.0,
                confirm_evals=evals,
                verdict=EvaluationVerdict.accepted,
            ),
            vanilla_train_mean=0.0,
            round_index=1,
        )
        assert list(archive.cells.values())[0].node_id == node.node_id

    def test_journal_records_explicit_verdicts(self, tmp_path):
        outcomes = [
            CandidateOutcome("a", NodeStatus.promoted_to_baseline, verdict=EvaluationVerdict.accepted),
            CandidateOutcome("r", NodeStatus.pruned_at_confirm, verdict=EvaluationVerdict.rejected),
            CandidateOutcome("f", NodeStatus.errored, verdict=EvaluationVerdict.failed),
            CandidateOutcome("i", NodeStatus.pruned_at_confirm, verdict=EvaluationVerdict.inconclusive),
        ]
        journal = RoundJournal(tmp_path / "rounds.jsonl")
        journal.append(
            RoundResult(
                round_index=1,
                parent_id="C0",
                next_parent_id="a",
                promoted=True,
                outcomes=outcomes,
            )
        )
        record = json.loads(journal.path.read_text())
        assert [candidate["verdict"] for candidate in record["candidates"]] == [
            "accepted",
            "rejected",
            "failed",
            "inconclusive",
        ]
