"""Evidence boundaries: no network, credentials, model calls or temporary dirs."""

import copy
import unittest
from unittest.mock import patch

from context_study_evidence import accounting, comparisons, diagnostics, digest, identity, validate_cells
from export_context_study import expected_outputs
from langfuse_codeflow import make_events


class EvidenceTests(unittest.TestCase):
    def manifest(self):
        return dict(
            model="test",
            tasks=[dict(spec=dict(task_id="one", payload={}), history=[])],
            variants={"baseline": "full", "governed": "compact"},
            planned_trials=2,
            source_hashes={"context/verifier.py": "a"},
        )

    def test_version_changes_with_config_prompt_verifier_and_dataset(self):
        original = self.manifest()
        for key, value in [
            ("model", "new"),
            ("source_hashes", {"context/verifier.py": "b"}),
            ("summary_output_tokens", 1000),
            ("output_contract_version", "new"),
        ]:
            changed = {**original, key: value}
            self.assertNotEqual(identity(original)["experiment_version"], identity(changed)["experiment_version"])
        changed = copy.deepcopy(original)
        changed["tasks"][0]["history"] = [{"role": "user", "content": "new decision"}]
        self.assertNotEqual(identity(original)["prompt_digest"], identity(changed)["prompt_digest"])
        self.assertFalse(identity(original)["runtime_config_complete"])

    def test_reject_duplicate_missing_and_unplanned_trials(self):
        rows = [dict(pair_id="one", arm=a, status="passed") for a in ("baseline", "governed")]
        validate_cells(self.manifest(), rows)
        for invalid in [rows[:1], [rows[0], rows[0]], [*rows, dict(pair_id="other", arm="baseline")]]:
            with self.assertRaises(ValueError):
                validate_cells(self.manifest(), invalid)

    def test_failed_latest_decision_is_evidence_not_proven_cause(self):
        detail = diagnostics(
            dict(
                status="task_failed",
                verifier_state="failed",
                findings=[],
                verifier_findings=["artifact_mismatch"],
                latest_decision_applied=False,
            )
        )
        self.assertIn("artifact_mismatch", detail["findings"])
        self.assertIn("latest_decision_applied_failed", detail["failure_categories"])
        self.assertEqual(detail["root_cause"], "unknown")
        self.assertEqual(detail["evidence_chain"]["summary_retention"], "unknown")
        self.assertIsNone(detail["early_constraint_retained"])

    def test_cached_usage_cost_and_failure_attempts_are_separate(self):
        row = dict(
            pair_id="one",
            arm="governed",
            usage_complete=True,
            trial_total_input_tokens=100,
            trial_total_output_tokens=10,
        )
        spans = [
            dict(
                attributes={
                    "span.type": "model",
                    "llm.usage.input_tokens": 20,
                    "llm.usage.cache_read_tokens": 80,
                    "llm.usage.cache_write_tokens": 0,
                    "llm.usage.output_tokens": 10,
                    "llm.usage.cost_total": 0.01,
                }
            )
        ]
        ledger = [
            dict(kind="reserved", trial_id="one:governed", request_id="a"),
            dict(kind="failed", request_id="a", charged_cny=0.2),
        ]
        result = accounting(row, spans, ledger)
        self.assertTrue(result["usage_reconciled"])
        self.assertEqual(result["recorded_runtime_estimate_usd"], 0.01)
        self.assertEqual(result["ledger_budget_charge_cny"], 0.2)
        self.assertFalse(result["ledger_usage_reconciled"])
        self.assertIsNone(result["actual_billed_cost_usd"])
        row["trial_total_input_tokens"] = 101
        self.assertIsNone(accounting(row, spans, ledger)["recorded_runtime_estimate_usd"])
        self.assertIsNone(accounting(row, [], [ledger[0]])["ledger_budget_charge_cny"])

    def test_total_cost_includes_failed_trials_unknown_is_not_zero(self):
        rows = [
            dict(
                arm="governed",
                scenario="test",
                status=s,
                verifier_state=v,
                diagnostics=diagnostics(dict(status=s, verifier_state=v)),
                accounting=dict(recorded_runtime_estimate_usd=c, trial_wall_latency_ms=10),
            )
            for s, v, c in [("passed", "passed", 1), ("task_failed", "failed", 2)]
        ]
        group = comparisons(rows)["overall"]["governed"]
        self.assertEqual(group["estimate_usd_per_verified_task_including_failures"], 3)
        rows[1]["accounting"]["recorded_runtime_estimate_usd"] = None
        self.assertIsNone(comparisons(rows)["overall"]["governed"]["recorded_runtime_estimate_usd"])

    def test_frozen_expected_output_is_checked_before_upload(self):
        manifest = self.manifest()
        task = manifest["tasks"][0]
        task.update(expected_output={"x": 1}, expected_digest=digest({"x": 1}))
        with patch("pathlib.Path.glob", return_value=[]):
            self.assertEqual(expected_outputs(manifest), {"one": {"x": 1}})
            task["expected_output"] = {"x": 2}
            with self.assertRaises(ValueError):
                expected_outputs(manifest)

    def test_export_scores_keep_unknown_missing_and_metadata_sanitized(self):
        result = dict(
            run_id="offline",
            task="test",
            task_id="one",
            commit="test",
            model="test",
            started_at="2026-11-01T00:00:00+00:00",
            finished_at="2026-11-01T00:00:02+00:00",
            task_latency_ms=2000,
            return_code=1,
            timed_out=False,
            verifier_passed=False,
            evidence_metadata={"failure_category": "artifact_mismatch", "api_key": "private"},
            evidence_scores={
                "latest_decision_applied": False,
                "unknown": None,
                "cost": float("nan"),
                "estimate_usd": 0.01,
            },
        )
        _, events = make_events(result, [])
        self.assertEqual(events[0].body.metadata["api_key"], "[REDACTED]")
        scores = {e.body.name: e.body.value for e in events if e.type == "score-create"}
        self.assertEqual(scores["latest_decision_applied"], 0)
        self.assertNotIn("unknown", scores)
        self.assertNotIn("cost", scores)
        self.assertEqual(scores["estimate_usd"], 0.01)


if __name__ == "__main__":
    unittest.main()
