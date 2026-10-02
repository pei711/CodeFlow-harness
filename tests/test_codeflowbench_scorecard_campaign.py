from __future__ import annotations

from pathlib import Path

from benchmarks.codeflowbench.campaign import (
    CampaignMode,
    estimate_worst_case_cost,
    load_campaign_suite,
)
from benchmarks.codeflowbench.scorecard_campaign import (
    DEFAULT_SCORECARD_SUITE_PATH,
    _runtime_subject_compatible,
    run_scorecard_campaign,
)


def test_scorecard_suite_excludes_completed_and_historical_packs() -> None:
    suite = load_campaign_suite(DEFAULT_SCORECARD_SUITE_PATH)

    assert suite.suite == "agent-application-scorecard-v1"
    assert suite.calibration.pack_ids == (
        "context-calibration",
        "tool-mcp-calibration",
    )
    assert suite.formal.pack_ids == (
        "context",
        "tool-mcp",
    )
    assert all(
        "semantic_memory" not in rule.metric and "tokenwise" not in rule.metric and "runtime" not in rule.metric
        for rule in suite.claim_rules
    )


def test_scorecard_suite_freezes_expected_denominators_and_budget() -> None:
    suite = load_campaign_suite(DEFAULT_SCORECARD_SUITE_PATH)
    estimate = estimate_worst_case_cost(
        suite,
        modes=(
            CampaignMode.CALIBRATION,
            CampaignMode.FORMAL,
        ),
    )

    assert suite.calibration.expected_trials == 32
    assert suite.formal.expected_trials == 96
    assert suite.calibration.expected_retrieval_cases == 0
    assert suite.formal.expected_retrieval_cases == 0
    trial_budgets = {
        budget.pack_id: budget.max_provider_calls_per_trial for budget in suite.budget.provider_trial_budgets
    }
    assert trial_budgets == {
        "context-calibration": 10,
        "context": 10,
        "tool-mcp-calibration": 6,
        "tool-mcp": 6,
    }
    assert estimate.estimated_cny <= suite.budget.hard_cap_cny
    assert suite.execution.provider_call_max_attempts == 4
    assert suite.budget.max_additional_provider_attempts == 96


def test_makefile_exposes_scorecard_estimate_and_ship() -> None:
    makefile = (Path(__file__).resolve().parents[1] / "Makefile").read_text(encoding="utf-8")

    assert "codeflowbench-scorecard-estimate:" in makefile
    assert "codeflowbench-scorecard-ship:" in makefile


def test_runtime_subject_continuity_accepts_identical_commit() -> None:
    commit = "a" * 40

    assert _runtime_subject_compatible(commit, commit) is True


def test_scorecard_campaign_runtime_evidence_is_optional() -> None:
    assert run_scorecard_campaign.__kwdefaults__ == {
        "runtime_evidence": None,
        "suite_path": DEFAULT_SCORECARD_SUITE_PATH,
    }
