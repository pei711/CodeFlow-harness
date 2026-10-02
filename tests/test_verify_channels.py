"""Tests for the V-C0 / V-S0 Channel evidence gate driver.

They pin the two things a gate can silently lose: an honest pass/fail
classification, and a selection that still points at tests that exist.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_channels import (
    _ALLOWLIST_TESTS,
    _CHECKS,
    _CONTRACT_TESTS,
    _ISOLATION_TESTS,
    _SDK_LAZINESS_TESTS,
    _SECURITY_TESTS,
    _classify,
    _counts,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_success_requires_a_clean_pass() -> None:
    assert _classify(0, "12 passed in 0.4s") == "passed"


def test_skip_and_expected_failure_cannot_pass_gate() -> None:
    assert _classify(0, "1 skipped in 0.01s") == "failed"
    assert _classify(0, "1 xfailed in 0.01s") == "failed"
    assert _classify(0, "1 xpassed in 0.01s") == "failed"


def test_mixed_pass_and_non_pass_cannot_pass_gate() -> None:
    assert _classify(0, "5 passed, 1 skipped in 0.1s") == "failed"
    assert _classify(0, "no tests ran in 0.01s") == "failed"


def test_failure_and_infrastructure_failure_stay_separate() -> None:
    assert _classify(1, "1 failed, 3 passed in 0.2s") == "failed"
    assert _classify(2, "infrastructure_failure: worker crashed") == "infrastructure_failure"


def test_counts_scrape_the_pytest_summary() -> None:
    counts = _counts("2 failed, 30 passed, 1 skipped, 4 deselected in 1.2s")
    assert counts["failed"] == 2
    assert counts["passed"] == 30
    assert counts["skipped"] == 1
    assert counts["deselected"] == 4


def test_gate_names_are_stable() -> None:
    assert [(name, gate) for name, gate, _tests in _CHECKS] == [("contract", "V-C0"), ("security", "V-S0")]


@pytest.mark.parametrize("selection", sorted(set(_CONTRACT_TESTS) | set(_SECURITY_TESTS)))
def test_every_selected_path_exists(selection: str) -> None:
    assert (_REPO_ROOT / selection.split("::", 1)[0]).is_file()
    assert "integration" not in selection


def test_security_bundle_covers_its_three_claims() -> None:
    """V-S0 is SDK laziness plus deny-by-default plus adapter isolation; losing
    any one of the three would still leave a runnable but weaker gate."""
    for claim in (_SDK_LAZINESS_TESTS, _ALLOWLIST_TESTS, _ISOLATION_TESTS):
        assert claim
        assert set(claim) <= set(_SECURITY_TESTS)


def test_security_selection_resolves_to_real_test_functions() -> None:
    """Node ids are brittle by design: a rename must fail the gate loudly here,
    not shrink the bundle in silence."""
    node_ids = [t for t in _SECURITY_TESTS if "::" in t]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_qq_and_wecom_adapter_files_are_in_the_contract_bundle() -> None:
    assert "tests/test_channels_qq.py" in _CONTRACT_TESTS
    assert "tests/test_channels_wecom.py" in _CONTRACT_TESTS
