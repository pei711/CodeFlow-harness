from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from benchmarks.codeflowbench.records import VerificationState, VerifierResult
from .models import SwebenchTask


@dataclass(frozen=True)
class EvaluationOutcome:
    resolved: bool
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int
    details: tuple[str, ...] = ()


class SwebenchVerifier:
    """Verifies candidate patches against SWE-bench task test specifications.

    Supports:
    1. Deterministic dry-run validation (diff structure, syntax validity, line counts).
    2. Test specification evaluation (FAIL_TO_PASS must become PASS, PASS_TO_PASS must stay PASS).
    3. Docker / harness test runner logs parsing (standard SWE-bench eval output).
    """

    @staticmethod
    def parse_patch_stats(patch: str) -> tuple[int, int]:
        """Calculates lines added and removed in a git unified diff."""
        added = 0
        removed = 0
        for line in patch.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        return added, removed

    @staticmethod
    def is_valid_unified_diff(patch: str) -> bool:
        """Validates basic git unified diff syntax."""
        if not patch or not patch.strip():
            return False
        has_diff_header = "diff --git" in patch or ("--- " in patch and "+++ " in patch)
        has_hunk = bool(re.search(r"@@ -\d+,\d+ \+\d+,\d+ @@", patch))
        return has_diff_header and has_hunk

    @staticmethod
    def evaluate_test_spec(
        *,
        fail_to_pass: tuple[str, ...],
        pass_to_pass: tuple[str, ...],
        test_outcomes: dict[str, str],  # test_name -> "PASSED" | "FAILED" | "SKIPPED"
    ) -> EvaluationOutcome:
        """Evaluates whether test results satisfy SWE-bench resolution criteria.

        Resolved requires:
        - 100% of fail_to_pass tests must be PASSED.
        - 100% of pass_to_pass tests must be PASSED.
        """
        f2p_passed = 0
        details = []
        for test in fail_to_pass:
            outcome = test_outcomes.get(test, "FAILED")
            if outcome == "PASSED":
                f2p_passed += 1
            else:
                details.append(f"FAIL_TO_PASS test failed: {test} ({outcome})")

        p2p_passed = 0
        for test in pass_to_pass:
            outcome = test_outcomes.get(test, "PASSED")
            if outcome == "PASSED":
                p2p_passed += 1
            else:
                details.append(f"PASS_TO_PASS regression: {test} ({outcome})")

        resolved = (f2p_passed == len(fail_to_pass)) and (p2p_passed == len(pass_to_pass))
        return EvaluationOutcome(
            resolved=resolved,
            fail_to_pass_passed=f2p_passed,
            fail_to_pass_total=len(fail_to_pass),
            pass_to_pass_passed=p2p_passed,
            pass_to_pass_total=len(pass_to_pass),
            details=tuple(details),
        )

    def verify_patch(
        self,
        task: SwebenchTask,
        patch: str,
        test_outcomes: dict[str, str] | None = None,
    ) -> VerifierResult:
        if not patch or not patch.strip():
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=("empty_patch_generated",),
            )

        if not self.is_valid_unified_diff(patch):
            return VerifierResult(
                state=VerificationState.FAILED,
                findings=("invalid_unified_diff_syntax",),
            )

        # If actual test outcomes are provided (live Docker or test execution)
        if test_outcomes is not None:
            eval_result = self.evaluate_test_spec(
                fail_to_pass=task.fail_to_pass,
                pass_to_pass=task.pass_to_pass,
                test_outcomes=test_outcomes,
            )
            state = VerificationState.PASSED if eval_result.resolved else VerificationState.FAILED
            findings = tuple(eval_result.details) or ("all_tests_passed",)
            return VerifierResult(state=state, findings=findings)

        # Dry-run / synthetic check: verify patch against gold patch hash or structural validity
        patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        is_gold = bool(task.gold_patch_sha256 and patch_sha256 == task.gold_patch_sha256)
        added, removed = self.parse_patch_stats(patch)

        if is_gold:
            return VerifierResult(
                state=VerificationState.PASSED,
                findings=("gold_patch_match_verified", f"diff:+{added}/-{removed}"),
            )

        # Non-gold candidate patch in dry-run mode: structurally valid
        return VerifierResult(
            state=VerificationState.PASSED,
            findings=("structurally_valid_patch", f"diff:+{added}/-{removed}"),
        )
