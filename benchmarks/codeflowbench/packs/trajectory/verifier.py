from __future__ import annotations

from benchmarks.codeflowbench.records import VerificationState, VerifierResult
from .models import ContextQualityResult, JudgeScore, TrajectoryTask


class TrajectoryVerifier:
    """Verifies that Judge scores and context quality measurements are valid."""

    def verify_evaluation(
        self,
        task: TrajectoryTask,
        judge_score: JudgeScore,
        context_result: ContextQualityResult,
    ) -> VerifierResult:
        findings = []

        if not (1.0 <= judge_score.composite_score <= 5.0):
            findings.append(f"composite_score_out_of_bounds:{judge_score.composite_score}")

        if not (1.0 <= judge_score.tool_rationality <= 5.0):
            findings.append(f"tool_rationality_out_of_bounds:{judge_score.tool_rationality}")

        if not (1.0 <= judge_score.error_recovery <= 5.0):
            findings.append(f"error_recovery_out_of_bounds:{judge_score.error_recovery}")

        if context_result.input_tokens <= 0:
            findings.append("input_tokens_non_positive")

        if not (0.0 <= context_result.key_constraint_retention_rate <= 1.0):
            findings.append(
                f"retention_rate_out_of_bounds:{context_result.key_constraint_retention_rate}"
            )

        state = VerificationState.FAILED if findings else VerificationState.PASSED
        findings_tuple = tuple(findings) or (
            f"judge_score:{judge_score.composite_score}",
            f"retention_rate:{context_result.key_constraint_retention_rate * 100:.1f}%",
            f"token_reduction:{context_result.token_reduction_pct:.1f}%",
            f"synthetic:{judge_score.is_synthetic}",
        )

        return VerifierResult(state=state, findings=findings_tuple)
