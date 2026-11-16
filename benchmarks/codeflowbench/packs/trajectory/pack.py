from __future__ import annotations

from typing import Protocol

from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    TrialStatus,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import PackDefinition, PairSpec, VariantSpec
from .judge import DeterministicJudge, TrajectoryJudge
from .models import TrajectoryTask, TrajectoryTrack
from .runner import DeterministicTrajectoryRunner, TrajectoryTrialRunner
from .tasks import load_trajectory_tasks, trajectory_task_set_digest

_REQUIRED_METRICS = frozenset(
    {
        "tool_rationality",
        "error_recovery",
        "constraint_preservation",
        "composite_judge_score",
        "early_constraint_retained",
        "active_constraint_applied",
        "latest_decision_applied",
        "key_constraint_retention_rate",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "context_strategy",
        "token_reduction_pct",
        "end_to_end_latency_ms",
    }
)


class TrajectoryPack:
    def __init__(
        self,
        track: TrajectoryTrack = TrajectoryTrack.FORMAL,
        *,
        runner: TrajectoryTrialRunner | None = None,
        judge: TrajectoryJudge | None = None,
    ) -> None:
        self.track = TrajectoryTrack(track)
        self._runner = runner or DeterministicTrajectoryRunner(judge=judge)
        self._tasks = {t.task_id: t for t in load_trajectory_tasks(self.track)}

    def definition(self) -> PackDefinition:
        pack_id = "trajectory" if self.track is TrajectoryTrack.FORMAL else "trajectory-smoke"
        return PackDefinition(
            pack_id=pack_id,
            tasks=tuple(t.to_task_spec() for t in self._tasks.values()),
            variants=(
                VariantSpec(
                    variant_id="trajectory-fifo",
                    settings={"history_manager": "fifo_tail"},
                ),
                VariantSpec(
                    variant_id="trajectory-curator",
                    settings={"history_manager": "curator"},
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="history_manager",
                    control_variant_id="trajectory-fifo",
                    treatment_variant_id="trajectory-curator",
                ),
            ),
            identity={
                "task_set_digest": trajectory_task_set_digest(self.track),
                "total_tasks": len(self._tasks),
                "result_scope": (
                    "trajectory_llm_judge_formal_pack"
                    if self.track is TrajectoryTrack.FORMAL
                    else "trajectory_smoke_pack"
                ),
                "verifier": "trajectory_llm_judge_and_context_v1",
                "claim_reducer": "trajectory_v1",
                "categories": [
                    "long_session",
                    "cross_turn_dependency",
                    "compression_degradation",
                ],
                "judge_rubric_dimensions": [
                    "tool_rationality",
                    "error_recovery",
                    "constraint_preservation",
                ],
                "scale": "1.0_to_5.0_likert",
                "baseline_control": "fifo_tail_compression",
                "treatment": "codeflow_layered_curator",
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        task = self._tasks.get(context.task.task_id)
        if task is None:
            return _infrastructure_failure(context, "unknown_trajectory_task")

        history_manager = context.variant.settings.get("history_manager")
        if not isinstance(history_manager, str):
            return _infrastructure_failure(context, "history_manager_not_declared")

        execution = await self._runner(
            context=context,
            task=task,
            strategy=history_manager,
        )

        missing = sorted(_REQUIRED_METRICS - execution.metrics.keys())
        if missing:
            return _infrastructure_failure(
                context,
                "missing_trajectory_metrics:" + ",".join(missing),
            )

        if dict(execution.observed_variant_settings) != dict(context.variant.settings):
            return _infrastructure_failure(
                context,
                "observed_variant_settings_drift",
            )

        return execution


def _infrastructure_failure(
    context: TrialContext,
    finding: str,
) -> TrialExecution:
    return TrialExecution(
        status=TrialStatus.INFRASTRUCTURE_FAILURE,
        runtime_state=None,
        delivery_state=None,
        verification=VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=(finding,),
        ),
        observed_variant_settings=dict(context.variant.settings),
        findings=(finding,),
    )


__all__ = ["TrajectoryPack", "TrajectoryTrialRunner"]
