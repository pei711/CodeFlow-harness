from __future__ import annotations

from typing import Protocol

from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    TrialStatus,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import PackDefinition, PairSpec, VariantSpec
from .models import SwebenchTask, SwebenchTrack
from .runner import DeterministicSwebenchRunner, SwebenchTrialRunner
from .tasks import load_swebench_tasks, swebench_task_set_digest

_REQUIRED_METRICS = frozenset(
    {
        "resolved",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "context_strategy",
        "patch_generated",
        "patch_lines_added",
        "patch_lines_removed",
        "fail_to_pass_passed",
        "fail_to_pass_total",
        "pass_to_pass_passed",
        "pass_to_pass_total",
        "turns_taken",
        "end_to_end_latency_ms",
    }
)


class SwebenchPack:
    def __init__(
        self,
        track: SwebenchTrack = SwebenchTrack.FORMAL,
        *,
        runner: SwebenchTrialRunner | None = None,
    ) -> None:
        self.track = SwebenchTrack(track)
        self._runner = runner or DeterministicSwebenchRunner()
        self._tasks = {task.task_id: task for task in load_swebench_tasks(self.track)}

    def definition(self) -> PackDefinition:
        pack_id = "swebench" if self.track is SwebenchTrack.FORMAL else "swebench-smoke"
        return PackDefinition(
            pack_id=pack_id,
            tasks=tuple(task.to_task_spec() for task in self._tasks.values()),
            variants=(
                VariantSpec(
                    variant_id="swebench-fifo",
                    settings={"history_manager": "fifo_tail"},
                ),
                VariantSpec(
                    variant_id="swebench-curator",
                    settings={"history_manager": "curator"},
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="history_manager",
                    control_variant_id="swebench-fifo",
                    treatment_variant_id="swebench-curator",
                ),
            ),
            identity={
                "task_set_digest": swebench_task_set_digest(self.track),
                "total_tasks": len(self._tasks),
                "result_scope": (
                    "swebench_verified_50_pack"
                    if self.track is SwebenchTrack.FORMAL
                    else "swebench_smoke_pack"
                ),
                "verifier": "swebench_test_evaluator_v1",
                "claim_reducer": "swebench_v1",
                "baseline_control": "threshold_triggered_compression (fifo_tail)",
                "treatment": "layered_governance (curator)",
                "target_metric": "resolution_rate_and_token_efficiency",
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        task = self._tasks.get(context.task.task_id)
        if task is None:
            return _infrastructure_failure(context, "unknown_swebench_task")

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
                "missing_swebench_metrics:" + ",".join(missing),
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


__all__ = ["SwebenchPack", "SwebenchTrialRunner"]
