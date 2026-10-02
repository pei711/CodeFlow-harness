from __future__ import annotations

from typing import Protocol

from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    TrialStatus,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import PackDefinition, PairSpec, VariantSpec

from .history import (
    CONTEXT_BENCHMARK_CURATOR_MAX_STEPS,
    context_engine_factory_for,
)
from .models import ContextTask, ContextTrack
from .runner import (
    CONTEXT_BENCHMARK_MAX_TOOL_ITERATIONS,
    CONTEXT_BENCHMARK_OUTPUT_TOKENS,
    CONTEXT_BENCHMARK_PROTECT_FIRST_N,
    CONTEXT_BENCHMARK_WINDOW_TOKENS,
)
from .tasks import context_task_set_digest, load_context_tasks

_REQUIRED_METRICS = frozenset(
    {
        "main_agent_input_tokens",
        "trial_total_input_tokens",
        "context_auxiliary_input_tokens",
        "usage_complete",
        "context_path",
        "early_constraint_retained",
        "artifact_valid_json",
        "active_constraint_applied",
        "latest_decision_applied",
        "artifact_exact",
        "forbidden_paths_clean",
        "capability_criteria_passed",
        "capability_criteria_total",
        "end_to_end_latency_ms",
    }
)


class ContextTrialRunner(Protocol):
    async def __call__(
        self,
        *,
        context: TrialContext,
        task: ContextTask,
        context_engine_factory,
    ) -> TrialExecution: ...


class ContextPack:
    def __init__(
        self,
        track: ContextTrack = ContextTrack.FORMAL,
        *,
        runner: ContextTrialRunner | None = None,
    ) -> None:
        self.track = ContextTrack(track)
        self._runner = runner
        self._tasks = {task.task_id: task for task in load_context_tasks(self.track)}

    def definition(self) -> PackDefinition:
        pack_id = "context" if self.track is ContextTrack.FORMAL else "context-calibration"
        return PackDefinition(
            pack_id=pack_id,
            tasks=tuple(task.to_task_spec() for task in self._tasks.values()),
            variants=(
                VariantSpec(
                    variant_id="context-fifo",
                    settings={"history_manager": "fifo_tail"},
                ),
                VariantSpec(
                    variant_id="context-curator",
                    settings={"history_manager": "curator"},
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="history_manager",
                    control_variant_id="context-fifo",
                    treatment_variant_id="context-curator",
                ),
            ),
            identity={
                "task_set_digest": context_task_set_digest(self.track),
                "result_scope": (
                    "exploratory_eight_task_pack" if self.track is ContextTrack.FORMAL else "calibration_only"
                ),
                "verifier": "external_sealed_json",
                "claim_reducer": "context_v2",
                "context_window_tokens": (CONTEXT_BENCHMARK_WINDOW_TOKENS),
                "reserved_output_tokens": (CONTEXT_BENCHMARK_OUTPUT_TOKENS),
                "protected_initial_turns": (CONTEXT_BENCHMARK_PROTECT_FIRST_N),
                "curator_max_steps": (CONTEXT_BENCHMARK_CURATOR_MAX_STEPS),
                "main_agent_max_tool_iterations": (CONTEXT_BENCHMARK_MAX_TOOL_ITERATIONS),
                "capability_criteria": (
                    "early_constraint_retained",
                    "active_constraint_applied",
                    "latest_decision_applied",
                    "artifact_exact",
                ),
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        if self._runner is None:
            return _infrastructure_failure(
                context,
                "context_trial_runner_not_configured",
            )
        task = self._tasks.get(context.task.task_id)
        if task is None:
            return _infrastructure_failure(context, "unknown_context_task")
        history_manager = context.variant.settings.get("history_manager")
        if not isinstance(history_manager, str):
            return _infrastructure_failure(
                context,
                "history_manager_not_declared",
            )
        try:
            factory = context_engine_factory_for(history_manager)
        except ValueError:
            return _infrastructure_failure(
                context,
                "history_manager_not_supported",
            )
        execution = await self._runner(
            context=context,
            task=task,
            context_engine_factory=factory,
        )
        missing = sorted(_REQUIRED_METRICS - execution.metrics.keys())
        if missing:
            return _infrastructure_failure(
                context,
                "missing_context_metrics:" + ",".join(missing),
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


__all__ = ["ContextPack", "ContextTrialRunner"]
