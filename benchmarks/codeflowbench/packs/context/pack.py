from __future__ import annotations

from typing import Protocol

from benchmarks.codeflowbench.canonical import canonical_digest
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
        "trial_total_output_tokens",
        "context_auxiliary_input_tokens",
        "repeated_tool_calls",
        "context_budget_ratio",
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
_REQUIRED_THRESHOLD_METRICS = frozenset(
    {
        "structured_compact_threshold",
        "normal_trim_count",
        "compact_count",
        "summary_model_call_count",
        "summary_fallback_count",
        "over_budget_count",
        "trigger_pressure",
        "history_pressure",
        "summary_call_latency_ms",
        "agent_steps",
        "effective_history_budget",
        "pressure_target",
        "main_agent_output_tokens",
        "context_auxiliary_output_tokens",
        "provider_reported_models",
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
        tasks: tuple[ContextTask, ...] | None = None,
        budget_ratios: tuple[float, ...] | None = None,
        compact_thresholds: tuple[float, ...] | None = None,
    ) -> None:
        self.track = ContextTrack(track)
        self._runner = runner
        selected_tasks = tasks if tasks is not None else load_context_tasks(self.track)
        if not selected_tasks:
            raise ValueError("ContextPack requires at least one task")
        self._tasks = {task.task_id: task for task in selected_tasks}
        if len(self._tasks) != len(selected_tasks):
            raise ValueError("ContextPack task ids must be unique")
        self._budget_ratios = budget_ratios
        self._compact_thresholds = compact_thresholds
        if budget_ratios is not None and (
            len(budget_ratios) < 2
            or len(set(budget_ratios)) != len(budget_ratios)
            or any(not 0 < ratio <= 1 for ratio in budget_ratios)
        ):
            raise ValueError("Context budget ratios must be unique values in (0, 1]")
        if compact_thresholds is not None and (
            len(compact_thresholds) < 2
            or len(set(compact_thresholds)) != len(compact_thresholds)
            or any(not 0 < ratio <= 1 for ratio in compact_thresholds)
        ):
            raise ValueError("Compact thresholds must be unique values in (0, 1]")
        if budget_ratios is not None and compact_thresholds is not None:
            raise ValueError("Choose budget ratios or Compact thresholds, not both")

    def definition(self) -> PackDefinition:
        if self._compact_thresholds is not None:
            variants = tuple(
                VariantSpec(
                    variant_id=f"compact-{round(ratio * 100)}",
                    settings={
                        "history_manager": "codeflow_history",
                        "structured_compact_threshold": ratio,
                        "context_budget_ratio": 0.80,
                    },
                )
                for ratio in self._compact_thresholds
            )
            control_id = variants[0].variant_id
            pair_specs = tuple(
                PairSpec(
                    treatment_axis="structured_compact_threshold",
                    control_variant_id=control_id,
                    treatment_variant_id=variant.variant_id,
                )
                for variant in variants[1:]
            )
            pack_id = "context-threshold-ablation"
            task_digest = canonical_digest(
                [
                    {
                        "task": task.to_task_spec(),
                        "history": task.materialize_history(),
                        "expected": task.expected_digest,
                    }
                    for task in self._tasks.values()
                ]
            )
            identity = {
                "task_set_digest": task_digest,
                "result_scope": "exploratory_structured_compact_threshold_ablation",
                "verifier": "external_sealed_json",
                "history_manager": "shipping_codeflow_history",
                "structured_compact_thresholds": list(self._compact_thresholds),
                "context_budget_ratio": 0.80,
                "context_window_tokens": CONTEXT_BENCHMARK_WINDOW_TOKENS,
                "reserved_output_tokens": CONTEXT_BENCHMARK_OUTPUT_TOKENS,
            }
        elif self._budget_ratios is not None:
            variants = tuple(
                VariantSpec(
                    variant_id=f"budget-{round(ratio * 100)}",
                    settings={
                        "history_manager": "curator",
                        "context_budget_ratio": ratio,
                    },
                )
                for ratio in self._budget_ratios
            )
            control_id = variants[0].variant_id
            pair_specs = tuple(
                PairSpec(
                    treatment_axis="context_budget_ratio",
                    control_variant_id=control_id,
                    treatment_variant_id=variant.variant_id,
                )
                for variant in variants[1:]
            )
            pack_id = "context-budget-study"
            task_digest = canonical_digest(
                [
                    {
                        "task": task.to_task_spec(),
                        "history": task.materialize_history(),
                        "expected": task.expected_digest,
                    }
                    for task in self._tasks.values()
                ]
            )
            identity = {
                "task_set_digest": task_digest,
                "result_scope": "exploratory_context_budget_study",
                "verifier": "external_sealed_json",
                "history_manager": "curator",
                "context_budget_ratios": list(self._budget_ratios),
                "context_window_tokens": CONTEXT_BENCHMARK_WINDOW_TOKENS,
                "reserved_output_tokens": CONTEXT_BENCHMARK_OUTPUT_TOKENS,
            }
        else:
            pack_id = "context" if self.track is ContextTrack.FORMAL else "context-calibration"
            variants = (
                VariantSpec(
                    variant_id="context-fifo",
                    settings={"history_manager": "fifo_tail"},
                ),
                VariantSpec(
                    variant_id="context-curator",
                    settings={"history_manager": "curator"},
                ),
            )
            pair_specs = (
                PairSpec(
                    treatment_axis="history_manager",
                    control_variant_id="context-fifo",
                    treatment_variant_id="context-curator",
                ),
            )
            identity = {
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
            }
        return PackDefinition(
            pack_id=pack_id,
            tasks=tuple(task.to_task_spec() for task in self._tasks.values()),
            variants=variants,
            pairs=pair_specs,
            identity=identity,
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
        if self._compact_thresholds is not None:
            missing = sorted(set(missing) | (_REQUIRED_THRESHOLD_METRICS - execution.metrics.keys()))
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
