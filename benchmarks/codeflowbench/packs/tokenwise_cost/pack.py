from __future__ import annotations

from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import TrialStatus, VerificationState, VerifierResult
from benchmarks.codeflowbench.schema import PackDefinition, PairSpec, TaskSpec, VariantSpec

from .reducer import (
    CACHE_POLICY_PREFIX_DISRUPTED,
    CACHE_POLICY_PREFIX_STABLE,
)

_WORKLOAD_SHAPES = (
    ("stable_dialogue", 6, 0, 0),
    ("long_history", 6, 16, 0),
    ("tool_accumulation", 6, 0, 1),
    ("intra_turn_tool_chain", 1, 0, 3),
)


class TokenWiseCostPack:
    def definition(self) -> PackDefinition:
        tasks = tuple(
            TaskSpec(
                task_id=f"tw-{workload_class.replace('_', '-')}-{case_index}",
                payload={
                    "workload_class": workload_class,
                    "case_index": case_index,
                    "turn_count": turn_count,
                    "seed_history_turns": seed_history_turns,
                    "expected_tool_calls_per_turn": tool_calls_per_turn,
                    "verifier": "sealed_exact_outcome_v1",
                },
            )
            for workload_class, turn_count, seed_history_turns, tool_calls_per_turn in _WORKLOAD_SHAPES
            for case_index in range(1, 4)
        )
        return PackDefinition(
            pack_id="tokenwise-cost",
            tasks=tasks,
            variants=tuple(
                VariantSpec(variant_id=variant_id, settings={"cache_policy": policy})
                for variant_id, policy in (
                    ("tokenwise-prefix-disrupted", CACHE_POLICY_PREFIX_DISRUPTED),
                    ("tokenwise-prefix-stable", CACHE_POLICY_PREFIX_STABLE),
                )
            ),
            pairs=(
                PairSpec(
                    treatment_axis="cache_policy",
                    control_variant_id="tokenwise-prefix-disrupted",
                    treatment_variant_id="tokenwise-prefix-stable",
                ),
            ),
            identity={
                "claim_reducer": "tokenwise_cost_v1",
                "workload_classes": [shape[0] for shape in _WORKLOAD_SHAPES],
                "cache_hit_denominator": "fresh_plus_cache_write_plus_cache_read",
                "primary_metric": "cost_per_verified_success_usd",
                "result_scope": "deepseek_automatic_cache_with_negative_control",
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        finding = "tokenwise_cost_live_runner_not_configured"
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


__all__ = ["TokenWiseCostPack"]
