from __future__ import annotations

from typing import Protocol

from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    TrialStatus,
    VerificationState,
    VerifierResult,
)
from benchmarks.codeflowbench.schema import PackDefinition, PairSpec, VariantSpec
from .models import SecurityScenario, SecurityTrack
from .runner import DeterministicSecurityRunner, SecurityTrialRunner
from .tasks import load_security_scenarios, security_task_set_digest

_REQUIRED_METRICS = frozenset(
    {
        "intercepted",
        "unapproved_dangerous_operations_executed",
        "outside_workspace_writes_executed",
        "category",
        "guard_mechanism",
        "repetition",
        "end_to_end_latency_ms",
    }
)


class SecurityPack:
    def __init__(
        self,
        track: SecurityTrack = SecurityTrack.FORMAL,
        *,
        runner: SecurityTrialRunner | None = None,
    ) -> None:
        self.track = SecurityTrack(track)
        self._runner = runner or DeterministicSecurityRunner()
        self._scenarios = {s.task_id: s for s in load_security_scenarios(self.track)}

    def definition(self) -> PackDefinition:
        pack_id = "security" if self.track is SecurityTrack.FORMAL else "security-smoke"
        return PackDefinition(
            pack_id=pack_id,
            tasks=tuple(s.to_task_spec() for s in self._scenarios.values()),
            variants=(
                VariantSpec(
                    variant_id="security-hardened",
                    settings={"security_policy": "hardened_boundary"},
                ),
                VariantSpec(
                    variant_id="security-permissive",
                    settings={"security_policy": "permissive_baseline"},
                ),
            ),
            pairs=(
                PairSpec(
                    treatment_axis="security_policy",
                    control_variant_id="security-permissive",
                    treatment_variant_id="security-hardened",
                ),
            ),
            identity={
                "task_set_digest": security_task_set_digest(self.track),
                "total_scenarios": len(self._scenarios),
                "result_scope": (
                    "security_30_scenarios_pack"
                    if self.track is SecurityTrack.FORMAL
                    else "security_smoke_pack"
                ),
                "verifier": "security_boundary_interceptor_v1",
                "claim_reducer": "security_v1",
                "categories": [
                    "unapproved_dangerous_command",
                    "outside_workspace_write",
                    "unauthorized_network_ssrf",
                    "sensitive_file_read",
                    "prompt_injection_escalation",
                ],
                "repetitions_per_scenario": 3,
                "target_metric": "zero_unapproved_operations_and_zero_escapes",
            },
        )

    async def run_trial(self, context: TrialContext) -> TrialExecution:
        scenario = self._scenarios.get(context.task.task_id)
        if scenario is None:
            return _infrastructure_failure(context, "unknown_security_scenario")

        execution = await self._runner(
            context=context,
            scenario=scenario,
            repetition=context.key.repetition,
        )

        missing = sorted(_REQUIRED_METRICS - execution.metrics.keys())
        if missing:
            return _infrastructure_failure(
                context,
                "missing_security_metrics:" + ",".join(missing),
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


__all__ = ["SecurityPack", "SecurityTrialRunner"]
