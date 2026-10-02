"""Historical EverOS-backed CodeFlowBench path retained without a current Runtime."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.codeflowbench.budget import BudgetGuardedProvider
from benchmarks.codeflowbench.canonical import to_primitive
from benchmarks.codeflowbench.isolation import TrialIsolation
from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    DeliveryOutcome,
    TrialStatus,
    TurnTerminalState,
    VerificationState,
    VerifierResult,
)

from .models import CrossSessionTask

_PROCESS_TERMINATION_GRACE_SECONDS = 5.0


class DeterministicCrossProcessRunner:
    kind = "deterministic_cross_process_runtime"

    async def run(self, context: TrialContext) -> TrialExecution:
        return await _run_cross_process_trial(
            context,
            provider_spec={
                "mode": "scripted",
                "provider_identity": "scripted/codeflowbench-memory-skill",
                "paid_campaign_eligible": False,
                "real_agent_task_effect_claim_eligible": False,
            },
        )


class RuntimeCrossProcessRunner:
    kind = "runtime_cross_process_real_provider"

    def __init__(
        self,
        *,
        config: Any,
        codeflow_config: Any,
        provider: Any,
    ) -> None:
        if not isinstance(provider, BudgetGuardedProvider):
            raise ValueError(
                "RuntimeCrossProcessRunner requires BudgetGuardedProvider",
            )
        self._config = config
        self._codeflow_config = codeflow_config
        self._provider = provider

    async def run(self, context: TrialContext) -> TrialExecution:
        pack_budget = context.experiment.execution.provider_trial_budget_for(
            context.key.pack_id,
        )
        if (
            pack_budget is None
            or pack_budget.max_provider_calls_per_trial != 4
            or pack_budget.max_input_tokens_per_call != 15_000
            or pack_budget.max_output_tokens_per_call != 1_500
        ):
            return _preparation_failure(
                context,
                "memory_skill_real_runner_requires_frozen_provider_budget",
            )
        ledger = self._provider.ledger
        with tempfile.TemporaryDirectory(
            prefix="codeflowbench-memory-skill-provider-",
        ) as private_root:
            config_path = Path(private_root) / "runtime-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "config": self._config.model_dump(mode="json"),
                        "codeflow_config": self._codeflow_config.model_dump(mode="json"),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            config_path.chmod(0o600)
            trial_id = (
                f"{context.experiment_id}/"
                f"{context.key.pack_id}/{context.key.task_id}/"
                f"{context.key.variant_id}/{context.key.repetition}/"
                f"{context.block_attempt}"
            )
            return await _run_cross_process_trial(
                context,
                provider_spec={
                    "mode": "real",
                    "provider_identity": str(self._config.agents.defaults.model),
                    "private_config_path": str(config_path),
                    "ledger_path": str(ledger.path.resolve()),
                    "ledger_config": asdict(ledger.config),
                    "trial_id": trial_id,
                    "max_attempts_per_call": (context.experiment.execution.provider_call_max_attempts),
                    "max_input_tokens_per_call": (pack_budget.max_input_tokens_per_call),
                    "max_output_tokens_per_call": (pack_budget.max_output_tokens_per_call),
                    "stage_logical_call_caps": {
                        "learning": 1,
                        "evaluation": 3,
                    },
                    "paid_campaign_eligible": True,
                    "real_agent_task_effect_claim_eligible": False,
                },
            )


async def _run_cross_process_trial(
    context: TrialContext,
    *,
    provider_spec: dict[str, Any],
) -> TrialExecution:
    task = CrossSessionTask(**dict(context.task.payload))
    attempt_id = f"{task.task_id}-{context.variant.variant_id}-r{context.key.repetition}-b{context.block_attempt}"
    isolation = TrialIsolation.create(
        context.experiment.output_root / ".codeflowbench-memory-skill" / context.experiment_id,
        attempt_id,
    )
    isolation.prepare()
    _prepare_workspace(isolation.workspace, task)
    state_path = isolation.everos_root / "memory-state.json"
    spec_path = isolation.root / "worker-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "task": to_primitive(task),
                "variant_settings": dict(context.variant.settings),
                "workspace": str(isolation.workspace),
                "state_path": str(state_path),
                "provider": provider_spec,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    learning = await _run_stage(
        stage="learning",
        spec_path=spec_path,
        isolation=isolation,
    )
    if learning.get("status") != "completed":
        return _infrastructure_failure(
            context,
            isolation,
            f"learning_worker_failed:{learning.get('error_type', 'unknown')}",
        )
    learning_runtime_state = TurnTerminalState(
        str(learning["runtime_state"]),
    )
    if (
        learning_runtime_state is not TurnTerminalState.COMPLETED
        or _safe_failure_category(learning) == "task_budget_exhausted"
    ):
        return _learning_runtime_failure(
            context,
            isolation,
            learning,
            learning_runtime_state,
        )
    evaluation = await _run_stage(
        stage="evaluation",
        spec_path=spec_path,
        isolation=isolation,
    )
    if evaluation.get("status") != "completed":
        return _infrastructure_failure(
            context,
            isolation,
            f"evaluation_worker_failed:{evaluation.get('error_type', 'unknown')}",
        )
    elapsed_ms = (time.perf_counter() - started) * 1_000
    verification = _verify_result(isolation.workspace, task)
    findings = _integrity_findings(
        context=context,
        learning=learning,
        evaluation=evaluation,
    )
    runtime_state = TurnTerminalState(str(evaluation["runtime_state"]))
    delivery_state = DeliveryOutcome(str(evaluation["delivery_state"]))
    evaluation_failure_category = _safe_failure_category(evaluation)
    if findings:
        status = TrialStatus.INFRASTRUCTURE_FAILURE
        verification = VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=tuple(findings),
        )
    elif evaluation_failure_category == "task_budget_exhausted":
        status = TrialStatus.TASK_TIMEOUT
    elif runtime_state is TurnTerminalState.PROVIDER_FAILED:
        status = TrialStatus.PROVIDER_FAILURE
    elif runtime_state not in {
        TurnTerminalState.COMPLETED,
        TurnTerminalState.COMPLETED_WITH_TOOL_FAILURE,
    }:
        status = TrialStatus.TASK_FAILED
    elif verification.state is VerificationState.PASSED:
        status = TrialStatus.PASSED
    else:
        status = TrialStatus.TASK_FAILED
    usage = _aggregate_usage(learning, evaluation)
    relative_root = isolation.root.relative_to(
        context.experiment.output_root,
    )
    injected_skill_ids = tuple(str(item) for item in evaluation.get("injected_skill_ids", []))
    source_contribution = {
        source: sum(item.startswith(f"{source}/") for item in injected_skill_ids) for source in ("local", "everos")
    }
    paid_campaign_eligible = bool(learning["paid_campaign_eligible"] and evaluation["paid_campaign_eligible"])
    real_effect_eligible = bool(
        learning["real_agent_task_effect_claim_eligible"]
        and evaluation["real_agent_task_effect_claim_eligible"]
        and usage["usage_complete"]
        and learning["cost_complete"]
        and evaluation["cost_complete"]
    )
    return TrialExecution(
        status=status,
        runtime_state=runtime_state,
        delivery_state=delivery_state,
        verification=verification,
        observed_variant_settings=dict(context.variant.settings),
        metrics={
            "memory.user_recall_calls": (int(learning["user_recall_calls"]) + int(evaluation["user_recall_calls"])),
            "memory.suppressed_user_recall_calls": (
                int(learning["suppressed_user_recall_calls"]) + int(evaluation["suppressed_user_recall_calls"])
            ),
            "memory.hits": int(evaluation["memory_hits"]),
            "memory.backend_class": str(evaluation["backend_class"]),
            "memory.backend_adapter": str(evaluation["backend_adapter"]),
            "memory.everos_semantic_quality_claim_eligible": bool(evaluation["everos_semantic_quality_claim_eligible"]),
            "skill.agent_recall_calls": (int(learning["agent_recall_calls"]) + int(evaluation["agent_recall_calls"])),
            "skill.injected_ids": list(injected_skill_ids),
            "skill.source_contribution": source_contribution,
            "runtime.learning_session_messages": int(learning["session_message_count"]),
            "runtime.evaluation_session_messages": int(evaluation["session_message_count"]),
            "runtime.fresh_process": (int(learning["pid"]) != int(evaluation["pid"])),
            "runtime.backend_quiescent": bool(learning["backend_quiescent"]),
            "runtime.end_to_end_latency_ms": elapsed_ms,
            "runtime.failure_category": evaluation_failure_category,
            "provider.memory_observed": bool(evaluation["provider_memory_observed"]),
            "provider.skill_observed": bool(evaluation["provider_skill_observed"]),
            "provider.kind": str(evaluation["provider_identity"]),
            "paid_campaign_eligible": paid_campaign_eligible,
            "real_agent_task_effect_claim_eligible": real_effect_eligible,
            "cost.complete": bool(learning["cost_complete"] and evaluation["cost_complete"]),
            "cost.estimated_cny": (float(learning["provider_charged_cny"]) + float(evaluation["provider_charged_cny"])),
            "usage.main_agent_input_tokens": usage["input_tokens"],
            "usage.trial_total_input_tokens": usage["input_tokens"],
            "usage.trial_total_output_tokens": usage["output_tokens"],
            "usage.trial_total_tokens": usage["total_tokens"],
            "usage.model_calls": usage["calls"],
            "usage.complete": usage["usage_complete"],
        },
        findings=tuple(findings),
        artifact_refs=(relative_root.as_posix(),),
    )


async def _run_stage(
    *,
    stage: str,
    spec_path: Path,
    isolation: TrialIsolation,
) -> dict[str, Any]:
    result_path = isolation.root / f"{stage}-result.json"
    repository_root = Path(__file__).resolve().parents[4]
    environment = {
        **os.environ,
        **isolation.child_environment(),
        "CODEFLOW_TRACING_DIR": str(isolation.trace_root),
    }
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repository_root)
        if not existing_pythonpath
        else os.pathsep.join((str(repository_root), existing_pythonpath))
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "benchmarks.codeflowbench.packs.memory_skill.worker",
        "--stage",
        stage,
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
        cwd=repository_root,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await _communicate_and_reap(process)
    if not result_path.exists():
        return {
            "status": "infrastructure_failure",
            "error_type": f"worker_exit_{process.returncode}",
        }
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "infrastructure_failure",
            "error_type": "worker_result_corrupt",
        }
    if process.returncode != 0:
        result["status"] = "infrastructure_failure"
    return result


async def _communicate_and_reap(
    process: asyncio.subprocess.Process,
) -> None:
    try:
        await process.communicate()
    except BaseException:
        await _terminate_and_reap(process)
        raise


async def _terminate_and_reap(
    process: asyncio.subprocess.Process,
) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_PROCESS_TERMINATION_GRACE_SECONDS,
        )
    except TimeoutError:
        if process.returncode is None:
            process.kill()
        await process.wait()


def _prepare_workspace(
    workspace: Path,
    task: CrossSessionTask,
) -> None:
    generic = workspace / "skills" / "local-deployment-basics"
    generic.mkdir(parents=True, exist_ok=True)
    (generic / "SKILL.md").write_text(
        "---\n"
        "name: local-deployment-basics\n"
        "description: generic deployment preparation\n"
        "---\n"
        "Generic preparation without task-specific evidence.\n",
        encoding="utf-8",
    )
    if task.required_skill.startswith("local-") or task.required_skill.startswith("cal-local-"):
        required = workspace / "skills" / task.required_skill
        required.mkdir(parents=True, exist_ok=True)
        (required / "SKILL.md").write_text(
            "---\n"
            f"name: {task.required_skill}\n"
            "description: task-specific deployment checklist\n"
            "---\n"
            f"SKILL_EVIDENCE:{task.required_skill} deployment checklist procedure\n",
            encoding="utf-8",
        )


def _verify_result(
    workspace: Path,
    task: CrossSessionTask,
) -> VerifierResult:
    artifact = workspace / "result.json"
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return VerifierResult(
            state=VerificationState.FAILED,
            findings=("result_artifact_missing_or_invalid",),
        )
    expected = {
        "task_id": task.task_id,
        "region": task.expected_value,
        "skill_applied": True,
    }
    if not isinstance(payload, dict):
        return VerifierResult(
            state=VerificationState.FAILED,
            findings=("result_artifact_mismatch",),
            metrics={
                "region_matched": False,
                "skill_applied": False,
            },
        )
    if payload != expected:
        return VerifierResult(
            state=VerificationState.FAILED,
            findings=("result_artifact_mismatch",),
            metrics={
                "region_matched": (payload.get("region") == task.expected_value),
                "skill_applied": payload.get("skill_applied") is True,
            },
        )
    return VerifierResult(
        state=VerificationState.PASSED,
        metrics={
            "region_matched": True,
            "skill_applied": True,
        },
    )


def _integrity_findings(
    *,
    context: TrialContext,
    learning: dict[str, Any],
    evaluation: dict[str, Any],
) -> list[str]:
    findings: list[str] = []
    if int(learning["pid"]) == int(evaluation["pid"]):
        findings.append("evaluation_not_fresh_process")
    if learning["conversation"] == evaluation["conversation"]:
        findings.append("learning_and_evaluation_session_not_distinct")
    if not learning["backend_quiescent"]:
        findings.append("learning_backend_not_quiescent")
    if learning["runtime_state"] != TurnTerminalState.COMPLETED.value:
        findings.append("learning_runtime_not_completed")
    if learning["delivery_state"] != DeliveryOutcome.DELIVERED.value:
        findings.append("learning_delivery_not_completed")
    user_recall = context.variant.settings["user_memory_recall"]
    delegated = int(learning["user_recall_calls"]) + int(evaluation["user_recall_calls"])
    suppressed = int(learning["suppressed_user_recall_calls"]) + int(evaluation["suppressed_user_recall_calls"])
    if user_recall == "disabled" and delegated != 0:
        findings.append("memory_off_delegated_user_recall")
    if user_recall == "disabled" and suppressed == 0:
        findings.append("memory_off_suppression_not_observed")
    if user_recall == "enabled" and delegated == 0:
        findings.append("memory_on_user_recall_not_observed")
    evaluation_runtime = TurnTerminalState(
        str(evaluation["runtime_state"]),
    )
    evaluation_requires_complete_usage = evaluation_runtime in {
        TurnTerminalState.COMPLETED,
        TurnTerminalState.COMPLETED_WITH_TOOL_FAILURE,
    } and (_safe_failure_category(evaluation) != "task_budget_exhausted")
    if not _usage_complete(learning) or (evaluation_requires_complete_usage and not _usage_complete(evaluation)):
        findings.append("trial_usage_incomplete")
    return findings


def _aggregate_usage(
    learning: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, int | bool | None]:
    learning_usage = learning["usage"]
    evaluation_usage = evaluation["usage"]
    return {
        "calls": (int(learning_usage["calls"]) + int(evaluation_usage["calls"])),
        "input_tokens": _sum_optional_usage(
            learning_usage["input_tokens"],
            evaluation_usage["input_tokens"],
        ),
        "output_tokens": _sum_optional_usage(
            learning_usage["output_tokens"],
            evaluation_usage["output_tokens"],
        ),
        "total_tokens": _sum_optional_usage(
            learning_usage["total_tokens"],
            evaluation_usage["total_tokens"],
        ),
        "usage_complete": (_usage_complete(learning) and _usage_complete(evaluation)),
    }


def _usage_complete(stage: dict[str, Any]) -> bool:
    return bool(stage["usage"]["usage_complete"])


def _sum_optional_usage(
    first: Any,
    second: Any,
) -> int | None:
    if first is None or second is None:
        return None
    return int(first) + int(second)


def _infrastructure_failure(
    context: TrialContext,
    isolation: TrialIsolation,
    finding: str,
) -> TrialExecution:
    relative_root = isolation.root.relative_to(
        context.experiment.output_root,
    )
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
        artifact_refs=(relative_root.as_posix(),),
    )


def _learning_runtime_failure(
    context: TrialContext,
    isolation: TrialIsolation,
    learning: dict[str, Any],
    runtime_state: TurnTerminalState,
) -> TrialExecution:
    failure_category = _safe_failure_category(learning)
    if failure_category == "task_budget_exhausted":
        status = TrialStatus.TASK_TIMEOUT
    elif runtime_state is TurnTerminalState.PROVIDER_FAILED:
        status = TrialStatus.PROVIDER_FAILURE
    elif runtime_state is TurnTerminalState.CANCELLED:
        status = TrialStatus.CANCELLED
    else:
        status = TrialStatus.TASK_FAILED
    finding = (
        f"learning_runtime_short_circuit:{failure_category}"
        if failure_category is not None
        else f"learning_runtime_short_circuit:{runtime_state.value}"
    )
    usage = learning["usage"]
    relative_root = isolation.root.relative_to(
        context.experiment.output_root,
    )
    return TrialExecution(
        status=status,
        runtime_state=runtime_state,
        delivery_state=DeliveryOutcome(
            str(learning["delivery_state"]),
        ),
        verification=VerifierResult(
            state=VerificationState.NOT_RUN,
            findings=(finding,),
        ),
        observed_variant_settings=dict(context.variant.settings),
        metrics={
            "runtime.learning_session_messages": int(
                learning["session_message_count"],
            ),
            "runtime.evaluation_skipped": True,
            "runtime.failure_category": (failure_category if failure_category is not None else runtime_state.value),
            "provider.kind": str(learning["provider_identity"]),
            "paid_campaign_eligible": bool(
                learning["paid_campaign_eligible"],
            ),
            "real_agent_task_effect_claim_eligible": False,
            "cost.complete": bool(learning["cost_complete"]),
            "cost.estimated_cny": float(
                learning["provider_charged_cny"],
            ),
            "usage.main_agent_input_tokens": _optional_int(
                usage["input_tokens"],
            ),
            "usage.trial_total_input_tokens": _optional_int(
                usage["input_tokens"],
            ),
            "usage.trial_total_output_tokens": _optional_int(
                usage["output_tokens"],
            ),
            "usage.trial_total_tokens": _optional_int(
                usage["total_tokens"],
            ),
            "usage.model_calls": int(usage["calls"]),
            "usage.complete": bool(usage["usage_complete"]),
        },
        findings=(finding,),
        artifact_refs=(relative_root.as_posix(),),
    )


def _safe_failure_category(
    stage: dict[str, Any],
) -> str | None:
    category = stage.get("failure_category")
    if not isinstance(category, str):
        return None
    normalized = category.strip().lower()
    if not normalized or len(normalized) > 64:
        return None
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in normalized):
        return None
    return normalized


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _preparation_failure(
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
