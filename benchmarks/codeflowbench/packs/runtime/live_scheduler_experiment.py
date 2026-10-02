from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from loguru import logger

from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import GenerationSettings, LLMProvider
from codeflow.spine import ChatType, Origin, OriginPools, Scheduler, Source, Text, TurnOutcome, TurnRequest

from ...artifacts import ArtifactStore
from ...budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from ...canonical import canonical_digest, to_primitive
from ...environment import capture_environment_identity
from ...schema import ExperimentRef
from .models import LatencySummary
from .scheduler_experiments import _GlobalFifoScheduler, _noop_emit

LIVE_SCHEDULER_SCHEMA = "codeflow.codeflowbench.runtime-live-scheduler.v2"
LIVE_TURN_RECORD_SCHEMA = "codeflow.codeflowbench.runtime-live-turn-record.v1"
_DISABLED_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "grep",
    "find",
    "exec",
    "web_search",
    "web_fetch",
    "message",
    "spawn",
    "ask_user",
    "cron",
]


@dataclass(frozen=True)
class LiveSchedulerConfig:
    repetitions: int = 80
    user_slots: int = 4
    system_slots: int = 1
    hot_turns: int = 2
    foreground_sessions: int = 24
    max_agent_iterations: int = 3
    max_output_tokens_per_call: int = 512
    max_input_tokens_per_call: int = 8_000
    max_logical_calls_per_turn: int = 4
    max_attempts_per_call: int = 2
    timeout_seconds_per_arm: int = 180
    hard_cap_cny: float = 320.0
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_806
    confidence_level: float = 0.95
    input_cache_miss_usd_per_million: float = 0.14
    output_usd_per_million: float = 0.28
    conservative_usd_to_cny_multiplier: float = 7.5

    def __post_init__(self) -> None:
        counts = (
            self.repetitions,
            self.user_slots,
            self.system_slots,
            self.hot_turns,
            self.foreground_sessions,
            self.max_agent_iterations,
            self.max_output_tokens_per_call,
            self.max_input_tokens_per_call,
            self.max_logical_calls_per_turn,
            self.max_attempts_per_call,
            self.timeout_seconds_per_arm,
            self.bootstrap_resamples,
        )
        if any(value < 1 for value in counts):
            raise ValueError("live scheduler experiment limits must be positive")
        if self.max_logical_calls_per_turn < self.max_agent_iterations + 1:
            raise ValueError("Provider call budget must include the exhaustion synthesis call")
        if self.repetitions % 2:
            raise ValueError("live scheduler repetitions must be even for balanced arm order")
        if not 0 < self.confidence_level < 1:
            raise ValueError("live scheduler confidence level must be between zero and one")
        if self.hard_cap_cny <= 0:
            raise ValueError("live scheduler experiment hard cap must be positive")
        if self.maximum_cost_cny > self.hard_cap_cny:
            raise ValueError("live scheduler experiment worst-case cost exceeds the hard cap")

    @property
    def turns_per_arm(self) -> int:
        return self.hot_turns + self.foreground_sessions

    @property
    def planned_turns(self) -> int:
        return self.repetitions * 2 * self.turns_per_arm

    @property
    def maximum_provider_request_attempts(self) -> int:
        return self.planned_turns * self.max_logical_calls_per_turn * self.max_attempts_per_call

    @property
    def maximum_cost_cny(self) -> float:
        usd_per_attempt = (
            self.max_input_tokens_per_call / 1_000_000 * self.input_cache_miss_usd_per_million
            + self.max_output_tokens_per_call / 1_000_000 * self.output_usd_per_million
        )
        return usd_per_attempt * self.conservative_usd_to_cny_multiplier * self.maximum_provider_request_attempts


@dataclass(frozen=True)
class _LiveWork:
    request: TurnRequest
    marker: str
    measured: bool


class _LiveRecorder:
    def __init__(self, work: list[_LiveWork]) -> None:
        self.work = {_message_id(item.request): item for item in work}
        self.submission_order = {_message_id(item.request): index for index, item in enumerate(work)}
        self.accepted_ns: dict[str, int] = {}
        self.started_ns: dict[str, int] = {}
        self.terminal_ns: dict[str, int] = {}
        self.end_to_end_ms: list[float] = []
        self.invocations: Counter[str] = Counter()
        self.task_failures: dict[str, str] = {}
        self.outcomes: dict[str, dict[str, Any]] = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def accept(self, request: TurnRequest) -> None:
        self.accepted_ns[_message_id(request)] = time.perf_counter_ns()

    def start(self, request: TurnRequest) -> None:
        self.started_ns[_message_id(request)] = time.perf_counter_ns()

    def fail(self, request: TurnRequest, reason: str) -> None:
        message_id = _message_id(request)
        self.terminal_ns[message_id] = time.perf_counter_ns()
        self.task_failures[message_id] = reason

    def complete(
        self,
        request: TurnRequest,
        outcome: TurnOutcome,
        response_text: str,
    ) -> None:
        message_id = _message_id(request)
        item = self.work[message_id]
        self.terminal_ns[message_id] = time.perf_counter_ns()
        if item.measured:
            self.end_to_end_ms.append((self.terminal_ns[message_id] - self.accepted_ns[message_id]) / 1_000_000)
        self.prompt_tokens += outcome.usage.prompt_tokens
        self.completion_tokens += outcome.usage.completion_tokens
        self.total_tokens += outcome.usage.total_tokens
        failures = []
        if item.marker not in response_text:
            failures.append("marker_missing")
        if outcome.usage.total_tokens <= 0:
            failures.append("usage_missing")
        if not outcome.explicit_reply:
            failures.append("reply_missing")
        if outcome.tool_calls or outcome.tool_failures:
            failures.append("unexpected_tool_activity")
        if failures:
            self.task_failures[message_id] = ",".join(failures)
        self.outcomes[message_id] = {
            "completion_tokens": outcome.usage.completion_tokens,
            "context_fallback_reason": outcome.context_fallback_reason,
            "context_path": outcome.context_path,
            "explicit_reply": outcome.explicit_reply,
            "memory_hits": outcome.memory_hits,
            "prompt_tokens": outcome.usage.prompt_tokens,
            "tool_calls": outcome.tool_calls,
            "tool_failures": outcome.tool_failures,
            "total_tokens": outcome.usage.total_tokens,
        }

    def turn_records(self) -> list[dict[str, Any]]:
        origin_ns = min(self.accepted_ns.values()) if self.accepted_ns else 0
        records = []
        for message_id, item in sorted(
            self.work.items(),
            key=lambda entry: self.submission_order[entry[0]],
        ):
            accepted_ns = self.accepted_ns.get(message_id)
            started_ns = self.started_ns.get(message_id)
            terminal_ns = self.terminal_ns.get(message_id)
            outcome = self.outcomes.get(message_id, {})
            failures = self.task_failures.get(message_id)
            records.append(
                {
                    "schema": LIVE_TURN_RECORD_SCHEMA,
                    "submission_index": self.submission_order[message_id],
                    "message_id": message_id,
                    "conversation": item.request.conversation,
                    "measured": item.measured,
                    "metric_eligible": item.measured and message_id in self.outcomes,
                    "accepted_offset_ms": _offset_ms(accepted_ns, origin_ns),
                    "started_offset_ms": _offset_ms(started_ns, origin_ns),
                    "terminal_offset_ms": _offset_ms(terminal_ns, origin_ns),
                    "queue_wait_ms": _duration_ms(accepted_ns, started_ns),
                    "execution_ms": _duration_ms(started_ns, terminal_ns),
                    "end_to_end_ms": _duration_ms(accepted_ns, terminal_ns),
                    "invocation_count": self.invocations[message_id],
                    "verifier_failures": failures.split(",") if failures else [],
                    **outcome,
                }
            )
        return records


class _LiveMeasuredRunner:
    def __init__(
        self,
        agent_loop: Any,
        recorder: _LiveRecorder,
        config: LiveSchedulerConfig,
        arm: str,
    ) -> None:
        self._delegate = AgentTurnRunner(agent_loop, stream=False)
        self._recorder = recorder
        self._config = config
        self._arm = arm

    async def run(self, req, emit, drain) -> TurnOutcome:
        message_id = _message_id(req)
        self._recorder.invocations[message_id] += 1
        self._recorder.start(req)
        response_parts: list[str] = []

        async def capture(event: object) -> None:
            if isinstance(event, Text):
                response_parts.append(event.content)
            await emit(event)

        try:
            with provider_call_budget_scope(
                trial_id=f"{self._arm}:{message_id}",
                max_logical_calls=self._config.max_logical_calls_per_turn,
                max_attempts_per_call=self._config.max_attempts_per_call,
                max_input_tokens_per_call=self._config.max_input_tokens_per_call,
                max_output_tokens_per_call=self._config.max_output_tokens_per_call,
            ):
                outcome = await self._delegate.run(req, capture, drain)
        except Exception as exc:
            self._recorder.fail(req, f"runner_exception:{type(exc).__name__}")
            raise
        self._recorder.complete(req, outcome, "\n".join(response_parts))
        return outcome


class _ObservedLiveProvider:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self.logical_calls = 0
        self.failure_categories: Counter[str] = Counter()
        self.actual_models: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def chat_with_retry(self, **kwargs: Any):
        self.logical_calls += 1
        try:
            response = await self._provider.chat_with_retry(**kwargs)
        except Exception as exc:
            classification = self._provider.classify_error(exc)
            self.failure_categories[classification.category] += 1
            raise
        if response.model:
            self.actual_models.add(response.model)
        if response.finish_reason == "error" or response.error_classification is not None:
            category = (
                response.error_classification.category if response.error_classification is not None else "unknown"
            )
            self.failure_categories[category] += 1
        return response


def build_live_scheduler_plan(
    config: LiveSchedulerConfig,
    *,
    repository_root: Path,
    provider_name: str,
    model: str,
) -> dict[str, Any]:
    return {
        "schema": LIVE_SCHEDULER_SCHEMA,
        "source_commit": _git(repository_root, "rev-parse", "HEAD"),
        "environment": capture_environment_identity(repository_root),
        "provider": {
            "name": provider_name,
            "model": model,
            "allow_fallback": False,
        },
        "workload": {
            "control": "strict_global_fifo_with_session_serialization",
            "treatment": "codeflow_per_session_lanes",
            "scenario": "adversarial_head_of_line_burst",
            "task_verifier": "unique_marker_and_complete_provider_usage",
            "primary_metric": "foreground_accept_to_terminal_p95_ms",
            "aggregation": "median_of_paired_repetition_reductions",
            "arm_order": "balanced_alternating",
            "raw_record_schema": LIVE_TURN_RECORD_SCHEMA,
        },
        "analysis": {
            "confidence_interval": "paired_repetition_bootstrap_percentile",
            "confidence_level": config.confidence_level,
            "resamples": config.bootstrap_resamples,
            "seed": config.bootstrap_seed,
        },
        "config": to_primitive(config),
        "budget": {
            "planned_turns": config.planned_turns,
            "maximum_provider_request_attempts": config.maximum_provider_request_attempts,
            "maximum_cost_cny": config.maximum_cost_cny,
            "hard_cap_cny": config.hard_cap_cny,
            "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        },
        "claim_gates": {
            "all_tasks_verified": True,
            "all_usage_complete": True,
            "provider_failures_equal_zero": True,
            "actual_model_verified": True,
            "foreground_p95_reduction_percent_greater_than": 0,
            "paired_reduction_ci_lower_bound_greater_than": 0,
            "clean_stable_checkout": True,
        },
    }


def write_live_scheduler_plan(
    config: LiveSchedulerConfig,
    *,
    repository_root: Path,
    output_root: Path,
    provider_name: str,
    model: str,
) -> tuple[dict[str, Any], Path]:
    if not _worktree_clean(repository_root):
        raise RuntimeError("live scheduler plan requires a clean worktree")
    plan = build_live_scheduler_plan(
        config,
        repository_root=repository_root,
        provider_name=provider_name,
        model=model,
    )
    plan_digest = canonical_digest(plan)
    root = output_root / plan["source_commit"] / plan_digest
    store = ArtifactStore(ExperimentRef(experiment_id=plan_digest, root=root))
    store.freeze_manifest(plan)
    return plan, store.manifest_path


async def run_live_scheduler_experiment(
    config: LiveSchedulerConfig,
    *,
    repository_root: Path,
    output_root: Path,
    base_config: Config,
    delegate_provider: LLMProvider,
    approval_digest: str,
    approved_cny: float,
) -> tuple[dict[str, Any], Path]:
    provider_name = base_config.get_provider_name(base_config.agents.defaults.model)
    model = base_config.agents.defaults.model
    plan, _manifest_path = write_live_scheduler_plan(
        config,
        repository_root=repository_root,
        output_root=output_root,
        provider_name=provider_name,
        model=model,
    )
    plan_digest = canonical_digest(plan)
    if approval_digest != plan_digest:
        raise RuntimeError("live scheduler approval digest does not match the frozen plan")
    if not math.isclose(approved_cny, config.hard_cap_cny, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError("live scheduler approved CNY amount does not match the hard cap")

    root = output_root / plan["source_commit"] / plan_digest
    store = ArtifactStore(ExperimentRef(experiment_id=plan_digest, root=root))
    ledger_path = root / "provider-budget.jsonl"
    with store.exclusive_run_lock():
        if ledger_path.exists() or ledger_path.with_suffix(".high-water.json").exists():
            raise RuntimeError("live scheduler plan already has a Provider budget ledger")
        bootstrap_config = _budget_config(config)
        bootstrap_ledger = ProviderBudgetLedger(ledger_path, bootstrap_config)
        prefix = bootstrap_ledger.snapshot()
        approved_budget = ProviderBudgetConfig(
            **{
                **to_primitive(bootstrap_config),
                "approval_digest": approval_digest,
                "ledger_prefix_event_count": prefix.ledger_event_count,
                "ledger_prefix_digest": prefix.ledger_digest,
                "ledger_prefix_charged_cny": prefix.provider_charged_cny,
            }
        )
        ledger = ProviderBudgetLedger(ledger_path, approved_budget)
    delegate_provider.generation = GenerationSettings(
        temperature=0,
        max_tokens=config.max_output_tokens_per_call,
    )
    provider = BudgetGuardedProvider(delegate_provider, ledger=ledger)
    observed_provider = _ObservedLiveProvider(provider)

    clean_before = _worktree_clean(repository_root)
    environment_before = capture_environment_identity(repository_root)
    source_commit = _git(repository_root, "rev-parse", "HEAD")
    pairs: list[dict[str, Any]] = []
    try:
        for repetition in range(config.repetitions):
            arm_order = ("global_fifo", "session_lanes") if repetition % 2 == 0 else ("session_lanes", "global_fifo")
            arms: dict[str, dict[str, Any]] = {}
            for arm in arm_order:
                work = _live_work(config, repetition)
                arms[arm] = await _run_live_arm(
                    config,
                    work=work,
                    scheduler_kind=arm,
                    base_config=base_config,
                    provider=observed_provider,
                    workspace=root / "state" / f"r{repetition:02d}" / arm,
                )
            control_p95 = _optional_p95(arms["global_fifo"])
            treatment_p95 = _optional_p95(arms["session_lanes"])
            paired_reduction = (
                (control_p95 - treatment_p95) / control_p95 * 100
                if control_p95 is not None and treatment_p95 is not None
                else None
            )
            pairs.append(
                {
                    "repetition": repetition,
                    "arm_order": list(arm_order),
                    "global_fifo": arms["global_fifo"],
                    "session_lanes": arms["session_lanes"],
                    "p95_reduction_percent": paired_reduction,
                }
            )
    finally:
        await _close_provider(delegate_provider)

    budget = ledger.snapshot()
    clean_after = _worktree_clean(repository_root)
    environment_after = capture_environment_identity(repository_root)
    final_commit = _git(repository_root, "rev-parse", "HEAD")
    all_tasks_verified = all(
        arm["task_failures"] == 0
        and arm["runner_exceptions"] == 0
        and arm["unexpected_duplicate_executions"] == 0
        and arm["missing_executions"] == 0
        for pair in pairs
        for arm in (pair["global_fifo"], pair["session_lanes"])
    )
    summary = _analyze_live_pairs(
        pairs,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    )
    reduction = summary["foreground_p95_reduction_percent"]
    confidence_interval = summary["paired_reduction_ci"]
    gates = {
        "all_tasks_verified": all_tasks_verified,
        "all_usage_complete": all(
            arm["total_tokens"] > 0 for pair in pairs for arm in (pair["global_fifo"], pair["session_lanes"])
        ),
        "provider_failures_equal_zero": not observed_provider.failure_categories,
        "actual_model_verified": bool(observed_provider.actual_models)
        and observed_provider.actual_models <= {model, model.split("/", 1)[-1]},
        "performance_direction": isinstance(reduction, int | float) and reduction > 0,
        "paired_reduction_ci_lower_bound_positive": (
            isinstance(confidence_interval["lower_percent"], int | float) and confidence_interval["lower_percent"] > 0
        ),
        "budget_accounting_complete": budget.accounting_complete and budget.open_reservations == 0,
        "clean_stable_checkout": (
            clean_before and clean_after and source_commit == final_commit and environment_before == environment_after
        ),
    }
    evidence_payload = {
        "schema": LIVE_SCHEDULER_SCHEMA,
        "plan_digest": plan_digest,
        "source_commit": source_commit,
        "environment": environment_before,
        "provider": {
            **plan["provider"],
            "actual_models": sorted(observed_provider.actual_models),
            "logical_calls": observed_provider.logical_calls,
            "failure_categories": dict(observed_provider.failure_categories),
        },
        "analysis": plan["analysis"],
        "evidence_scope": "live_provider_agent_end_to_end_scheduler_comparison",
        "repetitions": pairs,
        "summary": summary,
        "budget": to_primitive(budget),
        "gates": gates,
        "claim_eligible": all(gates.values()),
    }
    evidence_digest = canonical_digest(evidence_payload)
    evidence = {**evidence_payload, "evidence_digest": evidence_digest}
    evidence_path = root / "runs" / f"live-scheduler.{evidence_digest}.json"
    store.append_immutable(
        evidence_path,
        evidence,
    )
    return evidence, evidence_path


async def _run_live_arm(
    config: LiveSchedulerConfig,
    *,
    work: list[_LiveWork],
    scheduler_kind: str,
    base_config: Config,
    provider: LLMProvider,
    workspace: Path,
) -> dict[str, Any]:
    from codeflow.cli._runtime_assembly import assemble_runtime

    runtime_config = base_config.model_copy(deep=True)
    runtime_config.agents.defaults.workspace = str(workspace)
    runtime_config.agents.defaults.temperature = 0
    runtime_config.agents.defaults.max_tokens = config.max_output_tokens_per_call
    runtime_config.agents.defaults.max_tool_iterations = config.max_agent_iterations
    runtime_config.agents.defaults.enable_personalization = False
    runtime_config.tools.disabled_tools = list(_DISABLED_TOOLS)
    runtime_config.tools.mcp_servers = {}
    runtime_config.tools.sandbox.backend = "none"
    codeflow_config = CodeFlowConfig(base=runtime_config)
    codeflow_config.memory.backend = None
    codeflow_config.plugins.disabled = []
    codeflow_config.skill_forge.enabled = False
    codeflow_config.skill_forge.router.enabled = False
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.token_wise.enabled = False
    codeflow_config.context.fast_path_threshold = 1.0
    codeflow_config.runtime.checkpoint.policy = "never"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = assemble_runtime(
        runtime_config,
        codeflow_config,
        provider=provider,
        cron_service=None,
        interactive=False,
    )
    recorder = _LiveRecorder(work)
    runner = _LiveMeasuredRunner(runtime.agent_loop, recorder, config, scheduler_kind)
    if scheduler_kind == "global_fifo":
        scheduler: Any = _GlobalFifoScheduler(runner, config.user_slots)
    elif scheduler_kind == "session_lanes":
        scheduler = Scheduler(
            runner,
            OriginPools(user=config.user_slots, system=config.system_slots),
            _noop_emit,
        )
    else:
        raise ValueError(f"unknown live scheduler arm: {scheduler_kind}")

    handles = []
    started_ns = time.perf_counter_ns()
    try:
        for item in work:
            recorder.accept(item.request)
            handles.append(scheduler.submit(item.request))
        results = await asyncio.wait_for(
            asyncio.gather(*(handle.result() for handle in handles), return_exceptions=True),
            timeout=config.timeout_seconds_per_arm,
        )
    finally:
        if isinstance(scheduler, _GlobalFifoScheduler):
            await scheduler.close()
        else:
            await scheduler.shutdown(grace=0)
        await runtime.close()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    duplicate_executions = sum(max(0, count - 1) for count in recorder.invocations.values())
    runner_exceptions = sum(isinstance(result, BaseException) or result is None for result in results)
    return {
        "accepted_requests": len(work),
        "measured_foreground_requests": sum(item.measured for item in work),
        "elapsed_ms": elapsed_ms,
        "end_to_end_ms": to_primitive(LatencySummary.from_values(recorder.end_to_end_ms)),
        "task_failures": len(recorder.task_failures),
        "task_failure_categories": dict(Counter(recorder.task_failures.values())),
        "task_failure_details": dict(recorder.task_failures),
        "runner_exceptions": runner_exceptions,
        "unexpected_duplicate_executions": duplicate_executions,
        "missing_executions": len(work) - len(recorder.invocations),
        "prompt_tokens": recorder.prompt_tokens,
        "completion_tokens": recorder.completion_tokens,
        "total_tokens": recorder.total_tokens,
        "turn_records": recorder.turn_records(),
    }


def _live_work(config: LiveSchedulerConfig, repetition: int) -> list[_LiveWork]:
    work = []
    for index in range(config.hot_turns):
        marker = f"CODEFLOW_LIVE_PERF_R{repetition:02d}_HOT_{index:02d}"
        prompt = f"Reply with exactly {marker} and nothing else. Do not call tools."
        work.append(_make_work(marker, "hot", prompt, measured=False))
    for index in range(config.foreground_sessions):
        marker = f"CODEFLOW_LIVE_PERF_R{repetition:02d}_FG_{index:02d}"
        prompt = f"Reply with exactly {marker} and nothing else. Do not call tools."
        work.append(_make_work(marker, f"foreground-{index:02d}", prompt, measured=True))
    return work


def _make_work(marker: str, conversation: str, prompt: str, *, measured: bool) -> _LiveWork:
    return _LiveWork(
        request=TurnRequest(
            origin=Origin.USER,
            source=Source(
                channel="codeflowbench-live-scheduler",
                chat_id=conversation,
                sender_id="codeflowbench",
                chat_type=ChatType.DM,
            ),
            text=prompt,
            message_id=marker,
            conversation=f"codeflowbench-live-scheduler:{conversation}",
        ),
        marker=marker,
        measured=measured,
    )


def _budget_config(config: LiveSchedulerConfig) -> ProviderBudgetConfig:
    return ProviderBudgetConfig(
        hard_cap_cny=config.hard_cap_cny,
        external_service_reserve_cny=0,
        max_total_request_attempts=config.maximum_provider_request_attempts,
        max_input_tokens_per_call=config.max_input_tokens_per_call,
        max_output_tokens_per_call=config.max_output_tokens_per_call,
        input_cache_miss_usd_per_million=config.input_cache_miss_usd_per_million,
        output_usd_per_million=config.output_usd_per_million,
        conservative_usd_to_cny_multiplier=config.conservative_usd_to_cny_multiplier,
        max_additional_request_attempts=(
            config.planned_turns * config.max_logical_calls_per_turn * (config.max_attempts_per_call - 1)
        ),
    )


def _optional_p95(arm: dict[str, Any]) -> float | None:
    value = arm["end_to_end_ms"]["p95"]
    return float(value) if isinstance(value, int | float) and value > 0 else None


def _median_p95(pairs: list[dict[str, Any]], arm: str) -> float | None:
    values = [_optional_p95(pair[arm]) for pair in pairs]
    return (
        median(value for value in values if value is not None) if all(value is not None for value in values) else None
    )


def _analyze_live_pairs(
    pairs: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    reductions = [
        pair["p95_reduction_percent"] for pair in pairs if isinstance(pair["p95_reduction_percent"], int | float)
    ]
    reduction = median(reductions) if len(reductions) == len(pairs) and reductions else None
    confidence_interval = _bootstrap_median_interval(
        reductions,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return {
        "foreground_p95_reduction_percent": reduction,
        "paired_reduction_ci": confidence_interval,
        "control_p95_ms": _median_p95(pairs, "global_fifo"),
        "treatment_p95_ms": _median_p95(pairs, "session_lanes"),
        "accepted_turns": sum(
            arm["accepted_requests"] for pair in pairs for arm in (pair["global_fifo"], pair["session_lanes"])
        ),
        "verified_turns": sum(
            arm["accepted_requests"] - arm["task_failures"]
            for pair in pairs
            for arm in (pair["global_fifo"], pair["session_lanes"])
        ),
    }


def _bootstrap_median_interval(
    values: list[float],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not values:
        return {
            "method": "paired_repetition_bootstrap_percentile",
            "confidence_level": confidence_level,
            "resamples": resamples,
            "seed": seed,
            "lower_percent": None,
            "upper_percent": None,
        }
    generator = random.Random(seed)
    estimates = sorted(median(generator.choices(values, k=len(values))) for _ in range(resamples))
    tail = (1 - confidence_level) / 2
    return {
        "method": "paired_repetition_bootstrap_percentile",
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "lower_percent": _nearest_rank(estimates, tail),
        "upper_percent": _nearest_rank(estimates, 1 - tail),
    }


def verify_live_scheduler_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = dict(evidence)
    evidence_digest = payload.pop("evidence_digest", None)
    analysis = evidence.get("analysis", {})
    pairs = evidence.get("repetitions", [])
    raw_record_gates = []
    raw_aggregate_gates = []
    reduction_gates = []
    for pair in pairs:
        for arm_name in ("global_fifo", "session_lanes"):
            arm = pair[arm_name]
            records = arm.get("turn_records", [])
            measured = [
                record["end_to_end_ms"]
                for record in records
                if record.get("metric_eligible") and isinstance(record.get("end_to_end_ms"), int | float)
            ]
            submission_indices = [record.get("submission_index") for record in records]
            message_ids = [record.get("message_id") for record in records]
            raw_record_gates.append(
                len(records) == arm["accepted_requests"]
                and sum(bool(record.get("measured")) for record in records) == arm["measured_foreground_requests"]
                and submission_indices == list(range(len(records)))
                and len(set(message_ids)) == len(message_ids)
                and all(record.get("schema") == LIVE_TURN_RECORD_SCHEMA for record in records)
                and to_primitive(LatencySummary.from_values(measured)) == arm["end_to_end_ms"]
            )
            raw_aggregate_gates.append(
                sum(int(record.get("prompt_tokens", 0)) for record in records) == arm["prompt_tokens"]
                and sum(int(record.get("completion_tokens", 0)) for record in records) == arm["completion_tokens"]
                and sum(int(record.get("total_tokens", 0)) for record in records) == arm["total_tokens"]
                and sum(bool(record.get("verifier_failures")) for record in records) == arm["task_failures"]
                and sum(max(0, int(record.get("invocation_count", 0)) - 1) for record in records)
                == arm["unexpected_duplicate_executions"]
                and sum(int(record.get("invocation_count", 0)) == 0 for record in records) == arm["missing_executions"]
            )
        control_p95 = _optional_p95(pair["global_fifo"])
        treatment_p95 = _optional_p95(pair["session_lanes"])
        recomputed_reduction = (
            (control_p95 - treatment_p95) / control_p95 * 100
            if control_p95 is not None and treatment_p95 is not None
            else None
        )
        reduction_gates.append(_optional_float_equal(recomputed_reduction, pair["p95_reduction_percent"]))
    recomputed_summary = _analyze_live_pairs(
        pairs,
        bootstrap_resamples=int(analysis.get("resamples", 0)),
        bootstrap_seed=int(analysis.get("seed", 0)),
        confidence_level=float(analysis.get("confidence_level", 0)),
    )
    gates = {
        "evidence_digest_matches": isinstance(evidence_digest, str) and canonical_digest(payload) == evidence_digest,
        "raw_records_complete": bool(raw_record_gates) and all(raw_record_gates),
        "raw_aggregates_reproduce": bool(raw_aggregate_gates) and all(raw_aggregate_gates),
        "paired_reductions_reproduce": bool(reduction_gates) and all(reduction_gates),
        "summary_reproduces": recomputed_summary == evidence.get("summary"),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "recomputed_summary": recomputed_summary,
    }


def _nearest_rank(ordered: list[float], quantile: float) -> float:
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _optional_float_equal(left: float | None, right: object) -> bool:
    if left is None:
        return right is None
    return isinstance(right, int | float) and math.isclose(left, float(right), rel_tol=0, abs_tol=1e-9)


def _offset_ms(value_ns: int | None, origin_ns: int) -> float | None:
    return (value_ns - origin_ns) / 1_000_000 if value_ns is not None else None


def _duration_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    return (end_ns - start_ns) / 1_000_000 if start_ns is not None and end_ns is not None else None


async def _close_provider(provider: LLMProvider) -> None:
    client = getattr(provider, "_client", None)
    close = getattr(client, "close", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result


def _message_id(request: TurnRequest) -> str:
    if request.message_id is None:
        raise ValueError("live scheduler request requires message_id")
    return request.message_id


def _git(repository_root: Path, *args: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to bind live scheduler evidence")
    completed = subprocess.run(
        [git, *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _worktree_clean(repository_root: Path) -> bool:
    return not _git(repository_root, "status", "--short")


def _load_subject() -> tuple[Config, str, str]:
    from codeflow.config.loader import load_config

    config = load_config()
    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    provider_config = config.get_provider(model)
    if provider_config is None or not provider_config.api_key:
        raise RuntimeError("configured live Provider credential is unavailable")
    if provider_name != "deepseek" or model != "deepseek/deepseek-v4-flash":
        raise RuntimeError("live scheduler plan is frozen to deepseek/deepseek-v4-flash")
    return config, provider_name, model


def _format_metric(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, int | float) else "unavailable"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan, run, or verify the live Agent scheduler experiment")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".codeflow/evidence/codeflowbench-runtime-live-scheduler"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run = subparsers.add_parser("run")
    run.add_argument("--approval-digest", required=True)
    run.add_argument("--approved-cny", required=True, type=float)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--evidence", required=True, type=Path)
    return parser.parse_args()


async def _main() -> int:
    from codeflow.cli._helpers import make_provider

    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[4]
    if args.command == "verify":
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = verify_live_scheduler_evidence(evidence)
        summary = result["recomputed_summary"]
        print(f"evidence: {args.evidence}")
        print(f"verified: {str(result['passed']).lower()}")
        print(f"foreground_p95_reduction_percent: {_format_metric(summary['foreground_p95_reduction_percent'])}")
        interval = summary["paired_reduction_ci"]
        print(f"paired_reduction_ci_lower_percent: {_format_metric(interval['lower_percent'])}")
        print(f"paired_reduction_ci_upper_percent: {_format_metric(interval['upper_percent'])}")
        return 0 if result["passed"] else 1
    config = LiveSchedulerConfig()
    base_config, provider_name, model = _load_subject()
    plan, path = write_live_scheduler_plan(
        config,
        repository_root=repository_root,
        output_root=args.output_root,
        provider_name=provider_name,
        model=model,
    )
    plan_digest = canonical_digest(plan)
    if args.command == "plan":
        print(f"manifest: {path}")
        print(f"approval_digest: {plan_digest}")
        print(f"provider: {provider_name}")
        print(f"model: {model}")
        print(f"planned_turns: {config.planned_turns}")
        print(f"maximum_provider_request_attempts: {config.maximum_provider_request_attempts}")
        print(f"maximum_cost_cny: {config.maximum_cost_cny:.6f}")
        print(f"hard_cap_cny: {config.hard_cap_cny:.2f}")
        return 0

    logger.disable("codeflow")
    try:
        evidence, evidence_path = await run_live_scheduler_experiment(
            config,
            repository_root=repository_root,
            output_root=args.output_root,
            base_config=base_config,
            delegate_provider=make_provider(base_config),
            approval_digest=args.approval_digest,
            approved_cny=args.approved_cny,
        )
    finally:
        logger.enable("codeflow")
    summary = evidence["summary"]
    print(f"evidence: {evidence_path}")
    print(f"claim_eligible: {str(evidence['claim_eligible']).lower()}")
    print(f"verified_turns: {summary['verified_turns']}")
    print(f"control_p95_ms: {_format_metric(summary['control_p95_ms'])}")
    print(f"treatment_p95_ms: {_format_metric(summary['treatment_p95_ms'])}")
    print(f"foreground_p95_reduction_percent: {_format_metric(summary['foreground_p95_reduction_percent'])}")
    interval = summary["paired_reduction_ci"]
    print(f"paired_reduction_ci_lower_percent: {_format_metric(interval['lower_percent'])}")
    print(f"paired_reduction_ci_upper_percent: {_format_metric(interval['upper_percent'])}")
    print(f"provider_charged_cny: {evidence['budget']['provider_charged_cny']:.6f}")
    return 0 if evidence["claim_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
