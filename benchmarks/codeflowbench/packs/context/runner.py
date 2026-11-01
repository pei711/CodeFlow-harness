from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmarks.codeflowbench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.codeflowbench.isolation import TrialIsolation
from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    TrialStatus,
    TurnTerminalState,
    VerificationState,
)
from benchmarks.codeflowbench.usage import (
    ModelCallUsage,
    RecordingProvider,
    UsageRecorder,
    usage_scope,
)
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import (
    ErrorClassification,
    GenerationSettings,
    LLMProvider,
    LLMResponse,
)
from codeflow.spine import ChatType, Origin, Source, TurnRequest
from codeflow.spine.events import TurnFailed
from codeflow.utils.helpers import estimate_message_tokens

from .models import ContextTask
from .verifier import SealedContextTaskVerifier

_DISABLED_CONTEXT_TOOLS = [
    "ask_user",
    "edit_file",
    "exec",
    "find",
    "grep",
    "list_dir",
    "message",
    "skill_read",
    "spawn",
    "understand_media",
    "web_fetch",
    "web_search",
]
_CONTEXT_TOOL_NAMES = frozenset({"read_file", "write_file"})
CONTEXT_BENCHMARK_WINDOW_TOKENS = 8_192
CONTEXT_BENCHMARK_OUTPUT_TOKENS = 500
CONTEXT_BENCHMARK_MAX_TOOL_ITERATIONS = 6
CONTEXT_BENCHMARK_PROTECT_FIRST_N = 1
_TASK_EXECUTION_NOTE = (
    "All values required for the artifact are in the conversation history. "
    "The target artifact does not exist yet. Write it directly, then read it "
    "back. The workspace has no additional source of task facts. Do not "
    "create exploration or placeholder files."
)


class _CapturingProvider(RecordingProvider):
    def __init__(self, delegate: Any, *, recorder: UsageRecorder) -> None:
        super().__init__(delegate, recorder=recorder)
        self.message_windows: list[list[dict[str, Any]]] = []
        self.tool_invocations: list[tuple[str, str]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.message_windows.append([dict(message) for message in messages])
        response = await super().chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        for call in response.tool_calls or []:
            name = str(call.name)
            if name in _CONTEXT_TOOL_NAMES:
                arguments = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
                self.tool_invocations.append((name, arguments))
        return response


class _RoleScopedProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(self, delegate: LLMProvider, role: str, *, max_output_tokens: int | None = None) -> None:
        super().__init__(
            api_key=getattr(delegate, "api_key", None),
            api_base=getattr(delegate, "api_base", None),
        )
        self._delegate = delegate
        self._role = role
        self.generation = delegate.generation
        self._max_output_tokens = max_output_tokens or self.generation.max_tokens
        self._call_durations: list[float] | None = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        with usage_scope(call_role=self._role):
            try:
                return await self._delegate.chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=min(max_tokens, self._max_output_tokens),
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
            finally:
                if self._call_durations is not None:
                    self._call_durations.append((time.perf_counter() - started) * 1_000)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        with usage_scope(call_role=self._role):
            async for delta in self._delegate.chat_stream(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=min(max_tokens, self._max_output_tokens),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            ):
                yield delta

    def get_default_model(self) -> str:
        return self._delegate.get_default_model()

    def classify_error(
        self,
        exc: BaseException | None = None,
        content: str | None = None,
    ) -> ErrorClassification:
        return self._delegate.classify_error(exc, content)


class RuntimeContextTrialRunner:
    def __init__(
        self,
        *,
        config: Config,
        codeflow_config: CodeFlowConfig,
        provider: LLMProvider,
        pressure_profile: dict[str, float] | None = None,
        context_window_tokens: int = CONTEXT_BENCHMARK_WINDOW_TOKENS,
    ) -> None:
        self._config = config
        self._codeflow_config = codeflow_config
        self._provider = provider
        self._pressure_profile = dict(pressure_profile or {})
        self._context_window_tokens = context_window_tokens

    async def __call__(
        self,
        *,
        context: TrialContext,
        task: ContextTask,
        context_engine_factory,
    ) -> TrialExecution:
        attempt_identity = (
            f"{task.task_id}:{context.variant.variant_id}:{context.key.repetition}:{context.block_attempt}"
        )
        attempt_id = f"t-{hashlib.sha256(attempt_identity.encode('utf-8')).hexdigest()[:12]}"
        isolation = TrialIsolation.create(
            context.experiment.output_root / ".cf" / context.experiment_id,
            attempt_id,
        )
        isolation.prepare()
        config, codeflow_config = _trial_configs(
            self._config,
            self._codeflow_config,
            workspace=isolation.workspace,
            model=self._provider.get_default_model(),
            context_budget_ratio=float(
                context.variant.settings.get(
                    "context_budget_ratio",
                    self._codeflow_config.context.context_budget_ratio,
                )
            ),
            structured_compact_threshold=float(
                context.variant.settings.get(
                    "structured_compact_threshold",
                    self._codeflow_config.context.structured_compact_threshold,
                )
            ),
            context_window_tokens=self._context_window_tokens,
        )
        recorder = UsageRecorder()
        provider = _CapturingProvider(self._provider, recorder=recorder)
        role_call_durations: dict[str, list[float]] = {"context_auxiliary": []}
        provider.generation = GenerationSettings(
            temperature=config.agents.defaults.temperature,
            max_tokens=config.agents.defaults.max_tokens,
            reasoning_effort=config.agents.defaults.reasoning_effort,
        )
        context_metadata_records: list[dict[str, Any]] = []
        scoped_factory = _role_scoped_context_factory(
            context_engine_factory,
            metadata_records=context_metadata_records,
            role_call_durations=role_call_durations,
            max_output_tokens=(
                codeflow_config.context.compact_max_output_tokens
                if context.variant.settings.get("history_manager") == "codeflow_history"
                else None
            ),
        )
        outlet = RecordingOutlet("codeflowbench-context")
        verifier = SealedContextTaskVerifier.capture(task)
        environment = {
            **isolation.child_environment(),
            "CODEFLOW_TRACING_DIR": str(isolation.trace_root),
        }
        started: float | None = None
        observation = None
        with (
            patch.dict("os.environ", environment, clear=False),
            patch(
                "codeflow.cli._plugin_stack.build_plugin_tools",
                return_value=[],
            ),
        ):
            host = await RuntimeTrialHost.build(
                config=config,
                codeflow_config=codeflow_config,
                provider=provider,
                cron_service=None,
                outlet=outlet,
                context_engine_factory=scoped_factory,
            )
            try:
                _require_context_tool_registry(host)
                history = task.materialize_history()
                pressure_target = self._pressure_profile.get(task.task_id)
                history_budget = _effective_history_budget(host)
                if pressure_target is not None:
                    history = _pad_history_to_pressure(
                        history,
                        history_budget=history_budget,
                        target_pressure=pressure_target,
                    )
                _seed_history(
                    host.assembly.session_manager,
                    conversation=_conversation_id(task, context),
                    history=history,
                )
                with usage_scope(call_role="main_agent"):
                    started = time.perf_counter()
                    observation = await host.run(
                        _turn_request(task, context),
                    )
            finally:
                await host.close()
        latency_ms = (time.perf_counter() - (started or time.perf_counter())) * 1_000
        verification = await verifier.verify(isolation.workspace)
        artifact_diagnostics = verifier.diagnostic_metrics(
            isolation.workspace,
        )
        aggregate = recorder.aggregate()
        records = recorder.records()
        compact_events = [record for record in context_metadata_records if record.get("context_manager") == "codeflow"]
        compact_count = sum(bool(record.get("context_compacted")) for record in compact_events)
        normal_trim_count = sum(
            bool(record.get("history_trimmed")) or record.get("pressure_tier") in {"tier1_snip", "tier2_prune"}
            for record in compact_events
        )
        summary_model_calls = sum(
            record.call_role == "context_auxiliary" and record.provider_dispatched for record in records
        )
        findings: list[str] = []
        if observation.failure_category is not None:
            findings.append(f"runtime_failure_category:{observation.failure_category}")
        turn_failure = next(
            (event for event in observation.events if isinstance(event, TurnFailed)),
            None,
        )
        if turn_failure is not None:
            findings.append(f"runtime_failure_detail:{turn_failure.error[:300]}")
        if verification.infrastructure_error is not None:
            findings.append(verification.infrastructure_error)
        if not aggregate.usage_complete:
            findings.append("trial_usage_incomplete")
        status = _trial_status(
            runtime_state=observation.runtime_state,
            failure_category=observation.failure_category,
            verification_state=verification.result.state,
            verifier_infrastructure_error=verification.infrastructure_error,
        )
        if not aggregate.usage_complete and status in {TrialStatus.PASSED, TrialStatus.TASK_FAILED}:
            status = TrialStatus.INFRASTRUCTURE_FAILURE
        relative_root = isolation.root.relative_to(
            context.experiment.output_root,
        )
        early_constraint_retained = _constraint_retained(
            provider.message_windows,
            final_prompt=task.final_prompt,
            constraint=task.early_constraint,
        )
        capability_criteria_passed = sum(
            (
                early_constraint_retained,
                artifact_diagnostics["active_constraint_applied"],
                artifact_diagnostics["latest_decision_applied"],
                artifact_diagnostics["artifact_exact"],
            )
        )
        return TrialExecution(
            status=status,
            runtime_state=observation.runtime_state,
            delivery_state=observation.delivery_state,
            verification=verification.result,
            observed_variant_settings=dict(context.variant.settings),
            metrics={
                "main_agent_input_tokens": _role_total(
                    records,
                    role="main_agent",
                    field="input_tokens",
                ),
                "main_agent_output_tokens": _role_total(
                    records,
                    role="main_agent",
                    field="output_tokens",
                ),
                "trial_total_input_tokens": aggregate.input_tokens,
                "trial_total_output_tokens": aggregate.output_tokens,
                "context_auxiliary_input_tokens": _role_total(
                    records,
                    role="context_auxiliary",
                    field="input_tokens",
                ),
                "context_auxiliary_output_tokens": _role_total(
                    records,
                    role="context_auxiliary",
                    field="output_tokens",
                ),
                "repeated_tool_calls": _repeated_tool_call_count(provider.tool_invocations),
                "usage_complete": aggregate.usage_complete,
                "context_path": (observation.outcome.context_path if observation.outcome is not None else None),
                "context_budget_ratio": codeflow_config.context.context_budget_ratio,
                "structured_compact_threshold": codeflow_config.context.structured_compact_threshold,
                "normal_trim_count": normal_trim_count,
                "compact_count": compact_count,
                "summary_model_call_count": summary_model_calls,
                "summary_fallback_count": sum(
                    record.get("summary_mode") == "deterministic_fallback" for record in compact_events
                ),
                "over_budget_count": sum(not bool(record.get("validation_ok", True)) for record in compact_events),
                "trigger_pressure": [
                    record.get("pressure_ratio") for record in compact_events if record.get("context_compacted")
                ],
                "history_pressure": [record.get("pressure_ratio") for record in compact_events],
                "effective_history_budget": history_budget,
                "pressure_target": pressure_target,
                "summary_call_latency_ms": sum(role_call_durations["context_auxiliary"]),
                "agent_steps": sum(type(event).__name__ == "ToolCompleted" for event in observation.events),
                "provider_reported_models": sorted({record.model for record in records if record.model}),
                "early_constraint_retained": early_constraint_retained,
                **artifact_diagnostics,
                "capability_criteria_passed": capability_criteria_passed,
                "capability_criteria_total": 4,
                "end_to_end_latency_ms": latency_ms,
            },
            findings=tuple(findings),
            artifact_refs=(relative_root.as_posix(),),
        )


class _CapturingContextEngine:
    def __init__(self, delegate, metadata_records: list[dict[str, Any]]) -> None:
        self._delegate = delegate
        self._metadata_records = metadata_records

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    async def assemble(self, *args, **kwargs):
        assembled = await self._delegate.assemble(*args, **kwargs)
        self._metadata_records.append(dict(assembled.metadata or {}))
        return assembled


def _role_scoped_context_factory(
    factory,
    *,
    metadata_records: list[dict[str, Any]],
    role_call_durations: dict[str, list[float]],
    max_output_tokens: int | None = None,
):
    def build(**kwargs):
        scoped_provider = _RoleScopedProvider(
            kwargs["provider"],
            "context_auxiliary",
            max_output_tokens=max_output_tokens,
        )
        scoped_provider._call_durations = role_call_durations["context_auxiliary"]
        kwargs["provider"] = scoped_provider
        return _CapturingContextEngine(factory(**kwargs), metadata_records)

    return build


def _require_context_tool_registry(host: RuntimeTrialHost) -> None:
    actual = frozenset(host.assembly.agent_loop.tools.tool_names)
    if actual != _CONTEXT_TOOL_NAMES:
        raise RuntimeError(
            "context_trial_tool_registry_drift:"
            f"expected={','.join(sorted(_CONTEXT_TOOL_NAMES))}:"
            f"actual={','.join(sorted(actual))}",
        )


def _trial_configs(
    config: Config,
    codeflow_config: CodeFlowConfig,
    *,
    workspace: Path,
    model: str,
    context_budget_ratio: float = 0.80,
    structured_compact_threshold: float = 0.95,
    context_window_tokens: int = CONTEXT_BENCHMARK_WINDOW_TOKENS,
) -> tuple[Config, CodeFlowConfig]:
    trial_config = config.model_copy(deep=True)
    trial_config.agents.defaults.workspace = str(workspace)
    trial_config.agents.defaults.model = model
    trial_config.agents.defaults.context_window_tokens = context_window_tokens
    trial_config.agents.defaults.max_tokens = CONTEXT_BENCHMARK_OUTPUT_TOKENS
    trial_config.agents.defaults.max_tool_iterations = CONTEXT_BENCHMARK_MAX_TOOL_ITERATIONS
    trial_config.agents.defaults.enable_personalization = False
    trial_config.routing.enabled = False
    trial_config.tools.restrict_to_workspace = True
    trial_config.tools.disabled_tools = list(_DISABLED_CONTEXT_TOOLS)
    trial_config.tools.mcp_servers = {}
    trial_config.tools.tool_search.enabled = False

    trial_codeflow = codeflow_config.model_copy(deep=True)
    trial_codeflow.base = trial_config
    trial_codeflow.context.curator_model = model
    trial_codeflow.context.protect_first_n = CONTEXT_BENCHMARK_PROTECT_FIRST_N
    trial_codeflow.context.context_budget_ratio = context_budget_ratio
    trial_codeflow.context.structured_compact_threshold = structured_compact_threshold
    trial_codeflow.memory.backend = None
    # Token usage and the spend ceiling are recorded by CodeFlowBench's
    # provider wrappers. Disable this separate per-trial ledger to keep its
    # health-file writes from exceeding Windows' legacy path-length limit.
    trial_codeflow.call_efficiency.enabled = False
    trial_codeflow.call_efficiency.usage_tracking = False
    trial_codeflow.skill_forge.enabled = False
    trial_codeflow.skill_forge.router.enabled = False
    trial_codeflow.skill_forge.rewrite_enabled = False
    trial_codeflow.skill_forge.llm_gate_enabled = False
    trial_codeflow.token_wise.smart_routing.enabled = False
    trial_codeflow.runtime.checkpoint.policy = "never"
    return trial_config, trial_codeflow


def _effective_history_budget(host: RuntimeTrialHost) -> int:
    loop = host.assembly.agent_loop
    budget = loop._make_token_budget()
    engine = loop.context_engine
    _, sections = engine._section_budgets(budget)
    return max(1, min(budget.available_history, sections["history"]))


def _pad_history_to_pressure(
    history: list[dict[str, Any]],
    *,
    history_budget: int,
    target_pressure: float,
) -> list[dict[str, Any]]:
    """Add deterministic neutral turns so the live history builder sees a target pressure band."""
    if not 0 < target_pressure < 1:
        raise ValueError("target pressure must be in (0, 1)")
    suffix_size = min(6, len(history))
    prefix = [dict(message) for message in history[:-suffix_size]]
    suffix = [dict(message) for message in history[-suffix_size:]]
    target_tokens = int(history_budget * target_pressure)
    current_tokens = sum(estimate_message_tokens(message) for message in (*prefix, *suffix))
    index = 0

    def pair(repetitions: int, pair_index: int) -> list[dict[str, Any]]:
        body = "neutral telemetry context. " * repetitions
        return [
            {"role": "user", "content": f"[deterministic stress filler {pair_index}] {body}"},
            {"role": "assistant", "content": f"Recorded neutral context {pair_index}. {body}"},
        ]

    while current_tokens < target_tokens:
        remaining = target_tokens - current_tokens
        low, high = 0, max(1, remaining * 2)
        best: list[dict[str, Any]] = []
        best_tokens = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = pair(middle, index)
            gained = sum(estimate_message_tokens(message) for message in candidate)
            if 0 < gained <= remaining:
                best, best_tokens = candidate, gained
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            break
        prefix.extend(best)
        current_tokens += best_tokens
        index += 1
    return [*prefix, *suffix]


def _seed_history(
    session_manager,
    *,
    conversation: str,
    history: list[dict[str, Any]],
) -> None:
    session = session_manager.get_or_create(conversation)
    for message in history:
        session.record(dict(message))
    session_manager.save(session)


def _conversation_id(task: ContextTask, context: TrialContext) -> str:
    identity = f"{task.task_id}:{context.variant.variant_id}:{context.key.repetition}:{context.block_attempt}"
    return f"cfctx:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"


def _turn_request(
    task: ContextTask,
    context: TrialContext,
) -> TurnRequest:
    conversation = _conversation_id(task, context)
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="cfctx",
            chat_id=conversation.removeprefix("cfctx:"),
            sender_id="codeflowbench",
            chat_type=ChatType.DM,
        ),
        text=f"{task.final_prompt}\n\n{_TASK_EXECUTION_NOTE}\n\n{task.output_contract}",
        message_id=f"{task.task_id}-message",
        conversation=conversation,
    )


def _role_total(
    records: tuple[ModelCallUsage, ...],
    *,
    role: str,
    field: str,
) -> int | None:
    selected = [getattr(record, field) for record in records if record.call_role == role and record.provider_dispatched]
    if not selected:
        return 0
    if any(value is None for value in selected):
        return None
    return sum(selected)


def _repeated_tool_call_count(calls: list[tuple[str, str]]) -> int:
    counts = Counter(calls)
    return sum(count - 1 for count in counts.values())


def _constraint_retained(
    windows: list[list[dict[str, Any]]],
    *,
    final_prompt: str,
    constraint: str,
) -> bool:
    for messages in windows:
        if not any(
            message.get("role") == "user" and final_prompt in str(message.get("content", "")) for message in messages
        ):
            continue
        return any(constraint in str(message.get("content", "")) for message in messages)
    return False


def _trial_status(
    *,
    runtime_state: TurnTerminalState,
    failure_category: str | None,
    verification_state: VerificationState,
    verifier_infrastructure_error: str | None,
) -> TrialStatus:
    if verifier_infrastructure_error is not None:
        return TrialStatus.INFRASTRUCTURE_FAILURE
    if failure_category == "task_budget_exhausted":
        return TrialStatus.TASK_TIMEOUT
    if runtime_state is TurnTerminalState.PROVIDER_FAILED:
        return TrialStatus.PROVIDER_FAILURE
    if runtime_state is TurnTerminalState.CANCELLED:
        return TrialStatus.CANCELLED
    if runtime_state is TurnTerminalState.ERROR:
        return TrialStatus.INFRASTRUCTURE_FAILURE
    if runtime_state is TurnTerminalState.COMPLETED and verification_state is VerificationState.PASSED:
        return TrialStatus.PASSED
    return TrialStatus.TASK_FAILED


__all__ = [
    "CONTEXT_BENCHMARK_OUTPUT_TOKENS",
    "CONTEXT_BENCHMARK_MAX_TOOL_ITERATIONS",
    "CONTEXT_BENCHMARK_PROTECT_FIRST_N",
    "CONTEXT_BENCHMARK_WINDOW_TOKENS",
    "RuntimeContextTrialRunner",
]
