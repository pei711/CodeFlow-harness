from __future__ import annotations

import time
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
CONTEXT_BENCHMARK_WINDOW_TOKENS = 2_400
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
        return await super().chat(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )


class _RoleScopedProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(self, delegate: LLMProvider, role: str) -> None:
        super().__init__(
            api_key=getattr(delegate, "api_key", None),
            api_base=getattr(delegate, "api_base", None),
        )
        self._delegate = delegate
        self._role = role
        self.generation = delegate.generation

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
        with usage_scope(call_role=self._role):
            return await self._delegate.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=min(max_tokens, self.generation.max_tokens),
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )

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
                max_tokens=min(max_tokens, self.generation.max_tokens),
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
    ) -> None:
        self._config = config
        self._codeflow_config = codeflow_config
        self._provider = provider

    async def __call__(
        self,
        *,
        context: TrialContext,
        task: ContextTask,
        context_engine_factory,
    ) -> TrialExecution:
        attempt_id = f"{task.task_id}-{context.variant.variant_id}-r{context.key.repetition}-b{context.block_attempt}"
        isolation = TrialIsolation.create(
            context.experiment.output_root / ".codeflowbench-context" / context.experiment_id,
            attempt_id,
        )
        isolation.prepare()
        config, codeflow_config = _trial_configs(
            self._config,
            self._codeflow_config,
            workspace=isolation.workspace,
            model=self._provider.get_default_model(),
        )
        recorder = UsageRecorder()
        provider = _CapturingProvider(self._provider, recorder=recorder)
        provider.generation = GenerationSettings(
            temperature=config.agents.defaults.temperature,
            max_tokens=config.agents.defaults.max_tokens,
            reasoning_effort=config.agents.defaults.reasoning_effort,
        )
        scoped_factory = _role_scoped_context_factory(
            context_engine_factory,
        )
        outlet = RecordingOutlet("codeflowbench-context")
        verifier = SealedContextTaskVerifier.capture(task)
        environment = {
            **isolation.child_environment(),
            "CODEFLOW_TRACING_DIR": str(isolation.trace_root),
        }
        started = time.perf_counter()
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
                _seed_history(
                    host.assembly.session_manager,
                    conversation=_conversation_id(task, context),
                    history=task.materialize_history(),
                )
                with usage_scope(call_role="main_agent"):
                    observation = await host.run(
                        _turn_request(task, context),
                    )
            finally:
                await host.close()
        latency_ms = (time.perf_counter() - started) * 1_000
        verification = await verifier.verify(isolation.workspace)
        artifact_diagnostics = verifier.diagnostic_metrics(
            isolation.workspace,
        )
        aggregate = recorder.aggregate()
        records = recorder.records()
        findings: list[str] = []
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
                "trial_total_input_tokens": aggregate.input_tokens,
                "context_auxiliary_input_tokens": _role_total(
                    records,
                    role="context_auxiliary",
                    field="input_tokens",
                ),
                "usage_complete": aggregate.usage_complete,
                "context_path": (observation.outcome.context_path if observation.outcome is not None else None),
                "early_constraint_retained": early_constraint_retained,
                **artifact_diagnostics,
                "capability_criteria_passed": capability_criteria_passed,
                "capability_criteria_total": 4,
                "end_to_end_latency_ms": latency_ms,
            },
            findings=tuple(findings),
            artifact_refs=(relative_root.as_posix(),),
        )


def _role_scoped_context_factory(factory):
    def build(**kwargs):
        kwargs["provider"] = _RoleScopedProvider(
            kwargs["provider"],
            "context_auxiliary",
        )
        return factory(**kwargs)

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
) -> tuple[Config, CodeFlowConfig]:
    trial_config = config.model_copy(deep=True)
    trial_config.agents.defaults.workspace = str(workspace)
    trial_config.agents.defaults.model = model
    trial_config.agents.defaults.context_window_tokens = CONTEXT_BENCHMARK_WINDOW_TOKENS
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
    trial_codeflow.memory.backend = None
    trial_codeflow.skill_forge.enabled = False
    trial_codeflow.skill_forge.router.enabled = False
    trial_codeflow.skill_forge.rewrite_enabled = False
    trial_codeflow.skill_forge.llm_gate_enabled = False
    trial_codeflow.token_wise.smart_routing.enabled = False
    trial_codeflow.runtime.checkpoint.policy = "never"
    return trial_config, trial_codeflow


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
    return (
        f"codeflowbench-context:{task.task_id}-{context.variant.variant_id}-"
        f"r{context.key.repetition}-b{context.block_attempt}"
    )


def _turn_request(
    task: ContextTask,
    context: TrialContext,
) -> TurnRequest:
    conversation = _conversation_id(task, context)
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="codeflowbench-context",
            chat_id=conversation.removeprefix("codeflowbench-context:"),
            sender_id="codeflowbench",
            chat_type=ChatType.DM,
        ),
        text=f"{task.final_prompt}\n\n{_TASK_EXECUTION_NOTE}",
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
