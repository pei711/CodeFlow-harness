from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmarks.codeflowbench.budget import (
    _conservative_serialized_input_tokens,
)
from benchmarks.codeflowbench.canonical import (
    canonical_digest,
    to_primitive,
)
from benchmarks.codeflowbench.fixtures.mcp import (
    MCP_CATALOG_SIZE,
    catalog_definitions,
    catalog_digest,
)
from benchmarks.codeflowbench.host import RecordingOutlet, RuntimeTrialHost
from benchmarks.codeflowbench.isolation import TrialIsolation
from benchmarks.codeflowbench.protocol import TrialContext, TrialExecution
from benchmarks.codeflowbench.records import (
    TrialStatus,
    TurnTerminalState,
    VerificationState,
)
from codeflow.agent.tools.mcp import connect_mcp_servers
from codeflow.agent.tools.registry import ToolRegistry
from codeflow.agent.tools.tool_search import (
    ToolCallTool,
    ToolSearchController,
    ToolSearchTool,
)
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config, MCPServerConfig
from codeflow.providers.base import (
    ErrorClassification,
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from codeflow.spine import (
    ChatType,
    Origin,
    Source,
    ToolEvent,
    TurnRequest,
)

from .metrics import (
    TOOL_SCHEMA_ESTIMATOR_DIGEST,
    TOOL_SCHEMA_ESTIMATOR_ID,
    estimate_visible_tool_schema_tokens,
    normalize_target_calls,
)
from .models import MCPTransportSmokeResult, ToolMCPTask
from .verifier import SealedMCPReceiptVerifier

_DISABLED_DEFAULT_TOOLS = [
    "ask_user",
    "edit_file",
    "exec",
    "find",
    "grep",
    "list_dir",
    "message",
    "read_file",
    "skill_read",
    "spawn",
    "understand_media",
    "web_fetch",
    "web_search",
    "write_file",
]
_EXPECTED_CATALOG_NAMES = frozenset(definition.runtime_name for definition in catalog_definitions())
_MAX_OUTPUT_TOKENS = 1_500
TOOL_MCP_MAX_TOOL_ITERATIONS = 6


class _ScriptedToolMCPProvider(LLMProvider):
    def __init__(self, task: ToolMCPTask, disclosure: str) -> None:
        super().__init__()
        self.task = task
        self.disclosure = disclosure
        self._step = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        del messages, model, max_tokens, temperature, reasoning_effort
        del tool_choice, tools
        step = self._step
        self._step += 1
        if self.disclosure == "progressive_disclosure" and step == 0:
            first = self.task.targets[0]
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"{self.task.task_id}-search",
                        name="tool_search",
                        arguments={
                            "query": _search_query(first.tool_name),
                            "limit": 5,
                        },
                    )
                ],
                usage=_scripted_usage(),
                model=self.get_default_model(),
            )
        target_step = 1 if self.disclosure == "progressive_disclosure" else 0
        if step == target_step:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"{self.task.task_id}-target-{index}",
                        name=("tool_call" if self.disclosure == "progressive_disclosure" else target.runtime_name),
                        arguments=(
                            {
                                "name": target.runtime_name,
                                "arguments": dict(target.arguments),
                            }
                            if self.disclosure == "progressive_disclosure"
                            else dict(target.arguments)
                        ),
                    )
                    for index, target in enumerate(self.task.targets)
                ],
                usage=_scripted_usage(),
                model=self.get_default_model(),
            )
        return LLMResponse(
            content=f"verified {self.task.task_id}",
            usage=_scripted_usage(),
            model=self.get_default_model(),
        )

    def get_default_model(self) -> str:
        return "codeflowbench/tool-mcp-deterministic"


class _RecordingToolMCPProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(
        self,
        delegate: LLMProvider,
        *,
        model: str,
        generation: GenerationSettings,
    ) -> None:
        super().__init__()
        self.delegate = delegate
        self.model = model
        self.generation = _bounded_generation(generation)
        self.tool_payloads: list[list[dict[str, Any]]] = []
        self.call_records: list[dict[str, Any]] = []
        self._request_attempts: dict[str, int] = {}

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        if model is not None and model != self.model:
            raise RuntimeError(f"Tool/MCP model identity drift: {model!r} != {self.model!r}")
        effective_model = self.model
        tool_payload = to_primitive(tools or [])
        self.tool_payloads.append(tool_payload)
        request_digest = canonical_digest(
            {
                "messages": messages,
                "tools": tool_payload,
                "model": effective_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "tool_choice": tool_choice,
            }
        )
        attempt = self._request_attempts.get(request_digest, 0) + 1
        self._request_attempts[request_digest] = attempt
        started = time.perf_counter()
        base_record = {
            "call_id": f"main-agent-{len(self.call_records) + 1:04d}",
            "call_role": "main_agent",
            "attempt_kind": "initial" if attempt == 1 else "retry",
            "request_digest": request_digest,
            "model": effective_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "visible_tool_count": len(tool_payload),
            "visible_tool_names": sorted(definition["function"]["name"] for definition in tool_payload),
            "visible_tool_schema_digest": canonical_digest(tool_payload),
            "estimated_visible_tool_schema_tokens": (estimate_visible_tool_schema_tokens(tool_payload)),
            "conservative_serialized_input_tokens": (
                _conservative_serialized_input_tokens(
                    messages,
                    tool_payload,
                    model=effective_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    tool_choice=tool_choice,
                )
            ),
        }
        try:
            response = await self.delegate.chat(
                messages=messages,
                tools=tool_payload,
                model=effective_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
        except Exception as exc:
            error_category = self.delegate.classify_error(exc).category
            self.call_records.append(
                {
                    **base_record,
                    "model": None,
                    "requested_model": effective_model,
                    "latency_ms": (time.perf_counter() - started) * 1_000,
                    "finish_reason": "exception",
                    "tool_calls": [],
                    "usage": _normalized_usage({}),
                    "usage_complete": False,
                    "error_type": type(exc).__name__,
                    "error_category": error_category,
                }
            )
            raise
        normalized_usage = _normalized_usage(response.usage)
        actual_model = response.model
        self.call_records.append(
            {
                **base_record,
                "model": actual_model,
                "requested_model": effective_model,
                "latency_ms": (time.perf_counter() - started) * 1_000,
                "finish_reason": response.finish_reason,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": to_primitive(tool_call.arguments),
                    }
                    for tool_call in response.tool_calls
                ],
                "usage": normalized_usage,
                "usage_complete": _usage_record_complete(normalized_usage),
            }
        )
        return response

    def get_default_model(self) -> str:
        return self.model

    def classify_error(
        self,
        exc: BaseException | None = None,
        content: str | None = None,
    ) -> ErrorClassification:
        return self.delegate.classify_error(exc, content)


class MCPRuntimeTrialRunner:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        generation: GenerationSettings | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.generation = generation or provider.generation

    async def __call__(
        self,
        *,
        context: TrialContext,
        task: ToolMCPTask,
    ) -> TrialExecution:
        provider = _RecordingToolMCPProvider(
            self.provider,
            model=self.model,
            generation=self.generation,
        )
        return await _run_runtime_trial(
            context=context,
            task=task,
            provider=provider,
        )


class DeterministicMCPTrialRunner:
    async def __call__(
        self,
        *,
        context: TrialContext,
        task: ToolMCPTask,
    ) -> TrialExecution:
        disclosure = str(context.variant.settings["tool_disclosure"])
        delegate = _ScriptedToolMCPProvider(task, disclosure)
        provider = _RecordingToolMCPProvider(
            delegate,
            model=delegate.get_default_model(),
            generation=delegate.generation,
        )
        return await _run_runtime_trial(
            context=context,
            task=task,
            provider=provider,
        )


async def _run_runtime_trial(
    *,
    context: TrialContext,
    task: ToolMCPTask,
    provider: _RecordingToolMCPProvider,
) -> TrialExecution:
    disclosure = str(context.variant.settings["tool_disclosure"])
    isolation_root = context.experiment.output_root / ".codeflowbench-tool-mcp" / context.experiment_id
    attempt_id = f"{task.task_id}-{context.variant.variant_id}-r{context.key.repetition}-b{context.block_attempt}"
    isolation = TrialIsolation.create(isolation_root, attempt_id)
    isolation.prepare()
    receipt_path = isolation.evidence_root / "receipts.jsonl"
    verifier = SealedMCPReceiptVerifier.capture(task)
    config, codeflow_config = _runtime_config(
        workspace=isolation.workspace,
        receipt_path=receipt_path,
        provider=provider,
        progressive=disclosure == "progressive_disclosure",
    )
    outlet = RecordingOutlet("codeflowbench-tool-mcp")
    started = time.perf_counter()
    observation = None
    catalog_payloads: list[dict[str, Any]] = []
    environment = {
        **isolation.child_environment(),
        "CODEFLOW_TRACING_DIR": str(isolation.trace_root),
    }
    with patch.dict("os.environ", environment, clear=False):
        host = await RuntimeTrialHost.build(
            config=config,
            codeflow_config=codeflow_config,
            provider=provider,
            cron_service=None,
            outlet=outlet,
        )
        try:
            observation = await host.run(_turn_request(task))
            definitions = host.assembly.agent_loop.tools.get_definitions()
            catalog_payloads = [
                definition
                for definition in definitions
                if definition["function"]["name"].startswith("mcp_codeflowbench_catalog_probe_")
            ]
        finally:
            await host.close()
    latency_ms = (time.perf_counter() - started) * 1_000

    verification, receipts = verifier.verify(receipt_path)
    catalog_names = frozenset(definition["function"]["name"] for definition in catalog_payloads)
    actual_catalog_digest = _actual_catalog_digest(catalog_payloads)
    tool_events = tuple(event for event in observation.events if isinstance(event, ToolEvent))
    first_payload = provider.tool_payloads[0] if provider.tool_payloads else []
    initial_names = frozenset(definition["function"]["name"] for definition in first_payload)
    initially_visible_catalog = initial_names & _EXPECTED_CATALOG_NAMES
    normalized = normalize_target_calls(
        tool_events,
        catalog_names=catalog_names,
        initially_visible_names=initial_names,
        expected_first_target=task.targets[0].runtime_name,
    )
    schema_tokens = [estimate_visible_tool_schema_tokens(payload) for payload in provider.tool_payloads]
    usage = _aggregate_usage(provider.call_records)
    mcp_connected = catalog_names == _EXPECTED_CATALOG_NAMES
    findings: list[str] = []
    if not mcp_connected:
        findings.append("mcp_catalog_connection_incomplete")
    if not usage["usage_complete"]:
        findings.append("incomplete_main_agent_usage")
    failure_category = _effective_failure_category(
        observation.failure_category,
        provider.call_records,
    )
    status = _trial_status(
        runtime_state=observation.runtime_state,
        failure_category=failure_category,
        verification_state=verification.state,
        mcp_connected=mcp_connected,
    )
    return TrialExecution(
        status=status,
        runtime_state=observation.runtime_state,
        delivery_state=observation.delivery_state,
        verification=verification,
        observed_variant_settings=dict(context.variant.settings),
        metrics={
            "mcp_transport": "stdio",
            "mcp_connected": mcp_connected,
            "mcp_catalog_count": len(catalog_names),
            "mcp_catalog_digest": actual_catalog_digest,
            "mcp_catalog_source_digest": catalog_digest(),
            "mcp_verifier_digest": verifier.verifier_code_digest,
            "mcp_receipt_count": len(receipts),
            "initial_visible_tool_count": len(initial_names),
            "initial_visible_tool_names": sorted(initial_names),
            "initial_visible_catalog_tool_count": len(initially_visible_catalog),
            "initial_visible_catalog_tool_names": sorted(initially_visible_catalog),
            "visible_tool_schema_tokens_per_call": schema_tokens,
            "trial_total_estimated_visible_tool_schema_tokens": sum(schema_tokens),
            "schema_estimator_id": TOOL_SCHEMA_ESTIMATOR_ID,
            "schema_estimator_digest": TOOL_SCHEMA_ESTIMATOR_DIGEST,
            "meta_tool_invocations": normalized.meta_tool_invocations,
            "meta_tool_failures": normalized.meta_tool_failures,
            "first_target_accuracy": normalized.first_target_accuracy,
            "invalid_target_call_rate": normalized.invalid_target_call_rate,
            "exact_target_repeat_rate": normalized.exact_target_repeat_rate,
            "normalized_target_call_count": len(normalized.records),
            "target_call_records": [to_primitive(record) for record in normalized.records],
            "provider_model": provider.model,
            "actual_model_names": sorted(
                {str(record["model"]) for record in provider.call_records if record["model"] is not None}
            ),
            "generation_settings": to_primitive(provider.generation),
            "provider_call_max_attempts": len(provider._CHAT_RETRY_DELAYS) + 1,
            "model_call_records": to_primitive(provider.call_records),
            **usage,
            "end_to_end_latency_ms": latency_ms,
        },
        findings=tuple(findings),
        artifact_refs=(isolation.root.relative_to(context.experiment.output_root).as_posix(),),
    )


async def run_mcp_transport_smoke(root: Path) -> MCPTransportSmokeResult:
    state_root = root / "mcp-smoke"
    state_root.mkdir(parents=True, exist_ok=True)
    receipt_path = state_root / "receipts.jsonl"
    registry = ToolRegistry()
    stack = AsyncExitStack()
    await stack.__aenter__()
    transport_closed = False
    try:
        await connect_mcp_servers(
            {"codeflowbench": _mcp_server_config(receipt_path)},
            registry,
            stack,
        )
        names = frozenset(name for name in registry.tool_names if name in _EXPECTED_CATALOG_NAMES)
        if names != _EXPECTED_CATALOG_NAMES:
            raise RuntimeError(f"expected {MCP_CATALOG_SIZE} MCP tools, got {len(names)}")
        controller = ToolSearchController(
            registry,
            always_visible=set(),
            search_result_limit=5,
        )
        controller.refresh()
        search = ToolSearchTool(controller)
        call = ToolCallTool(controller)
        target = catalog_definitions()[0]
        raw_hits = await search.execute(
            query=_search_query(target.name),
            limit=5,
        )
        hits = json.loads(raw_hits)
        hit = next(candidate for candidate in hits if candidate["name"] == target.runtime_name)
        raw_receipt = await call.execute(
            name=hit["name"],
            arguments={
                "resource": "smoke-resource",
                "operation": "inspect",
                "value": 1,
            },
        )
        receipt = json.loads(str(raw_receipt))
    finally:
        await stack.aclose()
        transport_closed = True
    return MCPTransportSmokeResult(
        transport="stdio",
        catalog_count=len(names),
        search_hit_name=hit["name"],
        called_target_name=target.runtime_name,
        receipt=receipt,
        transport_closed=transport_closed,
    )


def _runtime_config(
    *,
    workspace: Path,
    receipt_path: Path,
    provider: LLMProvider,
    progressive: bool,
) -> tuple[Config, CodeFlowConfig]:
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = provider.get_default_model()
    config.agents.defaults.max_tokens = provider.generation.max_tokens
    config.agents.defaults.temperature = provider.generation.temperature
    config.agents.defaults.reasoning_effort = provider.generation.reasoning_effort
    config.agents.defaults.max_tool_iterations = TOOL_MCP_MAX_TOOL_ITERATIONS
    config.routing.enabled = False
    config.tools.restrict_to_workspace = True
    config.tools.disabled_tools = list(_DISABLED_DEFAULT_TOOLS)
    config.tools.mcp_servers = {"codeflowbench": _mcp_server_config(receipt_path)}
    config.tools.tool_search.enabled = progressive
    config.tools.tool_search.compaction_threshold = 50
    config.tools.tool_search.search_result_limit = 5
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.memory.backend = None
    codeflow_config.skill_forge.enabled = False
    codeflow_config.skill_forge.router.enabled = False
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.token_wise.smart_routing.enabled = False
    codeflow_config.runtime.checkpoint.policy = "never"
    return config, codeflow_config


def _mcp_server_config(receipt_path: Path) -> MCPServerConfig:
    server_path = Path(__file__).resolve().parents[2] / "fixtures" / "mcp" / "server.py"
    return MCPServerConfig(
        type="stdio",
        command=sys.executable,
        args=[str(server_path)],
        env={"CODEFLOWBENCH_MCP_RECEIPTS": str(receipt_path)},
        tool_timeout=10,
    )


def _turn_request(task: ToolMCPTask) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="codeflowbench-tool-mcp",
            chat_id=task.task_id,
            sender_id="codeflowbench",
            chat_type=ChatType.DM,
        ),
        text=task.prompt,
        message_id=f"{task.task_id}-message",
        conversation=f"codeflowbench-tool-mcp:{task.task_id}",
    )


def _actual_catalog_digest(
    definitions: list[dict[str, Any]],
) -> str:
    return canonical_digest(
        sorted(
            definitions,
            key=lambda definition: definition["function"]["name"],
        )
    )


def _search_query(tool_name: str) -> str:
    return f"catalog slot {tool_name.rsplit('_', 1)[-1]}"


def _bounded_generation(
    generation: GenerationSettings,
) -> GenerationSettings:
    return GenerationSettings(
        temperature=generation.temperature,
        max_tokens=min(generation.max_tokens, _MAX_OUTPUT_TOKENS),
        reasoning_effort=generation.reasoning_effort,
    )


def _scripted_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


def _normalized_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    input_tokens = _usage_value(
        usage,
        ("prompt_tokens",),
        ("input_tokens",),
    )
    output_tokens = _usage_value(
        usage,
        ("completion_tokens",),
        ("output_tokens",),
    )
    total_tokens = _usage_value(usage, ("total_tokens",))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": _usage_value(
            usage,
            ("cache_read_tokens",),
            ("cache_read_input_tokens",),
            ("prompt_cache_hit_tokens",),
            ("prompt_tokens_details", "cached_tokens"),
        ),
        "cache_write_tokens": _usage_value(
            usage,
            ("cache_write_tokens",),
            ("cache_creation_input_tokens",),
        ),
        "reasoning_tokens": _usage_value(
            usage,
            ("reasoning_tokens",),
            ("completion_tokens_details", "reasoning_tokens"),
        ),
        "cost_usd": _usage_value(
            usage,
            ("cost_usd",),
            ("estimated_cost_usd",),
            ("response_cost",),
        ),
        "raw": to_primitive(dict(usage)),
    }


def _usage_value(
    usage: Mapping[str, Any],
    *paths: tuple[str, ...],
) -> int | float | None:
    for path in paths:
        value: Any = usage
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                value = None
                break
            value = value[component]
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
    return None


def _usage_record_complete(usage: Mapping[str, Any]) -> bool:
    return all(usage.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens"))


def _aggregate_usage(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    usages = [record["usage"] for record in records]
    usage_complete = bool(records) and all(record["usage_complete"] for record in records)
    return {
        "main_agent_model_calls": len(records),
        "main_agent_input_tokens": _complete_sum(usages, "prompt_tokens"),
        "main_agent_output_tokens": _complete_sum(
            usages,
            "completion_tokens",
        ),
        "main_agent_total_tokens": _complete_sum(usages, "total_tokens"),
        "main_agent_cache_read_tokens": _complete_sum(
            usages,
            "cache_read_tokens",
        ),
        "main_agent_cache_write_tokens": _complete_sum(
            usages,
            "cache_write_tokens",
        ),
        "main_agent_reasoning_tokens": _complete_sum(
            usages,
            "reasoning_tokens",
        ),
        "main_agent_cost_usd": _complete_sum(usages, "cost_usd"),
        "usage_complete": usage_complete,
        "cost_complete": bool(records) and all(usage["cost_usd"] is not None for usage in usages),
    }


def _complete_sum(
    usages: list[dict[str, Any]],
    key: str,
) -> int | float | None:
    values = [usage[key] for usage in usages]
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _trial_status(
    *,
    runtime_state: TurnTerminalState,
    failure_category: str | None,
    verification_state: VerificationState,
    mcp_connected: bool,
) -> TrialStatus:
    if verification_state is VerificationState.NOT_RUN or not mcp_connected:
        return TrialStatus.INFRASTRUCTURE_FAILURE
    if failure_category == "task_budget_exhausted":
        return TrialStatus.TASK_TIMEOUT
    if runtime_state is TurnTerminalState.PROVIDER_FAILED:
        return TrialStatus.PROVIDER_FAILURE
    if runtime_state is TurnTerminalState.CANCELLED:
        return TrialStatus.CANCELLED
    if verification_state is VerificationState.PASSED and runtime_state is TurnTerminalState.COMPLETED:
        return TrialStatus.PASSED
    return TrialStatus.TASK_FAILED


def _effective_failure_category(
    runtime_failure_category: str | None,
    call_records: list[dict[str, Any]],
) -> str | None:
    if runtime_failure_category is not None:
        return runtime_failure_category
    if any(record.get("error_category") == "task_budget_exhausted" for record in call_records):
        return "task_budget_exhausted"
    return None


__all__ = [
    "DeterministicMCPTrialRunner",
    "MCPRuntimeTrialRunner",
    "run_mcp_transport_smoke",
]
