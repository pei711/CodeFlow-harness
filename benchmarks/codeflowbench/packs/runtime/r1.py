from __future__ import annotations

import asyncio
import re
from collections import Counter
from pathlib import Path

from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.agent.tools.base import Tool, ToolResult
from codeflow.cli._runtime_assembly import assemble_runtime
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from codeflow.session.manager import SessionManager
from codeflow.spine import (
    ChatType,
    Origin,
    OriginPools,
    Scheduler,
    Source,
    Text,
    ToolEvent,
    ToolPhase,
    TurnEnded,
    TurnFailed,
    TurnRequest,
    TurnStarted,
)
from codeflow.spine.delivery import (
    Capabilities,
    DeliveryHub,
    make_hub_sink,
)

from .models import R1RuntimeResult

R1_FULL_PATH_TURNS = 100

_MARKER_RE = re.compile(r"\[(r1-\d{3}):([a-z_]+)]")
_ID_RE = re.compile(r"r1-\d{3}")


class _DeterministicProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []
        self.cancel_entered: dict[str, asyncio.Event] = {}

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
        marker, scenario = _scenario_from_messages(messages)
        self.calls.append(marker)
        if scenario == "cancel":
            self.cancel_entered.setdefault(marker, asyncio.Event()).set()
            await asyncio.Event().wait()
        if scenario == "provider_failure":
            return LLMResponse(
                content="deterministic provider failure",
                finish_reason="error",
                error_classification=ErrorClassification("auth"),
                usage=_usage(),
            )
        if scenario in {"tool_success", "tool_failure"} and not any(
            message.get("role") == "tool" for message in messages
        ):
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id=f"{marker}-tool",
                        name="codeflowbench_probe",
                        arguments={"fail": scenario == "tool_failure"},
                    )
                ],
                usage=_usage(),
            )
        return LLMResponse(
            content=f"completed {marker}",
            usage=_usage(),
        )

    def get_default_model(self) -> str:
        return "codeflowbench/deterministic"


class _ProbeTool(Tool):
    @property
    def name(self) -> str:
        return "codeflowbench_probe"

    @property
    def description(self) -> str:
        return "Return a deterministic success or expected failure."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"fail": {"type": "boolean"}},
            "required": ["fail"],
        }

    async def execute(self, **kwargs) -> str:
        if kwargs["fail"]:
            return ToolResult("expected deterministic failure", failed=True)
        return "deterministic success"


class _RecordingOutlet:
    capabilities = Capabilities()

    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.received: list[object] = []

    async def deliver(self, out) -> None:
        if self.fail:
            raise RuntimeError("injected delivery failure")
        self.received.append(out)


class _RecordingDeliveryHub(DeliveryHub):
    def __init__(self) -> None:
        super().__init__(send_max_retries=0)
        self.outcomes: dict[str, str] = {}
        self.duplicate_outcomes = 0

    def _deliver_span(
        self,
        out,
        channel: str,
        outcome: str,
        *,
        attempts: int,
        error: str | None = None,
    ) -> None:
        conversation = out.conversation_id
        if conversation is not None:
            if conversation in self.outcomes:
                self.duplicate_outcomes += 1
            self.outcomes[conversation] = outcome
        super()._deliver_span(
            out,
            channel,
            outcome,
            attempts=attempts,
            error=error,
        )


class _R1Recorder:
    def __init__(self, requests: dict[str, TurnRequest]) -> None:
        self.requests = requests
        self.active: set[str] = set()
        self.terminals: dict[str, str] = {}
        self.tool_turns: set[str] = set()
        self.identity_contradictions = 0
        self.lifecycle_contradictions = 0

    async def record(self, event: object) -> None:
        conversation = getattr(event, "conversation_id", None)
        if conversation is None:
            return
        if conversation not in self.requests:
            self.identity_contradictions += 1
            return
        if isinstance(event, TurnStarted):
            if conversation in self.active:
                self.lifecycle_contradictions += 1
            self.active.add(conversation)
            return
        if isinstance(event, TurnEnded):
            self._terminal(
                conversation,
                "completed_with_tool_failure" if event.tool_failures else "completed",
            )
            return
        if isinstance(event, TurnFailed):
            if event.cancelled:
                status = "cancelled"
            elif event.error.startswith("provider_error:"):
                status = "provider_failed"
            else:
                status = "error"
            self._terminal(conversation, status)
            return
        if isinstance(event, Text):
            marker = _id_from_text(event.content)
            self._check_identity(marker, conversation)
            return
        if isinstance(event, ToolEvent):
            marker = _id_from_text(event.tool_call_id)
            self._check_identity(marker, conversation)
            if event.phase is ToolPhase.START:
                self.tool_turns.add(conversation)

    def _terminal(self, conversation: str, status: str) -> None:
        if conversation not in self.active or conversation in self.terminals:
            self.lifecycle_contradictions += 1
        self.active.discard(conversation)
        self.terminals[conversation] = status

    def _check_identity(self, marker: str | None, conversation: str) -> None:
        if marker is None or conversation != f"codeflowbench-r1:{marker}":
            self.identity_contradictions += 1


async def run_r1_full_runtime_track(root: Path) -> R1RuntimeResult:
    workspace = root / "r1-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    provider = _DeterministicProvider()
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = provider.get_default_model()
    config.agents.defaults.max_tool_iterations = 4
    config.tools.restrict_to_workspace = True
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.memory.backend = None
    codeflow_config.skill_forge.enabled = False
    codeflow_config.skill_forge.router.enabled = False
    codeflow_config.skill_forge.rewrite_enabled = False
    codeflow_config.skill_forge.llm_gate_enabled = False
    codeflow_config.runtime.checkpoint.policy = "never"

    runtime = assemble_runtime(
        config,
        codeflow_config,
        provider=provider,
        cron_service=None,
        interactive=False,
    )
    runtime.agent_loop.tools.register(_ProbeTool())
    runner = AgentTurnRunner(runtime.agent_loop, stream=False)
    requests = {f"codeflowbench-r1:r1-{index:03d}": _request(index) for index in range(R1_FULL_PATH_TURNS)}
    recorder = _R1Recorder(requests)
    hub = _RecordingDeliveryHub()
    delivered = _RecordingOutlet("delivered")
    failing = _RecordingOutlet("failing", fail=True)
    hub.register(delivered)
    hub.register(failing)
    hub_sink = make_hub_sink(hub)

    async def sink(event) -> None:
        await recorder.record(event)
        await hub_sink(event)

    scheduler = Scheduler(
        runner,
        OriginPools(user=16, system=4),
        sink,
    )
    handles = {}
    results = {}
    resources_closed = False
    try:
        for conversation, req in requests.items():
            handle = scheduler.submit(req)
            handles[conversation] = handle
            if _scenario(req) == "cancel":
                marker = req.message_id
                if marker is None:
                    raise ValueError("R1 requests require message_id")
                await asyncio.wait_for(
                    provider.cancel_entered.setdefault(
                        marker,
                        asyncio.Event(),
                    ).wait(),
                    timeout=5,
                )
                handle.cancel()
            results[conversation] = await asyncio.wait_for(
                handle.result(),
                timeout=10,
            )

        await scheduler.shutdown(grace=0)
        await hub.wait_idle("delivered")
        await hub.wait_idle("failing")
    finally:
        await runtime.close()
        await hub.aclose()
        resources_closed = (
            not hub._workers and runtime.agent_loop._mcp_stack is None and runtime.agent_loop._executor_stack is None
        )

    identity_contradictions = (
        recorder.identity_contradictions
        + _provider_identity_contradictions(provider.calls, requests)
        + _delivery_identity_contradictions(hub.outcomes, requests)
        + hub.duplicate_outcomes
    )
    completed = {
        conversation
        for conversation, status in recorder.terminals.items()
        if status in {"completed", "completed_with_tool_failure"}
    }
    readable_sessions = _readable_session_count(
        workspace,
        completed,
    )
    unresolved_handles = len(handles) - len(results)
    lifecycle_contradictions = recorder.lifecycle_contradictions + len(recorder.active)

    return R1RuntimeResult(
        turns=len(requests),
        submitted_through_scheduler=len(handles),
        runner_type=type(runner).__name__,
        terminal_counts=dict(Counter(recorder.terminals.values())),
        model_calls=len(provider.calls),
        tool_event_turns=len(recorder.tool_turns),
        readable_sessions=readable_sessions,
        delivery_counts=dict(Counter(hub.outcomes.values())),
        identity_contradictions=identity_contradictions,
        lifecycle_contradictions=lifecycle_contradictions,
        unresolved_handles=unresolved_handles,
        resources_closed=resources_closed,
    )


def _request(index: int) -> TurnRequest:
    marker = f"r1-{index:03d}"
    scenario, channel = _scenario_for_index(index)
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel=channel,
            chat_id=marker,
            sender_id="codeflowbench",
            chat_type=ChatType.DM,
        ),
        text=f"[{marker}:{scenario}] execute deterministic runtime probe",
        message_id=marker,
        conversation=f"codeflowbench-r1:{marker}",
    )


def _scenario_for_index(index: int) -> tuple[str, str]:
    if index < 70:
        return "clean", "delivered"
    if index < 80:
        return "tool_success", "delivered"
    if index < 88:
        return "tool_failure", "delivered"
    if index < 92:
        return "provider_failure", "delivered"
    if index < 96:
        return "cancel", "delivered"
    if index < 98:
        return "clean", "failing"
    return "clean", "missing"


def _scenario(req: TurnRequest) -> str:
    match = _MARKER_RE.search(req.text)
    if match is None:
        raise ValueError(f"missing R1 scenario marker: {req.text}")
    return match.group(2)


def _scenario_from_messages(messages: list[dict]) -> tuple[str, str]:
    for message in reversed(messages):
        content = str(message.get("content", ""))
        match = _MARKER_RE.search(content)
        if match is not None:
            return match.group(1), match.group(2)
    raise ValueError("deterministic Provider did not receive an R1 marker")


def _id_from_text(value: str) -> str | None:
    match = _ID_RE.search(value)
    return match.group(0) if match is not None else None


def _usage() -> dict[str, int]:
    return {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


def _provider_identity_contradictions(
    calls: list[str],
    requests: dict[str, TurnRequest],
) -> int:
    return sum(f"codeflowbench-r1:{marker}" not in requests for marker in calls)


def _delivery_identity_contradictions(
    outcomes: dict[str, str],
    requests: dict[str, TurnRequest],
) -> int:
    return sum(conversation not in requests for conversation in outcomes)


def _readable_session_count(
    workspace: Path,
    completed: set[str],
) -> int:
    manager = SessionManager(workspace)
    readable = 0
    for conversation in completed:
        marker = conversation.partition(":")[2]
        session = manager.get_or_create(conversation)
        contents = [str(message.get("content", "")) for message in session.messages]
        if any(marker in content for content in contents) and any(
            message.get("role") == "assistant" for message in session.messages
        ):
            readable += 1
    return readable
