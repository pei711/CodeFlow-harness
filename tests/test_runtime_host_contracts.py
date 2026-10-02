from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from codeflow.agent.tools.base import Tool
from codeflow.channels.contract import Capabilities, Channel
from codeflow.channels.intake import Intake
from codeflow.cli._gateway_spine import build_gateway
from codeflow.cli._repl_spine import build_repl
from codeflow.cli._runtime_assembly import assemble_runtime
from codeflow.config.codeflow import CodeFlowConfig
from codeflow.config.schema import Config
from codeflow.providers.base import LLMResponse, StreamDelta, ToolCallRequest
from codeflow.spine import ChatType, Origin, Source, TurnRequest
from codeflow.tui_rpc.dispatcher import Dispatcher
from codeflow.tui_rpc.methods import turn as turn_methods
from codeflow.tui_rpc.methods.turn import register_turn_methods
from codeflow.tui_rpc.spine import build_tui
from codeflow.tui_rpc.subscriptions import SubscriptionEmitter


class _ProbePluginTool(Tool):
    @property
    def name(self) -> str:
        return "probe_plugin"

    @property
    def description(self) -> str:
        return "A deterministic plugin tool."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        return "plugin-ok"


class _ProbeMcpTool(_ProbePluginTool):
    @property
    def name(self) -> str:
        return "mcp_docs_probe"


class _ScriptedProvider:
    def __init__(self) -> None:
        self._non_stream_calls = 0
        self._stream_calls = 0
        self.tool_schemas: list[set[str]] = []

    def get_default_model(self) -> str:
        return "probe/model"

    @staticmethod
    def _schema_names(tools: list[dict[str, Any]] | None) -> set[str]:
        return {
            str(tool["function"]["name"])
            for tool in tools or []
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }

    @staticmethod
    def _tool_response() -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="probe-tool-1",
                    name="read_file",
                    arguments={"path": "sentinel.txt"},
                )
            ],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )

    @staticmethod
    def _final_response() -> LLMResponse:
        return LLMResponse(
            content="HOST_CONTRACT_OK",
            finish_reason="stop",
            usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        )

    async def chat_with_retry(self, **kwargs: Any) -> LLMResponse:
        self.tool_schemas.append(self._schema_names(kwargs.get("tools")))
        response = self._tool_response() if self._non_stream_calls == 0 else self._final_response()
        self._non_stream_calls += 1
        return response

    async def chat_stream(self, **kwargs: Any):
        self.tool_schemas.append(self._schema_names(kwargs.get("tools")))
        if self._stream_calls == 0:
            response = self._tool_response()
            tool = response.tool_calls[0]
            yield StreamDelta(
                content=None,
                tool_call_delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool.id,
                            "function": {
                                "name": tool.name,
                                "arguments": json.dumps(tool.arguments),
                            },
                        }
                    ]
                },
                usage=response.usage,
            )
        else:
            response = self._final_response()
            yield StreamDelta(content=response.content, usage=response.usage)
        self._stream_calls += 1


class _Channel:
    name = "probe"

    def __init__(self) -> None:
        self.capabilities = Capabilities()
        self.sent: list[tuple[str, str, list[str] | None]] = []
        self.intake = Intake(
            "probe",
            SimpleNamespace(allow_from=["*"]),
        )

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, media: list[str] | None = None) -> None:
        self.sent.append((chat_id, content, media))


@dataclass
class _HostEvidence:
    tool_names: set[str]
    provider_tool_names: set[str]
    total_tokens: int
    tool_calls: int
    tool_failures: int


def _request(host: str) -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="probe" if host == "gateway" else host,
            chat_id="chat",
            sender_id="user",
            chat_type=ChatType.DM,
        ),
        text="Read sentinel.txt with the file tool, then report the marker.",
        conversation=f"{host}:chat",
    )


async def _run_host(host: str, runtime, provider: _ScriptedProvider) -> _HostEvidence:
    loop = runtime.agent_loop
    request = _request(host)
    if host == "cli":
        rendered: list[str] = []
        errors: list[str] = []
        scheduler, hub, teardown = build_repl(
            loop,
            "cli",
            rendered.append,
            render_error=errors.append,
        )
        try:
            outcome = await scheduler.submit(request).result()
            await hub.wait_idle("cli")
        finally:
            await teardown()
        assert rendered == ["HOST_CONTRACT_OK"]
        assert errors == []
    elif host == "tui":
        frames: list[dict[str, Any]] = []
        terminal = asyncio.Event()

        async def _send_frame(frame: dict[str, Any]) -> None:
            frames.append(frame)
            event = frame.get("params", {}).get("event", {})
            if event.get("type") in {"message.complete", "error"}:
                terminal.set()

        session_key = "tui:chat"
        emitter = SubscriptionEmitter(send_frame=_send_frame)
        scheduler, _hub, turn_ids, submission_ids, teardown = build_tui(
            loop,
            emitter,
            on_turn_end=turn_methods.clear_active,
        )
        dispatcher = Dispatcher()
        register_turn_methods(
            dispatcher,
            emitter=emitter,
            scheduler=scheduler,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )
        subscription_id = None
        try:
            subscribed = await dispatcher.dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "turn.subscribe",
                    "params": {"session_key": session_key},
                }
            )
            subscription_id = subscribed["result"]["subscription_id"]
            sent = await dispatcher.dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "turn.send",
                    "params": {
                        "session_key": session_key,
                        "submission_id": "submission-1",
                        "content": request.text,
                        "channel": "tui",
                        "chat_id": "chat",
                        "sender_id": "user",
                    },
                }
            )
            assert sent["result"]["accepted"] is True
            await asyncio.wait_for(terminal.wait(), timeout=5)
        finally:
            if subscription_id is not None:
                await emitter.unregister(subscription_id)
            turn_methods.clear_active(session_key)
            await teardown()
        events = [frame["params"]["event"] for frame in frames if frame.get("method") == "event"]
        types = [event["type"] for event in events]
        assert types == [
            "message.start",
            "tool.start",
            "tool.complete",
            "token.delta",
            "message.complete",
        ]
        complete = events[-1]
        assert complete["payload"]["usage"]["total_tokens"] == 6
        return _HostEvidence(
            tool_names=set(loop.tools.tool_names),
            provider_tool_names=provider.tool_schemas[0],
            total_tokens=complete["payload"]["usage"]["total_tokens"],
            tool_calls=sum(event["type"] == "tool.start" for event in events),
            tool_failures=sum(
                event["type"] == "tool.complete" and event["payload"].get("failed", False) for event in events
            ),
        )
    else:
        channel = _Channel()
        assert isinstance(channel, Channel)
        scheduler, hub, _readback, _sources, teardown = build_gateway(
            loop,
            {"probe": channel},
        )
        handles = []

        async def _submit(req) -> None:
            handles.append(scheduler.submit(req))

        try:
            channel.intake.set_submit(_submit)
            await channel.intake.publish(
                sender_id=request.source.sender_id,
                chat_id=request.source.chat_id,
                content=request.text,
                session_key=request.conversation,
            )
            assert len(handles) == 1
            outcome = await handles[0].result()
            await hub.wait_idle("probe")
        finally:
            await teardown()
        assert channel.sent == [("chat", "HOST_CONTRACT_OK", None)]

    assert outcome is not None
    return _HostEvidence(
        tool_names=set(loop.tools.tool_names),
        provider_tool_names=provider.tool_schemas[0],
        total_tokens=outcome.usage.total_tokens,
        tool_calls=outcome.tool_calls,
        tool_failures=outcome.tool_failures,
    )


async def test_protected_task_runs_through_cli_tui_and_gateway(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    mcp_connections: list[set[str]] = []

    async def _connect_mcp(servers, registry, stack, executor=None) -> None:
        mcp_connections.append(set(servers))
        registry.register(_ProbeMcpTool())

    monkeypatch.setattr("codeflow.agent.tools.mcp.connect_mcp_servers", _connect_mcp)
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.maybe_build_memory_backend",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_tools",
        lambda *args, **kwargs: [_ProbePluginTool()],
    )

    evidence: dict[str, _HostEvidence] = {}
    required = {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "grep",
        "find",
        "exec",
        "web_search",
        "web_fetch",
        "spawn",
        "ask_user",
        "probe_plugin",
        "mcp_docs_probe",
    }

    for host in ("cli", "tui", "gateway"):
        workspace = tmp_path / host
        workspace.mkdir()
        (workspace / "sentinel.txt").write_text("CODEFLOW_HOST_SENTINEL", encoding="utf-8")
        provider = _ScriptedProvider()
        config = Config()
        config.agents.defaults.workspace = str(workspace)
        config.agents.defaults.model = "probe/model"
        config.tools.restrict_to_workspace = True
        config.tools.mcp_servers = {"docs": object()}
        codeflow_config = CodeFlowConfig(base=config)
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
            interactive=host != "cli",
        )
        try:
            evidence[host] = await _run_host(host, runtime, provider)
            read_file = runtime.agent_loop.tools.get("read_file")
            exec_tool = runtime.agent_loop.tools.get("exec")
            assert read_file is not None
            assert exec_tool is not None
            assert "outside allowed directory" in await read_file.execute(str(tmp_path / "outside.txt"))
            assert "outside workspace" in await exec_tool.execute("pwd", working_dir=str(tmp_path))
        finally:
            await runtime.close()

    assert mcp_connections == [{"docs"}, {"docs"}, {"docs"}]
    assert all(item.total_tokens == 6 for item in evidence.values())
    assert all(item.tool_calls == 1 for item in evidence.values())
    assert all(item.tool_failures == 0 for item in evidence.values())
    assert all(required <= item.tool_names for item in evidence.values())
    assert {frozenset(item.tool_names) for item in evidence.values()} == {frozenset(evidence["cli"].tool_names)}
    assert {frozenset(item.provider_tool_names) for item in evidence.values()} == {
        frozenset(evidence["cli"].provider_tool_names)
    }
    assert required <= evidence["cli"].provider_tool_names
