"""Runtime Assembly deletion, capability, lifecycle, and Turn Runner contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, sentinel

import pytest

from codeflow.agent.spine_runner import AgentTurnRunner
from codeflow.agent.tools.base import Tool
from codeflow.cli._gateway_spine import GatewayTurnRunner
from codeflow.spine import (
    ChatType,
    Origin,
    Source,
    Text,
    TurnOutcome,
    TurnRequest,
    TurnRunner,
    Usage,
)
from codeflow.tui_rpc.spine import TuiTurnRunner


def _runtime_configs(tmp_path):
    defaults = SimpleNamespace(
        model="retained-model",
        max_tool_iterations=17,
        context_window_tokens=98_304,
        max_concurrent_subagents=3,
        max_subagent_spawns_per_hour=12,
        enable_personalization=True,
    )
    tools = SimpleNamespace(
        web=SimpleNamespace(
            search=SimpleNamespace(api_key="brave-key"),
            jina_api_key="jina-key",
            proxy="http://proxy.test",
        ),
        exec=sentinel.exec_config,
        restrict_to_workspace=True,
        mcp_servers={"docs": sentinel.mcp_server},
        disabled_tools=["write_file"],
        tool_search=sentinel.tool_search_config,
        sandbox=sentinel.sandbox_config,
    )
    config = SimpleNamespace(
        workspace_path=tmp_path,
        agents=SimpleNamespace(defaults=defaults),
        tools=tools,
        channels=sentinel.channels_config,
    )
    codeflow_config = SimpleNamespace(
        plugins=SimpleNamespace(disabled=[], config={}),
        memory=sentinel.memory_config,
        context=sentinel.context_config,
        call_efficiency=sentinel.call_efficiency_config,
        runtime=sentinel.runtime_config,
        skill_forge=SimpleNamespace(
            router=sentinel.skill_forge_router_config,
        ),
    )
    return config, codeflow_config


@pytest.mark.parametrize(
    ("host", "interactive", "router"),
    [
        ("cli_once", False, None),
        ("tui", True, None),
        ("gateway", True, sentinel.gateway_router),
    ],
)
def test_runtime_assembly_preserves_config_and_plugin_parity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    host: str,
    interactive: bool,
    router,
) -> None:
    from codeflow.cli import _runtime_assembly

    config, codeflow_config = _runtime_configs(tmp_path)
    provider = object()
    cron_service = object()
    session_manager = object()
    registry = object()
    backend = object()
    plugin_tools = [sentinel.plugin_tool]
    calls: list[tuple[str, object]] = []
    captured: dict = {}

    monkeypatch.setattr(
        "codeflow.session.manager.SessionManager",
        lambda workspace: calls.append(("session", workspace)) or session_manager,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        lambda cfg: calls.append(("registry", cfg)) or registry,
    )

    def _build_backend(workspace, cfg, *, registry=None):
        calls.append(("backend", registry))
        return backend

    def _build_tools(workspace, cfg, *, registry=None):
        calls.append(("tools", registry))
        return plugin_tools

    monkeypatch.setattr("codeflow.cli._plugin_stack.maybe_build_memory_backend", _build_backend)
    monkeypatch.setattr("codeflow.cli._plugin_stack.build_plugin_tools", _build_tools)
    monkeypatch.setattr(
        "codeflow.call_efficiency.CallEfficiency.from_config",
        lambda config, *, telemetry_dir, provider: sentinel.call_efficiency,
    )

    class _AgentLoopSpy:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.configure_personalization = MagicMock()

    monkeypatch.setattr("codeflow.agent.loop.AgentLoop", _AgentLoopSpy)

    runtime = _runtime_assembly.assemble_runtime(
        config,
        codeflow_config,
        provider=provider,
        cron_service=cron_service,
        interactive=interactive,
        router=router,
    )

    assert runtime.session_manager is session_manager
    assert runtime.backend is backend
    assert runtime.call_efficiency is sentinel.call_efficiency
    assert calls == [
        ("session", tmp_path),
        ("registry", codeflow_config),
        ("backend", registry),
        ("tools", registry),
    ]
    assert captured["provider"].delegate is provider
    captured["provider"] = provider
    assert captured == {
        "provider": provider,
        "workspace": tmp_path,
        "state": tmp_path,
        "model": "retained-model",
        "max_iterations": 17,
        "empty_recovery": _runtime_assembly.limits_from_defaults(
            config.agents.defaults,
        ),
        "context_window_tokens": 98_304,
        "max_concurrent_subagents": 3,
        "max_subagent_spawns_per_hour": 12,
        "brave_api_key": "brave-key",
        "jina_api_key": "jina-key",
        "web_proxy": "http://proxy.test",
        "exec_config": sentinel.exec_config,
        "cron_service": cron_service,
        "restrict_to_workspace": True,
        "session_manager": session_manager,
        "mcp_servers": {"docs": sentinel.mcp_server},
        "disabled_tools": ["write_file"],
        "tool_search_config": sentinel.tool_search_config,
        "sandbox_config": sentinel.sandbox_config,
        "channels_config": sentinel.channels_config,
        "router": router,
        "call_efficiency": sentinel.call_efficiency,
        "skill_forge_config": codeflow_config.skill_forge,
        "context_config": sentinel.context_config,
        "context_engine_factory": None,
        "runtime_config": sentinel.runtime_config,
        "interactive": interactive,
        "backend": backend,
        "memory_config": sentinel.memory_config,
        "skill_forge_router_config": sentinel.skill_forge_router_config,
        "plugin_tools": plugin_tools,
    }
    runtime.agent_loop.configure_personalization.assert_called_once_with(True)


def test_runtime_assembly_closes_call_efficiency_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from codeflow.cli import _runtime_assembly

    config, codeflow_config = _runtime_configs(tmp_path)
    controller = SimpleNamespace(close=MagicMock())
    monkeypatch.setattr(
        "codeflow.call_efficiency.CallEfficiency.from_config",
        lambda config, *, telemetry_dir, provider: controller,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        MagicMock(side_effect=RuntimeError("plugin failed")),
    )

    with pytest.raises(RuntimeError, match="plugin failed"):
        _runtime_assembly.assemble_runtime(
            config,
            codeflow_config,
            provider=object(),
            cron_service=object(),
            interactive=False,
        )

    controller.close.assert_called_once()


def test_runtime_assembly_preserves_construction_error_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from codeflow.cli import _runtime_assembly

    config, codeflow_config = _runtime_configs(tmp_path)
    controller = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("close failed")))
    monkeypatch.setattr(
        "codeflow.call_efficiency.CallEfficiency.from_config",
        lambda config, *, telemetry_dir, provider: controller,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        MagicMock(side_effect=RuntimeError("plugin failed")),
    )

    with pytest.raises(RuntimeError, match="plugin failed"):
        _runtime_assembly.assemble_runtime(
            config,
            codeflow_config,
            provider=object(),
            cron_service=object(),
            interactive=False,
        )

    controller.close.assert_called_once()


def test_runtime_assembly_forwards_context_engine_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from codeflow.cli import _runtime_assembly

    config, codeflow_config = _runtime_configs(tmp_path)
    captured: dict = {}

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
        lambda *args, **kwargs: [],
    )

    class _AgentLoopSpy:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.configure_personalization = MagicMock()

    monkeypatch.setattr("codeflow.agent.loop.AgentLoop", _AgentLoopSpy)

    _runtime_assembly.assemble_runtime(
        config,
        codeflow_config,
        provider=object(),
        cron_service=object(),
        interactive=False,
        session_manager=object(),
        context_engine_factory=sentinel.context_engine_factory,
    )

    assert captured["context_engine_factory"] is sentinel.context_engine_factory


def test_runtime_assembly_separates_workspace_from_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from codeflow.cli import _runtime_assembly
    from codeflow.config.paths import RuntimePaths

    config, codeflow_config = _runtime_configs(tmp_path / "configured")
    paths = RuntimePaths(
        workspace=tmp_path / "project",
        state=tmp_path / "project" / ".codeflow",
    )
    captured: dict = {}
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "codeflow.session.manager.SessionManager",
        lambda state: calls.append(("session", state)) or object(),
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.maybe_build_memory_backend",
        lambda workspace, *_args, **_kwargs: calls.append(("backend", workspace)) or None,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_tools",
        lambda workspace, *_args, **_kwargs: calls.append(("tools", workspace)) or [],
    )

    class _AgentLoopSpy:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.configure_personalization = MagicMock()

    monkeypatch.setattr("codeflow.agent.loop.AgentLoop", _AgentLoopSpy)

    _runtime_assembly.assemble_runtime(
        config,
        codeflow_config,
        provider=object(),
        cron_service=object(),
        interactive=True,
        paths=paths,
    )

    assert calls == [
        ("session", paths.state),
        ("backend", paths.workspace),
        ("tools", paths.workspace),
    ]
    assert captured["workspace"] == paths.workspace
    assert captured["state"] == paths.state


class _RetainedPluginTool(Tool):
    @property
    def name(self) -> str:
        return "retained_plugin_tool"

    @property
    def description(self) -> str:
        return "A retained plugin capability."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return "ok"


@pytest.mark.parametrize(
    ("host", "interactive", "router"),
    [
        ("cli", False, None),
        ("tui", True, None),
        ("gateway", True, sentinel.gateway_router),
    ],
)
async def test_runtime_hosts_keep_protected_tool_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    host: str,
    interactive: bool,
    router,
) -> None:
    from codeflow.cli._runtime_assembly import assemble_runtime
    from codeflow.config.codeflow import CodeFlowConfig
    from codeflow.config.schema import Config

    config = Config()
    config.agents.defaults.workspace = str(tmp_path / host)
    config.agents.defaults.model = "retained-model"
    config.tools.disabled_tools = ["write_file"]
    codeflow_config = CodeFlowConfig(base=config)
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        lambda cfg: object(),
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.maybe_build_memory_backend",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_tools",
        lambda *args, **kwargs: [_RetainedPluginTool()],
    )

    runtime = assemble_runtime(
        config,
        codeflow_config,
        provider=object(),
        cron_service=MagicMock(),
        interactive=interactive,
        router=router,
    )
    try:
        names = set(runtime.agent_loop.tools.tool_names)
        assert {
            "read_file",
            "edit_file",
            "exec",
            "web_search",
            "web_fetch",
            "message",
            "spawn",
            "ask_user",
            "cron",
            "retained_plugin_tool",
        } <= names
        assert "write_file" not in names
    finally:
        await runtime.close()


async def test_project_local_runtime_keeps_tools_and_state_in_their_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from codeflow.cli._runtime_assembly import assemble_runtime
    from codeflow.config.codeflow import CodeFlowConfig
    from codeflow.config.paths import RuntimePaths
    from codeflow.config.schema import Config

    project = tmp_path / "project"
    state = project / ".codeflow"
    project.mkdir()
    config = Config()
    config.agents.defaults.model = "retained-model"
    codeflow_config = CodeFlowConfig(base=config)
    codeflow_config.memory.backend = None
    paths = RuntimePaths(workspace=project, state=state)
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_registry",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        "codeflow.cli._plugin_stack.build_plugin_tools",
        lambda *args, **kwargs: [],
    )

    runtime = assemble_runtime(
        config,
        codeflow_config,
        provider=object(),
        cron_service=MagicMock(),
        interactive=True,
        paths=paths,
    )
    try:
        assert runtime.call_efficiency.mode == "observe"
        assert runtime.agent_loop.call_efficiency is runtime.call_efficiency
        assert runtime.agent_loop.subagents.provider is runtime.agent_loop.provider
        assert runtime.agent_loop.memory_consolidator.provider is runtime.agent_loop.provider
        read_file = runtime.agent_loop.tools.get("read_file")
        assert runtime.agent_loop.workspace == project
        assert runtime.agent_loop.state == state
        assert runtime.session_manager.workspace == state
        assert read_file._workspace == project
        assert runtime.agent_loop.context.memory.memory_file == state / "user_memory" / "profile" / "user.md"
        assert runtime.agent_loop.context.skills.registry.workspace_skills == state / "skills"
        assert runtime.agent_loop._checkpoint._git_dir == state / "shadow.git"
    finally:
        await runtime.close()


async def test_runtime_assembly_owns_memory_and_agent_resources_once() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    order: list[str] = []
    backend = SimpleNamespace(
        start=AsyncMock(side_effect=lambda: order.append("backend.start")),
        stop=AsyncMock(side_effect=lambda: order.append("backend.stop")),
    )
    agent_loop = SimpleNamespace(
        begin_close=MagicMock(),
        close=AsyncMock(side_effect=lambda: order.append("agent.close")),
    )
    call_efficiency = SimpleNamespace(close=MagicMock(side_effect=lambda: order.append("call_efficiency.close")))
    runtime = RuntimeAssembly(
        agent_loop=agent_loop,
        session_manager=object(),
        backend=backend,
        call_efficiency=call_efficiency,
    )

    assert await runtime.start_memory_backend() is True
    assert await runtime.start_memory_backend() is True
    await runtime.close()
    await runtime.close()

    assert order == ["backend.start", "agent.close", "call_efficiency.close", "backend.stop"]
    backend.start.assert_awaited_once()
    backend.stop.assert_awaited_once()
    agent_loop.close.assert_awaited_once()
    call_efficiency.close.assert_called_once()


async def test_runtime_assembly_serializes_concurrent_close() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    entered = asyncio.Event()
    release = asyncio.Event()
    close_calls = 0

    async def _close_agent() -> None:
        nonlocal close_calls
        close_calls += 1
        entered.set()
        await release.wait()

    backend = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    call_efficiency = SimpleNamespace(close=MagicMock())
    runtime = RuntimeAssembly(
        agent_loop=SimpleNamespace(begin_close=MagicMock(), close=_close_agent),
        session_manager=object(),
        backend=backend,
        call_efficiency=call_efficiency,
    )

    first = asyncio.create_task(runtime.close())
    await entered.wait()
    second = asyncio.create_task(runtime.close())
    asyncio.get_running_loop().call_soon(release.set)
    await asyncio.gather(first, second)

    assert close_calls == 1
    call_efficiency.close.assert_called_once()
    backend.stop.assert_awaited_once()


async def test_runtime_assembly_finishes_started_close_before_propagating_caller_cancellation() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    entered = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def _close_agent() -> None:
        entered.set()
        await release.wait()
        order.append("agent.close")

    runtime = RuntimeAssembly(
        agent_loop=SimpleNamespace(begin_close=MagicMock(), close=_close_agent),
        session_manager=object(),
        backend=SimpleNamespace(
            start=AsyncMock(),
            stop=AsyncMock(side_effect=lambda: order.append("backend.stop")),
        ),
        call_efficiency=SimpleNamespace(close=MagicMock(side_effect=lambda: order.append("call_efficiency.close"))),
    )

    shutdown = asyncio.create_task(runtime.close())
    await entered.wait()
    shutdown.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await shutdown

    assert order == ["agent.close", "call_efficiency.close", "backend.stop"]


async def test_tui_runtime_host_finishes_close_after_cancellation_during_build() -> None:
    from codeflow.cli._runtime_host import TuiRuntimeHost

    build_entered = asyncio.Event()
    build_release = asyncio.Event()
    runtime = SimpleNamespace(begin_close=MagicMock(), close=AsyncMock())
    host = TuiRuntimeHost(lambda: runtime)

    async def _build_runtime():
        build_entered.set()
        await build_release.wait()
        return runtime

    host._task = asyncio.create_task(_build_runtime())
    await build_entered.wait()
    shutdown = asyncio.create_task(host.close())
    asyncio.get_running_loop().call_soon(shutdown.cancel)
    asyncio.get_running_loop().call_soon(build_release.set)

    with pytest.raises(asyncio.CancelledError):
        await shutdown

    runtime.begin_close.assert_called_once()
    runtime.close.assert_awaited_once()
    assert host._closed is True


async def test_runtime_assembly_continues_backend_shutdown_after_call_efficiency_failure() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    backend = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    agent_loop = SimpleNamespace(begin_close=MagicMock(), close=AsyncMock())
    call_efficiency = SimpleNamespace(close=MagicMock(side_effect=RuntimeError("ledger failed")))
    runtime = RuntimeAssembly(
        agent_loop=agent_loop,
        session_manager=object(),
        backend=backend,
        call_efficiency=call_efficiency,
    )

    await runtime.close()
    await runtime.close()

    backend.stop.assert_awaited_once()
    agent_loop.close.assert_awaited_once()
    assert call_efficiency.close.call_count == 2


async def test_runtime_assembly_preserves_memory_start_failure() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    failure = RuntimeError("offline")
    backend = SimpleNamespace(
        start=AsyncMock(side_effect=failure),
        stop=AsyncMock(),
    )
    agent_loop = SimpleNamespace(begin_close=MagicMock(), close=AsyncMock())
    runtime = RuntimeAssembly(
        agent_loop=agent_loop,
        session_manager=object(),
        backend=backend,
    )

    with pytest.raises(RuntimeError, match="offline"):
        await runtime.start_memory_backend()
    with pytest.raises(RuntimeError, match="offline"):
        await runtime.start_memory_backend()
    await runtime.close()

    backend.start.assert_awaited_once()
    backend.stop.assert_awaited_once()


async def test_runtime_assembly_retries_agent_close_without_double_stopping_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeflow.cli import _runtime_assembly

    backend = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    agent_loop = SimpleNamespace(
        close=AsyncMock(
            side_effect=[RuntimeError("close failed"), None],
        ),
    )
    runtime = _runtime_assembly.RuntimeAssembly(
        agent_loop=SimpleNamespace(
            begin_close=MagicMock(),
            close=agent_loop.close,
        ),
        session_manager=object(),
        backend=backend,
    )
    log = MagicMock()
    monkeypatch.setattr(_runtime_assembly, "logger", log)

    await runtime.close()
    await runtime.close()

    agent_loop.close.assert_awaited()
    assert agent_loop.close.await_count == 2
    backend.stop.assert_awaited_once()
    log.exception.assert_called_once()


async def test_runtime_assembly_stops_memory_before_propagating_cancellation() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    backend = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    runtime = RuntimeAssembly(
        agent_loop=SimpleNamespace(
            begin_close=MagicMock(),
            close=AsyncMock(side_effect=asyncio.CancelledError),
        ),
        session_manager=object(),
        backend=backend,
    )

    with pytest.raises(asyncio.CancelledError):
        await runtime.close()

    backend.stop.assert_awaited_once()


async def test_runtime_assembly_exposes_memory_stop_failure_after_agent_close() -> None:
    from codeflow.cli._runtime_assembly import RuntimeAssembly

    order: list[str] = []

    async def _close_agent() -> None:
        order.append("agent.close")

    async def _stop_backend() -> None:
        order.append("backend.stop")
        raise RuntimeError("myna stop failed")

    runtime = RuntimeAssembly(
        agent_loop=SimpleNamespace(begin_close=MagicMock(), close=AsyncMock(side_effect=_close_agent)),
        session_manager=object(),
        backend=SimpleNamespace(start=AsyncMock(), stop=AsyncMock(side_effect=_stop_backend)),
    )

    with pytest.raises(RuntimeError, match="myna stop failed"):
        await runtime.close()

    assert order == ["agent.close", "backend.stop"]


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, session_key: str, event: dict) -> None:
        self.events.append((session_key, event))


class _RunnerLoop:
    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.tools = {}

    async def run_turn(self, req, emit, drain, *, stream, **kwargs):
        self.calls.append(stream)
        await emit(Text(content="retained", source=req.source))
        return TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)


@pytest.mark.parametrize(
    ("host", "expected_stream"),
    [
        ("cli", False),
        ("tui", True),
        ("gateway", False),
    ],
)
async def test_host_turn_runners_share_the_turn_runner_interface(
    host: str,
    expected_stream: bool,
) -> None:
    loop = _RunnerLoop()
    source = Source(
        channel=host,
        chat_id="chat",
        sender_id="user",
        chat_type=ChatType.DM,
    )
    request = TurnRequest(
        origin=Origin.USER,
        source=source,
        text="hello",
        conversation=f"{host}:chat",
    )
    emitter = _Emitter()
    if host == "cli":
        runner = AgentTurnRunner(loop, stream=False)
    elif host == "tui":
        runner = TuiTurnRunner(
            loop,
            emitter,
            {},
            {id(request): "turn-1"},
            {},
            submission_ids={id(request): "submission-1"},
        )
    else:
        runner = GatewayTurnRunner(loop, {}, {})
    events: list[Text] = []

    async def emit(event):
        events.append(event)

    assert isinstance(runner, TurnRunner)
    outcome = await runner.run(request, emit, lambda: [])

    assert loop.calls == [expected_stream]
    assert [event.content for event in events] == ["retained"]
    assert outcome == TurnOutcome(usage=Usage(1, 2, 3), explicit_reply=True)
