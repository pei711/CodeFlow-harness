"""Shared construction and resource ownership for the three Runtime hosts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from codeflow.agent.loop.recovery import limits_from_defaults
from codeflow.spine._barrier import finish_barrier

if TYPE_CHECKING:
    from codeflow.agent.loop import AgentLoop
    from codeflow.call_efficiency import CallEfficiency
    from codeflow.config.paths import RuntimePaths
    from codeflow.config.codeflow import CodeFlowConfig
    from codeflow.config.schema import Config
    from codeflow.context_engine.factory import ContextEngineFactory
    from codeflow.memory_engine import MemoryBackend
    from codeflow.session.manager import SessionManager


@dataclass
class RuntimeAssembly:
    agent_loop: AgentLoop
    session_manager: SessionManager
    backend: MemoryBackend | None
    call_efficiency: CallEfficiency | None = None
    _backend_start_attempted: bool = field(default=False, init=False)
    _backend_started: bool = field(default=False, init=False)
    _backend_start_error: BaseException | None = field(default=None, init=False)
    _agent_closed: bool = field(default=False, init=False)
    _call_efficiency_closed: bool = field(default=False, init=False)
    _backend_stopped: bool = field(default=False, init=False)
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def start_memory_backend(self) -> bool:
        if self.backend is None:
            return True
        if self._backend_start_attempted:
            if self._backend_start_error is not None:
                raise self._backend_start_error
            return self._backend_started
        self._backend_start_attempted = True
        try:
            await self.backend.start()
        except BaseException as exc:
            self._backend_start_error = exc
            raise
        self._backend_started = True
        return True

    def begin_close(self) -> None:
        self.agent_loop.begin_close()

    async def close(self) -> None:
        async with self._close_lock:
            self.begin_close()
            close_task = asyncio.create_task(self._close_once())
            await finish_barrier(close_task)

    async def _close_once(self) -> None:
        cancellation: BaseException | None = None
        if not self._agent_closed:
            try:
                await self.agent_loop.close()
            except Exception:
                logger.exception(
                    "agent runtime close failed; continuing shutdown",
                )
            except BaseException as exc:
                cancellation = exc
            else:
                self._agent_closed = True

        if self.call_efficiency is None:
            self._call_efficiency_closed = True
        elif not self._call_efficiency_closed:
            try:
                await asyncio.to_thread(self.call_efficiency.close)
            except Exception:
                logger.exception("call efficiency close failed; continuing shutdown")
            except BaseException as exc:
                cancellation = cancellation or exc
            else:
                self._call_efficiency_closed = True

        backend_error: BaseException | None = None
        if self.backend is None:
            self._backend_stopped = True
        elif not self._backend_stopped:
            try:
                await self.backend.stop()
            except BaseException as exc:
                logger.exception(
                    "memory backend stop failed after agent shutdown",
                )
                backend_error = exc
            else:
                self._backend_stopped = True

        if cancellation is not None:
            raise cancellation
        if backend_error is not None:
            raise backend_error


def assemble_runtime(
    config: Config,
    codeflow_config: CodeFlowConfig,
    *,
    provider: Any,
    cron_service: Any,
    interactive: bool,
    router: Any = None,
    session_manager: SessionManager | None = None,
    context_engine_factory: "ContextEngineFactory | None" = None,
    paths: "RuntimePaths | None" = None,
) -> RuntimeAssembly:
    from codeflow.agent.loop import AgentLoop
    from codeflow.call_efficiency import CallEfficiency, CallEfficiencyProvider
    from codeflow.cli._plugin_stack import (
        build_plugin_registry,
        build_plugin_tools,
        maybe_build_memory_backend,
    )
    from codeflow.config.paths import RuntimePaths
    from codeflow.session.manager import SessionManager

    paths = paths or RuntimePaths(
        workspace=config.workspace_path,
        state=config.workspace_path,
    )
    if session_manager is None:
        session_manager = SessionManager(paths.state)
    call_efficiency = CallEfficiency.from_config(
        codeflow_config.call_efficiency,
        telemetry_dir=paths.state / "telemetry",
        provider=provider,
    )
    runtime_provider = CallEfficiencyProvider(provider, call_efficiency)
    try:
        plugin_registry = build_plugin_registry(codeflow_config)
        backend = maybe_build_memory_backend(
            paths.workspace,
            codeflow_config,
            registry=plugin_registry,
        )
        plugin_tools = build_plugin_tools(
            paths.workspace,
            codeflow_config,
            registry=plugin_registry,
        )
        defaults = config.agents.defaults
        agent_loop = AgentLoop(
            provider=runtime_provider,
            workspace=paths.workspace,
            state=paths.state,
            model=defaults.model,
            max_iterations=defaults.max_tool_iterations,
            empty_recovery=limits_from_defaults(defaults),
            context_window_tokens=defaults.context_window_tokens,
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            max_subagent_spawns_per_hour=defaults.max_subagent_spawns_per_hour,
            brave_api_key=config.tools.web.search.api_key or None,
            jina_api_key=config.tools.web.jina_api_key or None,
            web_proxy=config.tools.web.proxy or None,
            exec_config=config.tools.exec,
            cron_service=cron_service,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            session_manager=session_manager,
            mcp_servers=config.tools.mcp_servers,
            disabled_tools=config.tools.disabled_tools,
            tool_search_config=config.tools.tool_search,
            sandbox_config=config.tools.sandbox,
            channels_config=config.channels,
            router=router,
            call_efficiency=call_efficiency,
            skill_forge_config=codeflow_config.skill_forge,
            context_config=codeflow_config.context,
            context_engine_factory=context_engine_factory,
            runtime_config=codeflow_config.runtime,
            interactive=interactive,
            backend=backend,
            memory_config=codeflow_config.memory,
            skill_forge_router_config=codeflow_config.skill_forge.router,
            plugin_tools=plugin_tools,
        )
        agent_loop.configure_personalization(
            defaults.enable_personalization,
        )
    except BaseException:
        try:
            call_efficiency.close()
        except Exception:
            logger.exception("call efficiency close failed during runtime assembly cleanup")
        raise
    return RuntimeAssembly(
        agent_loop=agent_loop,
        session_manager=session_manager,
        backend=backend,
        call_efficiency=call_efficiency,
    )


__all__ = ["RuntimeAssembly", "assemble_runtime"]
