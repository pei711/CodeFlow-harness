"""Top-level ``gateway`` command.

Spawns the CodeFlow gateway: agent loop + channel manager + cron service.
The bulk of the wiring lives in this command body.

``commands.py`` registers it via :func:`register`.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import typer
from loguru import logger
from rich.console import Console

from codeflow import __logo__
from codeflow.cli._helpers import (
    load_runtime_config,
    make_provider,
    print_deprecated_memory_window_notice,
)
from codeflow.utils.helpers import sync_workspace_templates

console = Console()


def build_model_routing(config, provider):
    """Return ``(router, provider)`` for the configured routing backend.

    - ``knn``: build a :class:`KNNModelRouter` and wrap ``provider`` in a
      :class:`PerModelProvider` so routed model names reach their endpoints
      (other models fall back to ``provider`` unchanged).
    - ``ecoclaw``: build the PinchBench :class:`ModelRouter`.
    - routing disabled, or ecoclaw with no API key: return ``(None, provider)``.
    """
    if not config.routing.enabled:
        return None, provider

    if config.routing.backend == "knn":
        from codeflow.providers.per_model_provider import PerModelProvider
        from codeflow.routing.knn_router import KNNModelRouter

        router = KNNModelRouter(config.routing, default_model=config.agents.defaults.model)
        return router, PerModelProvider(config.routing.models, fallback=provider)

    from codeflow.routing.router import ModelRouter

    api_key = config.routing.api_key or config.providers.openrouter.api_key or ""
    if not api_key:
        console.print("[yellow]⚠[/yellow] Routing enabled but no OpenRouter API key found — routing disabled")
        return None, provider

    from codeflow.routing.types import RoutingProfileName

    profile: RoutingProfileName = config.routing.profile  # type: ignore[assignment]
    router = ModelRouter(api_key=api_key, profile=profile, fallback_model=config.agents.defaults.model)
    return router, provider


_GATEWAY_IM_CHANNELS: tuple[str, ...] = (
    "feishu",
    "qq",
    "wecom",
)


def _build_gateway_channels(config) -> set[str]:
    """Build the ``allowed_channels`` set used by gateway's ``CronService`` — the
    enabled IM channels only.

    The gateway owns cron jobs for its IM channels. It does NOT claim ephemeral
    ``tui``/``cli`` jobs: those are fired by the interactive process that created
    them (the TUI / ``codeflow run`` session), so a TUI-set reminder always
    delivers to the TUI rather than racing the gateway and being forwarded to an
    IM channel. The trade-off is no cross-process fallback while that process is
    down; restoring "fire at origin, hand off only after the origin exits" is a
    deferred cron-delivery-ownership design, not this set.
    """
    return {name for name in _GATEWAY_IM_CHANNELS if getattr(getattr(config.channels, name, None), "enabled", False)}


async def _health_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer any request with a 200 ``{"status":"ok"}`` liveness body."""
    try:
        await reader.readline()
        body = b'{"status":"ok"}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: %d\r\nConnection: close\r\n\r\n%b" % (len(body), body)
        )
        await writer.drain()
    finally:
        writer.close()


async def _cleanup_gateway(
    *,
    run_error: BaseException | None,
    health_server: Any | None,
    cron: Any,
    question_broker: Any | None,
    channels: Any,
    gw_teardown: Callable[[], object] | None,
    agent: Any,
    runtime: Any,
) -> None:
    """Attempt every cleanup step, then raise the highest-priority failure."""
    first_error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None

    async def attempt(name: str, cleanup: Callable[[], object]) -> None:
        nonlocal cancellation, first_error
        try:
            result = cleanup()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError as exc:
            logger.opt(exception=exc).error("Gateway cleanup step {} was cancelled", name)
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            logger.opt(exception=exc).error("Gateway cleanup step {} failed", name)
            if first_error is None:
                first_error = exc

    if health_server is not None:
        await attempt("health server", health_server.close)
    await attempt("cron", cron.stop)
    if question_broker is not None:
        await attempt("question broker", question_broker.cancel_all)
    await attempt("channel intake", channels.quiesce_intake)
    if gw_teardown is not None:
        await attempt("spine", gw_teardown)
    await attempt("channel transports", channels.stop_all)
    await attempt("agent", agent.stop)
    await attempt("runtime", runtime.close)
    # 运行失败说明网关为何退出；清理失败会记录日志，且只有运行本身成功时才成为最终错误。
    if run_error is not None:
        raise run_error
    if cancellation is not None:
        raise cancellation
    if first_error is not None:
        raise first_error


def register(app: typer.Typer) -> None:
    """Attach the ``gateway`` command to ``app``."""

    @app.command()
    def gateway(
        port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
        workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
        config: str | None = typer.Option(None, "--config", help="Path to config file"),
    ):
        """Start the CodeFlow gateway."""
        from codeflow.channels.manager import ChannelManager
        from codeflow.cli._runtime_assembly import assemble_runtime
        from codeflow.config.paths import get_cron_dir, resolve_service_paths
        from codeflow.config.codeflow import load_codeflow_config
        from codeflow.proactive_engine.schedulers.cron.service import CronService

        # load_runtime_config 必须最先运行：它调用 set_config_path()，使后续 load_codeflow_config()
        # 从 --config 而非默认 ~/.codeflow/config.json 读取。否则 --config 中的 skill_forge 会被静默忽略。
        config = load_runtime_config(config, workspace)
        paths = resolve_service_paths(config)

        from codeflow.cli._log_file import redirect_loguru_to_file

        log_cfg = config.gateway.log
        log_path = redirect_loguru_to_file(
            "gateway.log",
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            file_level="DEBUG" if verbose else log_cfg.level,
            terminal_level="DEBUG" if verbose else log_cfg.console_level,
        )

        from codeflow.cli._gateway_lock import GatewayAlreadyRunningError, acquire

        # 整个进程期间持有；关闭该句柄或垃圾回收时释放锁。
        try:
            _lock_handle = acquire(now=time.time())
        except GatewayAlreadyRunningError as exc:
            since = datetime.fromtimestamp(exc.info.started_at).strftime("%Y-%m-%d %H:%M:%S")
            console.print(
                f"[red]✗[/red] CodeFlow gateway already running for this instance "
                f"(pid {exc.info.pid}, since {since}).\n"
                f"  Stop it first, or use --config to run a separate instance."
            )
            raise typer.Exit(code=1)

        ec_config = load_codeflow_config()
        print_deprecated_memory_window_notice(config)
        port = port if port is not None else config.gateway.port

        console.print(f"{__logo__} Starting CodeFlow gateway on port {port}...")
        console.print(f"[dim]📝 Logs → {log_path}[/dim]")
        sync_workspace_templates(paths.state)
        provider = make_provider(config)

        # 先创建 cron 服务，智能体创建后再设置回调。
        #
        # 限制为网关拥有适配器的渠道，防止网关与 REPL 竞速并抢走来源为 CLI 的提醒；REPL 能
        # 投递这些提醒，而网关不能（REPL 标准输出归 REPL 进程所有，网关没有 CLI 渠道）。
        # 若不限制，两者同时运行时会出现 "Unknown channel: cli" 警告并丢失 REPL 提醒。
        cron_store_path = get_cron_dir() / "jobs.json"
        gateway_channels = _build_gateway_channels(config)
        cron = CronService(cron_store_path, allowed_channels=gateway_channels)

        # 创建模型路由器；使用 knn 后端时还要包装提供商。
        router, provider = build_model_routing(config, provider)

        # 在运行时之前构造渠道，使损坏的适配器在插件资源被领取前禁用，且永不成为致命错误。
        channels = ChannelManager(config)

        runtime = assemble_runtime(
            config,
            ec_config,
            provider=provider,
            cron_service=cron,
            router=router,
            interactive=True,
            paths=paths,
        )
        agent = runtime.agent_loop
        session_manager = runtime.session_manager

        from codeflow.cli._cron_handler import make_on_cron_job

        if channels.enabled_channels:
            console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
        else:
            console.print("[yellow]Warning: No channels enabled[/yellow]")

        cron_status = cron.status()
        if cron_status["jobs"] > 0:
            console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

        async def run():
            health_server = None
            gw_teardown = None
            question_broker = None
            run_error: BaseException | None = None
            try:
                await runtime.start_memory_backend()
                # 为网关主机来源装配 Spine：cron 通过它提交，回复经逐渠道出口路由到渠道。必须在此处
                # 正在运行的事件循环内构建，而非同步命令序言；Scheduler 构造时会固定所属循环，
                # submit 必须来自该循环，而序言尚无事件循环。
                from codeflow.cli._gateway_spine import build_gateway

                gw_scheduler, gw_hub, gw_readback_texts, gw_sources, gw_teardown = build_gateway(
                    agent,
                    channels.channels,
                    user_pool=config.gateway.user_pool,
                    system_pool=config.gateway.system_pool,
                    send_max_retries=config.gateway.send_max_retries,
                )
                cron.on_job = make_on_cron_job(
                    gw_hub,
                    submit=gw_scheduler.submit,
                    readback_texts=gw_readback_texts,
                    channel_manager=channels,
                    session_manager=session_manager,
                    default_channel="cli",
                )

                # 子智能体结果回注会提交来源为 SUBAGENT 的轮次。
                agent.subagents.set_submit(gw_scheduler.submit)

                # 渠道侧的 ask_user 往返：QuestionBroker 把智能体的 clarify.request 渲染为发往会话渠道的
                # 出站 Text；下方入站门通过 reply() 路由用户下一条消息。问题在轮次中途发出，因此当前
                # 轮次的真实入站 Source 仍在以会话 ID 为键的 gw_sources 中；复用它可精确保留话题/
                # 线程地址，而无需从会话 ID 重建。
                from codeflow.spine import Text as _Text
                from codeflow.tui_rpc.question_broker import QuestionBroker

                async def _question_to_channel(frame: dict) -> None:
                    params = frame.get("params", {})
                    qcid = params.get("conversation_id", "")
                    source = gw_sources.get(qcid)
                    if source is None:
                        logger.warning(
                            "ask_user question for {} has no live source — dropping",
                            qcid,
                        )
                        return
                    body = params.get("question", "")
                    choices = params.get("choices") or []
                    if choices:
                        body += "\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices))
                    await gw_hub.dispatch(_Text(content=body, source=source))

                question_broker = QuestionBroker(send_frame=_question_to_channel)
                if (ask_tool := agent.tools.get("ask_user")) is not None and hasattr(ask_tool, "set_broker"):
                    ask_tool.set_broker(question_broker)

                # 渠道入站通过 spine：获准消息提交为 USER 轮次。/stop 和 /restart 是控制命令
                # （总线排空器的职责），在此拦截而不提交为轮次，否则智能体会回复命令文本。cid 与
                # 通道键（conversation 或 channel:chat_id）一致，也是总线路径 _handle_stop 使用的会话键。
                from dataclasses import replace

                from codeflow.spine import Text
                from codeflow.spine.turn import BusyPolicy

                async def _inbound_dispatch(req) -> None:
                    cmd = req.text.strip().lower()
                    cid = req.conversation or f"{req.source.channel}:{req.source.chat_id}"
                    if cmd == "/stop":
                        stopped = gw_scheduler.cancel_conversation(cid)
                        stopped += await agent.subagents.cancel_by_session(cid)
                        content = f"Stopped {stopped} task(s)." if stopped else "No active task to stop."
                        await gw_hub.dispatch(Text(content=content, source=req.source))
                    elif cmd == "/restart":
                        await gw_hub.dispatch(Text(content="Restarting...", source=req.source))

                        async def _do_restart() -> None:
                            import os
                            import sys

                            await asyncio.sleep(1)
                            os.execv(sys.executable, [sys.executable] + sys.argv)

                        asyncio.create_task(_do_restart())
                    elif question_broker.pending_req(cid) is not None:
                        # 当前会话阻塞在 ask_user 问题上；把答案路由到代理器以解决等待中的工具，
                        # 而不是启动或注入新轮次。
                        question_broker.reply(cid, req.text)
                    elif gw_scheduler.has_inflight(cid):
                        # 当前会话已有轮次在运行；以 BusyPolicy.INJECT 提交，使循环在下一次迭代合并该消息，
                        # 而不是排队新轮次。
                        gw_scheduler.submit(replace(req, busy=BusyPolicy.INJECT))
                    else:
                        gw_scheduler.submit(req)  # 发出后不等待，也不读回

                for _ch in channels.channels.values():
                    _ch.intake.set_submit(_inbound_dispatch)

                await cron.start()
                try:
                    health_server = await asyncio.start_server(_health_handler, "127.0.0.1", port)
                    console.print(f"[green]✓[/green] Health: http://127.0.0.1:{port}/health")
                except OSError as exc:
                    logger.warning(
                        "health endpoint unavailable on 127.0.0.1:{} ({}); gateway continues without it",
                        port,
                        exc,
                    )
                coros = [
                    agent.run(),
                    channels.start_all(),
                ]
                if health_server is not None:
                    coros.append(health_server.serve_forever())
                await asyncio.gather(*coros)
            except KeyboardInterrupt:
                console.print("\nShutting down...")
            except BaseException as exc:
                run_error = exc
            await _cleanup_gateway(
                run_error=run_error,
                health_server=health_server,
                cron=cron,
                question_broker=question_broker,
                channels=channels,
                gw_teardown=gw_teardown,
                agent=agent,
                runtime=runtime,
            )

        asyncio.run(run())


__all__ = ["register"]
