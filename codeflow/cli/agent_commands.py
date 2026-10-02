"""Top-level ``run`` command + its dedicated helpers.

This module owns:

- The interactive ``codeflow run`` REPL command body (multiline paste,
  history, agent-loop wiring).
- A small bundle of helpers used only by that command: prompt-toolkit
  session init, terminal restore, TTY-flush, response rendering, exit
  detection.

``commands.py`` registers the command via :func:`register`.
"""

from __future__ import annotations

import asyncio
import signal
import sys

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from codeflow import __logo__
from codeflow.cli._helpers import (
    load_runtime_config,
    make_provider,
    print_deprecated_memory_window_notice,
    warn_about_pending_cli_reminders,
)
from codeflow.utils.helpers import sync_workspace_templates

console = Console()


# ---------------------------------------------------------------------------
# 模块级状态（仅交互式 REPL）
# ---------------------------------------------------------------------------

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI 输入：prompt_toolkit 负责编辑、粘贴、历史和显示
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # 原始 termios 设置，退出时恢复


# ---------------------------------------------------------------------------
# 辅助方法（模块私有）
# ---------------------------------------------------------------------------


def _stdout_isatty() -> bool:
    """Whether stdout is an interactive TTY (seam for the onboarding gate test;
    CliRunner swaps ``sys.stdout`` for a non-TTY buffer)."""
    return sys.stdout.isatty()


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios

        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # 保存终端状态，以便退出时恢复。
    try:
        import termios

        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    from codeflow.config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,  # 单行模式下按 Enter 提交
    )


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} CodeFlow[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        # raw=True 会原样传递 ANSI 转义序列。否则，当用户停留在提示符时，后台 Cron 协程输出
        # Rich 样式内容会破坏 ESC 字节，显示成 ?[36m...?[0m 乱码。
        with patch_stdout(raw=True):
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    """Attach the ``run`` command to ``app``."""

    @app.command("run")
    def agent(
        message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
        session_id: str | None = typer.Option(
            None,
            "--session",
            "-s",
            help=(
                "Full session key (channel:chat_id), any channel. By default "
                "a fresh cli session is minted per invocation. The legacy "
                "'direct' session remains reachable via --resume direct."
            ),
        ),
        continue_: bool = typer.Option(False, "--continue", "-c", help="Continue the most recent cli session"),
        resume: str | None = typer.Option(None, "--resume", "-r", help="Resume session by bare id or unique prefix"),
        workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
        config: str | None = typer.Option(None, "--config", help="Config file path"),
        markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
        logs: bool = typer.Option(False, "--logs/--no-logs", help="Show CodeFlow runtime logs during chat"),
    ):
        """Interact with the agent directly."""
        if sum((session_id is not None, continue_, resume is not None)) > 1:
            raise typer.BadParameter("--session, --continue and --resume are mutually exclusive")

        # 启动门：缺少必要配置（提供商密钥和默认模型）时先运行引导向导。仅限交互式 TTY；
        # 脚本单次调用（`-m`）和非 TTY 管道应在后续明确失败，而不是阻塞在提示上。
        from codeflow.cli.onboard_commands import _is_config_populated

        if message is None and _stdout_isatty() and not _is_config_populated():
            from codeflow.cli.onboard_commands import ensure_configured_or_onboard

            ensure_configured_or_onboard()

        from loguru import logger

        from codeflow.cli._cron_handler import make_on_cron_job
        from codeflow.cli._runtime_assembly import assemble_runtime
        from codeflow.config.paths import get_cron_dir, resolve_foreground_paths
        from codeflow.config.codeflow import load_codeflow_config
        from codeflow.proactive_engine.schedulers.cron.service import CronService
        from codeflow.session.manager import SessionManager, new_chat_id

        # load_runtime_config 必须最先运行：它调用 set_config_path()，使之后的 load_codeflow_config()
        # 从 --config 而非默认 ~/.codeflow/config.json 读取。否则 --config 中的 skill_forge 会被静默忽略。
        config = load_runtime_config(config, workspace)
        paths = resolve_foreground_paths(config, workspace=workspace)
        ec_config = load_codeflow_config()
        print_deprecated_memory_window_notice(config)
        sync_workspace_templates(paths.state)

        provider = make_provider(config)
        session_manager = SessionManager(paths.state)

        # 默认新建会话，避免独立的单次调用相互污染。
        if resume is not None:
            from codeflow.cli.session_commands import resolve_session

            session_id = resolve_session(session_manager, resume)
        elif continue_:
            recent = session_manager.find_most_recent_chat_id("cli")
            if recent is None:
                console.print("[dim]no previous cli session — starting fresh[/dim]")
                recent = new_chat_id()
            session_id = f"cli:{recent}"
        elif session_id is None:
            session_id = f"cli:{new_chat_id()}"
        else:
            from codeflow.cli.session_commands import resolve_session_cross_channel

            session_id = resolve_session_cross_channel(session_manager, session_id)

        # 创建 cron 服务（智能体就绪后在下方设置回调）。allowed_channels={"cli"} 防止该 REPL
        # 领取消息渠道中创建的提醒；这些提醒应由已连接真实渠道适配器的网关投递。
        cron_store_path = get_cron_dir() / "jobs.json"
        cron = CronService(cron_store_path, allowed_channels={"cli"})

        if logs:
            logger.enable("codeflow")
        else:
            logger.disable("codeflow")

        runtime = assemble_runtime(
            config,
            ec_config,
            provider=provider,
            cron_service=cron,
            interactive=message is None,
            session_manager=session_manager,
            paths=paths,
        )
        agent_loop = runtime.agent_loop
        # REPL 没有真实 ChannelManager——提供最小垫片，仅报告 "cli" 已启用，使 CLI 提醒走
        # 直通路径（经 Spine 出口投递到 REPL 标准输出）。
        from types import SimpleNamespace

        cli_shim = SimpleNamespace(enabled_channels=["cli"])
        # spine 调度器就绪后，run_interactive 内才连接 cron.on_job；cron 提醒通过它提交 CRON 轮次。

        # 日志关闭时显示转圈动画（不会遮掉输出）；日志开启时跳过。
        def _thinking_ctx():
            if logs:
                from contextlib import nullcontext

                return nullcontext()
            # 动画转圈可安全配合 prompt_toolkit 输入处理使用。
            return console.status("[dim]CodeFlow is thinking...[/dim]", spinner="dots")

        if message:
            # 单消息模式——一个 USER 轮次通过 spine（submit -> lane -> run_turn -> hub ->
            # CliOutlet），并使用旧有 cli/direct 默认值（channel="cli"、chat_id="direct"、
            # session_key=session_id）。进度由 CliOutlet 渲染，并受总线路径沿用的两个配置开关
            # （send_progress / send_tool_hints）控制。
            from codeflow.cli._repl_spine import build_repl
            from codeflow.spine import ChatType, Origin, Source, TurnRequest

            async def run_once():
                teardown = None
                try:
                    await runtime.start_memory_backend()
                    # 在运行中的事件循环内构建：Scheduler 会在 __init__ 中固定所属循环，因此
                    # build_repl 不能在同步序言中运行。
                    ch = agent_loop.channels_config
                    scheduler, hub, teardown = build_repl(
                        agent_loop,
                        "cli",
                        lambda t: _print_agent_response(t, render_markdown=markdown),
                        render_notice=lambda c: console.print(f"  [dim]↳ {c}[/dim]"),
                        render_error=lambda c: console.print(f"[red]{c}[/red]"),
                        send_progress=bool(ch.send_progress) if ch else False,
                        send_tool_hints=bool(ch.send_tool_hints) if ch else False,
                    )
                    # 单次 spawn 很少能在下方硬退出前完成（与总线路径相同），但仍连接 submit，
                    # 使行为与 REPL/TUI 一致。
                    agent_loop.subagents.set_submit(scheduler.submit)
                    with _thinking_ctx():
                        handle = scheduler.submit(
                            TurnRequest(
                                origin=Origin.USER,
                                source=Source(
                                    channel="cli",
                                    chat_id="direct",
                                    sender_id="user",
                                    chat_type=ChatType.DM,
                                ),
                                text=message,
                                conversation=session_id,
                            )
                        )
                        outcome = await handle.result()
                    await hub.wait_idle("cli")  # 渲染屏障：等待 CliOutlet 追上
                    return outcome is not None
                finally:
                    try:
                        if teardown is not None:
                            await teardown()
                    finally:
                        await runtime.close()

            if not asyncio.run(run_once()):
                raise typer.Exit(1)
            # 智能体循环加载的原生运行时（lancedb 的 Rust/tokio 线程、torch）会在解释器收尾时
            # 段错误。该风险存在时，codeflow.cli.commands.run 的退出汇合点会硬退出、跳过收尾，
            # 因此此路径只需正常返回。
        else:
            # 交互模式——用户轮次通过 spine（submit -> lane -> hub -> CliOutlet）运行；
            # Cron 轮次使用同一 spine 和 hub。
            from codeflow.cli._repl_spine import build_repl, run_repl_loop

            _init_prompt_session()
            console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

            if ":" in session_id:
                cli_channel, cli_chat_id = session_id.split(":", 1)
            else:
                cli_channel, cli_chat_id = "cli", session_id

            def _handle_signal(signum, frame):
                sig_name = signal.Signals(signum).name
                _restore_terminal()
                console.print(f"\nReceived {sig_name}, goodbye!")
                sys.exit(0)

            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
            # Windows 不支持 SIGHUP。
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, _handle_signal)
            # 忽略 SIGPIPE，防止写入已关闭管道时进程静默终止；Windows 不支持 SIGPIPE。
            if hasattr(signal, "SIGPIPE"):
                signal.signal(signal.SIGPIPE, signal.SIG_IGN)

            async def run_interactive():
                runtime_task = None
                teardown = None
                try:
                    await runtime.start_memory_backend()
                    # 启动 cron 前先构建 spine：cron 任务通过该调度器提交 CRON 轮次，且 on_job 必须在
                    # cron.start() 前连接，确保立即触发的任务已有回调。Scheduler 在此固定所属事件循环
                    # （run_interactive 为异步）；不能移到同步序言。
                    _ch = agent_loop.channels_config
                    scheduler, hub, teardown = build_repl(
                        agent_loop,
                        cli_channel,
                        lambda t: _print_agent_response(t, render_markdown=markdown),
                        render_notice=lambda c: console.print(f"  [dim]↳ {c}[/dim]"),
                        render_error=lambda c: console.print(f"[red]{c}[/red]"),
                        send_progress=bool(_ch.send_progress) if _ch else False,
                        send_tool_hints=bool(_ch.send_tool_hints) if _ch else False,
                    )
                    # 子智能体结果回注会提交来源为 SUBAGENT 的轮次。
                    agent_loop.subagents.set_submit(scheduler.submit)
                    # Cron 提醒以来源为 CRON 的轮次通过 spine 调度器运行，并由 hub -> CliOutlet 投递。
                    cron.on_job = make_on_cron_job(
                        hub,
                        submit=scheduler.submit,
                        channel_manager=cli_shim,
                        session_manager=session_manager,
                        default_channel="cli",
                    )
                    # 启动 cron，使已调度提醒（“一分钟后提醒我”）真正触发。过去 REPL 虽创建 CronService，
                    # 却未启动其 tick 循环，任务只会停留在 jobs.json 中。
                    await cron.start()

                    # 仅在同步装配成功后启动保活任务。先让出一次执行权，避免 stop() 在 run() 建立运行状态前执行。
                    runtime_task = asyncio.create_task(agent_loop.run())
                    await asyncio.sleep(0)

                    def _on_exit() -> None:
                        _restore_terminal()
                        console.print("\nGoodbye!")

                    def _slash(command: str) -> bool:
                        from codeflow.cli._repl_slash import handle_repl_slash

                        return handle_repl_slash(command, console=console)

                    await run_repl_loop(
                        read_input=_read_interactive_input_async,
                        submit=scheduler.submit,
                        wait_idle=hub.wait_idle,
                        channel=cli_channel,
                        chat_id=cli_chat_id,
                        is_exit=_is_exit_command,
                        handle_slash=_slash,
                        thinking=_thinking_ctx,
                        on_exit=_on_exit,
                    )
                finally:
                    try:
                        cron.stop()
                        agent_loop.stop()
                        if teardown is not None:
                            await teardown()
                        if runtime_task is not None:
                            await asyncio.gather(runtime_task, return_exceptions=True)
                    finally:
                        await runtime.close()
                    warn_about_pending_cli_reminders(cron, config)

            asyncio.run(run_interactive())


__all__ = ["register"]
