"""Native TUI launcher for the bare ``codeflow`` entry point.

Interactive launches connect the Node child to an authenticated loopback
socket and run ``RpcServer`` in an asyncio task. Diagnostic launches inherit
stdio and skip RPC.

Exit codes (in addition to bootstrap's 0/1/2):
    3  — RPC handshake timeout / failure
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import typer

from codeflow.cli._log_file import _strip_tty_stream_handlers, redirect_loguru_to_file, redirect_terminal_fds_to_file
from codeflow.product import get_product_home
from codeflow.spine._barrier import finish_barrier

# 相对于本文件的 ui-tui/ 源码树路径：
# codeflow/cli/tui_commands.py -> ../../../ui-tui/。仅 `--dev` 路径（从源码运行 tsx）需要它；
# 该路径依赖 src/ 和 node_modules，wheel 中不存在。
_UI_TUI_DIR = Path(__file__).resolve().parent.parent.parent / "ui-tui"

# 已安装 wheel 内预构建、自包含包的位置：codeflow/cli/tui_commands.py ->
# ../ui-tui/dist/entry.js（即 codeflow/ui-tui/dist/entry.js）。pyproject 会强制把
# ui-tui/dist 包含到这里，使 `uv tool install` 无需源码检出也能附带 TUI。
_PACKAGED_DIST_ENTRY = Path(__file__).resolve().parent.parent / "ui-tui" / "dist" / "entry.js"

_MIN_NODE_VERSION = (22, 0, 0)


def resolve_dist_entry() -> Optional[Path]:
    """Locate the prebuilt ``entry.js`` bundle for production (non-dev) launch.

    Tries, in order:
      1. The packaged copy inside the installed wheel (``codeflow/ui-tui/dist``).
      2. The source-tree copy a developer built locally (``ui-tui/dist``).

    The bundle is self-contained (esbuild ``bundle: true``), so no sibling
    ``node_modules`` is needed — only a Node runtime. Returns the first path
    that exists, or ``None`` if neither does.
    """
    for candidate in (_PACKAGED_DIST_ENTRY, _UI_TUI_DIR / "dist" / "entry.js"):
        if candidate.exists():
            return candidate
    return None


def _stdout_isatty() -> bool:
    """Whether stdout is an interactive TTY (seam for the onboarding gate test;
    CliRunner swaps ``sys.stdout`` for a non-TTY buffer)."""
    return sys.stdout.isatty()


def find_node() -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
    """Find a usable node executable (>= 22).

    Returns (path, version_tuple) or (None, None) if not found.
    """
    # 优先级 1：CODEFLOW_NODE 环境变量——显式覆盖，不回退。用户设置 CODEFLOW_NODE 即表示强制
    # 使用指定二进制；若它缺失或不可用，绝不能静默回退到虚拟环境或 PATH，否则会掩盖错误配置。
    candidates: list[str] = []
    if env_node := os.environ.get("CODEFLOW_NODE"):
        candidates.append(env_node)
    else:
        # 优先级 2：当前虚拟环境
        if venv := os.environ.get("VIRTUAL_ENV"):
            if sys.platform == "win32":
                candidates.append(str(Path(venv) / "Scripts" / "node.exe"))
            else:
                candidates.append(str(Path(venv) / "bin" / "node"))

        # 优先级 3：PATH
        if path_node := shutil.which("node"):
            candidates.append(path_node)

        # 优先级 4：一行安装器安装到 ~/.codeflow/runtime/ 的 CodeFlow 私有运行时。这是零配置回退，
        # 即使用户没有系统 Node，安装器在此配置私有 Node 后仍可正常运行 `codeflow`。使用 glob
        # 以兼容带版本号的目录名。磁盘布局因操作系统而异：POSIX 压缩包把二进制放在 bin/
        # 下（node-v22.x.y-darwin-arm64/bin/node），Windows zip 则把 node.exe 放在顶层
        # （node-v22.x.y-win-x64/node.exe）；install.ps1 配置的是后者。
        runtime_root = get_product_home() / "runtime"
        if runtime_root.is_dir():
            if sys.platform == "win32":
                direct = runtime_root / "node" / "node.exe"
                if direct.exists():
                    candidates.append(str(direct))
                candidates.extend(str(p) for p in sorted(runtime_root.glob("node-*/node.exe")))
            else:
                direct = runtime_root / "node" / "bin" / "node"
                if direct.exists():
                    candidates.append(str(direct))
                candidates.extend(str(p) for p in sorted(runtime_root.glob("node-*/bin/node")))

    for node_path in candidates:
        if not Path(node_path).exists():
            continue
        try:
            proc = subprocess.run(
                [node_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            match = re.match(r"v(\d+)\.(\d+)\.(\d+)", proc.stdout.strip())
            if not match:
                continue
            version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return (node_path, version)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            continue

    return (None, None)


def run_subprocess(
    node_path: str,
    args: list[str],
    cwd: Path,
    forward_signals: bool = True,
) -> int:
    """Spawn node subprocess, inherit stdio, forward signals, return exit code."""
    proc = subprocess.Popen(
        [node_path, *args],
        cwd=str(cwd),
        stdin=None,
        stdout=None,
        stderr=None,
    )

    if forward_signals:

        def _forward(sig, _frame):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

        signal.signal(signal.SIGINT, _forward)
        signal.signal(signal.SIGTERM, _forward)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _forward)

    try:
        return proc.wait()
    except KeyboardInterrupt:
        # 上方已转发信号；短暂等待优雅退出。
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


# ---------------------------------------------------------------------------
# RPC 握手与 asyncio 服务器循环
# ---------------------------------------------------------------------------

# 握手时限：规范 5.1——Node 必须在启动后 5 秒内发送 `system.hello`，否则父进程以状态 3 中止。
_RPC_HANDSHAKE_TIMEOUT_S: float = 5.0
_RPC_HANDSHAKE_EXIT_CODE: int = 3

# Node 子进程连接 CODEFLOW_RPC_SOCKET 导出的 TCP 回环地址，并使用 CODEFLOW_RPC_TOKEN 共享密钥
# 认证。任何本地进程都能访问回环地址，因此服务器分发请求前需由令牌维持信任边界。
_RPC_SOCKET_ENV: str = "CODEFLOW_RPC_SOCKET"
_RPC_TOKEN_ENV: str = "CODEFLOW_RPC_TOKEN"  # noqa: S105 -- 这是环境变量名，不是密钥值


def _suppress_noisy_watchers() -> None:
    """Raise file-watcher loggers to INFO so ``watchfiles`` per-poll DEBUG
    chatter ('rust notify timeout') stays out of the log sink."""
    import logging as _stdlib_logging

    for _name in ("watchfiles", "watchfiles.main", "watchfiles.watcher", "watchdog", "notify"):
        _stdlib_logging.getLogger(_name).setLevel(_stdlib_logging.INFO)


def _drop_watcher_spam(record: dict) -> bool:
    """Sink filter dropping watchfiles poll-timeout chatter (TUI-only, so the
    shared gateway sink is unaffected)."""
    return "rust notify timeout" not in record["message"]


# 这里只收窄到表示可恢复初始化崩溃的异常类型：AgentLoop 构造器重构后的 kwargs 漂移、
# 配置模式重命名后的属性路径漂移、可选依赖的 ImportError、配置文件缺失和 Pydantic
# ValidationError。它们统一暴露为 -32603 ``internal_error``，并携带
# ``data.reason="tui_init_crash"``，让 UI 能与正常的 -32008 ``model_not_available``
# （未配置提供商）区分开。
_TUI_INIT_CRASH_TYPES: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    ImportError,
    FileNotFoundError,
    OSError,
)


async def _fanout_cron_delivered(emitter, *, job_id, name, text, fired_at) -> None:
    """Fan a ``cron.delivered`` event out to every active TUI session.

    Fan-out (rather than a session-keyed emit) is required because a cron turn
    runs in the ``cron:<job_id>`` conversation, which matches no user
    subscription key. TUI v0.1 is single-session per ``hermes-tui-rpc-architecture``
    5-domain fallback.
    """
    payload = {"job_id": job_id, "name": name, "text": text, "fired_at": fired_at}
    for session_key in list(emitter._by_session.keys()):
        await emitter.emit(session_key, {"type": "cron.delivered", "payload": payload})


def _build_cron_callback_spine(base_on_cron, emitter):
    """Wrap the spine cron callback so a delivering job's reply is fanned out as a
    ``cron.delivered`` event. ``base_on_cron`` (``make_on_cron_job`` with
    ``submit=``) runs the reminder as a CRON turn through the TUI scheduler and
    returns its reply (read back from the runner via ``readback_texts``); the cron
    turn's own hub deliverables target the ``cron:<job_id>`` conversation, which
    has no subscriber and so no-op, making this fan-out the only delivery path."""
    from datetime import datetime, timezone

    async def wrapped(job):
        response = await base_on_cron(job)
        if job.payload.deliver and response:
            await _fanout_cron_delivered(
                emitter,
                job_id=job.id,
                name=job.name,
                text=response,
                fired_at=datetime.now(timezone.utc).isoformat(),
            )
        return response

    return wrapped


def _build_tui_runtime():
    """Construct the Runtime served by ``turn.send``.

    Wires a TUI-scoped ``CronService(allowed_channels={"tui"})`` so the
    Runtime can register reminders from within a TUI turn. The provider remains
    lazy, and Cron delivery plus TUI interaction policy stay in this host.

    Raises ``InternalError`` (-32603) when AgentLoop construction fails —
    ``_run_rpc_server_until_done`` catches and latches the error onto the
    factory closure passed to ``register_aligned_methods_except_system`` so
    ``turn.send`` can emit it through the subscription emitter (the launcher
    has no live client connection at startup time).
    """
    from pydantic import ValidationError

    from codeflow.plugin.registry import PluginNotFoundError
    from codeflow.tui_rpc.errors import InternalError

    try:
        from codeflow.cli._helpers import load_runtime_config, make_lazy_provider
        from codeflow.cli._runtime_assembly import assemble_runtime
        from codeflow.config.paths import get_cron_dir, resolve_foreground_paths
        from codeflow.config.codeflow import load_codeflow_config
        from codeflow.proactive_engine.schedulers.cron.service import CronService
        from codeflow.proactive_engine.schedulers.cron.tool import CronTool
        from codeflow.utils.helpers import sync_workspace_templates

        config = load_runtime_config(None, None)
        paths = resolve_foreground_paths(config)
        ec_config = load_codeflow_config()
        sync_workspace_templates(paths.state, silent=True)

        provider = make_lazy_provider(config)

        cron = CronService(
            get_cron_dir() / "jobs.json",
            allowed_channels={"tui"},
        )

        runtime = assemble_runtime(
            config,
            ec_config,
            provider=provider,
            cron_service=cron,
            interactive=True,
            paths=paths,
        )
        agent_loop = runtime.agent_loop

        registered_cron_tool = agent_loop.tools.get("cron")
        if isinstance(registered_cron_tool, CronTool):
            registered_cron_tool.set_context("tui", "default")

        # spine 调度器就绪后，_run_rpc_server_until_done 才连接 cron.on_job：提醒以 CRON
        # 轮次通过调度器运行，其回复再扇出为 cron.delivered 事件。

        return runtime
    except PluginNotFoundError as e:
        from loguru import logger as _logger

        _logger.exception(
            "tui: plugin configuration failed; surfacing actionable internal_error",
        )
        raise InternalError(
            detail=str(e),
            data={
                "reason": "plugin_configuration_error",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "log_path": "~/.codeflow/logs/tui.log",
                "public_message": str(e),
            },
        ) from e
    except (*_TUI_INIT_CRASH_TYPES, ValidationError) as e:
        from loguru import logger as _logger

        _logger.exception(
            "tui: _build_tui_runtime init crash ({}); surfacing as -32603 internal_error",
            type(e).__name__,
        )
        raise InternalError(
            detail=str(e),
            data={
                "reason": "tui_init_crash",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "log_path": "~/.codeflow/logs/tui.log",
            },
        ) from e
    except Exception as e:
        from loguru import logger as _logger

        _logger.exception(
            "tui: _build_tui_runtime uncaught exception; surfacing as -32603 internal_error",
        )
        raise InternalError(
            detail=str(e),
            data={
                "reason": "uncaught",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "log_path": "~/.codeflow/logs/tui.log",
            },
        ) from e


def _build_tui_agent_loop():
    """Compatibility seam for tests and callers that only need the Agent Loop."""
    return _build_tui_runtime().agent_loop


async def _run_rpc_server_until_done(
    conn: socket.socket,
    auth_token: str,
    handshake_deadline_s: float,
    proc_done: asyncio.Event,
) -> bool:
    """Run RpcServer until the child exits or we abort on handshake timeout.

    Returns True if handshake succeeded (system.hello was received within the
    deadline); False if it timed out.
    """
    # 延迟导入：从不使用 TUI 的用户（例如纯 CLI 工作流）导入 tui_commands 时无需加载 tui_rpc。
    from codeflow.tui_rpc.confirm_broker import ConfirmBroker
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.methods import register_aligned_methods_except_system
    from codeflow.tui_rpc.methods.system import (
        system_hello as _orig_hello,
    )
    from codeflow.tui_rpc.methods.system import (
        system_ping,
        system_version,
    )
    from codeflow.tui_rpc.question_broker import QuestionBroker
    from codeflow.tui_rpc.server import RpcServer
    from codeflow.tui_rpc.spine import build_tui
    from codeflow.tui_rpc.subscriptions import SubscriptionEmitter

    handshake_done = asyncio.Event()

    async def hello_then_signal(params: dict) -> dict:
        result = await _orig_hello(params)
        handshake_done.set()
        return result

    dispatcher = Dispatcher()
    # 服务器在总注册前构建，使 SubscriptionEmitter 能把其 send_frame 方法绑定为通知出口。
    # serve_forever() 仍最后启动（所有处理器注册后）；在它建立写传输前，RpcServer.send_frame
    # 会抛错，但发射器只会在订阅调用后发射，而订阅只能发生在握手和服务启动之后。
    server = RpcServer(dispatcher=dispatcher, sock=conn, auth_token=auth_token)
    emitter = SubscriptionEmitter(send_frame=server.send_frame)
    # ConfirmBroker 共用同一 send_frame 出口，使运行时确认可以发出 confirm.request 并等待
    # confirm.respond。连接断开时，finally 中的 cancel_all() 会解决所有待处理确认。
    confirm_broker = ConfirmBroker(send_frame=server.send_frame)
    # QuestionBroker 共用同一 send_frame 出口：ask_user 工具发出 clarify.request 并等待
    # clarify.respond，与 ConfirmBroker 的行为对称。
    question_broker = QuestionBroker(send_frame=server.send_frame)

    from codeflow.cli._runtime_host import TuiRuntimeHost
    from codeflow.tui_rpc.errors import InternalError, RpcError

    agent_loop = None
    build_error: RpcError | None = None
    runtime_host = TuiRuntimeHost(_build_tui_runtime)

    def _agent_loop_factory():
        if agent_loop is not None:
            return agent_loop
        if build_error is not None:
            raise build_error
        return None

    turn_scheduler = None
    turn_ids: dict[int, str] = {}
    submission_ids: dict[int, str] = {}
    turn_teardown = None
    serve_task: asyncio.Task | None = None
    runtime_bind_task: asyncio.Task | None = None
    runtime_services_task: asyncio.Task | None = None
    runtime_bound = asyncio.Event()
    backend_ready = asyncio.Event()
    backend_start_error: InternalError | None = None
    cleanup_task: asyncio.Task[None] | None = None

    async def _bind_runtime() -> None:
        nonlocal agent_loop, build_error, turn_scheduler, turn_teardown
        try:
            runtime = await runtime_host.acquire()
            agent_loop = runtime.agent_loop
            if (ask_tool := agent_loop.tools.get("ask_user")) is not None and hasattr(ask_tool, "set_broker"):
                ask_tool.set_broker(question_broker)

            from types import SimpleNamespace

            from codeflow.cli._cron_handler import make_on_cron_job
            from codeflow.tui_rpc.methods import turn as turn_module

            cron_readback: dict[str, str] = {}
            turn_scheduler, turn_hub, _, _, turn_teardown = build_tui(
                agent_loop,
                emitter,
                on_turn_end=turn_module.clear_active,
                readback_texts=cron_readback,
                await_runtime_ready=_await_runtime_ready,
                turn_ids=turn_ids,
                submission_ids=submission_ids,
            )
            agent_loop.subagents.set_submit(turn_scheduler.submit)
            if agent_loop.cron_service is not None:
                base_on_cron = make_on_cron_job(
                    turn_hub,
                    submit=turn_scheduler.submit,
                    readback_texts=cron_readback,
                    channel_manager=SimpleNamespace(enabled_channels=["tui"]),
                    default_channel="tui",
                )
                agent_loop.cron_service.on_job = _build_cron_callback_spine(base_on_cron, emitter)
        except RpcError as exc:
            build_error = exc
        except Exception as exc:
            build_error = InternalError(
                detail="runtime binding failed",
                data={"reason": "runtime_binding_failed"},
            )
            from loguru import logger as _logger

            _logger.exception("tui: runtime binding failed: {}", type(exc).__name__)
        finally:
            runtime_bound.set()

    async def _acquire_turn_scheduler():
        await runtime_bound.wait()
        if build_error is not None:
            raise build_error
        if turn_scheduler is None:
            raise InternalError(
                detail="runtime scheduler unavailable",
                data={"reason": "runtime_scheduler_unavailable"},
            )
        return turn_scheduler

    async def _start_runtime_services() -> None:
        nonlocal backend_start_error
        try:
            await handshake_done.wait()
            await runtime_bound.wait()
            if build_error is not None:
                return
            runtime = runtime_host.get_now()
            if runtime is None or agent_loop is None:
                return
            await runtime.start_memory_backend()
            _strip_tty_stream_handlers()
            if agent_loop.cron_service is not None:
                await agent_loop.cron_service.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            backend_start_error = InternalError(
                detail="runtime services failed to start",
                data={"reason": "runtime_start_failed"},
            )
            from loguru import logger as _logger

            _logger.exception("tui: runtime service startup failed")
        finally:
            backend_ready.set()

    async def _await_runtime_ready() -> None:
        await backend_ready.wait()
        if backend_start_error is not None:
            raise backend_start_error

    async def _cleanup_once() -> None:
        cancellation: asyncio.CancelledError | None = None
        if serve_task is not None:
            serve_task.cancel()
            try:
                await serve_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            confirm_broker.cancel_all()
        except Exception:
            pass
        try:
            question_broker.cancel_all()
        except Exception:
            pass
        if runtime_services_task is not None:
            if not runtime_services_task.done():
                runtime_services_task.cancel()
            try:
                await runtime_services_task
            except (asyncio.CancelledError, Exception):
                pass
        if runtime_bind_task is not None:
            try:
                await runtime_bind_task
            except (asyncio.CancelledError, Exception):
                pass
        if agent_loop is not None and agent_loop.cron_service is not None:
            try:
                agent_loop.cron_service.stop()
            except Exception:
                pass
        if turn_teardown is not None:
            try:
                await turn_teardown()
            except asyncio.CancelledError as exc:
                cancellation = exc
            except Exception:
                pass
        try:
            await runtime_host.close()
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation

    async def _cleanup() -> None:
        nonlocal cleanup_task
        if cleanup_task is None:
            cleanup_task = asyncio.create_task(_cleanup_once())
        await finish_barrier(cleanup_task)

    try:
        # 包装 system.hello 以锁存握手事件；下方总入口注册保留的会话 RPC 接口。
        dispatcher.register("system.hello", hello_then_signal)
        dispatcher.register("system.ping", system_ping)
        dispatcher.register("system.version", system_version)
        register_aligned_methods_except_system(
            dispatcher,
            emitter=emitter,
            agent_loop_factory=_agent_loop_factory,
            confirm_broker=confirm_broker,
            question_broker=question_broker,
            scheduler_factory=_acquire_turn_scheduler,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
        )

        from codeflow.config.paths import get_logs_dir

        # 子进程已继承终端。将运行时启动和清理置于此次重定向内，避免后端日志破坏终端画面。
        with redirect_terminal_fds_to_file(get_logs_dir() / "tui.log"):
            _strip_tty_stream_handlers()
            serve_task = asyncio.create_task(server.serve_forever())
            runtime_bind_task = asyncio.create_task(_bind_runtime())
            runtime_services_task = asyncio.create_task(_start_runtime_services())

            try:
                # 等待握手完成、超时或子进程退出，任一发生即结束等待。
                done, pending = await asyncio.wait(
                    {
                        asyncio.create_task(handshake_done.wait()),
                        asyncio.create_task(proc_done.wait()),
                    },
                    timeout=handshake_deadline_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                # 等待已取消任务收尾，避免产生警告。
                for t in pending:
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

                if not handshake_done.is_set():
                    return False
                # 持续服务直到子进程退出。
                await proc_done.wait()
                return True
            finally:
                await _cleanup()
    finally:
        await _cleanup()


def _spawn_with_rpc_socket(
    node_path: str,
    args: list[str],
    cwd: Path,
) -> tuple[subprocess.Popen[bytes], socket.socket, str]:
    """Spawn `[node_path, *args]` with a per-session TCP-loopback socket.

    Topology (cross-platform: macOS / Linux / Windows):

        parent: socket()/bind()/listen() on 127.0.0.1:<ephemeral>; exports the
                ``host:port`` as ``CODEFLOW_RPC_SOCKET`` and a random shared secret
                as ``CODEFLOW_RPC_TOKEN``
        child:  reads ``CODEFLOW_RPC_SOCKET``, ``net.createConnection({host,port})``,
                sends ``CODEFLOW_RPC_TOKEN`` as the first line, then emits JSON-RPC
                frames + reads responses on the same socket

    Loopback (127.0.0.1) is reachable by any local process, so the token --
    known only to the child we spawn (passed via env) -- is what keeps a rogue
    local process from talking to us; the parent validates it as the first line
    (see ``RpcServer(auth_token=...)``) before any dispatch.

    Returns ``(popen, listening_server_socket, auth_token)``. Caller must
    eventually ``server_sock.close()``. The accepted client connection is NOT
    created here -- that's done by :func:`_run_rpc_server_until_done` once the
    asyncio loop is up so the accept can be cancelled cleanly on handshake
    timeout.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    host, port = server_sock.getsockname()[:2]

    token = secrets.token_hex(32)
    env = os.environ.copy()
    env[_RPC_SOCKET_ENV] = f"{host}:{port}"
    env[_RPC_TOKEN_ENV] = token

    proc = subprocess.Popen(
        [node_path, *args],
        cwd=str(cwd),
        stdin=None,
        stdout=None,
        stderr=None,
        env=env,
    )

    return proc, server_sock, token


async def _accept_with_timeout(
    server_sock: socket.socket,
    timeout_s: float,
) -> socket.socket | None:
    """Accept one connection on `server_sock` or return None on timeout.

    Uses ``loop.sock_accept`` so the wait is cooperatively cancellable.
    """
    server_sock.setblocking(False)
    loop = asyncio.get_running_loop()
    try:
        conn, _addr = await asyncio.wait_for(loop.sock_accept(server_sock), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None
    conn.setblocking(False)
    return conn


def run_subprocess_with_rpc(
    node_path: str,
    args: list[str],
    cwd: Path,
    forward_signals: bool = True,
) -> int:
    """Spawn Node child with a per-session TCP-loopback socket; run RpcServer; enforce handshake.

    Cross-platform transport (macOS / Linux / Windows): the parent listens on
    ``127.0.0.1:<ephemeral>`` and exports the ``host:port`` via
    ``CODEFLOW_RPC_SOCKET`` plus a random shared secret via ``CODEFLOW_RPC_TOKEN``.
    The Node child connects, sends the token as the first newline-terminated
    line, then speaks newline-JSON frames. The accepted socket *object* is
    handed to ``RpcServer`` (no os.dup of a socket fd -- unsupported on
    Windows), which wires it via ``loop.connect_accepted_socket`` and validates
    the token before any dispatch.

    Returns the child's exit code, OR ``_RPC_HANDSHAKE_EXIT_CODE`` (3) if
    the handshake times out (either because the child never connected, or
    because it connected but never sent the token + ``system.hello``).
    """
    proc, server_sock, auth_token = _spawn_with_rpc_socket(node_path, args, cwd)

    if forward_signals:

        def _forward(sig, _frame):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

        signal.signal(signal.SIGINT, _forward)
        signal.signal(signal.SIGTERM, _forward)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _forward)

    proc_done = asyncio.Event()

    def _waiter() -> None:
        try:
            proc.wait()
        finally:
            try:
                loop = _loop_holder.get("loop")
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(proc_done.set)
            except RuntimeError:
                pass

    _loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    _conn_holder: dict[str, socket.socket] = {}
    _flags = {"handed_off": False}

    async def _main() -> bool:
        _loop_holder["loop"] = asyncio.get_running_loop()

        # 在握手时限内等待子进程连接。让 `accept` 与 `proc.wait()` 竞速，使提前退出的
        # 子进程立即返回，而不是停满 5 秒。
        accept_task = asyncio.create_task(_accept_with_timeout(server_sock, _RPC_HANDSHAKE_TIMEOUT_S))
        proc_done_task = asyncio.create_task(proc_done.wait())
        done, pending = await asyncio.wait(
            {accept_task, proc_done_task},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=_RPC_HANDSHAKE_TIMEOUT_S,
        )
        for t in pending:
            t.cancel()
        for t in pending:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if accept_task not in done:
            return False
        conn = accept_task.result()
        if conn is None:
            return False
        _conn_holder["conn"] = conn

        # 将已连接的套接字对象直接交给 RpcServer。不要对文件描述符调用 os.dup
        # （Windows 不支持）；connect_accepted_socket 会取得该套接字所有权并在清理时关闭，
        # 因此外层清理不得重复关闭。
        _flags["handed_off"] = True
        return await _run_rpc_server_until_done(conn, auth_token, _RPC_HANDSHAKE_TIMEOUT_S, proc_done)

    waiter = threading.Thread(target=_waiter, daemon=True)
    waiter.start()

    handshake_ok = False
    try:
        handshake_ok = asyncio.run(_main())
    finally:
        # 1）仅当接收的连接从未交给 RpcServer 时才关闭。移交后服务器通过
        # connect_accepted_socket 拥有它并在清理时关闭；此处重复关闭会产生竞态。
        if "conn" in _conn_holder and not _flags["handed_off"]:
            try:
                _conn_holder["conn"].close()
            except OSError:
                pass
        # 2）关闭监听套接字。
        try:
            server_sock.close()
        except OSError:
            pass

    if not handshake_ok:
        print(
            f"✗ RPC handshake timeout ({_RPC_HANDSHAKE_TIMEOUT_S:.0f}s); is the Node side using the new IPC bridge?",
            file=sys.stderr,
        )
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        return _RPC_HANDSHAKE_EXIT_CODE

    waiter.join(timeout=5)
    return proc.returncode if proc.returncode is not None else 0


def _print_node_help(out=None) -> None:
    """Print the friendly Node-missing error message."""
    msg = (
        "✗ TUI 启动失败：未找到 Node.js ≥ 22。\n"
        "  安装：https://nodejs.org/  或  brew install node@22  或  nvm install 22\n"
        "  或：临时使用行式 REPL  ->  codeflow run --legacy-repl\n"
    )
    typer.echo(msg, file=out)


def _is_abnormal_child_exit(exit_code: int) -> bool:
    """A child exit worth surfacing to the user: not a clean exit, not one of
    the signals the parent forwards for a graceful shutdown, and not the
    RPC-handshake failure path (which reports itself).

    The graceful signal codes mirror ui-tui gracefulExit.ts, which maps SIGHUP
    to 129 (terminal closed), SIGINT to 130 (Ctrl+C) and SIGTERM to 143
    (kill / process manager) through one clean-exit path. A hard SIGKILL (137)
    is left abnormal. ``_RPC_HANDSHAKE_EXIT_CODE`` is 3."""
    return exit_code not in (0, 129, 130, 143, _RPC_HANDSHAKE_EXIT_CODE)


def launch_tui(
    *,
    check: bool = False,
    dev: bool = False,
    color: Optional[str] = None,
    print_colors: bool = False,
    preview_colors: bool = False,
) -> None:
    """Launch CodeFlow native TUI."""
    # 启动门：缺少必要配置（提供商密钥和默认模型）时先启动引导向导。无 TTY 的诊断启动
    # （--check / --print-colors / --preview-colors）会跳过。
    if not (check or print_colors or preview_colors) and _stdout_isatty():
        from codeflow.cli.onboard_commands import (
            _is_config_populated,
            ensure_configured_or_onboard,
        )

        if not _is_config_populated():
            ensure_configured_or_onboard()

    node_path, version = find_node()
    if node_path is None:
        _print_node_help()
        raise typer.Exit(code=1)
    if version is None or version < _MIN_NODE_VERSION:
        ver_str = ".".join(map(str, version)) if version else "<unknown>"
        typer.echo(
            f"✗ Node 版本过低（找到 {ver_str}，需要 >= 22）。\n  请升级：nvm install 22  或  brew upgrade node\n",
        )
        raise typer.Exit(code=1)

    # `--dev` 从源码树运行 tsx，因此需要 ui-tui/ 检出。生产路径会单独解析打包或源码构建的
    # bundle（见 resolve_dist_entry），绝不能硬性依赖源码树；wheel 安装本就没有 ui-tui/ 源码目录。
    if dev and not _UI_TUI_DIR.exists():
        print(f"✗ TUI 源码缺失（--dev 需要源码树）：{_UI_TUI_DIR}", file=sys.stderr)
        raise typer.Exit(code=2)

    # 颜色覆盖通过环境变量传给子进程（entry.tsx -> colorTier.ts）。仅在传入 --color 时设置，
    # 避免 "auto" 默认值覆盖 shell 层的 CODEFLOW_TUI_COLOR。
    if color is not None:
        os.environ["CODEFLOW_TUI_COLOR"] = color

    # `--check` 是冒烟测试：让子进程启动并证明运行时初始化成功，然后以 0 退出
    # （不渲染 Ink，也不需要交互 TTY）。子进程从继承的环境中读取 CODEFLOW_TUI_CHECK；
    # 对应处理器见 ui-tui/src/entry.tsx。
    if check:
        os.environ["CODEFLOW_TUI_CHECK"] = "1"

    # `--print-colors` / `--preview-colors` 是无 IPC 诊断：子进程输出解析后的调色板
    # （色块/上下文预览）后退出。它们与 --check 一样跳过 RPC 握手。
    if print_colors:
        os.environ["CODEFLOW_TUI_PRINT_COLORS"] = "1"
    if preview_colors:
        os.environ["CODEFLOW_TUI_COLOR_PREVIEW"] = "1"

    # --check / --print-colors / --preview-colors 都是不使用 RPC、仅使用标准输入输出的启动方式。
    no_rpc = check or print_colors or preview_colors

    # 将父进程 loguru 重定向到文件，避免 RPC 服务器日志破坏 Ink 协调器。无 RPC 路径会在
    # Ink 渲染前退出，因此跳过此操作。文件在此启动时创建；正常运行/退出只写入 INFO
    # 生命周期记录，所以仅在子进程异常退出时向用户显示路径（见下方）。
    if not no_rpc:
        _suppress_noisy_watchers()
        log_path = redirect_loguru_to_file(
            "tui.log",
            retention=3,
            record_filter=_drop_watcher_spam,
        )

    if dev:
        # 通过本地 node_modules 运行 tsx。根据已验证的 node_path 推导 npx，使 CODEFLOW_NODE 的
        # 版本固定语义端到端生效。仅当推导路径不存在时才回退到 PATH（少见，例如运维人员
        # 将 CODEFLOW_NODE 指向没有同级 npx 的独立 node 二进制）。
        derived_npx = Path(node_path).parent / "npx"
        if derived_npx.exists():
            npx = str(derived_npx)
        else:
            fallback = shutil.which("npx")
            if fallback is not None and os.environ.get("CODEFLOW_NODE"):
                typer.echo(
                    f"⚠ CODEFLOW_NODE was set but no `npx` next to {node_path};\n"
                    f"  falling back to PATH npx at {fallback}. Node version\n"
                    f"  used by tsx may differ from the validated one.\n",
                    err=True,
                )
            npx = fallback or "npx"
        # 使用 npx 运行 tsx（源码模式，无构建步骤）；node >= 22 自带 npx。刻意不使用
        # `--watch`：交互路径要求一次性 RPC 握手（父进程只接受一个套接字连接），监听触发的
        # 重启会断开连接。--check 保持普通启动，因为 entry.tsx 会在套接字防护前根据
        # CODEFLOW_TUI_CHECK 短路，无需 RPC 服务器；交互路径必须打开套接字，否则 entry.tsx
        # 以状态 2 退出（"CODEFLOW_RPC_SOCKET env var required"）。
        tsx_args = ["tsx", "src/entry.tsx"]
        if no_rpc:
            exit_code = run_subprocess(npx, tsx_args, cwd=_UI_TUI_DIR)
        else:
            exit_code = run_subprocess_with_rpc(npx, tsx_args, cwd=_UI_TUI_DIR)
    else:
        dist_entry = resolve_dist_entry()
        if dist_entry is None:
            print(
                f"✗ TUI 构建产物缺失：{_PACKAGED_DIST_ENTRY}（或源码树 {_UI_TUI_DIR / 'dist' / 'entry.js'}）\n"
                f"  开发者请运行：cd {_UI_TUI_DIR} && npm install && npm run build\n"
                "  用户请从 CodeFlow 仓库重新运行安装脚本。\n",
                file=sys.stderr,
            )
            raise typer.Exit(code=2)
        # 自包含 bundle 不需要 node_modules；从其所在目录运行，确保相对资源解析有明确基准。
        dist_cwd = dist_entry.parent
        # `--check` 冒烟路径保留简单的纯标准输入输出启动，使不支持 JSON-RPC 的早期引导测试
        # 仍能通过；交互运行路径则打开 RPC 套接字并强制握手。
        if no_rpc:
            exit_code = run_subprocess(node_path, [str(dist_entry)], cwd=dist_cwd)
        else:
            exit_code = run_subprocess_with_rpc(node_path, [str(dist_entry)], cwd=dist_cwd)

            # 正常运行时 tui.log 保持静默；仅在子进程异常退出时显示（见 _is_abnormal_child_exit）。
    if not no_rpc and _is_abnormal_child_exit(exit_code):
        typer.echo(f"📝 TUI logs → {log_path} (exit {exit_code})", err=True)

    raise typer.Exit(code=exit_code)
