"""CLI subcommands for sandbox VM inspection and interaction."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codeflow.sandbox._async_utils import cancel_and_collect as _cancel_and_collect

console = Console()
logger = logging.getLogger(__name__)


class _SocketClosedError(Exception):
    """Raised when the debug server closes the connection without sending a response."""


sandbox_app = typer.Typer(
    name="sandbox",
    help="Inspect and interact with sandbox VMs (requires sandbox.debug=true).",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# 辅助方法
# ---------------------------------------------------------------------------


def _get_socket_path() -> Path:
    """Resolve the debug socket path from config (falls back to defaults)."""
    from codeflow.config.paths import get_data_dir
    from codeflow.sandbox.debug_server import SandboxDebugServer

    debug_socket = "sandbox/debug.sock"
    try:
        from codeflow.config.loader import load_config

        cfg = load_config()
        debug_socket = cfg.tools.sandbox.debug.socket
    except FileNotFoundError:
        # 预期路径没有配置文件，使用默认套接字。
        pass
    except Exception as exc:
        # 配置存在但加载或解析失败。不要让 CLI 崩溃，但需明确警告，方便用户关联错误的套接字查找。
        logger.warning("Failed to load sandbox debug config (%s); using default socket path", exc)

    return SandboxDebugServer.resolve_socket_path(debug_socket, get_data_dir())


def _check_socket(path: Path) -> None:
    """Exit with a clear error if the socket file is missing or inaccessible."""
    if not path.exists():
        console.print(f"[red]Debug socket not found at {path}.[/red]")
        console.print(
            "[dim]Is codeflow running with sandbox.debug.enabled=true? "
            "(If it is, the debug server may have failed to start — check the "
            "agent logs/output for a '[Sandbox debug]' message.)[/dim]"
        )
        raise typer.Exit(1)
    if not os.access(path, os.R_OK | os.W_OK):
        console.print(f"[red]Cannot connect to debug socket at {path}: permission denied[/red]")
        raise typer.Exit(1)


async def _connect(path: Path) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a connection to the debug socket."""
    try:
        return await asyncio.open_unix_connection(str(path))
    except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
        console.print(f"[red]Cannot connect to debug socket at {path}: {exc}[/red]")
        raise typer.Exit(1) from exc


async def _send(writer: asyncio.StreamWriter, obj: dict) -> None:
    writer.write((json.dumps(obj) + "\n").encode())
    await writer.drain()


async def _recv(reader: asyncio.StreamReader) -> dict:
    line = await reader.readline()
    if not line:
        raise _SocketClosedError("server closed connection without sending a response")
    try:
        return json.loads(line.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # 帧格式仅用于客户端和服务器内部，实际不应出错；若确实发生（行截断、垃圾数据、协议
        # 不匹配），应展示清晰错误，而不是在用户终端输出回溯。
        raise _SocketClosedError(f"server sent malformed response: {exc}") from exc


def _close(writer: asyncio.StreamWriter) -> None:
    try:
        writer.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 列表 / ls
# ---------------------------------------------------------------------------


def _run_list() -> None:
    socket_path = _get_socket_path()
    _check_socket(socket_path)

    async def _do() -> None:
        reader, writer = await _connect(socket_path)
        try:
            await _send(writer, {"cmd": "list"})
            try:
                msg = await _recv(reader)
            except _SocketClosedError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                raise typer.Exit(1) from exc
        finally:
            _close(writer)

        if msg.get("type") == "error":
            console.print(f"[red]Error: {msg.get('message')}[/red]")
            raise typer.Exit(1)

        vms = msg.get("vms", [])
        if not vms:
            console.print("[dim]No VMs found.[/dim]")
            return

        table = Table(title="Sandbox VMs")
        table.add_column("", style="bold", no_wrap=True)  # 所有权标记
        table.add_column("ID", style="cyan", no_wrap=True)
        # table.add_column("Name")  # 当前 VM 没有名称，支持命名后恢复
        table.add_column("State")
        table.add_column("Image")
        table.add_column("CPUs", justify="right")
        table.add_column("Mem MiB", justify="right")
        table.add_column("Created At")

        for vm in vms:
            owned_marker = "[green]*[/green]" if vm.get("owned") else "[dim]-[/dim]"
            status = vm.get("status", "")
            status_styled = f"[green]{status}[/green]" if status == "running" else f"[dim]{status}[/dim]"
            created = (vm.get("created_at") or "")[:19].replace("T", " ")
            table.add_row(
                owned_marker,
                vm.get("id", ""),
                # vm.get("name") or "",  # 当前 VM 没有名称，支持命名后恢复
                status_styled,
                vm.get("image", ""),
                str(vm.get("cpus", "")),
                str(vm.get("memory_mib", "")),
                created,
            )
        console.print(table)

    asyncio.run(_do())


@sandbox_app.command("list")
def sandbox_list() -> None:
    """List all sandbox VMs (owned VMs marked with *)."""
    _run_list()


@sandbox_app.command("ls")
def sandbox_ls() -> None:
    """Alias for 'list'."""
    _run_list()


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


@sandbox_app.command("exec", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def sandbox_exec(
    ctx: typer.Context,
    vm: str | None = typer.Option(None, "--vm", help="Target VM — ID or name (auto-select if only one running VM)."),
) -> None:
    """Run a command inside a sandbox VM and stream its output."""
    cmd_args = ctx.args
    if not cmd_args:
        console.print("[red]Error: a command to execute is required.[/red]")
        console.print("Usage: codeflow sandbox exec [--vm VM_REF] CMD [ARG ...]")
        raise typer.Exit(1)

    program, *args = cmd_args
    socket_path = _get_socket_path()
    _check_socket(socket_path)

    async def _do() -> int:
        reader, writer = await _connect(socket_path)
        # 为按行添加标准错误前缀而缓冲：数据块不保证与行对齐（boxlite 可能拆分单行或合并多行），
        # 因此按 "\n" 分行加前缀，而不是逐块添加，避免在行中间出现 "[stderr] " 标记。
        stderr_buf = bytearray()

        def _emit_stderr(chunk: bytes) -> None:
            stderr_buf.extend(chunk)
            while True:
                nl = stderr_buf.find(b"\n")
                if nl < 0:
                    break
                sys.stderr.buffer.write(b"[stderr] " + bytes(stderr_buf[: nl + 1]))
                del stderr_buf[: nl + 1]
            sys.stderr.buffer.flush()

        def _flush_stderr_tail() -> None:
            # 刷新最后一个没有换行符的不完整行，避免 VM 进程在两次写入间退出时静默丢失。
            if stderr_buf:
                sys.stderr.buffer.write(b"[stderr] " + bytes(stderr_buf))
                sys.stderr.buffer.write(b"\n")
                sys.stderr.buffer.flush()
                stderr_buf.clear()

        try:
            await _send(
                writer,
                {
                    "cmd": "exec",
                    "vm_ref": vm,
                    "program": program,
                    "args": args,
                },
            )
            exit_code = 1
            while True:
                try:
                    msg = await _recv(reader)
                except _SocketClosedError as exc:
                    _flush_stderr_tail()
                    console.print(f"[red]Error: {exc}[/red]")
                    return 1
                mtype = msg.get("type")
                if mtype == "stdout":
                    data = base64.b64decode(msg["data"])
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
                elif mtype == "stderr":
                    _emit_stderr(base64.b64decode(msg["data"]))
                elif mtype == "exit":
                    _flush_stderr_tail()
                    exit_code = msg.get("code", 0)
                    break
                elif mtype == "error":
                    _flush_stderr_tail()
                    console.print(f"[red]Error: {msg.get('message')}[/red]")
                    exit_code = 1
                    break
            return exit_code
        finally:
            _close(writer)

    code = asyncio.run(_do())
    raise typer.Exit(code)


# ---------------------------------------------------------------------------
# Shell 交互
# ---------------------------------------------------------------------------


@sandbox_app.command("shell")
def sandbox_shell(
    vm: str | None = typer.Option(None, "--vm", help="Target VM — ID or name (auto-select if only one running VM)."),
    shell_path: str = typer.Option("/bin/sh", "--shell", help="Shell binary inside the VM."),
) -> None:
    """Open an interactive shell inside a sandbox VM."""
    import fcntl
    import signal
    import struct
    import termios
    import tty

    socket_path = _get_socket_path()
    _check_socket(socket_path)

    async def _do() -> int:
        reader, writer = await _connect(socket_path)
        try:
            await _send(writer, {"cmd": "shell", "vm_ref": vm, "shell": shell_path})

            # 进入原始模式前等待就绪或错误。
            try:
                first = await _recv(reader)
            except _SocketClosedError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                return 1
            if first.get("type") == "error":
                console.print(f"[red]Error: {first.get('message')}[/red]")
                return 1
            if first.get("type") != "ready":
                console.print(f"[red]Unexpected server response: {first}[/red]")
                return 1

            # 保存终端状态并进入原始模式。
            fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(fd)

            def _restore():
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
                except Exception:
                    pass

            tty.setraw(fd)

            exit_code = 1
            done = asyncio.Event()
            loop = asyncio.get_running_loop()
            stdin_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

            def _on_readable() -> None:
                # 文件描述符可读时由事件循环触发；当前已在循环线程，可安全直接调用 put_nowait。
                try:
                    chunk = os.read(fd, 4096)
                    stdin_queue.put_nowait(chunk if chunk else None)
                except OSError:
                    stdin_queue.put_nowait(None)

            loop.add_reader(fd, _on_readable)

            async def _do_send_resize(rows: int, cols: int) -> None:
                # 包装后，关闭期间的管道破裂错误不会在垃圾回收时显示为“从未获取任务异常”。
                try:
                    await _send(writer, {"cmd": "resize", "rows": rows, "cols": cols})
                except Exception:
                    pass

            def _send_resize():
                try:
                    winsz = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
                    rows, cols = struct.unpack("HHHH", winsz)[:2]
                    loop.create_task(_do_send_resize(rows, cols))
                except Exception:
                    pass

                # SIGWINCH 可能在任意线程上下文触发；转回事件循环，确保 create_task 和 writer
                # 访问发生在循环线程。

            original_sigwinch = signal.getsignal(signal.SIGWINCH)
            signal.signal(
                signal.SIGWINCH,
                lambda *_: loop.call_soon_threadsafe(_send_resize),
            )

            # 发送初始终端尺寸。
            _send_resize()

            async def _recv_loop():
                nonlocal exit_code
                try:
                    while True:
                        msg = await _recv(reader)
                        mtype = msg.get("type")
                        if mtype == "stdout":
                            data = base64.b64decode(msg["data"])
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        elif mtype == "exit":
                            exit_code = msg.get("code", 0)
                            break
                        elif mtype == "error":
                            _restore()
                            console.print(f"\r\n[red]Error: {msg.get('message')}[/red]")
                            break
                except _SocketClosedError as exc:
                    # 区分服务器断开与正常退出，使用户看到 Shell 结束原因（智能体崩溃、服务器
                    # 重启、协议格式错误等），而不是静默关闭。
                    _restore()
                    console.print(f"\r\n[red]Error: {exc}[/red]")
                except Exception as exc:
                    # 其他异常（格式错误但合法 JSON 载荷引发的 KeyError、base64 解码失败等）都是真实
                    # 缺陷；应显示足够信息供后续报告，而不是让用户静默回到状态码为 1 的已关闭终端。
                    _restore()
                    console.print(f"\r\n[red]Internal error in sandbox shell: {type(exc).__name__}: {exc}[/red]")
                    logger.exception("sandbox shell recv loop failed")
                finally:
                    done.set()

            async def _stdin_loop():
                # 让每次 queue.get() 与 done.wait() 竞速，使取消立即生效；无需 50 毫秒轮询延迟，
                # 也不会浪费唤醒。
                done_task = asyncio.create_task(done.wait())
                try:
                    while True:
                        get_task = asyncio.create_task(stdin_queue.get())
                        finished, _ = await asyncio.wait(
                            {get_task, done_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if done_task in finished:
                            await _cancel_and_collect(get_task)
                            break
                        chunk = get_task.result()
                        if not chunk:
                            # 本地标准输入到达 EOF（如管道输入结束或终端关闭）。转发最后一个 EOT 字节
                            # （\x04），使 VM 侧 PTY 的行规程收到 VEOF 并终止 Shell；行为与
                            # `docker exec -it` 在 Ctrl-D 或管道结束时一致。
                            data = base64.b64encode(b"\x04").decode()
                            try:
                                await _send(writer, {"cmd": "stdin", "data": data})
                            except (ConnectionResetError, BrokenPipeError):
                                pass
                            break
                        data = base64.b64encode(chunk).decode()
                        await _send(writer, {"cmd": "stdin", "data": data})
                except Exception:
                    pass
                finally:
                    await _cancel_and_collect(done_task)

            try:
                recv_task = asyncio.create_task(_recv_loop())
                stdin_task = asyncio.create_task(_stdin_loop())
                await done.wait()
            finally:
                recv_task.cancel()
                stdin_task.cancel()
                await asyncio.gather(recv_task, stdin_task, return_exceptions=True)
                loop.remove_reader(fd)
                _restore()
                signal.signal(signal.SIGWINCH, original_sigwinch)

            return exit_code
        finally:
            _close(writer)

    code = asyncio.run(_do())
    raise typer.Exit(code)
