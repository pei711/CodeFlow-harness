"""通过 Unix Domain Socket 检查并交互 Sandbox VM 的 Debug Server。

Server 提供 List、Exec 与 Interactive Shell 三类本地调试命令，协议使用 Newline-delimited JSON，二进制
Stream 放在 Base64 ``data`` 字段。它只允许同时一个 Client，并只允许 Attach 当前 Process 在
`owned_ids` 中声明拥有的 Running VM，避免调试端跨进程误操作其他 CodeFlow 实例。

Socket 权限会设为 ``0600``，但 Debug Interface 仍可在 VM 内执行任意命令；启用者必须保护路径所在
目录与本机账户。它是运维入口，不属于 Agent 自主 Tool Surface。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

from codeflow.sandbox._async_utils import cancel_and_collect as _cancel_and_collect

logger = logging.getLogger(__name__)


class SandboxDebugServerError(RuntimeError):
    """Debug Server 无法启动时抛出，例如 Socket 已被 Live Process 使用。

    该异常区分可向 Operator 解释的启动冲突与普通 Client Protocol Error；调用方应停止启动此 Debug
    Endpoint，而不是删除一个确认仍活跃的 Socket。
    """


class SandboxDebugServer:
    """监听 Unix Domain Socket，并服务 Sandbox Debug Commands。

    这是 Single-client Server：同一时间最多接受一个 List / Exec / Shell Connection。第二个 Concurrent
    Connection 会收到明确错误，避免两个 Debug Clients 在同一 VM 上 Race，也符合 Interactive Debugger
    一次附着一个 Process 的语义。

    Protocol 是 Newline-delimited JSON，Binary Payloads 在 ``data`` Field 使用 Base64。
    ``max_message_bytes`` 通过传给 ``asyncio.start_unix_server`` 的 `StreamReader` Buffer Limit 强制，
    保护服务端免受超大单行消息。实例生命周期由 `start`/`stop` 控制，不拥有 VM 本身。
    """

    # start() 探测已有套接字文件的最长等待时间，用于判断它属于
    # 正在运行的服务器（拒绝启动），还是可删除的过期文件。
    _PROBE_TIMEOUT_SEC = 0.5

    def __init__(
        self,
        socket_path: Path,
        owned_ids: set[str],
        max_message_bytes: int = 1048576,
    ) -> None:
        self._socket_path = socket_path
        self._owned_ids = owned_ids
        self._max_message_bytes = max_message_bytes
        self._server: asyncio.AbstractServer | None = None
        self._active_client: asyncio.StreamWriter | None = None

    @staticmethod
    def resolve_socket_path(debug_socket: str, data_dir: Path) -> Path:
        """把 ``debug_socket`` 解析成 Absolute Path，并创建 Parent Directories。

        Relative Path 与 `data_dir` Join，Absolute Path 原样使用；两种情况都自动创建父目录。函数不创建
        Socket File，也不验证目录权限，真正 Bind 与 ``0600`` Permission 在 `start` 中完成。
        """
        p = Path(debug_socket)
        if not p.is_absolute():
            p = data_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    async def start(self) -> None:
        """Bind Unix Socket，并拒绝 Clobber 其他 Process 的 Live Socket。

        Socket File 已存在时先 Probe：若 Server 正在 Listening，抛出 `SandboxDebugServerError`，让 Caller
        告知 “another CodeFlow process owns the socket”；若 Connection 因 ECONNREFUSED / No Listener 失败，
        才把它视为 Stale File 并 Unlink。成功 Bind 后把权限设为 Owner-only ``0600``。
        """
        if self._socket_path.exists():
            if await self._probe_alive():
                raise SandboxDebugServerError(
                    f"Sandbox debug socket already in use at {self._socket_path}: "
                    "another CodeFlow process is running with debug enabled. "
                    "Stop it, or set tools.sandbox.debug.socket to a different path."
                )
            self._socket_path.unlink(missing_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
            limit=self._max_message_bytes,
        )
        os.chmod(self._socket_path, 0o600)
        logger.info("Sandbox debug server listening at %s", self._socket_path)

    async def _probe_alive(self) -> bool:
        """探测 ``self._socket_path`` 当前是否有 Server Accepting。

        在 `_PROBE_TIMEOUT_SEC` 内成功建立 Unix Connection 即返回 `True`，随后立即关闭 Probe Writer。
        Refused、Missing、Timeout 或 OS Error 返回 `False`。它只探测 Listener 存活，不验证对端一定是同
        协议版本的 CodeFlow Debug Server。
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._socket_path)),
                timeout=self._PROBE_TIMEOUT_SEC,
            )
        except (ConnectionRefusedError, FileNotFoundError, asyncio.TimeoutError, OSError):
            return False
        try:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        except Exception:
            pass
        return True

    async def stop(self) -> None:
        """停止接受 Connections，并移除 Socket File。

        已启动时先 Close Server 并 Await `wait_closed`，再 Unlink Path。方法不主动终止当前 Client 正在
        操作的 VM Execution；Connection Handler 会在其自身生命周期完成清理。
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._socket_path.unlink(missing_ok=True)
        logger.info("Sandbox debug server stopped")

    # ------------------------------------------------------------------
    # 按连接分派
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """处理 Exactly One Client Connection：读取一个 Command、Dispatch、然后 Close。

        Single-client Invariant 要求已有 Client Attached 时立即拒绝新连接，并返回解释性 Error，确保两个
        Debug CLIs 不会在同一 VM 上 Race。首行超过限制、不是有效 JSON/Dict 或 Command Unknown 都转换
        成协议错误；Connection Reset 静默结束。`finally` 始终释放 Active-client Slot 并关闭 Writer。
        """
        if self._active_client is not None:
            try:
                await _send(
                    writer,
                    {
                        "type": "error",
                        "message": (
                            "Sandbox debug server already has an active client. "
                            "Only one sandbox CLI may connect at a time — "
                            "wait for the other session to finish, or stop it."
                        ),
                    },
                )
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
            return

        self._active_client = writer
        try:
            try:
                line = await reader.readline()
            except ValueError:
                await _send(writer, {"type": "error", "message": "Message too large."})
                return

            if not line:
                return

            try:
                msg = json.loads(line.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                await _send(writer, {"type": "error", "message": "Invalid JSON."})
                return

            if not isinstance(msg, dict):
                await _send(writer, {"type": "error", "message": "Invalid JSON."})
                return

            cmd = msg.get("cmd")
            if cmd == "list":
                await self._handle_list(writer)
            elif cmd == "exec":
                await self._handle_exec(msg, reader, writer)
            elif cmd == "shell":
                await self._handle_shell(msg, reader, writer)
            else:
                await _send(writer, {"type": "error", "message": f"Unknown command: '{cmd}'."})
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception:
            logger.exception("SandboxDebugServer: unexpected error handling client")
        finally:
            if self._active_client is writer:
                self._active_client = None
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 列出虚拟机
    # ------------------------------------------------------------------

    async def _handle_list(self, writer: asyncio.StreamWriter) -> None:
        try:
            import boxlite  # noqa: F401 — 可用性探测
        except ImportError:
            await _send(writer, {"type": "error", "message": "boxlite is not installed."})
            return

        from codeflow.sandbox._runtime import get_boxlite_runtime

        try:
            runtime = get_boxlite_runtime()
            boxes = await runtime.list_info()
        except Exception as exc:
            await _send(writer, {"type": "error", "message": f"Failed to list VMs: {exc}"})
            return

        vms = []
        for info in boxes:
            vms.append(
                {
                    "id": info.id,
                    "name": getattr(info, "name", None),
                    "owned": info.id in self._owned_ids,
                    "status": info.state.status,
                    "image": info.image,
                    "cpus": info.cpus,
                    "memory_mib": info.memory_mib,
                    "created_at": getattr(info, "created_at", None),
                }
            )
        await _send(writer, {"type": "vm_list", "vms": vms})

    # ------------------------------------------------------------------
    # 虚拟机解析（exec 和 shell 共用）
    # ------------------------------------------------------------------

    async def _resolve_vm(
        self,
        vm_ref: str | None,
        writer: asyncio.StreamWriter,
        boxes: list,
    ):
        """把 ``vm_ref`` 解析成 `BoxInfo`；失败时发送 Error 并返回 `None`。

        `boxes` 是历史称为 ``runtime.list``、当前由 ``runtime.list_info`` 返回的 Runtime List Result，
        由 Caller 传入以便复用同一 Snapshot。未指定 Ref 时只在当前
        Process Owned 且 Running 的 VMs 中自动选择，并要求唯一；指定时先精确 Match ID，再在 Owned VMs
        中 Match Name。Not Owned、Not Running、Ambiguous 或 Missing 都拒绝，防止跨进程 Attach。
        """
        if vm_ref is None:
            # 自动选择当前进程拥有且正在运行的虚拟机
            candidates = [b for b in boxes if b.id in self._owned_ids and b.state.status == "running"]
            if not candidates:
                await _send(writer, {"type": "error", "message": "No running VMs. Start codeflow run/gateway first."})
                return None
            if len(candidates) > 1:
                await _send(writer, {"type": "error", "message": "Multiple running VMs; use --vm to specify one."})
                return None
            return candidates[0]

        # 在全部虚拟机中匹配 ID
        by_id = [b for b in boxes if b.id == vm_ref]
        if by_id:
            box = by_id[0]
            if box.id not in self._owned_ids:
                await _send(writer, {"type": "error", "message": f"VM {box.id} is not owned by this process."})
                return None
            if box.state.status != "running":
                await _send(writer, {"type": "error", "message": f"VM is not running: {box.id}."})
                return None
            return box

        # 只在当前进程拥有的虚拟机中匹配名称
        by_name = [b for b in boxes if getattr(b, "name", None) == vm_ref]
        if len(by_name) > 1:
            await _send(
                writer, {"type": "error", "message": f"Ambiguous: multiple VMs named '{vm_ref}', use VM ID instead."}
            )
            return None
        if len(by_name) == 1:
            box = by_name[0]
            if box.id not in self._owned_ids:
                await _send(writer, {"type": "error", "message": f"VM {box.id} is not owned by this process."})
                return None
            if box.state.status != "running":
                await _send(writer, {"type": "error", "message": f"VM is not running: {box.id}."})
                return None
            return box

        await _send(writer, {"type": "error", "message": f"No VM found with ID or name '{vm_ref}'."})
        return None

    async def _attach_box(self, vm_ref: str | None, writer: asyncio.StreamWriter):
        """解析 ``vm_ref`` 并返回 Live ``boxlite.Box``，失败时返回 `None`。

        BoxLite Missing、``list_info`` Failure、Resolution Error 或 Runtime ``get`` Failure 都会先向 Client
        发送 Error Message。Caller 收到 `None` 后只需 Return，不应再发第二个冲突错误。成功返回已检查
        Ownership 与 Running State，但 VM 仍可能在随后操作前退出。
        """
        try:
            import boxlite  # noqa: F401 — 可用性探测（与 _handle_list 保持一致）
        except ImportError:
            await _send(writer, {"type": "error", "message": "boxlite is not installed."})
            return None

        from codeflow.sandbox._runtime import get_boxlite_runtime

        try:
            runtime = get_boxlite_runtime()
            boxes = await runtime.list_info()
        except Exception as exc:
            await _send(writer, {"type": "error", "message": f"Failed to list VMs: {exc}"})
            return None

        box_info = await self._resolve_vm(vm_ref, writer, boxes)
        if box_info is None:
            return None

        try:
            return await runtime.get(box_info.id)
        except Exception as exc:
            await _send(writer, {"type": "error", "message": f"Failed to attach to VM: {exc}"})
            return None

    # ------------------------------------------------------------------
    # 执行命令
    # ------------------------------------------------------------------

    async def _handle_exec(
        self,
        msg: dict,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        program = msg.get("program")
        if not program:
            await _send(writer, {"type": "error", "message": "exec: 'program' must be a non-empty string."})
            return
        args = msg.get("args") or []

        box = await self._attach_box(msg.get("vm_ref"), writer)
        if box is None:
            return

        try:
            execution = await box.exec(program, list(args))
        except Exception as exc:
            await _send(writer, {"type": "error", "message": f"Failed to start execution: {exc}"})
            return

        async def _stream_stdout():
            try:
                async for chunk in execution.stdout():
                    data = base64.b64encode(chunk.encode() if isinstance(chunk, str) else chunk).decode()
                    await _send(writer, {"type": "stdout", "data": data})
            except (ConnectionResetError, BrokenPipeError):
                pass

        async def _stream_stderr():
            try:
                async for chunk in execution.stderr():
                    data = base64.b64encode(chunk.encode() if isinstance(chunk, str) else chunk).decode()
                    await _send(writer, {"type": "stderr", "data": data})
            except (ConnectionResetError, BrokenPipeError):
                pass

        async def _watch_disconnect():
            # P1.3：客户端断开检测。exec 协议在初始命令后不再期待客户端向服务端发送数据，
            # 因此忽略 readline() 成功读到的意外数据；空结果表示客户端关闭了它的套接字，
            # 此时必须停止等待长时间运行的虚拟机进程，并终止该进程。
            try:
                while await reader.readline():
                    pass
            except (ConnectionResetError, BrokenPipeError, ValueError):
                pass

        async def _both_streams():
            await asyncio.gather(_stream_stdout(), _stream_stderr())

        async def _do_wait():
            # boxlite 的 Execution.wait() 返回 Future 而非协程，create_task() 不能直接接收；
            # 因此用异步函数包装后再 await，同时保持单元测试的 AsyncMock 路径可用。
            return await execution.wait()

        streams_task = asyncio.create_task(_both_streams())
        wait_task = asyncio.create_task(_do_wait())
        watcher = asyncio.create_task(_watch_disconnect())

        try:
            done, _ = await asyncio.wait({wait_task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if wait_task in done:
                # 进程先退出时，先有界地排空剩余标准输出和标准错误，避免异常流永久占用连接，
                # 然后发送退出码。
                try:
                    await asyncio.wait_for(asyncio.shield(streams_task), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception:
                    pass
                try:
                    result = wait_task.result()
                    await _send(writer, {"type": "exit", "code": result.exit_code})
                except (ConnectionResetError, BrokenPipeError):
                    pass
                except Exception as exc:
                    try:
                        await _send(writer, {"type": "error", "message": f"Execution failed: {exc}"})
                    except (ConnectionResetError, BrokenPipeError):
                        pass
            else:
                # 客户端先断开时，停止虚拟机进程，避免泄漏。
                try:
                    await execution.kill()
                except Exception:
                    pass
        finally:
            for t in (streams_task, wait_task, watcher):
                t.cancel()
            await asyncio.gather(streams_task, wait_task, watcher, return_exceptions=True)

    # ------------------------------------------------------------------
    # 交互式 Shell
    # ------------------------------------------------------------------

    async def _handle_shell(
        self,
        msg: dict,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        shell_path = msg.get("shell")
        if not shell_path:
            await _send(writer, {"type": "error", "message": "shell: 'shell' must be a non-empty string."})
            return

        box = await self._attach_box(msg.get("vm_ref"), writer)
        if box is None:
            return

        try:
            execution = await box.exec(shell_path, [], tty=True)
        except Exception as exc:
            await _send(writer, {"type": "error", "message": f"Failed to start shell: {exc}"})
            return

        await _send(writer, {"type": "ready"})

        done_event = asyncio.Event()
        stdout_task: asyncio.Task | None = None

        async def _stdout_task_fn():
            try:
                async for chunk in execution.stdout():
                    raw = chunk.encode() if isinstance(chunk, str) else chunk
                    await _send(writer, {"type": "stdout", "data": base64.b64encode(raw).decode()})
            except (ConnectionResetError, BrokenPipeError):
                pass

        async def _wait_task():
            try:
                result = await execution.wait()
                # P1.4：通知退出前先排空剩余标准输出。否则客户端在收到 `exit` 后就会中断，
                # 丢失仍排队在 stdout_task 中的最后几个数据块。等待必须有界，
                # 避免卡住的标准输出迭代器永久死锁会话。
                if stdout_task is not None:
                    try:
                        await asyncio.wait_for(asyncio.shield(stdout_task), timeout=1.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    except Exception:
                        pass
                await _send(writer, {"type": "exit", "code": result.exit_code})
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                done_event.set()

        # 记录尚未完成的即发即忘 resize 任务，以便外层清理取消并等待它们；
        # 否则销毁时仍在等待的慢速 resize_tty 会让 `execution` 比处理器活得更久。
        resize_tasks: list[asyncio.Task] = []

        async def _stdin_task():
            stdin_writer = execution.stdin()

            async def _do_resize(r: int, c: int) -> None:
                # Resize 采用即发即忘，绝不用它阻塞标准输入转发；否则慢速或卡住的
                # resize_tty（例如 Shell 启动期间）会让后续每次按键都停滞。
                try:
                    await execution.resize_tty(rows=r, cols=c)
                except Exception:
                    pass

            # P1.2：无论因正常 EOF、异常还是取消离开此循环，都必须触发 done_event，
            # 使客户端已离开的空闲 Shell 被销毁，而不是成为虚拟机内的孤儿进程。
            #
            # 让每次 readline() 与 done_event.wait() 竞速，使 _wait_task 触发的退出能立即被观察到，
            # 而不必等到下一个轮询周期。
            done_wait = asyncio.create_task(done_event.wait())
            try:
                while True:
                    read_task = asyncio.create_task(reader.readline())
                    finished, _ = await asyncio.wait(
                        {read_task, done_wait},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if done_wait in finished:
                        await _cancel_and_collect(read_task)
                        break
                    line = read_task.result()
                    if not line:
                        break
                    try:
                        client_msg = json.loads(line.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    ccmd = client_msg.get("cmd")
                    if ccmd == "stdin":
                        raw = base64.b64decode(client_msg.get("data", ""))
                        if raw:
                            await stdin_writer.send_input(raw)
                    elif ccmd == "resize":
                        rows = client_msg.get("rows", 0)
                        cols = client_msg.get("cols", 0)
                        if rows >= 1 and cols >= 1:
                            resize_tasks.append(asyncio.create_task(_do_resize(rows, cols)))
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                await _cancel_and_collect(done_wait)
                done_event.set()

        stdout_task = asyncio.create_task(_stdout_task_fn())
        tasks = [
            stdout_task,
            asyncio.create_task(_wait_task()),
            asyncio.create_task(_stdin_task()),
        ]
        try:
            await done_event.wait()
        finally:
            for t in (*tasks, *resize_tasks):
                t.cancel()
            await asyncio.gather(*tasks, *resize_tasks, return_exceptions=True)
            try:
                await execution.kill()
            except Exception:
                pass


# ------------------------------------------------------------------
# 共用分帧辅助函数（exec 和 shell 处理器也使用）
# ------------------------------------------------------------------


async def _send(writer: asyncio.StreamWriter, obj: dict) -> None:
    writer.write((json.dumps(obj) + "\n").encode())
    await writer.drain()
