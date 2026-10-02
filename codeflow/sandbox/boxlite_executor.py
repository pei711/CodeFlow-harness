"""基于 BoxLite MicroVM 的 `SandboxExecutor` Implementation。

该 Backend 为 Agent Command 创建独立 VM，把 Host Workspace 以 Read-write `/workspace` Volume 挂载，
并可限制 CPU、Memory、Disk、Network 与额外 Volumes。它同时支持 One-shot Exec 和 Long-running MCP
Stdio Process；真正隔离边界来自 BoxLite VM，而不是 Shell Wrapper。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from codeflow.sandbox.interfaces import ExecResult, SandboxExecutor, SandboxInitError

logger = logging.getLogger(__name__)


class BoxliteExecutor(SandboxExecutor):
    """管理一个生命周期绑定到 `AgentLoop` 的 BoxLite Box，即 Per-loop Instance。

    Executor 通过 CodeFlow BoxLite Runtime 使用 Raw ``boxlite.Box`` API，而不是 `SimpleBox`，使 One-shot
    ``exec()`` 与 Streaming ``start_process()`` 共享同一 Running VM。`SimpleBox.exec()` 直接返回
    `ExecResult`，不暴露 Stdin/Stdout Streaming 所需的 `Execution` Object。

    Workspace 以 RW 挂载到 ``/workspace``，所有 Command 默认在那里运行。Box 在 `start()` 内 Eagerly
    Created，使 VM Startup Error 在 Agent Loop 开始前暴露。Executor 还跟踪 MCP Executions、Bridge
    Tasks 与 Async Cleanup Stack，Stop 时按顺序回收。

    历史 ``BoxOptions.volumes`` / ``BoxOptions.env`` Contract 要求 Volumes/Env 使用 Tuple Lists；当前 BoxLite 0.8.2 接受 Volume Dicts，
    Exec-level Env Dict 会转为 Tuples。CWD 通过当前 SDK 的 ``cwd`` 参数传入，Timeout 仍由
    ``asyncio.wait_for`` 在 Host 侧强制。理解版本差异很重要，不能把 SimpleBox-only Kwargs 直接传给
    Raw Box API。
    """

    WORKSPACE_MOUNT = "/workspace"

    def __init__(
        self,
        image: str,
        workspace: Path,
        cpus: int = 2,
        memory_mib: int = 2048,
        disk_size_gb: int | None = None,
        allow_net: bool | list[str] = True,
        extra_volumes: list[list[str]] | None = None,
        default_timeout: int = 120,
        verify_timeout: int = 30,
        create_timeout: int = 300,
        owned_ids: set[str] | None = None,
    ):
        self._image = image
        self._workspace = workspace
        self._cpus = cpus
        self._memory_mib = memory_mib
        self._disk_size_gb = disk_size_gb
        self._allow_net = allow_net
        self._extra_volumes = extra_volumes or []
        self._default_timeout = default_timeout
        self._verify_timeout = verify_timeout
        self._create_timeout = create_timeout
        self._owned_ids = owned_ids

        self._box: Any | None = None  # 延迟导入的 boxlite.Box
        self._stack = AsyncExitStack()
        self._init_lock = asyncio.Lock()
        self._process_tasks: list[asyncio.Task] = []
        self._process_executions: list[Any] = []  # boxlite.Execution 列表

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """创建 Working Box，并验证它可以响应命令。

        Common Case ``allow_net=True`` 时直接创建 Working Box。Network Restricted 时先用 Throwaway
        `SimpleBox` 通过 Host Unrestricted Network 拉取 Image，再按受限 Sandbox Network Policy 创建
        真正 Box。

        Box 创建后的任意 Failure，包括 ``box.start()`` 或 `_verify`，都会先执行 Partial-start Cleanup
        再重新抛出。Caller 的 `AsyncExitStack` 在 ``__aenter__`` / `start()` 抛错时不会调用
        ``__aexit__``，所以清理必须在这里完成。成功返回表示 VM 通过 ``echo ok`` Probe，不代表目标
        Command 或 Network Policy 已被业务验证。
        """
        try:
            await self._start_inner()
        except BaseException:
            # 捕获 BaseException 而不是 Exception，使 KeyboardInterrupt、SystemExit 和
            # asyncio.CancelledError 也能触发清理。泄漏半启动的虚拟机比清理失败更糟；
            # 清理后立即重新抛出，以保留原始信号。
            await self._cleanup_after_failed_start()
            raise

    async def _start_inner(self) -> None:
        if self._allow_net is not True:
            await self._pull_image()
        try:
            box = await asyncio.wait_for(self._ensure_box(), timeout=self._create_timeout)
        except asyncio.TimeoutError:
            raise SandboxInitError(
                f"Sandbox VM creation timed out after {self._create_timeout}s "
                "(image pull or VM initialisation took too long).\n"
                "  • Check network connectivity and registry availability\n"
                "  • Increase create_timeout in sandbox config for large images\n"
                "  • Pre-pull the image to skip network delay: boxlite pull <image>"
            )
        await self._verify(box)

    async def _cleanup_after_failed_start(self) -> None:
        """运行 Partial Start 期间已经 Queue 的所有 Cleanup Callbacks。

        逻辑 Mirrors `stop()`，但只处理 Failure 前可能建立的 Lifecycle Pieces；此时尚无 MCP Bridges 或
        ``process_executions``。Cleanup Error 记录 Warning 而不覆盖原 Startup Exception。
        """
        try:
            await self._stack.aclose()
        except Exception as exc:
            logger.warning("Error during failed-start cleanup: %s", exc)

    async def _pull_image(self) -> None:
        """通过 Unrestricted Networking 的 Throwaway `SimpleBox` Pre-pull OCI Image。

        SimpleBox 使用与 Working Box 相同的 CodeFlow Runtime Home，确保 Image 进入同一 Cache；启动后执行
        ``echo ok`` 验证拉取和最小运行能力。整个步骤受 `create_timeout` 限制，Timeout、依赖或 Platform
        Error 会转换成含操作建议的 `SandboxInitError`。Throwaway Box 离开 Context 后自动销毁。
        """
        import boxlite

        from codeflow.sandbox._runtime import get_boxlite_runtime

        async def _do_pull() -> None:
            # 使用 CodeFlow 的运行时，让镜像进入工作 Box 读取的同一缓存；
            # 否则 SimpleBox 会默认使用 ~/.boxlite。
            async with boxlite.SimpleBox(
                image=self._image,
                cpus=1,
                memory_mib=256,
                runtime=get_boxlite_runtime(),
            ) as pull_box:
                result = await pull_box.exec("sh", "-c", "echo ok", timeout=15)
                if result.exit_code != 0 or result.stdout.strip() != "ok":
                    raise SandboxInitError(
                        f"boxlite image pre-pull check returned unexpected result "
                        f"(exit_code={result.exit_code}, stdout={result.stdout!r})"
                    )

        try:
            await asyncio.wait_for(_do_pull(), timeout=self._create_timeout)
        except asyncio.TimeoutError:
            raise SandboxInitError(
                f"Image pre-pull timed out after {self._create_timeout}s.\n"
                "  • Check network connectivity and registry availability\n"
                "  • Increase create_timeout in sandbox config for large images\n"
                "  • Pre-pull the image on a network-connected machine: boxlite pull <image>"
            )
        except SandboxInitError:
            raise
        except Exception as exc:
            raise SandboxInitError(
                f"Cannot initialise sandbox (image pre-pull failed): {exc}\n"
                "  • Source checkout: uv sync --extra sandbox\n"
                "  • Tool install: uv tool install --force 'codeflow-harness[channels,sandbox]'\n"
                f"  • macOS: requires Apple Silicon M1+ and macOS 12+\n"
                f"  • Linux: requires /dev/kvm accessible to the current user"
            ) from exc

    async def _verify(self, box: Any) -> None:
        """在 Working Box 内运行 ``echo ok``，确认 VM 已 Live。

        方法并发收集 Stdout/Stderr 并等待 Exit，要求 Exit Code 为零且 Stdout Strip 后等于 ``ok``。
        `verify_timeout` 到期会 Kill Execution 并抛出带资源建议的 `SandboxInitError`；其他 SDK Error 也
        统一转换。Probe 只验证最小命令通路，不验证用户 Workspace、额外 Volumes 或 External Network。
        """
        execution: Any = None

        async def _check() -> tuple[str, int]:
            nonlocal execution
            execution = await box.exec("sh", ["-c", "echo ok"])
            stdout_str, _ = await asyncio.gather(
                self._collect(execution.stdout()),
                self._collect(execution.stderr()),
            )
            result = await execution.wait()
            return stdout_str, result.exit_code

        try:
            stdout_str, exit_code = await asyncio.wait_for(_check(), timeout=self._verify_timeout)
        except asyncio.TimeoutError:
            if execution is not None:
                try:
                    await execution.kill()
                except Exception:
                    pass
            raise SandboxInitError(
                f"Sandbox verification timed out after {self._verify_timeout}s "
                "— VM may be unresponsive.\n"
                "  • Check available memory and CPU on the host\n"
                "  • Try restarting the boxlite runtime\n"
                "  • Increase verify_timeout in sandbox config if the host is slow"
            )
        except Exception as exc:
            raise SandboxInitError(
                f"Cannot initialise sandbox: {exc}\n"
                "  • Source checkout: uv sync --extra sandbox\n"
                "  • Tool install: uv tool install --force 'codeflow-harness[channels,sandbox]'\n"
                f"  • macOS: requires Apple Silicon M1+ and macOS 12+\n"
                f"  • Linux: requires /dev/kvm accessible to the current user"
            ) from exc
        if exit_code != 0 or stdout_str.strip() != "ok":
            raise SandboxInitError(
                f"Sandbox verification returned unexpected result (exit_code={exit_code}, stdout={stdout_str!r})"
            )

    async def stop(self) -> None:
        """Kill MCP Server Processes、Cancel Bridge Tasks，再清理 Box。

        顺序先终止 VM 内 Long-running Executions，再取消并 Gather Host Bridge Tasks，最后关闭 Async
        Cleanup Stack、清空 Box Reference。单项 Kill Error 被吞掉以继续释放其余资源；返回后 Owned ID
        应已由 `_cleanup_box` 移除。方法不保留 VM 内非挂载文件的数据。
        """
        for exec_ in self._process_executions:
            try:
                await exec_.kill()
            except Exception:
                pass
        self._process_executions.clear()

        for task in self._process_tasks:
            task.cancel()
        if self._process_tasks:
            await asyncio.gather(*self._process_tasks, return_exceptions=True)
        self._process_tasks.clear()

        await self._stack.aclose()
        self._box = None
        logger.info("Sandbox stopped")

    async def _ensure_box(self) -> Any:
        async with self._init_lock:
            if self._box is not None:
                return self._box

            import boxlite

            # boxlite 0.8.2 中 volumes 是字典，不是元组
            volumes = [
                {"host": str(self._workspace), "guest": self.WORKSPACE_MOUNT, "readonly": False},
                *[{"host": e[0], "guest": e[1], "readonly": e[2] == "ro"} for e in self._extra_volumes],
            ]
            # boxlite 0.8.2 中 network 是字符串字段，allow_net 是独立的列表字段；
            # 该版本不存在 NetworkSpec
            extra_kwargs: dict = {}
            if self._allow_net is False:
                extra_kwargs["network"] = "none"
            elif isinstance(self._allow_net, list):
                extra_kwargs["allow_net"] = self._allow_net
            # 其余情况下 allow_net 为 True，完全开放网络，无需额外参数

            options = boxlite.BoxOptions(
                image=self._image,
                cpus=self._cpus,
                memory_mib=self._memory_mib,
                disk_size_gb=self._disk_size_gb,
                volumes=volumes,
                **extra_kwargs,
            )
            from codeflow.sandbox._runtime import get_boxlite_runtime

            runtime = get_boxlite_runtime()
            self._box = await runtime.create(options)
            # 在 start() 之前注册清理回调，这样 start() 内部失败时仍会销毁 Box，
            # 避免泄漏部分启动的虚拟机。
            if self._owned_ids is not None:
                self._owned_ids.add(self._box.id)
            self._stack.push_async_callback(self._cleanup_box)
            await self._box.start()  # create() 不会自动启动，必须显式调用 start()
            logger.info("Sandbox started (image=%s)", self._image)
            return self._box

    async def _cleanup_box(self) -> None:
        if self._box is not None:
            box_id = self._box.id
            try:
                from codeflow.sandbox._runtime import get_boxlite_runtime

                await self._box.stop()
                try:
                    # Box 在 0.8.2 中没有 remove()，需通过运行时删除
                    await get_boxlite_runtime().remove(box_id)
                except Exception:
                    pass  # stop() 后 Box 可能已经被删除
            except Exception as exc:
                logger.warning("Error cleaning up sandbox box: %s", exc)
            finally:
                # 只在虚拟机完全销毁后才移除所有权。如果提前移除，与清理并发的
                # 调试客户端会看到一台仍在运行但标记为无主的虚拟机，并可能尝试删除它。
                if self._owned_ids is not None:
                    self._owned_ids.discard(box_id)
                self._box = None

    @staticmethod
    async def _collect(stream: AsyncIterator[str]) -> str:
        """把 ``Execution.stdout()`` 或 ``stderr()`` Async Iterator 收集成 Single String。

        假设 BoxLite Yield 已解码 `str` Lines，而不是 Bytes。函数兼容两种 SDK Convention：已有 Trailing
        ``\n`` 的 Line 原样保留，缺少时补一个换行。空 Stream 返回空字符串；它会把全部输出放进内存，
        不适合无界 Long-running Process。
        """
        lines = [line async for line in stream]
        if not lines:
            return ""
        return "".join(line if line.endswith("\n") else line + "\n" for line in lines)

    # ------------------------------------------------------------------
    # 核心执行逻辑
    # ------------------------------------------------------------------

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        box = await self._ensure_box()
        effective_timeout = self._default_timeout if timeout is None else timeout
        vm_cwd = self._translate_cwd(cwd)

        # boxlite 0.8.2 的 Box.exec() 可直接接收 cwd，无需注入 Shell 命令
        env_tuples = list(env.items()) if env else None

        execution: Any = None

        async def _run() -> ExecResult:
            nonlocal execution
            if env_tuples:
                execution = await box.exec("sh", ["-c", command], env_tuples, cwd=vm_cwd)
            else:
                execution = await box.exec("sh", ["-c", command], cwd=vm_cwd)
            stdout_str, stderr_str = await asyncio.gather(
                self._collect(execution.stdout()),
                self._collect(execution.stderr()),
            )
            result = await execution.wait()
            return ExecResult(
                stdout=stdout_str,
                stderr=stderr_str,
                exit_code=result.exit_code,
            )

        try:
            return await asyncio.wait_for(_run(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            if execution is not None:
                try:
                    await execution.kill()
                except Exception:
                    pass
            return ExecResult(
                stdout="",
                stderr=f"Command timed out after {effective_timeout}s",
                exit_code=-1,
            )

    # ------------------------------------------------------------------
    # 启动进程（MCP stdio 服务器）
    # ------------------------------------------------------------------

    @property
    def supports_process_spawning(self) -> bool:
        return True

    async def start_process(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> tuple[Any, Any]:
        """在 VM 内启动 Long-running Process，并返回 MCP-compatible Streams。

        Returns ``(read_stream, write_stream)``：Read Stream 是
        ``anyio MemoryObjectReceiveStream[JSONRPCMessage | Exception]``，Write Stream 是
        ``anyio MemoryObjectSendStream[JSONRPCMessage]``。两者都兼容 MCP SDK `ClientSession`
        Constructor；Caller Owns ``write_send.aclose()``，通常由 ``MCP ClientSession.__aexit__`` 处理。

        方法为 Stdout JSON Lines、Stderr Logging 与 Stdin JSONRPC 建立三个 Bridge Tasks，并跟踪底层
        Execution 供 `stop` 回收。返回 Streams 只表示 Process 已启动，不表示 MCP Handshake 已成功。
        """
        import anyio
        from mcp.shared.message import SessionMessage
        from mcp.types import JSONRPCMessage

        box = await self._ensure_box()

        env_tuples = list(env.items()) if env else None
        if env_tuples:
            execution = await box.exec(command, list(args), env_tuples)
        else:
            execution = await box.exec(command, list(args))
        self._process_executions.append(execution)

        read_send, read_recv = anyio.create_memory_object_stream(max_buffer_size=16)
        write_send, write_recv = anyio.create_memory_object_stream(max_buffer_size=16)

        stdout_iter = execution.stdout()
        stderr_iter = execution.stderr()
        stdin_writer = execution.stdin()

        async def _stdout_bridge() -> None:
            # MCP SDK 1.x 的读流传递包装 JSONRPCMessage 的 SessionMessage。
            # boxlite stdout 输出的是数据块，大消息不保证按行对齐，因此需先缓冲，
            # 再按 '\n' 分割以重建完整 JSON 行。
            buf = ""
            try:
                async for chunk in stdout_iter:
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rpc_msg = JSONRPCMessage.model_validate_json(line)
                            await read_send.send(SessionMessage(message=rpc_msg))
                        except Exception as parse_exc:
                            # 非 JSON 的标准输出（如启动横幅）只记录日志并跳过。
                            logger.debug(
                                "MCP stdout [%s]: skipping non-JSON line %r (%s)",
                                command,
                                line[:80],
                                parse_exc,
                            )
                # 流结束后处理没有末尾换行符的剩余内容
                line = buf.strip()
                if line:
                    try:
                        rpc_msg = JSONRPCMessage.model_validate_json(line)
                        await read_send.send(SessionMessage(message=rpc_msg))
                    except Exception:
                        pass
            except (anyio.ClosedResourceError, anyio.EndOfStream):
                pass
            except Exception as exc:
                logger.error("MCP stdout bridge error: %s", exc)
            finally:
                await read_send.aclose()

        async def _stderr_bridge() -> None:
            """把 VM Stderr 转发到 Logger 的 WARNING Level。

            每个非空 Line 带 Command Context 记录；Stream 正常关闭静默结束，其他异常记录 Error。该
            Bridge 只提供诊断，不把 Stderr 发送到 MCP JSONRPC Read Stream。
            """
            try:
                async for raw_line in stderr_iter:
                    line = raw_line.strip()
                    if line:
                        logger.warning("MCP server stderr [%s]: %s", command, line)
            except (anyio.ClosedResourceError, anyio.EndOfStream):
                pass
            except Exception as exc:
                logger.error("MCP stderr bridge error: %s", exc)

        async def _stdin_bridge() -> None:
            # MCP SDK 1.x 的写流传递 SessionMessage；将其内部 JSONRPCMessage
            # 序列化为换行分隔的 JSON，再写入虚拟机进程的标准输入。
            try:
                async for session_msg in write_recv:
                    rpc_msg = session_msg.message if isinstance(session_msg, SessionMessage) else session_msg
                    line = rpc_msg.model_dump_json(exclude_none=True, by_alias=True) + "\n"
                    await stdin_writer.send_input(line.encode("utf-8"))
            except (anyio.ClosedResourceError, anyio.EndOfStream):
                pass
            except Exception as exc:
                logger.error("MCP stdin bridge error: %s", exc)
            finally:
                await write_recv.aclose()

        # create_task 和 extend 之间没有 await；asyncio 单线程执行，此处不会切换上下文，
        # 因此 stop() 一定能看到这三个任务。
        tasks = [
            asyncio.create_task(_stdout_bridge()),
            asyncio.create_task(_stderr_bridge()),
            asyncio.create_task(_stdin_bridge()),
        ]
        self._process_tasks.extend(tasks)

        return read_recv, write_send

    def _translate_cwd(self, cwd: str | None) -> str:
        """把 Host Workspace Path 映射为 VM ``/workspace/...`` Path。

        `cwd=None` 或 Workspace Root 映射到 Mount Root；内部子路径保留 Relative Suffix。解析后位于
        Workspace 外的 Path 不会映射任意 Host Directory，而是记录 Warning 并回退到 ``/workspace``，
        防止 CWD 绕过已挂载工作区边界。
        """
        if cwd is None:
            return self.WORKSPACE_MOUNT
        host_path = Path(cwd).resolve()
        try:
            rel = host_path.relative_to(self._workspace.resolve())
            rel_str = str(rel)
            return self.WORKSPACE_MOUNT if rel_str == "." else f"{self.WORKSPACE_MOUNT}/{rel_str}"
        except ValueError:
            logger.warning("cwd '%s' is outside workspace; falling back to /workspace", cwd)
            return self.WORKSPACE_MOUNT
