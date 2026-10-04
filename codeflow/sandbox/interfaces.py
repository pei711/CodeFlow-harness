"""Sandbox Package 的 Public Interfaces。

Caller 应从这里或 ``codeflow.sandbox`` 导入，Never 直接依赖 ``boxlite_executor.py`` /
``direct_executor.py``，这样 `ExecTool`、Agent Loop 与 MCP 连接逻辑只面向稳定 ABC，不耦合 Concrete
Backends。接口同时明确隔离状态、Process Spawning Capability 与 Start/Stop Lifecycle。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class SandboxInitError(RuntimeError):
    """Sandbox Backend 无法 Start 或 Probe 时抛出。

    类型定义在 ``interfaces.py``，而不是 ``boxlite_executor.py``，使调用方在未安装 BoxLite 时也能安全
    Import。``mcp.py`` 与 ``loop.py`` 依赖此类型做 Error Handling，它们不能只因 Optional BoxLite
    Absent 就在模块导入阶段失败。
    """


@dataclass
class ExecResult:
    """一次 Sandbox Command Execution 的结构化 Result。

    `stdout`、`stderr` 与 `exit_code` 保留执行收据；`as_text` 再组合成适合送回 Agent Context 的文本，并
    对超长输出保留 Head/Tail。对象名不保证实际 Isolated，`DirectExecutor` 也返回同一类型，隔离事实
    必须查看 Executor `is_sandboxed`。
    """

    stdout: str
    stderr: str
    exit_code: int

    def as_text(self, max_chars: int = 10_000) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr}")
        parts.append(f"\nExit code: {self.exit_code}")
        result = "\n".join(parts)
        if len(result) > max_chars:
            half = max_chars // 2
            result = result[:half] + f"\n\n... ({len(result) - max_chars:,} chars truncated) ...\n\n" + result[-half:]
        return result


class SandboxExecutor(ABC):
    """Command Execution 的 Backend Abstraction。

    Implementations 包括 BoxLite MicroVM 的 `BoxliteExecutor`、SRT 命令隔离的 `SrtExecutor` 与 Host Fallback 的 `DirectExecutor`。
    `ExecTool` 只持有该 Interface，不知道 Concrete Backend。生命周期是构造、`start`、多次 `exec` 或
    `start_process`、最后 `stop`；Async Context Manager 自动遵循这一顺序。

    名称中的 Sandbox 不等于所有实现都隔离，Caller 必须读取 `is_sandboxed`。
    """

    @property
    def is_sandboxed(self) -> bool:
        """命令在 Isolated Environment 而非 Host Process 中运行时为 `True`。

        `ExecTool` 用此 Flag 决定是否应用 Regex Deny-list Guard。`DirectExecutor` 明确 Override 为
        `False`，其他实现默认 `True`。Base Class 刻意默认 True：Custom Executor 忘记 Override 时会跳过
        Denylist，而设计假设其真实隔离来自 Sandbox；必须 Opt Out 的唯一内置实现是 Host Execution。

        自定义实现若没有真实隔离却继承默认值会造成危险误报，因此实现者必须把此属性当作 Security
        Contract，而不是便利标签。
        """
        return True

    @property
    def supports_process_spawning(self) -> bool:
        """实现 Long-running Child Process 的 ``start_process()`` 时为 `True`。

        ``connect_mcp_servers()`` 在 Stdio MCP Branch 检查此 Flag，而不是使用 `isinstance`，使 Caller 与
        Concrete Executor Types 解耦。`DirectExecutor` 与 Base Class 默认 `False`；声明 True 的实现必须
        同时负责 Streams 与 Child Lifecycle Cleanup。
        """
        return False

    @abstractmethod
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """执行 Shell Command，并返回 ``stdout`` / ``stderr`` / ``exit_code``。

        `cwd`、`timeout` 与 `env` 由具体 Backend 解释；实现必须在 Timeout 后终止或隔离残余 Process，
        并把输出封装为 `ExecResult`。Abstract Method 不规定命令是否在 Host 或 VM，隔离由属性说明。
        """

    async def start_process(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> tuple[Any, Any]:
        """启动 Long-running Child Process，返回 ``(read_stream, write_stream)``。

        Streams 是兼容 MCP SDK `ClientSession` 的 AnyIO `MemoryObjectReceiveStream` /
        `MemoryObjectSendStream`。只有把 `supports_process_spawning` Override 为 `True` 的 Executor 才应
        实现；Base Method 抛出 `NotImplementedError`，防止 Stdio MCP 被错误地认为已沙箱化。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support process spawning. "
            "Stdio MCP servers cannot be sandboxed with this executor."
        )

    async def start(self) -> None:
        """Lifecycle：在 First Exec 前调用一次；Default 为 No-op。

        隔离 Backend 可在此创建 VM 并 Probe 可用性。失败应抛出 `SandboxInitError`，且负责清理部分启动
        资源。
        """

    async def stop(self) -> None:
        """Lifecycle：Graceful Shutdown 时调用；Default 为 No-op。

        实现应终止 Child Processes、Bridge Tasks 与 VM，并使重复清理尽量安全。Async Context Manager 的
        `__aexit__` 会自动调用它。
        """

    async def __aenter__(self) -> SandboxExecutor:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()
