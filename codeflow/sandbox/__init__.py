"""Sandbox Package，为 Python Agents 提供 Self-contained Isolated Command Execution。

Public API 应全部从这里导入，不直接依赖 Sub-modules：

- `SandboxInitError`：Sandbox Backend 启动失败时抛出；
- `ExecResult`：一次 ``exec()`` Call 的结果；
- `SandboxExecutor`：Executor Implementations 的 ABC；
- `SandboxConfig`：Pydantic Config Model；
- `DirectExecutor`：在 Host Process 直接执行的 Fallback，**No Isolation**；
- ``build_executor()``：把 `SandboxConfig` 构造成 `SandboxExecutor` 的 Factory。

Sandbox 只隔离经 Executor 发出的命令。选择 `none` 会让 Prompt-injected Command 获得完整 Host
Privileges，不能把统一接口误认为已经建立隔离边界。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from codeflow.sandbox.config import SandboxConfig
from codeflow.sandbox.direct_executor import DirectExecutor
from codeflow.sandbox.interfaces import ExecResult, SandboxExecutor, SandboxInitError

# 每个进程只警告一次：进程生命周期内会创建多个执行器（AgentLoop 及各子 Agent），
# 但“未使用沙箱”的风险提示只需输出一次。
_warned_no_sandbox = False

__all__ = [
    "ExecResult",
    "SandboxExecutor",
    "SandboxInitError",
    "SandboxConfig",
    "DirectExecutor",
    "build_executor",
]


def build_executor(
    sandbox_cfg: SandboxConfig | None,
    workspace: Path,
    owned_ids: set[str] | None = None,
) -> SandboxExecutor:
    """根据 Config 同步构造 Executor，不在此启动 Backend。

    Object Creation 始终 Sync and Cheap；Probe / VM Initialisation 延后到 ``executor.start()`` 或
    ``__aenter__()``。Backend 无法启动时，后续 Start 会传播 `SandboxInitError`。`none` 返回
    `DirectExecutor` 并在 Process 内只警告一次；`auto` / `boxlite` 要求可导入 BoxLite，否则给出明确
    安装命令。Unknown Backend 立即抛错。

    `owned_ids` 是 Optional Shared Set，`BoxliteExecutor` 在 Start 时加入自己的 VM ID、Stop 时移除。
    `SandboxDebugServer` 用它区分本 Process 拥有的 VMs 与其他 Process 的 VMs。Factory 返回只表示对象
    已创建，不证明 VM 已就绪或命令已经隔离。
    """
    backend = sandbox_cfg.backend if sandbox_cfg else "none"

    if backend == "none":
        global _warned_no_sandbox
        if not _warned_no_sandbox:
            _warned_no_sandbox = True
            logger.warning(
                "Sandbox backend is 'none' — agent commands run directly on the "
                "host with no isolation. Prompt-injected commands execute with "
                "full host privileges. Set tools.sandbox.backend to 'auto' or "
                "'boxlite' to contain them."
            )
        return DirectExecutor()

    if backend in ("auto", "boxlite"):
        try:
            import boxlite as _  # noqa: F401 — 构造执行器前探测可用性

            from codeflow.sandbox.boxlite_executor import BoxliteExecutor
        except ImportError as exc:
            raise SandboxInitError(
                f"No sandbox backend available: {exc}\n"
                "  • Source checkout: uv sync --extra sandbox\n"
                "  • Tool install: uv tool install --force 'codeflow-harness[channels,sandbox]'"
            ) from exc
        return BoxliteExecutor(
            image=sandbox_cfg.image,
            workspace=workspace,
            cpus=sandbox_cfg.cpus,
            memory_mib=sandbox_cfg.memory_mib,
            disk_size_gb=sandbox_cfg.disk_size_gb,
            allow_net=sandbox_cfg.allow_net,
            extra_volumes=sandbox_cfg.extra_volumes,
            default_timeout=sandbox_cfg.default_timeout,
            verify_timeout=sandbox_cfg.verify_timeout,
            create_timeout=sandbox_cfg.create_timeout,
            owned_ids=owned_ids,
        )

    raise SandboxInitError(f"Unknown sandbox backend: {backend!r}. Valid values: 'none', 'auto', 'boxlite'.")
