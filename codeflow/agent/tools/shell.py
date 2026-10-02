"""实现带 Sandbox、Workspace 与危险命令防护的 Shell Execution Tool。

`ExecTool` 把模型 command 交给注入的 SandboxExecutor；默认 DirectExecutor 走 Host-side deny/
allow list 与 Workspace 检查，真实 Sandbox 已有 VM 隔离时跳过 pattern deny，但仍尊重 Operator
配置的 Workspace boundary。Timeout、PATH 注入和输出长度统一受限，非零 exit code 通过
`ToolResult.failed` 明确表达。
"""

import os
import re
import shlex
from pathlib import Path
from typing import Any

from codeflow.agent.tools.base import Tool, ToolResult
from codeflow.sandbox import DirectExecutor, SandboxExecutor


class ExecTool(Tool):
    """在受控 Working Directory 中执行一条 Shell command 并返回有限输出。

    构造参数可设置默认 timeout、cwd、deny/allow regex、Workspace restriction、PATH append 与
    Executor。单调用 timeout 最大 600 秒，Registry 外层上限 660 秒只捕获 Executor 本身卡死；
    输出经 Executor result 截到 10,000 字符。Direct 模式使用环境白名单，绝不把完整
    ``os.environ`` 交给执行器；Sandbox 模式在 VM 内包装 PATH，避免 Host credential 泄漏。

    Safety guard 是 best-effort，不替代 OS/VM 权限。危险 pattern、path traversal、Workspace 外
    cwd 或命令中的绝对外部路径会返回 Error；执行异常也转成文本。返回成功文本只证明 command
    结束，``exit_code != 0`` 会显式标为 failed。
    """

    # 高于 exec 内部 600 秒上限（``_MAX_TIMEOUT``）的兜底值；执行器自身超时会先触发，
    # 此值只用于捕获完全卡死的执行器。
    timeout_seconds = 660.0

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        executor: SandboxExecutor | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # 匹配 rm -r、rm -rf、rm -fr
            r"\bdel\s+/[fq]\b",  # 匹配 del /f、del /q
            r"\brmdir\s+/s\b",  # 匹配 rmdir /s
            r"(?:^|[;&|]\s*)format\b",  # format（仅作为独立命令时）
            r"\b(mkfs|diskpart)\b",  # 磁盘操作
            r"\bdd\s+if=",  # 匹配 dd
            r">\s*/dev/sd",  # 写入磁盘
            r"\b(shutdown|reboot|poweroff)\b",  # 系统电源
            r":\(\)\s*\{.*\};\s*:",  # fork 炸弹
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append
        self._executor: SandboxExecutor = executor if executor is not None else DirectExecutor()

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()

        if not self._executor.is_sandboxed:
            # 非沙箱模式：完整保护，同时应用拒绝列表模式和工作区限制。
            guard_error = self._guard_command(command, cwd)
            if guard_error:
                return guard_error
        elif self.restrict_to_workspace:
            # 沙箱模式：微型虚拟机已提供真隔离，因此跳过拒绝列表；仍强制工作区限制，
            # 以遵守操作者设定的边界。
            workspace_error = self._check_workspace_restriction(command, cwd)
            if workspace_error:
                return workspace_error

        # 使用 `is None` 检查，因为 `timeout or default` 会将 timeout=0 误视为假值。
        effective_timeout = min(self.timeout if timeout is None else timeout, self._MAX_TIMEOUT)

        env: dict[str, str] | None = None
        if self.path_append:
            if self._executor.is_sandboxed:
                # 通过命令包装在虚拟机内注入路径；绝不向沙箱执行器传入 os.environ，
                # 否则会将宿主机凭据泄漏到虚拟机。
                command = f'export PATH="$PATH:{shlex.quote(self.path_append)}" && {command}'
            else:
                # 只传入 PATH 覆盖值。此处复制 os.environ 会把完整宿主机环境交给 DirectExecutor，
                # 破坏其基线白名单隔离；其余变量由执行器提供。
                base_path = os.environ.get("PATH", "")
                env = {"PATH": base_path + os.pathsep + self.path_append}

        try:
            result = await self._executor.exec(command, cwd=cwd, timeout=effective_timeout, env=env)
        except Exception as e:
            return f"Error executing command: {str(e)}"
        return ToolResult(
            result.as_text(self._MAX_OUTPUT),
            failed=result.exit_code != 0,
        )

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """对 Direct execution 执行危险 pattern、allowlist 与 Workspace 的 best-effort guard。

        Command 先 strip/lower，再依次匹配 deny_patterns；命中即拒绝。配置 allow_patterns 时至少
        命中一项，否则拒绝。最后委托 `_check_workspace_restriction` 验证路径。返回 ``None`` 表示
        这些静态检查通过，不证明命令无副作用，也不解析 Shell 的完整动态语义。
        """
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns and not any(re.search(pattern, lower) for pattern in self.allow_patterns):
            return "Error: Command blocked by safety guard (not in allowlist)"

        workspace_error = self._check_workspace_restriction(command, cwd)
        if workspace_error:
            return workspace_error

        return None

    def _check_workspace_restriction(self, command: str, cwd: str) -> str | None:
        """只检查 Workspace 路径边界，不应用 deny/allow-list。

        未启用 ``restrict_to_workspace`` 时立即允许。启用后拒绝 ``../``、``..\\`` traversal，要求
        resolved cwd 位于配置 Workspace 内，并从 command 提取 Windows、POSIX 与 Home absolute
        paths；环境变量和 ``~`` 展开后落在 Workspace 外的路径也拒绝。无法解析单个路径时跳过，
        因此这是防护层而非完整 Shell parser。
        """
        if not self.restrict_to_workspace:
            return None

        cmd = command.strip()
        if "..\\" in cmd or "../" in cmd:
            return "Error: Command blocked by safety guard (path traversal detected)"

        workspace_path = Path(self.working_dir or os.getcwd()).expanduser().resolve()
        cwd_path = Path(cwd).expanduser().resolve()
        if not cwd_path.is_relative_to(workspace_path):
            return "Error: Command blocked by safety guard (working directory outside workspace)"

        for raw in self._extract_absolute_paths(cmd):
            try:
                expanded = os.path.expandvars(raw.strip())
                p = Path(expanded).expanduser().resolve()
            except Exception:
                continue
            if p.is_absolute() and not p.is_relative_to(workspace_path):
                return "Error: Command blocked by safety guard (path outside workspace)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command)
        return win_paths + posix_paths + home_paths
