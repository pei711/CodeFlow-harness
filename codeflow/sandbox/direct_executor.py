"""`DirectExecutor`：直接在 Host Process 上运行 Commands，**No Isolation**。

这是 Sandbox Backend 为 ``none`` 时的兼容 Fallback。它仍限制单次 Timeout，并只把明确 Allowlist 的
基础环境变量加上 Caller 显式 Env 传给子进程，降低 Prompt Injection 直接读取 API Keys/Cloud
Credentials 的风险；但 Command 拥有当前用户对 Host Filesystem 与 Network 的全部权限，环境过滤不
等于沙箱。
"""

from __future__ import annotations

import asyncio
import os

from codeflow.sandbox.interfaces import ExecResult, SandboxExecutor

_DEFAULT_TIMEOUT = 60
_MAX_TIMEOUT = 600

# DirectExecutor 直接在宿主机上运行，没有隔离。如果 Agent 受到提示词注入诱导，
# 其执行的命令原本会继承宿主机的全部环境变量，包括凭据。因此只传入
# 最小的非敏感基线环境，以及调用方明确提供的变量。
_ENV_ALLOWLIST = (
    # 区域设置和 Shell 基础变量
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "USER",
    "LOGNAME",
    "SHELL",
    "PWD",
    "TZ",
    "TMPDIR",
    # 语言运行时（保证 Python、Node 和基于虚拟环境的工具可正常解析）
    "PYTHONPATH",
    "VIRTUAL_ENV",
    # TLS 信任配置和代理（使 git、curl 和 HTTPS 工具能在企业网络中工作）。
    # 这些是配置，而非高价值密钥；API 密钥、云凭据和 SSH 信息被明确排除。
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    # Windows 基础环境变量：POSIX 上不存在，会被 _baseline_env 过滤；Windows 上的
    # cmd.exe、PowerShell 及子工具需要它们来定位临时目录、用户配置和系统 DLL。
    # 如果省略，子进程将缺少 SystemRoot、TEMP 等变量，临时文件会落到当前目录，
    # SSL、Winsock 和 .NET 工具也会失败。这些变量都不是高价值密钥。
    "SystemRoot",
    "SystemDrive",
    "windir",
    "COMSPEC",
    "ComSpec",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "USERNAME",
    "USERDOMAIN",
)


def _baseline_env() -> dict[str, str]:
    return {k: v for k in _ENV_ALLOWLIST if (v := os.environ.get(k)) is not None}


class DirectExecutor(SandboxExecutor):
    """No-op Sandbox，在 Host 直接执行命令的 Executor。

    `is_sandboxed` 明确返回 `False`。`exec` 使用 Shell Subprocess，捕获 Stdout/Stderr，Timeout 最多限制
    为 600 Seconds；超时会 Kill Process 并返回 Exit Code -1。输出以 UTF-8 ``errors="replace"`` 解码。

    它不实现 Long-running Process Spawning，也不隔离 CWD、Filesystem 或 Network。只有用户明确选择
    Host Execution、或运行环境无法使用 BoxLite 时才应使用，并把命令视为等同当前账户手动执行。
    """

    @property
    def is_sandboxed(self) -> bool:
        return False

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        effective_timeout = min(
            _DEFAULT_TIMEOUT if timeout is None else timeout,
            _MAX_TIMEOUT,
        )
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env={**_baseline_env(), **(env or {})},
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return ExecResult(stdout="", stderr=f"Timed out after {effective_timeout}s", exit_code=-1)
        return ExecResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
        )
