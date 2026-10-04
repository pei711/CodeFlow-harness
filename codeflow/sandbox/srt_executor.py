"""Shell executor backed by Anthropic Sandbox Runtime (SRT).

SRT isolates one-shot shell commands using the host operating system's sandbox.
It does not provide a persistent VM or long-running process/MCP support.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from codeflow.sandbox.direct_executor import _baseline_env
from codeflow.sandbox.interfaces import ExecResult, SandboxExecutor, SandboxInitError

_DEFAULT_TIMEOUT = 120
_MAX_TIMEOUT = 600
_VERIFY_TIMEOUT = 30


class SrtExecutor(SandboxExecutor):
    """Run commands through the SRT CLI with a workspace-scoped policy."""

    def __init__(
        self,
        workspace: Path,
        cli_path: Path | None = None,
        settings_path: Path | None = None,
    ) -> None:
        self._workspace = workspace.resolve()
        package_root = Path(__file__).resolve().parents[2]
        self._cli_path = Path(
            cli_path or package_root / "node_modules" / "@anthropic-ai" / "sandbox-runtime" / "dist" / "cli.js"
        ).expanduser().resolve()
        self._settings_path = Path(
            settings_path or Path(__file__).with_name("srt-settings.json")
        ).expanduser().resolve()

    @property
    def supports_process_spawning(self) -> bool:
        return False

    def _runtime(self, env: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
        child_env = {**_baseline_env(), **(env or {})}
        node_path = shutil.which("node", path=child_env.get("PATH"))
        if not node_path:
            raise SandboxInitError("SRT sandbox unavailable: Node.js was not found on PATH")
        if not self._cli_path.is_file():
            raise SandboxInitError(
                "SRT sandbox unavailable: CLI not found; run `npm install` in the CodeFlow-harness checkout"
            )
        if not self._settings_path.is_file():
            raise SandboxInitError(f"SRT sandbox unavailable: settings file not found: {self._settings_path}")
        return node_path, child_env

    async def start(self) -> None:
        """Fail early when Node, the SRT package, or OS sandbox setup is unavailable."""
        try:
            result = await self.exec("echo codeflow-srt-ready", cwd=str(self._workspace), timeout=_VERIFY_TIMEOUT)
        except SandboxInitError:
            raise
        except Exception as exc:
            raise SandboxInitError(f"SRT sandbox initialization failed: {exc}") from exc
        if result.exit_code != 0 or result.stdout.strip() != "codeflow-srt-ready":
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.exit_code}"
            raise SandboxInitError(f"SRT sandbox initialization failed: {detail}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        node_path, child_env = self._runtime(env)
        working_dir = Path(cwd).expanduser().resolve() if cwd else self._workspace
        if not working_dir.is_relative_to(self._workspace):
            raise SandboxInitError(f"SRT command working directory escapes workspace: {working_dir}")
        effective_timeout = min(_DEFAULT_TIMEOUT if timeout is None else timeout, _MAX_TIMEOUT)
        process = await asyncio.create_subprocess_exec(
            node_path,
            str(self._cli_path),
            "--settings",
            str(self._settings_path),
            "-c",
            command,
            cwd=str(working_dir),
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=effective_timeout)
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return ExecResult(stdout="", stderr=f"Command timed out after {effective_timeout}s", exit_code=-1)
        return ExecResult(
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            exit_code=process.returncode,
        )
