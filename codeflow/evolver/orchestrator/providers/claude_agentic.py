"""运行拥有完整分析能力、但被锁定为 read-only 的 Claude Code Agentic session。

The map-reduce diagnosis reads every failing trajectory once, shallowly. The
``agentic`` analysis mode replaces that front half with one Claude Code session
that investigates the run the way an engineer would — ledger first, then
deep-reads of representative transcripts, then the harness source — and returns
one structured diagnosis (meta-harness style, but constrained to our taxonomy
so the failure_map/history/GSME machinery downstream is unchanged).

Safety model (the session must not be able to leave the safe zone):

- The tool whitelist is **hard-coded** to ``Read, Glob, Grep`` — no Bash, no
  Write/Edit, no Agent, and callers cannot widen it. In ``-p`` mode any tool
  outside ``--allowedTools`` is auto-denied, and ``--dangerously-skip-permissions``
  is never passed, so the session can inspect but cannot modify or execute.
- ``cwd`` is the assembled analysis workspace (digest + run data + a pinned
  harness worktree), not the live repo — no CLAUDE.md/project injection and
  nothing load-bearing to touch even if a write slipped through.
* 该 mode 只允许 Claude model，本模块强制验证；analyst 像 :mod:`.claude_cli` 一样使用 local
  CLI 已登录 subscription。

session 返回 structured diagnosis 只表示分析调用完成，不是 benchmark verdict、candidate
promotion 或任务完成证据。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

AGENTIC_TOOLS = ("Read", "Glob", "Grep")


def claude_cli_available(claude_bin: str = "claude") -> bool:
    """仅当 Claude CLI 在 ``PATH`` 且 ``--version`` 成功时返回 ``True``。

    binary 缺失、timeout 或任意 probe exception 都返回 ``False``；该探测不验证 subscription
    quota 或后续 model call 一定成功。
    """
    if shutil.which(claude_bin) is None:
        return False
    try:
        r = subprocess.run([claude_bin, "--version"], capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001 — 任意探测失败都表示不可用
        return False
    return r.returncode == 0


def require_claude_for_agentic(model: str, claude_bin: str = "claude") -> None:
    """校验 ``analysis_mode="agentic"`` 只使用 Claude model 与可工作的 local CLI。

    model name 不以 ``claude`` 开头时抛出 ``ValueError``；CLI probe 失败时抛出
    ``RuntimeError``。函数不启动 analysis session。
    """
    if not str(model).startswith("claude"):
        raise ValueError(
            f"analysis_mode='agentic' only runs on Claude models via the local "
            f"CLI (got model={model!r}); use analysis_mode='mapreduce' for other drivers"
        )
    if not claude_cli_available(claude_bin):
        raise RuntimeError(
            f"analysis_mode='agentic' requires the '{claude_bin}' CLI on PATH "
            "(logged-in Claude Code); it was not found or does not answer --version"
        )


def run_agentic_session(
    prompt: str,
    *,
    system_prompt: str,
    cwd: str | Path,
    model: str = "claude-opus-4-8",
    claude_bin: str = "claude",
    timeout: float = 1800.0,
    add_dirs: tuple = (),
    run: Optional[Callable[..., "subprocess.CompletedProcess"]] = None,
) -> str:
    """运行一次 read-only Claude Code Agentic session，返回 final text。

    ``--append-system-prompt`` (not ``--system-prompt``) keeps Claude Code's
    native agentic system prompt so Read/Glob/Grep are actually driven well;
    our instructions ride on top.

    ``add_dirs`` grants READ access to trees outside ``cwd``: print-mode file
    access is confined to the cwd subtree by *resolved* path, so workspace
    symlinks pointing at run data are denied without it (observed live: the
    analyst reported the data 'absent' and degraded to signature analogy).
    The tool whitelist stays read-only, so the grant widens visibility, not
    write reach。非零 exit、``is_error`` 或 empty result 都抛出 ``RuntimeError``。
    """
    require_claude_for_agentic(model, claude_bin)
    _run = run or subprocess.run
    argv = [
        claude_bin,
        "-p",
        "--model",
        model,
        "--output-format",
        "json",
        "--allowedTools",
        ",".join(AGENTIC_TOOLS),
        "--disallowedTools",
        "Bash,Write,Edit,NotebookEdit,Agent,WebFetch,WebSearch",
        "--append-system-prompt",
        system_prompt,
    ]
    for d in add_dirs:
        argv += ["--add-dir", str(d)]
    proc = _run(
        argv,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"agentic session exit {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"agentic session is_error: {str(data.get('result'))[:400]}")
    result = data.get("result")
    if not isinstance(result, str) or not result.strip():
        raise RuntimeError(f"agentic session empty result: {proc.stdout[:400]}")
    return result


__all__ = [
    "AGENTIC_TOOLS",
    "claude_cli_available",
    "require_claude_for_agentic",
    "run_agentic_session",
]
