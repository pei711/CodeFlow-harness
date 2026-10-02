"""通过 local Claude CLI 为 coding-plan subscription 构造 driver ``call_fn``。

Every evolver driver role — diagnose / taxonomy induction / design (bash-editor)
/ verdict — consumes the same seam, the sync ``CallFn`` from
:mod:`~codeflow.evolver.orchestrator.nodes.semantic`. :func:`make_claude_call_fn`
returns that CallFn backed by the local ``claude`` CLI in print mode, so driver
calls ride the logged-in Claude Code subscription instead of a per-token API
endpoint. The FSM keeps control: ``--allowedTools ""`` disables Claude's own
tools, making each call a pure chat completion (Claude-as-completion inside the
existing loops, isomorphic to the qwen path in :mod:`.openai_compat`).

Operational notes baked in:

- The coding plan rate-limits under fan-out (diagnose runs a thread pool), so
  each CallFn carries its own concurrency semaphore (default 4) plus
  backoff-and-retry that stretches the delay on rate-limit signatures.
- The prompt goes via stdin, not argv — rendered trajectories overflow argv.
- Multi-turn conversations (the bash-editor accumulates action/observation
  turns) are serialized into one role-tagged transcript; print mode accepts a
  single prompt and cannot replay assistant turns natively。

该 transport 只返回 raw assistant text；schema validation 与 candidate evidence 属于 downstream。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from typing import Callable, Optional, Sequence

from codeflow.evolver.orchestrator.nodes.semantic import CallFn, Messages

_CONTINUE_RULE = (
    "You are the assistant in the conversation transcript below. Reply with the "
    "assistant's NEXT message only — no role tag, no commentary about the "
    "transcript format."
)


def render_messages(messages: Messages) -> tuple[str, str]:
    """把 chat messages 拆成 ``claude -p`` 使用的 ``(system_prompt, user_prompt)``。

    A single user message passes through verbatim. A multi-turn history is
    serialized with role tags plus a continue-instruction, since print mode
    accepts one prompt, not a message array。函数不调用 CLI。
    """
    system = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1:
        return system, str(turns[0].get("content", ""))
    lines = [_CONTINUE_RULE, ""]
    for m in turns:
        lines.append(f"[{str(m.get('role', 'user')).upper()}]")
        lines.append(str(m.get("content", "")))
    return system, "\n".join(lines)


def _looks_rate_limited(text: str) -> bool:
    t = text.lower()
    return "rate limit" in t or "429" in t or "overloaded" in t or "usage limit" in t


def make_claude_call_fn(
    model: str,
    *,
    claude_bin: str = "claude",
    timeout: float = 300.0,
    retry_delays: Sequence[float] = (5.0, 15.0, 45.0, 90.0),
    rate_limit_delay: float = 120.0,
    max_concurrency: int = 4,
    extra_args: Sequence[str] = (),
    cwd: Optional[str] = None,
    run: Optional[Callable[..., "subprocess.CompletedProcess"]] = None,
) -> CallFn:
    """构造每次运行 ``claude -p --model <model>`` 的同步 ``call_fn``。

    ``run`` overrides ``subprocess.run`` for tests. The semaphore is
    per-CallFn: two roles (e.g. haiku diagnose + opus design) each get their
    own lane, but one fan-out caller (``classify_failures``'s thread pool)
    cannot exceed ``max_concurrency`` concurrent CLI processes on the shared
    plan quota.

    ``cwd`` defaults to a fresh empty temp dir, NEVER the inherited process
    cwd: Claude Code injects the cwd's project context (CLAUDE.md, env info)
    into the model even under ``--system-prompt``. Inheriting the orchestrator
    repo's cwd leaked the real repo path to the design driver, which then
    ``cd`` 出 sandbox worktree 浏览（verified live 2026-07-09）。每个 CallFn 拥有自己的
    semaphore 与 retry；耗尽后抛出 ``RuntimeError``，不返回空内容。
    """
    _run = run or subprocess.run
    workdir = cwd or tempfile.mkdtemp(prefix="claude-driver-")
    gate = threading.Semaphore(max_concurrency)

    def _once(system: str, prompt: str) -> str:
        argv = [
            claude_bin,
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--allowedTools",
            "",
            *(["--system-prompt", system] if system else []),
            *extra_args,
        ]
        with gate:
            proc = _run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
            )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p exit {proc.returncode}: {(proc.stderr or proc.stdout)[:400]}")
        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise RuntimeError(f"claude -p is_error: {str(data.get('result'))[:400]}")
        result = data.get("result")
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(f"claude -p empty result: {proc.stdout[:400]}")
        return result

    def call_fn(messages: Messages) -> str:
        system, prompt = render_messages(messages)
        last: Exception | None = None
        for attempt in range(len(retry_delays) + 1):
            try:
                return _once(system, prompt)
            except Exception as exc:  # noqa: BLE001 — 重试后明确暴露错误
                last = exc
                if attempt >= len(retry_delays):
                    break
                delay = retry_delays[attempt]
                if _looks_rate_limited(str(exc)):
                    delay = max(delay, rate_limit_delay)
                time.sleep(delay)
        raise RuntimeError(f"claude driver ({model}) failed after {len(retry_delays) + 1} attempts") from last

    return call_fn


__all__ = ["make_claude_call_fn", "render_messages"]
