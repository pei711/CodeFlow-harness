"""为 semantic-node layer 提供同步 OpenAI-compatible ``call_fn``。

The driver models are served behind OpenAI-compatible ``/v1`` endpoints
(self-hosted Qwen / Kimi via vLLM). :func:`make_call_fn` returns the sync
``CallFn`` a :class:`~codeflow.evolver.orchestrator.nodes.semantic.SemanticNode`
expects — messages in, assistant text out.

Two behaviours matter for these specific models:

- **Reasoning models.** Qwen3.5/3.6 emit a large ``reasoning`` field before the
  real answer lands in ``message.content``. A small ``max_tokens`` gets consumed
  entirely by reasoning and returns ``content == null``. So the default token
  budget here is deliberately generous, and an empty/None content is retried
  rather than parsed.
- **No key required.** vLLM accepts any bearer token; ``api_key`` defaults to
  ``"EMPTY"`` and can be overridden from the environment.

这是 plain sync httpx transport，可与 synchronous Orchestrator FSM 组合，无需贯穿 event loop。
成功只表示得到 non-empty assistant content，不代表 semantic parse 或 evidence gate 通过。
"""

from __future__ import annotations

import os
import time
from typing import Optional

from codeflow.evolver.orchestrator.nodes.semantic import CallFn, Messages


class EndpointError(RuntimeError):
    """endpoint 在全部 retry 后仍未返回 usable content 时抛出的异常。"""


def _extract_content(data: dict) -> Optional[str]:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def make_call_fn(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    api_key_env: str = "EVOLVER_DRIVER_API_KEY",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    timeout: float = 180.0,
    retry_delays: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> CallFn:
    """构造绑定到一个 OpenAI-compatible chat endpoint 的同步 ``call_fn``。

    ``base_url`` is the ``/v1`` root; ``/chat/completions`` is appended. Empty or
    missing content is retried with backoff (reasoning models occasionally emit
    no answer); after the final attempt an :class:`EndpointError` is raised so a
    node failure 会 loud fail，而不是返回 silent empty string。API key 顺序为 explicit、env、
    ``EMPTY``；函数构造阶段不访问 network。
    """
    key = api_key or os.environ.get(api_key_env) or "EMPTY"
    url = base_url.rstrip("/") + "/chat/completions"

    def call_fn(messages: Messages) -> str:
        import httpx  # 延迟导入：单元测试会注入自己的 call_fn

        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for delay in retry_delays:
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    content = _extract_content(resp.json())
                if content and content.strip():
                    return content
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
            time.sleep(delay)

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            content = _extract_content(resp.json())
        if content and content.strip():
            return content
        raise EndpointError(
            f"endpoint {model!r} returned empty content after "
            f"{len(retry_delays) + 1} attempts" + (f"; last exc: {last_exc!r}" if last_exc else "")
        )

    return call_fn


__all__ = ["make_call_fn", "EndpointError"]
