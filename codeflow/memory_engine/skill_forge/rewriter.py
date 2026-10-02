"""Query Rewriter：判断是否需要 Skill Retrieval，并把 Verbose Query 改写为 Concise Routing Query。

一次 LLM Call 同时完成两件事：一是判断 User Query 是否需要 Skill Retrieval，Chat / Greetings / General
Knowledge 可完全 Skip Router Fan-out；二是在 Retrieval **IS** Needed 时移除 Paths、IDs、Timestamps 等
Noise，保留 Task Type + Domain + Required Capabilities，让 BM25 / Dense Fan-outs 命中相关 Skills。

Failure 默认 ``need_retrieval=True``，这是 Safe Fallback：继续检索而非让 Flaky Provider Silently Turn
Off Skill Lane。改写结果只影响检索 Query，不回答或执行用户任务。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeflow.tracing import semconv, trace

if TYPE_CHECKING:
    from codeflow.providers.base import LLMProvider

log = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
Given a user query, first decide if it needs external skill/tool retrieval. \
Casual chat, greetings, simple follow-ups, and general knowledge tasks do not. \
Specialized tools, domain-specific workflows, or specific frameworks/APIs do.

If retrieval is needed, rewrite the query for skill retrieval. \
Remove noise (paths, IDs, timestamps, boilerplate). \
Keep task type, domain, required capabilities, and key technical details. \
Do NOT answer or solve the query — only rewrite it.

When in doubt, choose retrieval.

Return JSON: {{"need_retrieval": true/false, "rewritten_query": "..." or null}}

{query}"""

_QUERY_MAX_LENGTH = 2000
_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class RewriteResult:
    need_retrieval: bool
    rewritten_query: str | None = None


class QueryRewriter:
    """通过 Agent Shared :class:`LLMProvider` 判断 Retrieval Necessity 并 Rewrite Query。

    调用统一走 ``chat_with_retry``，使 Retry Policy、Generation Defaults 与 Provider Extras，如 Cache
    Control、Routing Affinity，与 Agent Main Path 一致。`analyze` 最多读取 Query 前 2000 Characters，并
    要求 JSON ``need_retrieval`` / ``rewritten_query``；Timeout、Provider Error、Invalid JSON 都 Fail Open
    到 Retrieval。实例不持久化 Query 或结果。
    """

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> None:
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature

    @trace.instrument("skill.rewrite", kind="skill", extract=semconv.skill_rewrite)
    async def analyze(
        self,
        query: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> RewriteResult:
        truncated = (query or "").strip()[:_QUERY_MAX_LENGTH]
        if not truncated:
            return RewriteResult(need_retrieval=False)

        prompt = _REWRITE_PROMPT.format(query=truncated)
        try:
            resp = await asyncio.wait_for(
                self._provider.chat_with_retry(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                ),
                timeout=_TIMEOUT_S,
            )
            content = resp.content or ""
            if getattr(resp, "finish_reason", None) == "error":
                if diagnostics is not None:
                    classification = getattr(resp, "error_classification", None)
                    diagnostics["fallback_reason"] = "provider_failure"
                    diagnostics["provider_error_category"] = getattr(classification, "category", None) or "unknown"
                raise RuntimeError(content or "provider error")
        except Exception as e:
            if diagnostics is not None and "fallback_reason" not in diagnostics:
                diagnostics["fallback_reason"] = "provider_exception"
                diagnostics["failure_type"] = type(e).__name__
            log.warning("query rewrite failed (%s); defaulting to retrieval", e)
            return RewriteResult(need_retrieval=True)
        return self._parse(content, diagnostics=diagnostics)

    @staticmethod
    def _parse(
        content: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> RewriteResult:
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if diagnostics is not None:
                diagnostics["fallback_reason"] = "invalid_response"
            log.warning("rewrite response not JSON; defaulting to retrieval")
            return RewriteResult(need_retrieval=True)

        if not isinstance(data, dict):
            if diagnostics is not None:
                diagnostics["fallback_reason"] = "invalid_response"
            return RewriteResult(need_retrieval=True)

        need = bool(data.get("need_retrieval", True))
        if not need:
            return RewriteResult(need_retrieval=False)

        rewritten = data.get("rewritten_query")
        if isinstance(rewritten, str):
            rewritten = rewritten.strip() or None
        else:
            rewritten = None
        return RewriteResult(need_retrieval=True, rewritten_query=rewritten)


__all__ = ["QueryRewriter", "RewriteResult"]
