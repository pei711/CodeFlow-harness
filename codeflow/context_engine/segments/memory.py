"""构建 Segment 3，把 Host Memory 与 Plugin recall 合并到一个 ``# Memory``。

这是唯一 composite Memory Segment：Host 的慢变化 ``user.md`` dump 提供长期背景，Backend
根据当前 User query 返回 query-conditioned recall hits。两种来源由同一个
`MemorySegmentBuilder` 排序和渲染，因此 System Prompt 只出现一个标题，也只有一个
``memory_hits`` 证据所有者。

Backend 未接线时不执行外部召回；启用后的 recall 硬失败会向上传播，不能
静默假装“没有记忆”。Host 内容与命中均为空时返回空文本和零/实际命中 metadata，不影响
Local Skill availability 或 CodeFlow History 管理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeflow.context_engine.base import AssemblyContext, Segment
from codeflow.context_engine.segments import render
from codeflow.context_engine.turn_summaries import TurnSummaryStore
from codeflow.security.trust import wrap_untrusted
from codeflow.tracing import semconv, trace

if TYPE_CHECKING:
    from codeflow.memory_engine.backend import MemoryBackend
    from codeflow.memory_engine.consolidate.consolidator import MemoryStore


class MemorySegmentBuilder:
    name = "memory"
    order = 3
    needs_prefix = False

    def __init__(
        self,
        memory_store: "MemoryStore",
        backend: "MemoryBackend | None" = None,
        user_id: str = "default",
        memory_top_k: int = 5,
        enabled: bool = True,
    ) -> None:
        self._memory_store = memory_store
        self._backend = backend
        self._user_id = user_id
        self._memory_top_k = memory_top_k
        self._enabled = enabled

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        if not self._enabled:
            return Segment(text="", meta={"memory_hits": 0})
        # 合并 Host 直接读取和插件召回。召回发生硬失败时继续向上传播，
        # 让后端故障在 AgentLoop 暴露，而不是静默丢弃记忆。
        host = self._memory_store.get_memory_context(current_message=ctx.current_message)
        recall_hits = await self._recall(ctx.current_message)
        recall_bullets = render.render_recalled_memory(recall_hits)
        sections = [s for s in (host, recall_bullets) if s]
        meta: dict[str, Any] = {"memory_hits": len(recall_hits)}
        if not sections:
            return Segment(text="", meta=meta)
        return Segment(text="# Memory\n\n" + "\n\n".join(sections), meta=meta)

    @trace.instrument("memory.recall", extract=semconv.memory_recall)
    async def _recall(self, query: str) -> list[Any]:
        if self._backend is None:
            return []
        return list(
            await self._backend.recall(
                query=query,
                user_id=self._user_id,
                top_k=self._memory_top_k,
            )
        )


class RelevantTurnMemorySegmentBuilder:
    """Retrieve up to three completed-turn summaries using semantic + BM25 RRF."""

    name = "relevant_memory"
    order = 4.5
    needs_prefix = False

    def __init__(self, state_root) -> None:
        self._store = TurnSummaryStore(state_root)
        self._current_messages: dict[str, str] = {}

    async def build(self, ctx: AssemblyContext) -> Segment:
        self._current_messages[ctx.session_key] = ctx.current_message
        hits, mode = self._store.retrieve(ctx.current_message, limit=3)
        lines = ["# Relevant Turn Memory"]
        if hits:
            lines.append(wrap_untrusted("\n".join(f"- {item['text']}" for item in hits), source="recalled conversation"))
        if not hits:
            return Segment(text="", meta={"relevant_turn_memory_hits": 0, "relevant_turn_memory_mode": mode})
        return Segment(
            text="\n".join(lines),
            meta={
                "relevant_turn_memory_hits": len(hits),
                "relevant_turn_memory_mode": mode,
                "relevant_turn_memory": [item["summary_id"] for item in hits],
            },
        )

    async def after_turn(self, session_key: str, outcome: dict[str, Any], usage=None) -> None:
        request = self._current_messages.pop(session_key, "")
        answer = str(outcome.get("final_content", "") or "").strip()
        if request and answer:
            self._store.add(session_key, str(outcome.get("turn_id", "") or ""), request, answer)
