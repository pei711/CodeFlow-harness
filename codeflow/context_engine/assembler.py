"""实现唯一 Context Engine，把统一 SegmentBuilder 列表组装成本轮模型消息。

Phase A 并行运行所有 ``needs_prefix=False`` 的 Builder，即 seg1–5：identity、bootstrap、
memory、active-skills、skills。各自 ``text`` 按 order 连接成 System prefix，``meta`` 合并为
组装证据。Phase B 再运行 ``needs_prefix=True`` 的 CodeFlow History manager；此时 ``ctx.prefix`` 已含精确的
System prefix、User message 与 Tool definitions，所以它能用固定开销预算 ``*history``，并
贡献 segment 6 与唯一 History slot。

User message 是结构内建项，每个 Turn 恰好一个，不是可插拔 Builder；Tool 走 side channel，
随 messages 一起交给 LLM 并计入预算，但永不渲染成 Segment。`ContextAssembler` 最终只产出
``[system, *history, user]`` 和 metadata，不执行 Memory/Skill 的业务选择，也不调用主 Agent。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from codeflow.context_engine.base import (
    AssembledPrefix,
    AssemblyContext,
    ContextEngine,
    SegmentBuilder,
)
from codeflow.context_engine.segments import render
from codeflow.memory_engine.base import AssembledContext, TokenBudget

if TYPE_CHECKING:
    from codeflow.context_engine.curator import TurnContext


class ContextAssembler(ContextEngine):
    """把 SegmentBuilders 的两阶段产物合并为单 Turn Context 的唯一 Engine。

    构造时按 ``order`` 排序，并依据 ``needs_prefix`` 固定 Phase A/Phase B；`assemble` 每轮并行
    运行独立贡献者、建立 `AssembledPrefix`、再运行依赖固定开销的贡献者。实例长期由
    AgentLoop 持有，可通过 `replace_model` 把模型变化转发给需要它的 Builder。

    Engine ``owns_compaction=True``，因为 CodeFlow History manager 自行压缩并维护边界；Host 必须传完整
    append-only 候选并跳过 MemoryConsolidator。最终 metadata 会带 ``engine`` 名称，便于 Turn
    evidence 确认实际组装路径。
    """

    def __init__(
        self,
        builders: list[SegmentBuilder],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._builders = sorted(builders, key=lambda b: b.order)
        self._phase_a = [b for b in self._builders if not b.needs_prefix]
        self._phase_b = [b for b in self._builders if b.needs_prefix]
        self.get_tool_definitions = get_tool_definitions
        self._now_fn = now_fn or datetime.now

    @property
    def name(self) -> str:
        return "context_assembler"

    @property
    def owns_compaction(self) -> bool:
        # CodeFlow 路径自行压缩历史，因此 AgentLoop 向其传入完整的追加式日志，
        # 并跳过 Host 的 MemoryConsolidator。
        return True

    def replace_model(self, model: str) -> None:
        for builder in self._builders:
            replace_model = getattr(builder, "replace_model", None)
            if callable(replace_model):
                replace_model(model)

    async def assemble(
        self,
        session_key: str,
        session_messages: list[dict[str, Any]],
        budget: TokenBudget,
        *,
        turn: "TurnContext",
    ) -> AssembledContext:
        effective_context_tokens = max(
            1,
            int(budget.context_length) - int(budget.reserved_output) - int(budget.reserved_tools),
        )
        # Keep CodeFlow's conservative half-window working set and section
        # priorities while still applying Harness's provider-aware final gate.
        context_budget_tokens = max(1, int(effective_context_tokens * 0.50))
        section_weights = {
            "prefix": 0.20,
            "memory": 0.13,
            "skills": 0.07,
            "relevant_memory": 0.10,
            "history": 0.50,
        }
        section_budgets = {
            name: max(1, int(context_budget_tokens * weight))
            for name, weight in section_weights.items()
        }
        adjusted_budget = replace(
            budget,
            available_history=min(budget.available_history, section_budgets["history"]),
        )
        ctx = AssemblyContext(
            session_key=session_key,
            current_message=turn.current_message,
            media=turn.media,
            channel=turn.channel,
            chat_id=turn.chat_id,
            session_messages=session_messages,
            budget=adjusted_budget,
        )

        # ── 阶段 A——相互独立的片段构建器，并发执行 ──────
        a_segs = await asyncio.gather(*[b.build(ctx) for b in self._phase_a])
        meta: dict[str, Any] = {}
        prefix_parts: list[str] = []
        remaining_chars = {name: tokens * 4 for name, tokens in section_budgets.items()}
        clipped_sections: dict[str, int] = {}
        for builder, seg in zip(self._phase_a, a_segs):
            if seg is None:
                continue
            meta |= seg.meta
            if seg.text:
                section = self._section_for_builder(builder.name)
                limit = remaining_chars.get(section)
                text = seg.text
                if limit is not None and len(text) > limit:
                    clipped_sections[section] = clipped_sections.get(section, 0) + len(text) - limit
                    text = text[: max(0, limit - 3)] + ("..." if limit >= 3 else "")
                if limit is not None:
                    remaining_chars[section] = max(0, limit - len(text))
                prefix_parts.append(text)
        system_prefix = "\n\n---\n\n".join(prefix_parts)

        user_msg = self._build_user(ctx)

        # ── 阶段 B——依赖前缀的历史管理器，串行执行 ─────────
        ctx_b = replace(
            ctx,
            prefix=AssembledPrefix(
                system_prefix=system_prefix,
                user_message=user_msg,
                tool_defs=self.get_tool_definitions(),
            ),
        )
        b_segs = await asyncio.gather(*[b.build(ctx_b) for b in self._phase_b])

        system = system_prefix
        history: list[dict[str, Any]] = []
        seg6_parts: list[str] = []
        for seg in b_segs:
            if seg is None:
                continue
            meta |= seg.meta
            if seg.text:
                seg6_parts.append(seg.text)
            if seg.history is not None:
                history = seg.history
        history_summary_chars = section_budgets["history"] * 4
        for text in seg6_parts:
            if len(text) > history_summary_chars:
                clipped_sections["history_summary"] = len(text) - history_summary_chars
                text = text[: max(0, history_summary_chars - 3)] + "..."
            system = system + "\n\n---\n\n" + text

        messages = [{"role": "system", "content": system}, *history, user_msg]
        return AssembledContext(
            messages=messages,
            metadata=meta
            | {
                "engine": self.name,
                "context_budget_tokens": context_budget_tokens,
                "context_section_budgets": section_budgets,
                "context_clipped_chars": clipped_sections,
            },
        )

    @staticmethod
    def _section_for_builder(name: str) -> str:
        if name in {"identity", "bootstrap"}:
            return "prefix"
        if name == "memory":
            return "memory"
        if name in {"active_skills", "skills"}:
            return "skills"
        if name == "relevant_memory":
            return "relevant_memory"
        return "prefix"

    async def after_turn(
        self,
        session_key: str,
        outcome: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        # 委托给需要维护每 Turn 状态的 builder（例如轮次摘要记忆）。
        for builder in self._builders:
            hook = getattr(builder, "after_turn", None)
            if hook is not None:
                await hook(session_key, outcome, usage)

    def _build_user(self, ctx: AssemblyContext) -> dict[str, Any]:
        """构造唯一结构化 User message，把运行时上下文放在真实内容之前。

        `render.build_runtime_context` 根据当前时间、Channel 与 Chat 生成每轮环境前缀，
        `render.build_user_content` 则把 ``current_message`` 和 Media 转成 Provider 可接受内容。
        纯文本用两个换行连接；多模态列表在首位插入 runtime text block，保持图片等后续块顺序。

        返回固定 ``{"role": "user", "content": merged}`` 形状。该运行时前缀只供本轮模型使用，
        Session 持久化会把它剥离，避免每轮动态时间污染长期用户历史。
        """
        runtime_ctx = render.build_runtime_context(self._now_fn, ctx.channel, ctx.chat_id)
        user_content = render.build_user_content(ctx.current_message, ctx.media)
        if isinstance(user_content, str):
            merged: Any = f"{runtime_ctx}\n\n{user_content}"
        else:
            merged = [{"type": "text", "text": runtime_ctx}] + user_content
        return {"role": "user", "content": merged}


__all__ = ["ContextAssembler"]
