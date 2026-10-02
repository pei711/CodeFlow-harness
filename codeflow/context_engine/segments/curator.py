"""把 Curator 实现为同时产出 Segment 6 与 History slot 的 SegmentBuilder。

Curator 与其他贡献者共享 :class:`SegmentBuilder` 协议，但声明 ``order=6``、
``needs_prefix=True``。一次计算同时返回 ``# Curator Working State``（System slot 的 segment 6）
和 budget-trimmed ``*history``（History slot），两者装在同一 :class:`Segment` 的 ``text`` 与
``history`` 中，避免工作状态与所选历史来自不同计划。

由于需要 prefix，:class:`ContextAssembler` 只在 Phase B 运行它，此时 ``ctx.prefix`` 已含
seg1–5、User 与 Tools。内部 budget Tool 因而按 exact fixed overhead 选择 History。Fast path
直接使用结构清理后的 History；Slow path 运行有界内部 LLM；timeout、Provider failure、非法
plan 或 step exhaustion 都回到 deterministic fallback，而不会阻断主 Agent 获得 Context。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from codeflow.agent.tools.registry import ToolRegistry
from codeflow.config.codeflow import ContextConfig
from codeflow.context_engine.base import AssemblyContext, Segment
from codeflow.context_engine.curator import (
    CuratorArchiveMessagesTool,
    CuratorArchiveStore,
    CuratorAssembler,
    CuratorBuildContextTool,
    CuratorCheckBudgetTool,
    CuratorReadMemoryTool,
    CuratorRetrieveArchivedTool,
    CuratorSearchHistoryTool,
    CuratorSetRelevanceTool,
    CuratorState,
    CuratorUpdateWorkingStateTool,
    TurnContext,
    _curator_input_payload,
    _trace_messages,
)
from codeflow.memory_engine.consolidate.consolidator import MemoryStore
from codeflow.providers.base import LLMProvider
from codeflow.tracing import semconv, trace
from codeflow.utils.helpers import build_assistant_message


class CuratorSegmentBuilder:
    """选择 ``*history``，并从同一计划渲染 ``# Curator Working State``。

    每轮先建立 Manifest 和独立 trace id。History Token 低于阈值时走 fast path；否则在 timeout
    与 ``max_steps`` 内运行只具备 Curator Tool 的 slow path，接受通过预算与结构验证的
    ContextPlan；任何受控失败再使用 protected、relevant、recent 的 deterministic fallback。

    Builder 长期持有 ArchiveStore 与 CuratorAssembler，但每轮 State、plan 与 trace 隔离。
    `after_turn` 只把主 Agent outcome 追加到对应 trace。它不回答用户、不执行外部 Tool，也不
    越过 ContextAssembler 直接修改最终消息。
    """

    name = "curator"
    order = 6
    needs_prefix = True

    def __init__(
        self,
        workspace: Path,
        config: ContextConfig,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        now_fn: Callable[[], datetime] | None = None,
        max_steps: int = 12,
        memory_enabled: bool = True,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.provider = provider
        self.model = model
        self.curator_model = config.curator_model or model
        self.context_window_tokens = context_window_tokens
        self.get_tool_definitions = get_tool_definitions
        self.max_steps = max_steps
        self.memory_enabled = memory_enabled
        self.archive = CuratorArchiveStore(workspace, config, now_fn=now_fn)
        self.assembler = CuratorAssembler(
            provider,
            model,
            get_tool_definitions,
            context_window_tokens,
        )
        self._turn_ids: dict[str, str] = {}

    def replace_model(self, model: str) -> None:
        self.model = model
        self.curator_model = self.config.curator_model or model
        self.assembler.replace_model(model)

    async def build(self, ctx: AssemblyContext) -> Segment | None:
        if ctx.prefix is None:
            raise RuntimeError("CuratorSegmentBuilder requires ctx.prefix (phase B)")

        session_key = ctx.session_key
        turn_id = uuid.uuid4().hex
        self._turn_ids[session_key] = turn_id

        manifest = self.archive.build_manifest(session_key, ctx.session_messages)
        turn = TurnContext(
            current_message=ctx.current_message,
            media=ctx.media,
            channel=ctx.channel,
            chat_id=ctx.chat_id,
        )
        state = CuratorState(
            session_key=session_key,
            session_messages=ctx.session_messages,
            budget=ctx.budget,
            turn=turn,
            manifest=manifest,
            prefix=ctx.prefix,
        )
        self.archive.append_trace(
            session_key,
            turn_id,
            "curator_start",
            {
                "budget": asdict(ctx.budget),
                "message_count": len(ctx.session_messages),
                "max_steps": self.max_steps,
            },
        )

        history_tokens = sum(item.tokens for item in manifest)
        threshold = int(ctx.budget.available_history * self.config.fast_path_threshold)
        if history_tokens < threshold:
            history = self._history_from_messages(ctx.session_messages)
            meta = {
                "path": "fast",
                "history_tokens": history_tokens,
                "threshold_tokens": threshold,
                "trace_path": str(self.archive.trace_path(session_key, turn_id)),
            }
            self.archive.append_trace(session_key, turn_id, "fast_path", meta)
            return Segment(text="", history=history, meta=meta)

        fallback_meta: dict[str, Any] = {}
        try:
            async with asyncio.timeout(self.config.curator_timeout_seconds):
                seg, fallback_meta = await self._slow_path(state, turn_id)
            if seg is not None:
                return seg
        except TimeoutError:
            fallback_meta = {"fallback_reason": "timeout"}
            logger.warning("Curator slow path timed out; using deterministic fallback")
            self.archive.append_trace(session_key, turn_id, "slow_path_timeout", {})
        except Exception as exc:
            fallback_meta = {
                "fallback_reason": "slow_path_exception",
                "failure_type": type(exc).__name__,
            }
            logger.exception("Curator slow path failed; using deterministic fallback")
            self.archive.append_trace(session_key, turn_id, "slow_path_exception", fallback_meta)

        plan = self.assembler.fallback_plan(state)
        assembled, validation = self.assembler.build(state, plan)
        meta = {
            "path": "fallback",
            "trace_path": str(self.archive.trace_path(session_key, turn_id)),
            **fallback_meta,
        }
        self.archive.append_trace(
            session_key,
            turn_id,
            "fallback",
            {
                "plan": asdict(plan),
                "validation": validation,
                "diagnostic": fallback_meta,
            },
        )
        return Segment(
            text=self.assembler.working_state_segment(plan.working_state_injection or None),
            history=assembled.messages[1:-1],
            meta=meta,
        )

    async def after_turn(
        self,
        session_key: str,
        response: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        turn_id = self._turn_ids.get(session_key)
        if not turn_id:
            return
        traced_response = dict(response)
        messages = traced_response.get("messages")
        if isinstance(messages, list):
            traced_response["messages"] = _trace_messages(
                [message for message in messages if isinstance(message, dict)]
            )
        self.archive.append_trace(
            session_key,
            turn_id,
            "main_agent_result",
            {
                "response": traced_response,
                "usage": usage or {},
            },
        )

    # ------------------------------------------------------------------
    # 慢路径（有界的 Curator 内部 LLM 循环）
    # ------------------------------------------------------------------

    @trace.instrument("context.curate", kind="memory", extract=semconv.context_curate)
    async def _slow_path(
        self,
        state: CuratorState,
        turn_id: str,
    ) -> tuple[Segment | None, dict[str, Any]]:
        registry = self._make_tools(state)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": json.dumps(_curator_input_payload(state, self.archive), ensure_ascii=False)},
        ]
        for step in range(1, self.max_steps + 1):
            self.archive.append_trace(
                state.session_key,
                turn_id,
                "curator_llm_request",
                {
                    "step": step,
                    "messages": _trace_messages(messages),
                    "tools": registry.tool_names,
                },
            )
            try:
                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=registry.get_definitions(),
                    model=self.curator_model,
                    max_tokens=2048,
                    temperature=0.1,
                )
            except Exception as exc:
                return None, {
                    "fallback_reason": "provider_exception",
                    "provider_error_type": type(exc).__name__,
                }
            self.archive.append_trace(
                state.session_key,
                turn_id,
                "curator_llm_response",
                {
                    "step": step,
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                    "tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                },
            )
            if response.finish_reason == "error":
                classification = response.error_classification
                return None, {
                    "fallback_reason": "provider_failure",
                    "provider_error_category": (classification.category if classification is not None else "unknown"),
                }
            if not response.has_tool_calls:
                return None, {"fallback_reason": "invalid_plan"}

            tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
            messages.append(
                build_assistant_message(
                    response.content,
                    tool_calls=tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
            )
            for tool_call in response.tool_calls:
                result = await registry.execute(tool_call.name, tool_call.arguments, tool_call.id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    }
                )
                self.archive.append_trace(
                    state.session_key,
                    turn_id,
                    "curator_tool_result",
                    {
                        "step": step,
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": _json_or_text(result),
                    },
                )
                if tool_call.name == "curator_build_context" and state.final_plan is not None:
                    assembled, validation = self.assembler.build(state, state.final_plan)
                    if validation.get("ok"):
                        self.archive.append_trace(
                            state.session_key,
                            turn_id,
                            "slow_path_accepted",
                            {
                                "plan": asdict(state.final_plan),
                                "validation": validation,
                            },
                        )
                        return (
                            Segment(
                                text=self.assembler.working_state_segment(
                                    state.final_plan.working_state_injection or None
                                ),
                                history=assembled.messages[1:-1],
                                meta={
                                    "path": "slow",
                                    "trace_path": str(self.archive.trace_path(state.session_key, turn_id)),
                                    "curator_steps": step,
                                },
                            ),
                            {},
                        )
        return None, {"fallback_reason": "step_exhausted"}

    def _make_tools(self, state: CuratorState) -> ToolRegistry:
        registry = ToolRegistry()
        tools = [
            CuratorCheckBudgetTool(state, self.assembler),
            CuratorArchiveMessagesTool(state, self.archive),
            CuratorRetrieveArchivedTool(self.archive),
            CuratorSearchHistoryTool(state),
            CuratorSetRelevanceTool(state, self.archive),
            CuratorUpdateWorkingStateTool(state, self.archive),
            CuratorBuildContextTool(state, self.assembler),
        ]
        if self.memory_enabled:
            tools.insert(4, CuratorReadMemoryTool(state, self.archive, MemoryStore(self.workspace)))
        for tool in tools:
            registry.register(tool)
        return registry

    @staticmethod
    def _system_prompt() -> str:
        return """You are CodeFlow Curator, an internal context manager.

Your only job is to build the next main-agent LLM context window.
Never answer the user. Never invent message content. Never call external tools.

Rules:
- Preserve the current user message; Python will add it after your plan.
- Preserve valid tool-call adjacency by selecting related message ids together.
- Prefer recent messages, explicit user constraints, unresolved tasks, decisions, and facts referenced by the current user message.
- When messages conflict about the same subject, keep the latest explicit user decision active and omit the superseded decision when structure permits.
- Store only the active version of a decision in working state.
- Archive old low-relevance messages losslessly before dropping them from live context when useful.
- Retrieve archived content only when needed.
- Finish by calling curator_build_context.
"""

    @staticmethod
    def _history_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed = {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content", "thinking_blocks"}
        out: list[dict[str, Any]] = []
        for message in messages:
            entry = {k: v for k, v in message.items() if k in allowed}
            if entry.get("role"):
                out.append(entry)
        for idx, msg in enumerate(out):
            if msg.get("role") == "user":
                return out[idx:]
        return []


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
