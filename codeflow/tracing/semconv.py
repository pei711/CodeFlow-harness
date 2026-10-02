"""CodeFlow Tracing Standard 的 Semantic Conventions。

模块同时拥有 Standard Span Attribute/Artifact *Builders*（Low-level Helpers），以及
``@trace.instrument(extract=...)`` 使用的 Per-span-kind *Extractors*。Extractor 根据参数名与 Duck-typed
Shape 读取 Runtime 对象，保持 Framework-agnostic；只要 Documented Shapes 不变，就能承受 CodeFlow Refactor。
历史说明见 ``docs/TRACING_STANDARD_API.md``。

这里定义“记录什么”和字段名，不负责 Span Lifecycle/Storage。Extractor Failure 必须由 Tracing Layer
隔离，不能改变被观测 Host Behavior；Artifact 存在也不等于任务或交付成功。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from . import config
from . import usage as usage_mod
from .store import preview_text

_FILE_READ_TOOLS = {"read_file"}


def _preview(value: Any, n: int | None = None) -> str:
    return preview_text(value, n if n is not None else config.preview_len())


def _split_session_key(session_key: str | None) -> tuple[str | None, str | None]:
    if session_key and ":" in session_key:
        channel, _, chat_id = session_key.partition(":")
        return channel or None, chat_id or None
    return None, None


def _turn_capabilities(loop: Any) -> dict[str, Any]:
    """Snapshot 本 Turn Agent 已加载的 Tools、Plugin Backend/Tools 与 Available Skills。

    数据在 Turn Probe 从 `AgentLoop` ``self`` 读取。每部分 Best-effort：Missing Attr 只 Omit Field，Never
    Break Turn Span。由此 Trace 可 Self-describing，例如 TUI Trace 会明确显示 ``backend=null`` / No
    Plugins。Snapshot 反映当时可见 Capability，不证明具体 Tool/Skill 被使用。
    """
    caps: dict[str, Any] = {}
    try:
        names = loop.tools.tool_names
        caps["turn.tools"] = list(names)
        caps["turn.tool_count"] = len(names)
    except Exception:  # noqa: BLE001
        pass
    try:
        backend = getattr(loop, "backend", None)
        caps["turn.plugin.backend"] = type(backend).__name__ if backend is not None else None
    except Exception:  # noqa: BLE001
        pass
    try:
        ptools = getattr(loop, "plugin_tools", None) or []
        caps["turn.plugin.tools"] = [getattr(t, "name", None) for t in ptools]
    except Exception:  # noqa: BLE001
        pass
    try:
        cat = getattr(getattr(loop, "context", None), "skills", None)
        reg = getattr(cat, "registry", None) or getattr(cat, "_registry", None)
        metas = list(reg.list_all()) if reg is not None else []
        caps["turn.skills"] = [getattr(m, "name", None) for m in metas][:50]
        caps["turn.skill_count"] = len(metas)
    except Exception:  # noqa: BLE001
        pass
    return caps


def _provider_label(model: str | None, provider_class: str | None) -> str | None:
    """返回一次 Call 的 Logical Routing Backend Label。

    CodeFlow 通过同一 ``LiteLLMProvider`` Class 访问多个 Gateways，Class Name 会隐藏真实 Backend。LiteLLM 在
    Model Prefix 中编码它，例如 ``openrouter/anthropic/claude-...`` 的 First Segment 是
    ``openrouter``。Model 无 Prefix 时回退到 Provider Class Name，例如 Native Provider；两者都缺失返回
    `None`。
    """
    if model and "/" in model:
        return model.split("/", 1)[0]
    return provider_class


def _llm_attrs(resp: Any, provider: str, model: str | None, provider_class: str | None = None) -> dict[str, Any]:
    attrs: dict[str, Any] = {"llm.provider": provider, "llm.model": model}
    if provider_class and provider_class != provider:
        attrs["llm.provider_class"] = provider_class
    if resp is None:
        return attrs
    attrs["llm.finish_reason"] = getattr(resp, "finish_reason", None)
    attrs["llm.output_preview"] = _preview(getattr(resp, "content", None))
    tool_calls = getattr(resp, "tool_calls", None) or []
    attrs["llm.tool_call_count"] = len(tool_calls)
    if tool_calls:
        attrs["llm.tool_names"] = [getattr(t, "name", None) for t in tool_calls]
    u = usage_mod.normalize(getattr(resp, "usage", None), model)
    attrs["llm.usage.input_tokens"] = u["input_tokens"]
    attrs["llm.usage.output_tokens"] = u["output_tokens"]
    attrs["llm.usage.cache_read_tokens"] = u["cache_read_tokens"]
    attrs["llm.usage.cache_write_tokens"] = u["cache_write_tokens"]
    attrs["llm.usage.total_tokens"] = u["total_tokens"]
    attrs["llm.usage.cost_total"] = u["cost_usd"]
    reasoning = getattr(resp, "reasoning_content", None)
    if reasoning:
        attrs["llm.reasoning_preview"] = _preview(reasoning)
    return attrs


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        import json as _json

        return _json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(value)


def _llm_input_payload(
    provider: str, model: str | None, messages: Any, tools: Any, provider_class: str | None = None
) -> dict:
    """构造 Model-input Viewer Card 的 Artifact Payload。

    CodeFlow 传入 ONE Flat ``messages`` List，包含 System + Prior Turns + Current。函数拆成 Three
    Non-overlapping Views：``systemPrompt`` 是第一条 System Message；``prompt`` 是 Latest User Message；
    ``historyMessages`` 是 Prior Turns，即 **EXCEPT** 所有 System 与该 Latest User，避免重复。``messages`` 仍保留
    Full Raw List，作为实际发送内容的 Ground Truth；``tools`` 也原样附带。
    """
    msgs = messages if isinstance(messages, list) else []
    system_prompt = ""
    user_prompt = ""
    system_idxs: set[int] = set()
    last_user_idx: int | None = None
    for i, m in enumerate(msgs):
        if isinstance(m, dict) and m.get("role") == "system":
            system_idxs.add(i)
            if not system_prompt:
                system_prompt = _coerce_text(m.get("content"))
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, dict) and m.get("role") == "user":
            user_prompt = _coerce_text(m.get("content"))
            last_user_idx = i
            break
    history = [m for i, m in enumerate(msgs) if i not in system_idxs and i != last_user_idx]
    return {
        "provider": provider,
        "providerClass": provider_class,
        "model": model,
        "systemPrompt": system_prompt,
        "prompt": user_prompt,
        "historyMessages": history,
        "messages": messages,
        "tools": tools,
    }


def _llm_output_payload(resp: Any) -> Any:
    if resp is None:
        return None
    content = getattr(resp, "content", None)
    return {
        "content": content,
        "output": content,  # 共享 viewer 的模型输出卡片读取该字段
        "finish_reason": getattr(resp, "finish_reason", None),
        "tool_calls": [
            {"id": getattr(t, "id", None), "name": getattr(t, "name", None), "arguments": getattr(t, "arguments", None)}
            for t in (getattr(resp, "tool_calls", None) or [])
        ],
        "reasoning_content": getattr(resp, "reasoning_content", None),
        "usage": getattr(resp, "usage", None),
    }


def _skill_name_from_path(path: str | None) -> str | None:
    """把 ``…/skills/weather/SKILL.md`` 转成 Skill Dir Name ``weather``。

    同时兼容 Windows/Posix Separator；Path 不以 ``SKILL.md`` 结尾时返回 Last Segment，Empty 输入返回
    `None`。该名称用于 Trace Label，不执行 Registry Validation。
    """
    if not path:
        return None
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    if len(parts) >= 2 and parts[-1].lower() == "skill.md":
        return parts[-2]
    return parts[-1] if parts else None


def _skill_read_path(name: str, params: Any) -> str | None:
    """``read_file`` Target 是 ``SKILL.md`` 时返回 Path，否则 `None`。

    这是 Discovery→Injection Follow-through：CodeFlow Summary Mode 告诉 Agent 用 ``read_file`` 读取 Skill Body，
    只获得 Skill *Catalog* 的 Subagents 也如此。这些 Reads 把真实 Body 带入 Context，却表面像普通 File
    Read，因此 Tracing 将其 Re-type 为 ``skill.read``。函数只按 Tool Name/Path Suffix 识别，不证明读取成功。
    """
    if name not in _FILE_READ_TOOLS:
        return None
    path = params.get("path") if isinstance(params, dict) else (params if isinstance(params, str) else None)
    if isinstance(path, str) and path.replace("\\", "/").lower().rstrip("/").endswith("skill.md"):
        return path
    return None


# 公共别名（标准中稳定的属性/载荷 builder）。
provider_label = _provider_label
llm_attrs = _llm_attrs
llm_input_payload = _llm_input_payload
llm_output_payload = _llm_output_payload


SPINE_COMPLETED = "completed"
SPINE_COMPLETED_WITH_TOOL_FAILURE = "completed_with_tool_failure"
SPINE_PROVIDER_FAILED = "provider_failed"
SPINE_ERROR = "error"
SPINE_CANCELLED = "cancelled"

# 固定的 Turn 终态分类；参见 docs/specs/turn-evidence-correlation.md。
SPINE_OUTCOMES = (
    SPINE_COMPLETED,
    SPINE_COMPLETED_WITH_TOOL_FAILURE,
    SPINE_PROVIDER_FAILED,
    SPINE_ERROR,
    SPINE_CANCELLED,
)

CHANNEL_DELIVERED = "delivered"
CHANNEL_DROPPED = "dropped"
CHANNEL_NO_OUTLET = "no_outlet"

CHANNEL_OUTCOMES = (CHANNEL_DELIVERED, CHANNEL_DROPPED, CHANNEL_NO_OUTLET)


def spine_turn_open(req: Any, conversation_id: str) -> dict[str, Any]:
    """构造 ``spine.turn`` Root Span 的 Opening Attributes。

    从 Turn Request 提取 Conversation ID、Origin、Source Channel 与 Busy Policy。返回 Dict 只描述 Turn
    Start Context，不包含 Terminal Outcome。
    """
    source = getattr(req, "source", None)
    return {
        "spine.conversation_id": conversation_id,
        "spine.origin": str(getattr(req, "origin", "") or "") or None,
        "spine.channel": getattr(source, "channel", None),
        "spine.busy_policy": str(getattr(req, "busy", "") or "") or None,
    }


def spine_turn_failed(exc: BaseException, *, started: bool) -> dict[str, Any]:
    """构造 Runner Raised 的 Turn Terminal Attributes。

    ``ProviderTurnError`` 按 Class Name 匹配，使 Tracing 保持 No-import-of-host Discipline；其 ``category``
    使用 Provider Own Failure Label，Never Exception Message。其他 Exception 标为 Generic Error。
    `started` 决定 Terminal Event 是 ``TurnFailed`` 还是尚未真正开始的 ``TurnStarted`` Failure。
    """
    error_class = type(exc).__name__
    outcome = SPINE_PROVIDER_FAILED if error_class == "ProviderTurnError" else SPINE_ERROR
    attrs: dict[str, Any] = {
        "spine.outcome": outcome,
        "spine.terminal_event": "TurnFailed" if started else "TurnStarted",
        "spine.error_class": error_class,
    }
    category = getattr(exc, "category", None)
    if isinstance(category, str):
        attrs["spine.provider_error_category"] = category
    return attrs


def spine_turn_cancelled(*, started: bool) -> dict[str, Any]:
    return {
        "spine.outcome": SPINE_CANCELLED,
        "spine.terminal_event": "TurnFailed" if started else "TurnStarted",
    }


def spine_turn_ended(outcome: Any, latency_ms: float) -> dict[str, Any]:
    """构造到达 ``TurnEnded`` 的 Terminal Attributes。

    即使 Tool Broken，Turn 仍可能绕过失败给出 Answer，属于 Completion，但证据不同于 Clean One，因此
    使用 ``completed_with_tool_failure``。同时记录 Tool Calls/Failures、Explicit Reply 与 Latency。到达
    TurnEnded 不等于 Channel Delivery 成功。
    """
    tool_failures = int(getattr(outcome, "tool_failures", 0) or 0)
    return {
        "spine.outcome": SPINE_COMPLETED_WITH_TOOL_FAILURE if tool_failures else SPINE_COMPLETED,
        "spine.terminal_event": "TurnEnded",
        "spine.tool_calls": int(getattr(outcome, "tool_calls", 0) or 0),
        "spine.tool_failures": tool_failures,
        "spine.explicit_reply": bool(getattr(outcome, "explicit_reply", False)),
        "spine.latency_ms": int(latency_ms),
    }


def channel_deliver(
    *,
    channel: str,
    event: str,
    conversation_id: str | None,
    outcome: str,
    attempts: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    """构造一次 ``channel.deliver`` Span 的 Attributes，即 Terminal Deliverable Evidence。

    字段包括 Channel、Event、Conversation ID、Outcome、Attempts/Derived Retries 与 Error。Caller 必须传入
    ``delivered`` / ``dropped`` / ``no_outlet`` 等真实终态；Span 创建本身不能把 Attempt 变成 Delivery。
    """
    return {
        "channel.name": channel,
        "channel.event": event,
        "channel.conversation_id": conversation_id,
        "channel.outcome": outcome,
        "channel.attempts": attempts,
        "channel.retries": max(0, attempts - 1),
        "channel.error": error,
    }


__all__ = [
    "llm_attrs",
    "llm_input_payload",
    "llm_output_payload",
    "provider_label",
    "llm_call",
    "llm_call_stream",
    "tool_call",
    "turn_seed",
    "turn_open",
    "turn",
    "memory_recall",
    "memory_store",
    "memory_feedback",
    "memory_extract",
    "memory_profile_refresh",
    "memory_consolidate",
    "plugin_load",
    "skill_inject_active",
    "skill_inject_skills",
    "subagent",
    "spine_turn_open",
    "spine_turn_ended",
    "spine_turn_failed",
    "spine_turn_cancelled",
    "channel_deliver",
    "SPINE_OUTCOMES",
    "CHANNEL_OUTCOMES",
]


def subagent(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """``SubagentManager._run_subagent`` 的 Extractor。

    Subagent 使用同一 Decorated Primitives ``chat_with_retry`` / ``tools.execute``，所以 LLM/Tool Spans 自动
    Capture；Spawn 时 ``asyncio.create_task`` 取得 Contextvars Snapshot，使它们 Nest 在本 Node、再位于
    Spawning Turn 下。此 Node 只描述 Spawn Task/Label/Origin/Status，不重复记录 Child Calls。
    """
    origin = bound.get("origin") or {}
    outcome_status = getattr(result, "status", None)
    status = getattr(outcome_status, "value", None)
    if isinstance(exc, asyncio.CancelledError):
        status = "cancelled"
    elif exc is not None and not isinstance(status, str):
        status = "failed"
    attributes = {
        "subagent.task_id": bound.get("task_id"),
        "subagent.task": _preview(bound.get("task"), 300),
        "subagent.label": bound.get("label"),
        "subagent.origin_session": origin.get("session_key") if isinstance(origin, dict) else None,
    }
    if isinstance(status, str):
        attributes["subagent.status"] = status
    span.set(attributes)
    if getattr(result, "failed", None) is True:
        span.error(status if isinstance(status, str) else "failed")


def plugin_load(contribution: str):
    """返回 Sync ``PluginRegistry.build_*`` Factory Call 的 Extractor。

    Closure 固定 Contribution Kind，并记录 Plugin Name、Result Type 与 Opt-out。它观测 Construction，不
    表示 Backend ``start`` 或 Tool Registration 已完成。
    """

    def _extract(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
        span.set(
            {
                "plugin.contribution": contribution,
                "plugin.name": bound.get("name"),
                "plugin.result_type": type(result).__name__ if result is not None else None,
                "plugin.opt_out": exc is None and result is None,
            }
        )

    return _extract


def _skill_inject_fill(span, *, via: str, names: list, ids: list, sources: dict, body_len: int) -> None:
    span.set(
        {
            "skill.inject.via": via,
            "skill.inject.count": len(ids),
            "skill.inject.names": names,
            "skill.inject.ids": ids,
            "skill.inject.sources": sources,
            "skill.inject.body_len": body_len,
        }
    )
    span.artifact(
        "skill.inject",
        {
            "via": via,
            "skills": [{"name": n, "id": i} for n, i in zip(names, ids)],
            "sources": sources,
            "body_len": body_len,
        },
    )


def skill_inject_active(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """记录 ``# Active Skills``；只有 Always-on Skills 真正 Force-injected 时 Emit。

    Extractor 重读 Catalog 的 Always Skills、应用 ``always_max``，再记录 Names/IDs/Sources/Body Length。
    Result 无 Text 或无 Metas 时 Cancel Span，避免把“检查过”误报成“已注入”。
    """
    if result is not None and getattr(result, "text", ""):
        self = bound.get("self")
        metas = list(self._skills.get_always_skills() or [])
        cfg = getattr(self._skills, "_config", None)
        always_max = getattr(cfg, "always_max", 5) or 5
        if always_max:
            metas = metas[:always_max]
        if metas:
            _skill_inject_fill(
                span,
                via="active_skills",
                names=[getattr(m, "name", None) for m in metas],
                ids=[str(getattr(m, "id", "")) for m in metas],
                sources=dict(Counter((getattr(m, "source", None) or "?") for m in metas)),
                body_len=len(result.text),
            )
            return
    span.cancel()


def skill_inject_skills(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """记录 ``# Skills``；只有 Gate-selected Skill Bodies 已 Render 时 Emit。

    `injected_skill_ids` 非空才写入 Skills Segment Evidence，并从 Segment Meta 记录 Source Breakdown；否则
    Cancel Span。该证据证明 Body 进入 Segment，不证明模型遵循或成功执行 Skill。
    """
    seg_meta = (getattr(result, "meta", None) or {}) if result is not None else {}
    ids = list(seg_meta.get("injected_skill_ids") or [])
    if ids:
        _skill_inject_fill(
            span,
            via="skills_segment",
            names=[str(i).split("/")[-1] for i in ids],
            ids=ids,
            sources=dict(seg_meta.get("skill_hits_by_source") or {}),
            body_len=len(getattr(result, "text", "") or ""),
        )
    else:
        span.cancel()


def _hit_ref(hit: Any) -> dict[str, Any]:
    """返回 `RouterHit` Candidate 的 Lightweight Serializable View，避免 Dump Bodies。

    只保留 ID/Name/Source/Score，供 Gate Input/Output Artifact；缺字段按 Duck Typing 返回 `None`。
    """
    return {
        "id": getattr(hit, "id", None) or getattr(hit, "skill_id", None),
        "name": getattr(hit, "name", None),
        "source": getattr(hit, "source", None),
        "score": getattr(hit, "score", None),
    }


def skill_rewrite(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """记录 Skill Retrieval 前的 ``QueryRewriter.analyze`` Judgment + Query Rewrite。

    Inner Model Call Nest 在此 Span 下，并把它作为 Invocation Source。Extractor 写 Query Preview、
    ``need_retrieval``、Rewritten Query 与完整 Input/Output Artifact；判断为 False 只表示 Skip Retrieval。
    """
    need = getattr(result, "need_retrieval", None)
    rewritten = getattr(result, "rewritten_query", None)
    span.set(
        {
            "skill.rewrite.query_preview": _preview(bound.get("query")),
            "skill.rewrite.need_retrieval": need,
            "skill.rewrite.rewritten": _preview(rewritten),
        }
    )
    span.artifact("skill.rewrite.input", {"query": bound.get("query")})
    span.artifact("skill.rewrite.output", {"need_retrieval": need, "rewritten_query": rewritten})


def skill_gate(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """记录 ``LLMGateFilter.filter`` 如何把 Skill Candidates 收窄到 Selected Few。

    Span 保存 Task Preview、Candidate/Selected Count，并以轻量 Hit Refs 记录 Input、Available Tools 与
    Output。它反映 Gate Result，不等于后续 Body Hydration/Injection。
    """
    candidates = bound.get("candidates") or []
    selected = result if isinstance(result, list) else []
    span.set(
        {
            "skill.gate.task_preview": _preview(bound.get("task")),
            "skill.gate.candidate_count": len(candidates),
            "skill.gate.selected_count": len(selected),
        }
    )
    span.artifact(
        "skill.gate.input",
        {
            "task": bound.get("task"),
            "candidates": [_hit_ref(h) for h in candidates],
            "available_tools": bound.get("available_tools"),
        },
    )
    span.artifact("skill.gate.output", {"selected": [_hit_ref(h) for h in selected]})


def context_curate(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """记录 ``CuratorSegmentBuilder._slow_path`` 的 Bounded Internal Curator LLM Loop。

    Per-step Model + Tool Calls Nest 在此 Node。Extractor 记录 Turn/Session、是否 Produced、History Length 与
    Working State；Produced=False 表示没有 Curated Segment，不一定是 Error。
    """
    seg = result
    state = bound.get("state")
    history = getattr(seg, "history", None) or [] if seg is not None else []
    span.set(
        {
            "context.curate.produced": seg is not None,
            "context.curate.history_len": len(history),
        }
    )
    span.artifact(
        "context.curate.input",
        {"turn_id": bound.get("turn_id"), "session_key": getattr(state, "session_key", None)},
    )
    span.artifact(
        "context.curate.output",
        {"produced": seg is not None, "history_len": len(history), "working_state": getattr(seg, "text", None)},
    )


def personalize(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """记录 Personalizer Steps：Classify / Question / Extract / ``post_learn``。

    这些步骤直接调用 ``provider.chat``，而非 Instrumented ``chat_with_retry``，所以 Step 本身就是 Traced
    Node：Input 是 Call Arguments，Output 是 Step Result。Raw Model Round-trip 在此 Summarized，而不是
    Child Span。``personalize.ok`` 只反映 Exception 是否发生。
    """
    args = {k: v for k, v in bound.items() if k != "self"}
    span.set({"personalize.step": span.name.split(".", 1)[-1], "personalize.ok": exc is None})
    span.artifact("personalize.input", args)
    span.artifact("personalize.output", {"result": result})


def memory_recall(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    self = bound.get("self")
    span.set(
        {
            "memory.query": _preview(bound.get("query"), 300),
            "memory.scope": "user",
            "memory.user_id": getattr(self, "_user_id", None),
            "memory.top_k": getattr(self, "_memory_top_k", None),
        }
    )
    hits = list(result or [])
    span.set({"memory.hits": len(hits)})
    span.artifact(
        "memory.recall",
        [
            {
                "text": getattr(m, "text", None),
                "score": getattr(m, "score", None),
                "metadata": getattr(m, "metadata", None),
            }
            for m in hits
        ],
    )


def memory_store(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    msgs = bound.get("messages_slice") or []
    span.set({"memory.session_id": bound.get("session_key"), "memory.message_count": len(msgs)})
    span.artifact("memory.store", {"session_id": bound.get("session_key"), "messages": msgs})


def memory_feedback(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    span.set(
        {
            "memory.session_id": bound.get("session_key"),
            "memory.injected": bound.get("injected_skill_ids"),
            "memory.used": bound.get("used_skill_ids"),
        }
    )


def memory_extract(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    msgs = bound.get("messages") or []
    span.set(
        {
            "memory.surface": "host",
            "memory.model": bound.get("model"),
            "memory.message_count": len(msgs),
            "memory.enable_foresight": bound.get("enable_foresight"),
            "memory.annotated": bool(result),
        }
    )


def memory_profile_refresh(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    span.set(
        {
            "memory.model": bound.get("model"),
            "memory.threshold": bound.get("threshold"),
            "memory.sections_rewritten": result if isinstance(result, int) else None,
        }
    )


def memory_consolidate(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    session = bound.get("session")
    span.set(
        {
            "memory.session_key": getattr(session, "key", None),
            "memory.last_consolidated": getattr(session, "last_consolidated", None),
            "memory.message_count": len(getattr(session, "messages", []) or []),
        }
    )


def _turn_request(bound: dict[str, Any]) -> Any:
    """取得 Turn Payload ``TurnRequest``，优先按 Name，再按 ``self`` 后 Position。

    兼容参数名 ``req`` / ``msg``；都缺失时取 Bound Values 第二项。仅供 Duck-typed Extractor，找不到返回
    `None` 而不影响 Host。
    """
    for key in ("req", "msg"):
        if key in bound:
            return bound[key]
    vals = list(bound.values())
    return vals[1] if len(vals) > 1 else None


def _turn_ids(bound: dict[str, Any]) -> tuple[Any, Any, Any]:
    req = _turn_request(bound)
    sk = bound.get("session_key") or getattr(req, "session_key", None)
    channel = getattr(req, "channel", None)
    chat_id = getattr(req, "chat_id", None)
    if not channel or not chat_id:
        ch2, cid2 = _split_session_key(sk)
        channel = channel or ch2
        chat_id = chat_id or cid2
    return sk, channel, chat_id


def _turn_input(bound: dict[str, Any]) -> Any:
    req = _turn_request(bound)
    text = getattr(req, "text", None)
    return text if text is not None else getattr(req, "content", None)


def turn_seed(bound: dict[str, Any]) -> dict[str, Any]:
    """为 Root Turn Span Seed Session Identity，使每个 Child Span Inherit。

    返回 Session Key、Channel、Chat ID；缺显式 Channel/Chat 时从 ``channel:chat_id`` Session Key 拆分。
    """
    sk, channel, chat_id = _turn_ids(bound)
    return {"session_key": sk, "channel": channel, "chat_id": chat_id}


def turn_open(span, bound: dict[str, Any]) -> None:
    """记录 Turn Input，并 Emit In-progress Root，使 Mid-turn Children 有 Parent。

    设置 Input Preview/``turn.in_progress=True``，写完整 Content/Channel/Chat/Media Artifact，再执行
    Checkpoint。此时尚无 Final Output 或 Terminal Outcome。
    """
    _, channel, chat_id = _turn_ids(bound)
    user_input = _turn_input(bound)
    req = _turn_request(bound)
    span.set({"turn.input_preview": _preview(user_input), "turn.in_progress": True})
    span.artifact(
        "turn.input",
        {"content": user_input, "channel": channel, "chat_id": chat_id, "media": getattr(req, "media", None)},
    )
    span.checkpoint()


def turn(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """Finalize Turn Span，记录 Output 与 Capabilities Snapshot。

    将 ``turn.in_progress`` 置 False，写 Input/Output Preview、当前 Tools/Plugins/Skills 与 Full Output
    Artifact。Span Finalization 表示 Runtime Call 返回，不自动证明 Channel Delivered。
    """
    user_input = _turn_input(bound)
    out_content = getattr(result, "content", None) if result is not None else None
    span.set(
        {
            "turn.input_preview": _preview(user_input),
            "turn.output_preview": _preview(out_content),
            "turn.in_progress": False,
        }
    )
    span.set(_turn_capabilities(bound.get("self")))
    span.artifact("turn.output", {"content": out_content})


def _finish_error(span, result) -> None:
    """Model 返回 Soft Error Response 时把 Span 标为 ERROR。

    判断 ``finish_reason == "error"``，使用最多 200 Characters Content 作为 Error；即使 Provider 没抛
    Exception，也不会把软失败记成成功。
    """
    if getattr(result, "finish_reason", None) == "error":
        span.error((getattr(result, "content", "") or "")[:200])


def llm_call(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """Non-streaming Provider Call Extractor，其中 ``self`` 是 Provider。

    记录 Ground-truth Input Artifact、Logical Provider/Model、Finish Reason、Output Preview、Tool Calls、
    Normalized Usage/Cost、Reasoning 与 Output Artifact。``llm.call_id`` 使用 Span ID，并继承 Invocation
    Source。Usage/Cost 是 Provider/Estimator Evidence，不是任务完成结论。
    """
    provider = bound.get("self")
    messages = bound.get("messages")
    tools = bound.get("tools")
    model = bound.get("model")
    provider_class = type(provider).__name__ if provider is not None else None
    eff_model = model or getattr(provider, "default_model", None)
    pname = provider_label(eff_model, provider_class)
    span.artifact("llm.input", llm_input_payload(pname, eff_model, messages, tools, provider_class))
    attrs = llm_attrs(result, pname, eff_model, provider_class)
    attrs["llm.call_id"] = span.span_id
    if span.invocation_source:
        attrs["llm.invocation_source"] = span.invocation_source
    span.set(attrs)
    span.artifact("llm.output", llm_output_payload(result))
    _finish_error(span, result)


def llm_call_stream(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """Streaming Call Extractor，其中 ``self`` 是 AgentLoop，Provider 来自 ``self.provider``。

    字段与 Non-streaming Path 对齐，并额外设置 ``llm.stream=True``。Extractor 处理 Aggregated Result，而
    不是每个 Delta；Stream Incomplete/Failure 的终态仍由 Result/Exception Lifecycle 决定。
    """
    loop = bound.get("self")
    provider = getattr(loop, "provider", None)
    messages = bound.get("messages")
    tools = bound.get("tools")
    model = bound.get("model")
    provider_class = type(provider).__name__ if provider is not None else "stream"
    eff_model = model or getattr(provider, "default_model", None)
    pname = provider_label(eff_model, provider_class)
    span.artifact("llm.input", llm_input_payload(pname, eff_model, messages, tools, provider_class))
    attrs = llm_attrs(result, pname, eff_model, provider_class)
    attrs["llm.call_id"] = span.span_id
    attrs["llm.stream"] = True
    if span.invocation_source:
        attrs["llm.invocation_source"] = span.invocation_source
    span.set(attrs)
    span.artifact("llm.output", llm_output_payload(result))
    _finish_error(span, result)


def tool_call(span, bound: dict[str, Any], result: Any, exc: BaseException | None) -> None:
    """``ToolRegistry.execute`` 的 Extractor。

    Explicit ``skill_read`` 与 Target ``SKILL.md`` 的 ``read_file`` Calls 会 Re-type 为 ``skill.read``；其他
    保持 ``tool.call``，Originating Tool 保存在 ``skill.read.via_tool``。普通 Tool 记录 Name、Args/Result
    Preview、Duration、Input/Output Artifacts 与 Exception/Explicit Failed Signal。

    Tool 返回 ``Error...`` 或 Result ``failed=True`` 时，即使没有 Exception 也标 Error。Tool Result 是执行
    收据，不代表用户目标已完成。
    """
    name = bound.get("name")
    params = bound.get("params")
    call_id = bound.get("call_id")
    explicit_failed = getattr(result, "failed", None)
    result_failed = (
        explicit_failed if isinstance(explicit_failed, bool) else isinstance(result, str) and result.startswith("Error")
    )
    if call_id:
        span.set({"tool.call_id": call_id})
    skill_name = params.get("name") if name == "skill_read" and isinstance(params, dict) else None
    skill_read_path = _skill_read_path(name, params)
    if isinstance(skill_name, str) and skill_name:
        span.retype("skill.read", "skill")
        span.set(
            {
                "skill.tool": name,
                "skill.read.via_tool": "skill_read",
                "skill.injected_via": "skill_read",
                "skill.name": skill_name,
                "skill.result_preview": _preview(result),
            }
        )
    elif skill_read_path:
        span.retype("skill.read", "skill")
        span.set(
            {
                "skill.tool": name,
                "skill.read.via_tool": "read_file",
                "skill.injected_via": "read_file",
                "skill.path": skill_read_path,
                "skill.name": _skill_name_from_path(skill_read_path),
                "skill.result_preview": _preview(result),
            }
        )
    else:
        err = None
        if exc is not None:
            err = repr(exc)
        elif result_failed:
            err = _preview(result, 200)
        span.set(
            {
                "tool.name": name,
                "tool.args_preview": _preview(params, 300),
                "tool.result_preview": _preview(result),
                "tool.error": err,
            }
        )
    span.set({"tool.duration_ms": span.elapsed_ms()})
    span.artifact("tool.input", {"name": name, "params": params})
    span.artifact("tool.output", {"result": result})
    if exc is None and result_failed:
        span.error(_preview(result, 200))
