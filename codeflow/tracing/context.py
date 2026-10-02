"""通过 `contextvars` 传播的 Per-turn Trace Context。

`contextvars` 能跨 ``await`` 保留，并在 ``asyncio.create_task`` Fork Child Task 时 Snapshot。因此 Mid-turn
Spawn 的 Subagent（P1）会自动继承 Turn Span 作为 Parent，Nested LLM/Tool Calls 挂到正确 Node。Context
携带 Trace/Session/Channel/Turn Correlation 与当前 Non-model Source。

该传播只覆盖同一 Python Context Lineage；跨 Process/External Queue 仍需显式传递 ID。Context 存在不
保证 Span 已持久化。
"""

from __future__ import annotations

import contextlib
import contextvars
import secrets
import time
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True)
class TraceCtx:
    trace_id: str
    session_key: str | None = None
    channel: str | None = None
    chat_id: str | None = None
    parent_span_id: str | None = None
    turn_span_id: str | None = None
    # 最近一层非模型 span 的名称，表示模型调用服务的目的
    # （turn / memory.extract / skill.gate / ...）。模型类 span 继承该来源，
    # 而不是把自身作为来源，因此嵌套调用
    # ``llm.call`` 无需遍历树即可自行标记；通用跨度没有采用方。
    # 的名称无需硬编码在此处。
    source: str | None = None


_CTX: contextvars.ContextVar[TraceCtx | None] = contextvars.ContextVar("codeflow_tracing_ctx", default=None)


def current() -> TraceCtx | None:
    return _CTX.get()


def new_trace_id() -> str:
    return f"trace-{int(time.time() * 1000):x}-{secrets.token_hex(4)}"


def new_span_id() -> str:
    return f"span-{int(time.time() * 1000):x}-{secrets.token_hex(3)}"


@contextlib.contextmanager
def turn_scope(
    *,
    session_key: str | None,
    channel: str | None,
    chat_id: str | None,
    root_span_id: str,
) -> Iterator[TraceCtx]:
    """为一个 Turn 打开 Fresh Trace，使 Child Spans Parent 到 ``root_span_id``。

    方法生成 New Trace ID，绑定 Session/Channel/Chat 与 Root Parent，并在 Context Manager 退出时恢复此前
    Context。Yield 的 `TraceCtx` 是 Frozen Snapshot；Scope 本身不创建或写入 Root Span Record。
    """
    ctx = TraceCtx(
        trace_id=new_trace_id(),
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
        parent_span_id=root_span_id,
    )
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def push(
    *,
    trace_id: str,
    span_id: str,
    name: str | None = None,
    kind: str | None = None,
    session_key: str | None = None,
    channel: str | None = None,
    chat_id: str | None = None,
):
    """设置 Active Context，使 Descendants Parent 到 ``span_id``，并返回 Reset Token。

    ``trace.span`` Facade 用于 Manual Instrumentation，显式控制 Enter/Exit 而非 ``with`` Block；必须与
    :func:`reset` Pair。``name``/``kind`` 还传播 :class:`TraceCtx` 的 Enclosing ``source``：Non-model Span
    成为后代 Source，Model Span 继承 Parent Source，永不把自身误标为 Invocation Source。

    ``session.turn`` 会把当前 Span 同时记为 ``turn_span_id``；其他 Span 继承已有 Turn Correlation。
    """
    cur = _CTX.get()
    parent_source = cur.source if cur else None
    source = parent_source if (kind == "model" or not name) else name
    return _CTX.set(
        TraceCtx(
            trace_id=trace_id,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            parent_span_id=span_id,
            turn_span_id=span_id if name == "session.turn" else (cur.turn_span_id if cur else None),
            source=source,
        )
    )


def reset(token) -> None:
    _CTX.reset(token)


@contextlib.contextmanager
def child_scope(span_id: str) -> Iterator[TraceCtx]:
    """把 Descendants Re-parent 到 ``span_id``，供 Subagent Probe P1 使用。

    当前 Context 缺失时创建 Fresh Trace，存在时保留其他字段只替换 Parent Span。退出时恢复原 Context，
    避免后续 Sibling Calls 错挂到 Child。Scope 不验证 Parent Span 是否真实存在于 Store。
    """
    cur = _CTX.get() or TraceCtx(trace_id=new_trace_id())
    token = _CTX.set(replace(cur, parent_span_id=span_id))
    try:
        yield _CTX.get()  # type: ignore[misc]
    finally:
        _CTX.reset(token)
