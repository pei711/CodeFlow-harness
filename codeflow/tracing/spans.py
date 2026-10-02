"""``audit.span.v1`` Span Construction 与 Best-effort Emission。

所有 Collector 写同一 Schema，因此 One Viewer 可以渲染 Any Framework Traces。CodeFlow 在 Original Five
``span.type`` 之外增加 ``memory``、``plugin``、``skill``，后者覆盖 ``skill.read`` / Inject 等事件。
`build_span` 统一共享字段，`emit` 追加 Span，`persist_artifact` 保存 Full Payload 并返回可挂到 Span 的
Reference。

Store 采用 Lazy Singleton。所有写入异常都吞掉，因为 Tracing 不能影响 Host；代价是调用方不能只凭函数
返回判断 Evidence Durable，需检查 Trace Store/Health。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import config
from .store import TraceStore

SCHEMA_VERSION = "audit.span.v1"
FRAMEWORK = "codeflow"

_store: TraceStore | None = None


def _get_store() -> TraceStore:
    global _store
    if _store is None:
        _store = TraceStore(config.state_dir())
    return _store


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_span(
    name: str,
    span_type: str,
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    session_key: str | None = None,
    channel: str | None = None,
    chat_id: str | None = None,
    start_time: str,
    end_time: str | None = None,
    status_code: str = "OK",
    status_message: str = "",
    attributes: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "span.type": span_type,
        "framework": FRAMEWORK,
        # session.id + channel.id 是共享 viewer 的分组键，
        # （audit.span.v1 公共属性）。将 session_key/channel 同步写入其中。
        # 使 CodeFlow trace 与其他来源一样按 conversation -> turn 分组。
        "session.id": session_key,
        "session.key": session_key,
        "channel": channel,
        "channel.id": channel,
        "chat_id": chat_id,
        "audit.schema_version": SCHEMA_VERSION,
    }
    if attributes:
        attrs.update(attributes)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent_span_id,
        "name": name,
        "kind": "INTERNAL",
        "startTime": start_time,
        "endTime": end_time or start_time,
        "status": {"code": status_code, "message": status_message},
        "attributes": attrs,
        "events": events or [],
    }


def emit(span: dict[str, Any]) -> None:
    try:
        _get_store().append_span(span)
    except Exception:  # noqa: BLE001 — 追踪绝不能影响宿主
        pass


def persist_artifact(kind: str, meta: dict[str, Any], payload: Any, *, label: str | None = None):
    try:
        return _get_store().persist_artifact(kind, meta, payload, label=label, preview_length=config.preview_len())
    except Exception:  # noqa: BLE001
        return None


def artifact_attributes(prefix: str, artifact: dict[str, Any] | None) -> dict[str, Any]:
    return TraceStore.artifact_attributes(prefix, artifact)
