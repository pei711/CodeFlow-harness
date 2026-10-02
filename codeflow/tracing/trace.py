"""Standard Adopters 调用的 Public Instrumentation API。

历史说明见 ``docs/TRACING_STANDARD_API.md``。这是 ``context`` / ``spans`` / ``store`` 上的 Thin Facade：
Enter 时 Open Span，Exit 时 Finalize + Emit；Contextvars 自动 Nest，内层 Span 成为外层 Child。

Safety Guarantees：Disabled 时 **No-op**，Yield No-op Handle 且无 IO；Tracing-internal Failure **Never Breaks
Caller**，但 Application Own Exception 在记录 ``status=ERROR`` 后原样 Re-raise；Import/Call **Import-safe**，
无需 Config File 存在。这些保证优先于观测完整性，因此 API Success 不保证 Span Durable。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Iterator

from codeflow.utils.persisted_payload import sanitize_persisted_payload

from . import config
from . import context as _ctx
from . import spans as _spans

_log = logging.getLogger("codeflow.tracing")

# 名称域 -> 粗粒度 kind（span.type）；自定义节点显式传入 kind=。
_KIND_BY_DOMAIN = {
    "session": "session",
    "spine": "session",
    "llm": "model",
    "tool": "tool",
    "subagent": "subagent",
    "skill": "skill",
    "memory": "memory",
    "plugin": "plugin",
    "tracing": "plugin",
    "channel": "channel",
}


def _derive_kind(name: str) -> str:
    domain = name.split(".", 1)[0]
    return _KIND_BY_DOMAIN.get(domain, domain or "internal")


class Span:
    """一个 Open Span 的 Mutable Handle；所有公开操作 No-throw 并返回 ``self``。

    Handle 拥有 Identity、Parent、Session Correlation、Start Time、Attributes/Events、Status 与 Cancellation。
    调用方可链式 Set/Artifact/Event/Error/Retype/Checkpoint；真正 Final Emit 由 `span` Context Manager 负责。
    """

    __slots__ = (
        "name",
        "kind",
        "trace_id",
        "span_id",
        "_parent",
        "_start",
        "_attrs",
        "_events",
        "_status_code",
        "_status_message",
        "_session_key",
        "_channel",
        "_chat_id",
        "_perf0",
        "_cancelled",
        "_source",
    )

    def __init__(self, name, kind, *, trace_id, span_id, parent, session_key, channel, chat_id, start, source=None):
        self.name = name
        self.kind = kind
        self.trace_id = trace_id
        self.span_id = span_id
        self._parent = parent
        self._session_key = session_key
        self._channel = channel
        self._chat_id = chat_id
        self._start = start
        self._source = source
        self._perf0 = time.monotonic()
        self._cancelled = False
        self._attrs: dict[str, Any] = {}
        self._events: list[dict[str, Any]] = []
        self._status_code = "OK"
        self._status_message = ""

    def set(self, attributes: dict[str, Any] | None = None, **kw) -> "Span":
        """把 Attributes Merge 到 Span。

        Standard Keys 使用 Fully-qualified Dotted Names，如 ``llm.provider``、``tool.name``，应以 Mapping
        传入：``s.set({"llm.provider": p})``。Bare Keywords ``s.set(foo=1)`` 为 Convenience 接受，但 Verbatim
        存储，不自动 Namespace。Later Value 覆盖 Same Key Earlier Value。
        """
        if attributes:
            self._attrs.update(attributes)
        if kw:
            self._attrs.update(kw)
        return self

    def artifact(self, key: str, payload: Any, *, kind: str = "json") -> "Span":
        """把 Large Payload Out-of-line Persist，并挂接 ``<key>.artifact_*`` + Preview。

        写入前通过 `sanitize_persisted_payload` 移除 Inline Image Data。Storage Failure 仅 Debug Log，不抛错；
        ``kind`` 保留在接口中，当前底层根据 Payload Type 选择文件扩展。Artifact Attributes 出现才表示返回
        Reference，仍需检查 Error/Path。
        """
        try:
            meta = {"traceId": self.trace_id, "sessionKey": self._session_key}
            art = _spans.persist_artifact(key, meta, sanitize_persisted_payload(payload), label=key)
            self._attrs.update(_spans.artifact_attributes(key, art))
        except Exception:  # noqa: BLE001 — tracing 不得破坏 Host
            _log.debug("tracing: artifact(%s) failed", key, exc_info=True)
        return self

    def event(self, name: str) -> "Span":
        self._events.append({"time": _spans.now_iso(), "name": name})
        return self

    def error(self, exc: BaseException | str) -> "Span":
        self._status_code = "ERROR"
        self._status_message = repr(exc) if isinstance(exc, BaseException) else str(exc)
        return self

    def retype(self, name: str, kind: str | None = None) -> "Span":
        """在 Emit 前 Runtime 修改 Span Name/Kind。

        用于 True Type 只有 Call 后才知道的情况，例如读取 ``SKILL.md`` 的 ``tool.call`` 应成为
        ``skill.read``。Span ID 固定，因此 Nesting 不受影响；Kind 为 `None` 时只改 Name。
        """
        self.name = name
        if kind:
            self.kind = kind
        return self

    @property
    def invocation_source(self) -> str | None:
        """返回 Nearest Enclosing Non-model Span Name，即本 Span 代表哪个 Operation 工作。

        Root 返回 `None`。Model Call 可借此 Self-label Purpose，无需 Extractor Walk Tree；值由 Context Push
        时继承。
        """
        return self._source

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._perf0) * 1000)

    def cancel(self) -> "Span":
        """Drop 此 Span，使 Close 时 Nothing Emitted。

        用于 Conditional Spans，例如只有真实注入时才保留 ``skill.inject``。Cancel 不回滚已经单独 Persist
        的 Artifact，也不影响 Child Spans；Detached 模式用于避免 Child 挂到可能被取消的 Parent。
        """
        self._cancelled = True
        return self

    def checkpoint(self) -> "Span":
        """立即 Emit Span 当前 In-progress State，但不 Close。

        Checkpoint 与 Final Emit 使用 Same ``span_id``，Viewer 按 ID Dedup 并保留 Last Write。Long Root Span
        如 Turn 用它让 Mid-flight Children 在 Turn 未结束时已有 Root 可 Group。Emit Failure 被吞掉；后续
        Final Close 仍会尝试写入。
        """
        try:
            _spans.emit(
                _spans.build_span(
                    self.name,
                    self.kind,
                    trace_id=self.trace_id,
                    span_id=self.span_id,
                    parent_span_id=self._parent,
                    session_key=self._session_key,
                    channel=self._channel,
                    chat_id=self._chat_id,
                    start_time=self._start,
                    end_time=_spans.now_iso(),
                    status_code=self._status_code,
                    status_message=self._status_message,
                    attributes=self._attrs,
                    events=self._events,
                )
            )
        except Exception:  # noqa: BLE001
            _log.debug("tracing: checkpoint(%s) failed", self.name, exc_info=True)
        return self


class _NoopSpan:
    """Tracing Disabled 或 Internal Open Failed 时返回的 No-op Handle。

    它实现与 `Span` 相同的 Chainable Surface，Identity 为空、Elapsed 为零，所有方法无副作用，使 Caller
    无需分支判断。
    """

    trace_id = ""
    span_id = ""
    name = ""
    invocation_source = None

    def set(self, *_a, **_k):
        return self

    def artifact(self, *_a, **_k):
        return self

    def event(self, *_a):
        return self

    def error(self, *_a):
        return self

    def retype(self, *_a, **_k):
        return self

    def elapsed_ms(self):
        return 0

    def checkpoint(self, *_a, **_k):
        return self

    def cancel(self, *_a, **_k):
        return self


@contextlib.contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    kind: str | None = None,
    session_key: str | None = None,
    channel: str | None = None,
    chat_id: str | None = None,
    detached: bool = False,
    root: bool = False,
    **kw,
) -> Iterator[Any]:
    """为 ``name``（``<domain>.<verb>``）Open Span，并 Yield :class:`Span` Handle。

    ``attributes`` 使用 Fully-qualified Dotted Keys；调用形式是 ``with span(...):``，例如
    ``{"llm.provider": p, "llm.model": m}``；``**kw`` 接受 Bare Keys。With Block 内 Child 自动 Nest。Body
    Exception 会把 Span 标 ``ERROR``，再 Unchanged Re-raise；`CancelledError` 也明确记录 Cancelled。

    ``root=True`` 拒绝 Inherited Context，始终 Mint Fresh Trace 且无 Parent。``detached=True`` 让 Span 不成为
    Active Parent，适合可能 Cancel 的 Leaf。Disabled/Open Failure Yield `_NoopSpan`。Final Emission
    Best-effort，不改变 Application Return/Exception。
    """
    if not config.enabled():
        yield _NoopSpan()
        return

    try:
        cur = None if root else _ctx.current()
        trace_id = cur.trace_id if cur else _ctx.new_trace_id()
        parent = cur.parent_span_id if cur else None
        # 传入 session 身份时以此创建根 span（Turn）；否则继承活动上下文，
        # 使子 span 携带 Turn 身份。
        session_key = session_key if session_key is not None else (cur.session_key if cur else None)
        channel = channel if channel is not None else (cur.channel if cur else None)
        chat_id = chat_id if chat_id is not None else (cur.chat_id if cur else None)
        span_id = _ctx.new_span_id()
        handle = Span(
            name,
            kind or _derive_kind(name),
            trace_id=trace_id,
            span_id=span_id,
            parent=parent,
            session_key=session_key,
            channel=channel,
            chat_id=chat_id,
            start=_spans.now_iso(),
            source=cur.source if cur else None,
        )
        handle.set(attributes, **kw)
        # detached span 是叶节点标记，不会成为活动父节点；其中执行的工作会
        # 挂到它的父节点，而不是它自身。这是可取消 span（如 skill.inject）
        # 所必需的，否则取消前打开的子节点会悬挂在一个永不发出的 span 上。
        token = (
            None
            if detached
            else _ctx.push(
                trace_id=trace_id,
                span_id=span_id,
                name=handle.name,
                kind=handle.kind,
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
            )
        )
    except Exception:  # noqa: BLE001 — 打开追踪绝不能影响宿主
        _log.debug("tracing: span(%s) open failed", name, exc_info=True)
        yield _NoopSpan()
        return

    try:
        yield handle
    # 必须捕获 BaseException 而非 Exception：asyncio.CancelledError 继承自
    # BaseException。只捕获 Exception 会把已取消 span 以 OK 关闭，导致记录
    # 感知不到取消。
    except BaseException as exc:
        handle.error("cancelled" if isinstance(exc, asyncio.CancelledError) else exc)
        raise
    finally:
        try:
            if token is not None:
                _ctx.reset(token)
            if not handle._cancelled:
                _spans.emit(
                    _spans.build_span(
                        handle.name,
                        handle.kind,
                        trace_id=handle.trace_id,
                        span_id=handle.span_id,
                        parent_span_id=handle._parent,
                        session_key=handle._session_key,
                        channel=handle._channel,
                        chat_id=handle._chat_id,
                        start_time=handle._start,
                        end_time=_spans.now_iso(),
                        status_code=handle._status_code,
                        status_message=handle._status_message,
                        attributes=handle._attrs,
                        events=handle._events,
                    )
                )
        except Exception:  # noqa: BLE001
            _log.debug("tracing: span(%s) emit failed", name, exc_info=True)


@contextlib.contextmanager
def attach(trace_id: str | None, parent_span_id: str | None = None) -> Iterator[None]:
    """在未继承 Context 的 Task 上 Re-enter Existing Trace。

    ``contextvars`` 只在 Task Creation 时传播；一个服务 Many Turns 的 *Resident* Worker 会携带启动它的
    某一 Turn Context。Worker 在 Hand-off 捕获 IDs，并在此 Attach，使新 Span Join Originating Trace，而
    非 Mint Unrelated Trace 或 Reuse Stale One。

    Falsy ``trace_id`` 表示 Capture 时 Tracing Off，方法 No-op。Push/Reset Failure 只 Debug Log；Scope
    退出恢复此前 Context。
    """
    if not config.enabled() or not trace_id:
        yield
        return
    token = None
    try:
        token = _ctx.push(trace_id=trace_id, span_id=parent_span_id)
    except Exception:  # noqa: BLE001 — 附加追踪绝不能影响宿主
        _log.debug("tracing: attach(%s) failed", trace_id, exc_info=True)
    try:
        yield
    finally:
        if token is not None:
            try:
                _ctx.reset(token)
            except Exception:  # noqa: BLE001
                _log.debug("tracing: attach reset failed", exc_info=True)


def instrument(name: str, *, kind: str | None = None, detached: bool = False, seed=None, on_open=None, extract=None):
    """Decorator：Wrap Sync/Async Method，使每次 Call Emit ``name`` Span。

    Adopter Integration Surface 只需 Annotate Method，不改 Body：

        @trace.instrument("llm.call", extract=semconv.llm_call)
        async def chat_with_retry(self, ...): ...

    Optional Hooks 都接收按 Name Bind 的 ``bound_args``：``seed(bound) -> dict`` 提供
    ``session_key``/``channel``/``chat_id``，给 Root Turn Seed Identity；``on_open(span, bound)`` 在 Body 前
    记录 Input，并可调用 ``span.checkpoint()``；``extract(span, bound, result, exc)`` 在 ``finally`` 填 Final Attributes/Artifacts，
    Error 时 Result 为 `None`，Success 时 Exc 为 `None`。

    Wrapper 同时支持 Coroutine 与 Sync Function。所有 Tracing Work No-throw，Disabled 时 No-op；Application
    Own BaseException 记录后原样传播，Wrapped Method Behavior 不被改变。
    """
    import functools
    import inspect

    def decorate(func):
        sig = inspect.signature(func)

        def _bind(args, kwargs):
            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            return b.arguments

        def _seed(args, kwargs) -> dict:
            if seed is None:
                return {}
            try:
                return seed(_bind(args, kwargs)) or {}
            except Exception:  # noqa: BLE001
                _log.debug("tracing: seed for %s failed", name, exc_info=True)
                return {}

        def _open(s, args, kwargs) -> None:
            if on_open is not None:
                try:
                    on_open(s, _bind(args, kwargs))
                except Exception:  # noqa: BLE001
                    _log.debug("tracing: on_open for %s failed", name, exc_info=True)

        def _close(s, args, kwargs, result, exc) -> None:
            if extract is not None:
                try:
                    extract(s, _bind(args, kwargs), result, exc)
                except Exception:  # noqa: BLE001 — 提取失败不得破坏 Host
                    _log.debug("tracing: extract for %s failed", name, exc_info=True)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                if not config.enabled():
                    return await func(*args, **kwargs)
                with span(name, kind=kind, detached=detached, **_seed(args, kwargs)) as s:
                    _open(s, args, kwargs)
                    result = exc = None
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except BaseException as e:  # noqa: BLE001 — 记录后重新抛出
                        exc = e
                        raise
                    finally:
                        _close(s, args, kwargs, result, exc)

            return awrapper

        @functools.wraps(func)
        def swrapper(*args, **kwargs):
            if not config.enabled():
                return func(*args, **kwargs)
            with span(name, kind=kind, detached=detached, **_seed(args, kwargs)) as s:
                _open(s, args, kwargs)
                result = exc = None
                try:
                    result = func(*args, **kwargs)
                    return result
                except BaseException as e:  # noqa: BLE001 — 记录后重新抛出
                    exc = e
                    raise
                finally:
                    _close(s, args, kwargs, result, exc)

        return swrapper

    return decorate


def current() -> Any | None:
    """返回 Active Trace Context；不存在时为 `None`。

    供需要显式关联 Usage/Delivery 的 Adopters 读取；返回对象是 Context Snapshot，不应修改。
    """
    return _ctx.current()


def enabled() -> bool:
    return config.enabled()
