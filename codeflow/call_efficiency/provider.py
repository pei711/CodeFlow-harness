"""把 `CallEfficiency` 应用于每次 Runtime-owned Call 的 Provider Decorator。

`CallEfficiencyProvider` 包装真实 `LLMProvider`，在请求前调用 Controller `prepare` 统一 Cache Policy，
在普通响应、Retry Attempt、Stream 完成、异常和取消等终点调用 `record`。Decorator 保留底层
Provider 的 Chat Interface，因此 Agent Loop 不需要为计量逻辑增加另一条调用路径。

包装器通过快照与锁支持运行期替换 Delegate，并避免重复包装形成嵌套 Controller。它记录的是每次
实际 Attempt 的模型、Usage 与 Outcome；发生异常时会先尽力写下失败证据再重新抛出，绝不把计量
行为变成对 Provider Error 的吞没。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from threading import RLock
from typing import Any

from codeflow.call_efficiency.runtime import CallEfficiency
from codeflow.providers.base import ErrorClassification, LLMProvider, LLMResponse, StreamDelta
from codeflow.tracing import trace


class CallEfficiencyProvider(LLMProvider):
    def __init__(self, delegate: LLMProvider, controller: CallEfficiency) -> None:
        super().__init__()
        self.controller = controller
        self._replace_lock = RLock()
        self.replace(delegate)

    def replace(self, delegate: LLMProvider) -> None:
        if isinstance(delegate, CallEfficiencyProvider):
            delegate = delegate._delegate_snapshot()
        with self._replace_lock:
            self.delegate = delegate
            self.controller.provider = delegate
            self.generation = getattr(delegate, "generation", self.generation)

    def _delegate_snapshot(self) -> LLMProvider:
        with self._replace_lock:
            return self.delegate

    def get_default_model(self) -> str:
        get_default_model = getattr(self._delegate_snapshot(), "get_default_model", None)
        return get_default_model() if callable(get_default_model) else ""

    def supports_explicit_cache_control(self, model: str) -> bool:
        supports = getattr(self._delegate_snapshot(), "supports_explicit_cache_control", None)
        return bool(supports(model)) if callable(supports) else False

    def classify_error(self, *args: Any, **kwargs: Any):
        return self._delegate_snapshot().classify_error(*args, **kwargs)

    @staticmethod
    def _failure(exc: BaseException, delegate: LLMProvider) -> tuple[str, ErrorClassification]:
        if isinstance(exc, GeneratorExit):
            return "incomplete", ErrorClassification("stream_incomplete")
        try:
            import asyncio

            if isinstance(exc, asyncio.CancelledError):
                return "cancelled", ErrorClassification("cancelled")
            return "error", delegate.classify_error(exc)
        except Exception:
            return "error", ErrorClassification(type(exc).__name__)

    def _record_exception(
        self,
        exc: BaseException,
        *,
        delegate: LLMProvider,
        requested_model: str,
        attempted_model: str,
        cache_policy: str,
        ctx: Any,
        usage: dict[str, Any] | None = None,
        actual_model: str | None = None,
    ) -> None:
        outcome, classification = self._failure(exc, delegate)
        self.controller.record(
            LLMResponse(
                content=None,
                finish_reason="error",
                usage=usage or {},
                model=actual_model or attempted_model,
                error_classification=classification,
            ),
            requested_model=requested_model,
            attempted_model=attempted_model,
            session_key=getattr(ctx, "session_key", None),
            cache_policy=cache_policy,
            trace_id=getattr(ctx, "trace_id", None),
            turn_span_id=getattr(ctx, "turn_span_id", None),
            outcome=outcome,
        )

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        *,
        fallback_models: list[str] | None = None,
        request_transform: Callable[
            [list[dict[str, Any]], list[dict[str, Any]] | None, str | None],
            tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str | None],
        ]
        | None = None,
        response_observer: Callable[[LLMResponse, str | None], Awaitable[None]] | None = None,
        attempt_started: Callable[[str | None], None] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        delegate = self._delegate_snapshot()
        requested_model = model or delegate.get_default_model()
        prepared_by_model: dict[str | None, Any] = {}
        active_attempt: str | None = None
        ctx = trace.current()

        def _prepare(
            attempt_messages: list[dict[str, Any]],
            attempt_tools: list[dict[str, Any]] | None,
            attempt_model: str | None,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, str | None]:
            if request_transform is not None:
                attempt_messages, attempt_tools, attempt_model = request_transform(
                    attempt_messages,
                    attempt_tools,
                    attempt_model,
                )
            prepared = self.controller.prepare(
                attempt_messages,
                attempt_tools,
                attempt_model or requested_model,
                provider=delegate,
            )
            prepared_by_model[prepared.model] = prepared
            return prepared.messages, prepared.tools, prepared.model

        def _started(attempt_model: str | None) -> None:
            nonlocal active_attempt
            if attempt_started is not None:
                attempt_started(attempt_model)
            active_attempt = attempt_model

        async def _observe(response: LLMResponse, attempt_model: str | None) -> None:
            nonlocal active_attempt
            active_attempt = None
            prepared = prepared_by_model.get(attempt_model)
            policy = prepared.cache_policy if prepared is not None else "observe_only"
            response.call_record = self.controller.record(
                response,
                requested_model=requested_model,
                attempted_model=attempt_model or requested_model,
                session_key=getattr(ctx, "session_key", None),
                cache_policy=policy,
                trace_id=getattr(ctx, "trace_id", None),
                turn_span_id=getattr(ctx, "turn_span_id", None),
            )
            if response_observer is not None:
                await response_observer(response, attempt_model)

        delegate_retry = getattr(delegate, "chat_with_retry")
        try:
            return await delegate_retry(
                messages=messages,
                tools=tools,
                model=requested_model,
                fallback_models=fallback_models,
                request_transform=_prepare,
                response_observer=_observe,
                attempt_started=_started,
                **kwargs,
            )
        except BaseException as exc:
            if active_attempt is not None:
                prepared = prepared_by_model.get(active_attempt)
                self._record_exception(
                    exc,
                    delegate=delegate,
                    requested_model=requested_model,
                    attempted_model=active_attempt,
                    cache_policy=(prepared.cache_policy if prepared is not None else "observe_only"),
                    ctx=ctx,
                )
            raise

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        delegate = self._delegate_snapshot()
        requested_model = model or delegate.get_default_model()
        prepared = self.controller.prepare(messages, tools, requested_model, provider=delegate)
        ctx = trace.current()
        try:
            response = await delegate.chat(
                messages=prepared.messages,
                tools=prepared.tools,
                model=prepared.model,
                **kwargs,
            )
        except BaseException as exc:
            self._record_exception(
                exc,
                delegate=delegate,
                requested_model=requested_model,
                attempted_model=prepared.model,
                cache_policy=prepared.cache_policy,
                ctx=ctx,
            )
            raise
        if response.finish_reason == "error" and response.error_classification is None:
            response.error_classification = delegate.classify_error(content=response.content)
        response.call_record = self.controller.record(
            response,
            requested_model=requested_model,
            attempted_model=prepared.model,
            session_key=getattr(ctx, "session_key", None),
            cache_policy=prepared.cache_policy,
            trace_id=getattr(ctx, "trace_id", None),
            turn_span_id=getattr(ctx, "turn_span_id", None),
        )
        return response

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        delegate = self._delegate_snapshot()
        requested_model = model or delegate.get_default_model()
        prepared = self.controller.prepare(messages, tools, requested_model, provider=delegate)
        ctx = trace.current()
        content: list[str] = []
        usage: dict[str, Any] = {}
        actual_model: str | None = None
        finish_reason = "stop"
        error_classification: ErrorClassification | None = None
        try:
            async for delta in delegate.chat_stream(
                messages=prepared.messages,
                tools=prepared.tools,
                model=prepared.model,
                **kwargs,
            ):
                if delta.content:
                    content.append(delta.content)
                if delta.usage is not None:
                    usage = delta.usage
                if delta.model:
                    actual_model = delta.model
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                if delta.error_classification is not None:
                    error_classification = delta.error_classification
                delta.cache_policy = prepared.cache_policy
                yield delta
        except BaseException as exc:
            self._record_exception(
                exc,
                delegate=delegate,
                requested_model=requested_model,
                attempted_model=prepared.model,
                cache_policy=prepared.cache_policy,
                ctx=ctx,
                usage=usage,
                actual_model=actual_model,
            )
            raise
        if finish_reason == "error" and error_classification is None:
            error_classification = delegate.classify_error(content="".join(content))
        record = self.controller.record(
            LLMResponse(
                content="".join(content),
                finish_reason=finish_reason,
                usage=usage,
                model=actual_model or prepared.model,
                error_classification=error_classification,
            ),
            requested_model=requested_model,
            attempted_model=prepared.model,
            session_key=getattr(ctx, "session_key", None),
            cache_policy=prepared.cache_policy,
            trace_id=getattr(ctx, "trace_id", None),
            turn_span_id=getattr(ctx, "turn_span_id", None),
        )
        yield StreamDelta(
            content=None,
            model=actual_model or prepared.model,
            cache_policy=prepared.cache_policy,
            call_record=record,
        )


__all__ = ["CallEfficiencyProvider"]
