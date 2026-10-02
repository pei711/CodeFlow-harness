from __future__ import annotations

import asyncio
import contextlib
import contextvars
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator

from codeflow.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
    StreamDelta,
)

from .budget import ProviderRequestNotDispatchedError

_CALL_ROLE = contextvars.ContextVar("codeflowbench_call_role", default="other")
_ATTEMPT_KIND = contextvars.ContextVar("codeflowbench_attempt_kind", default="initial")


@dataclass(frozen=True)
class ModelCallUsage:
    call_id: str
    model: str | None
    requested_model: str | None
    call_role: str
    attempt_kind: str
    provider_dispatched: bool
    succeeded: bool
    error_category: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None


@dataclass(frozen=True)
class AggregateUsage:
    calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    usage_complete: bool


class UsageRecorder:
    def __init__(self) -> None:
        self._records: list[ModelCallUsage] = []

    def record_response(
        self,
        response: LLMResponse,
        *,
        model: str | None,
        call_id: str | None = None,
    ) -> ModelCallUsage:
        usage = response.usage or {}
        record = ModelCallUsage(
            call_id=call_id or uuid.uuid4().hex,
            model=response.model,
            requested_model=model,
            call_role=_CALL_ROLE.get(),
            attempt_kind=_ATTEMPT_KIND.get(),
            provider_dispatched=True,
            succeeded=response.finish_reason != "error",
            error_category=(
                response.error_classification.category if response.error_classification is not None else None
            ),
            input_tokens=_optional_int(usage, "prompt_tokens"),
            output_tokens=_optional_int(usage, "completion_tokens"),
            total_tokens=_optional_int(usage, "total_tokens"),
            cache_read_tokens=_optional_int(usage, "cache_read_input_tokens"),
            cache_write_tokens=_optional_int(usage, "cache_creation_input_tokens"),
            reasoning_tokens=_reasoning_tokens(usage),
        )
        self._records.append(record)
        return record

    def record_failure(
        self,
        exc: BaseException,
        *,
        model: str | None,
        error_category: str | None,
        call_id: str | None = None,
        provider_dispatched: bool = True,
    ) -> ModelCallUsage:
        record = ModelCallUsage(
            call_id=call_id or uuid.uuid4().hex,
            model=None,
            requested_model=model,
            call_role=_CALL_ROLE.get(),
            attempt_kind=_ATTEMPT_KIND.get(),
            provider_dispatched=provider_dispatched,
            succeeded=False,
            error_category=error_category or type(exc).__name__,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            reasoning_tokens=None,
        )
        self._records.append(record)
        return record

    def records(self) -> tuple[ModelCallUsage, ...]:
        return tuple(self._records)

    def has_error_category(self, category: str) -> bool:
        return any(record.error_category == category for record in self._records)

    def aggregate(self) -> AggregateUsage:
        all_records = self.records()
        records = tuple(record for record in all_records if record.provider_dispatched)
        if all_records and not records:
            return AggregateUsage(
                calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
                usage_complete=True,
            )
        return AggregateUsage(
            calls=len(records),
            input_tokens=_sum_complete(records, "input_tokens"),
            output_tokens=_sum_complete(records, "output_tokens"),
            total_tokens=_sum_complete(records, "total_tokens"),
            cache_read_tokens=_sum_known(records, "cache_read_tokens"),
            cache_write_tokens=_sum_known(records, "cache_write_tokens"),
            reasoning_tokens=_sum_known(records, "reasoning_tokens"),
            usage_complete=bool(records)
            and all(
                record.input_tokens is not None and record.output_tokens is not None and record.total_tokens is not None
                for record in records
            ),
        )


@contextlib.contextmanager
def usage_scope(*, call_role: str, attempt_kind: str = "initial") -> Iterator[None]:
    role_token = _CALL_ROLE.set(call_role)
    attempt_token = _ATTEMPT_KIND.set(attempt_kind)
    try:
        yield
    finally:
        _ATTEMPT_KIND.reset(attempt_token)
        _CALL_ROLE.reset(role_token)


class RecordingProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(self, delegate: Any, *, recorder: UsageRecorder) -> None:
        super().__init__(
            api_key=getattr(delegate, "api_key", None),
            api_base=getattr(delegate, "api_base", None),
        )
        self._delegate = delegate
        self._recorder = recorder
        if hasattr(delegate, "generation"):
            self.generation = delegate.generation

    def get_default_model(self) -> str:
        return self._delegate.get_default_model()

    def classify_error(
        self,
        exc: BaseException | None = None,
        content: str | None = None,
    ) -> ErrorClassification:
        return self._delegate.classify_error(exc, content)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        try:
            response = await self._delegate.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
        except asyncio.CancelledError as exc:
            self._record_failure(exc, model=model)
            raise
        except Exception as exc:
            self._record_failure(exc, model=model)
            raise
        self._recorder.record_response(response, model=model)
        return response

    def _record_failure(
        self,
        exc: BaseException,
        *,
        model: str | None,
    ) -> None:
        classify_error = getattr(self._delegate, "classify_error", None)
        classification = classify_error(exc) if callable(classify_error) else LLMProvider.classify_error(exc)
        self._recorder.record_failure(
            exc,
            model=model,
            error_category=classification.category,
            provider_dispatched=not _contains_not_dispatched_error(exc),
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamDelta]:
        final_usage: dict[str, Any] | None = None
        actual_model: str | None = None
        try:
            async for delta in self._delegate.chat_stream(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            ):
                if delta.usage is not None:
                    final_usage = delta.usage
                if delta.model is not None:
                    actual_model = delta.model
                yield delta
        except asyncio.CancelledError as exc:
            self._record_failure(exc, model=model)
            raise
        except Exception as exc:
            self._record_failure(exc, model=model)
            raise
        response = LLMResponse(
            content=None,
            usage=final_usage or {},
            model=actual_model,
        )
        self._recorder.record_response(response, model=model)


def _contains_not_dispatched_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderRequestNotDispatchedError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _optional_int(usage: dict[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if value is not None else None


def _reasoning_tokens(usage: dict[str, Any]) -> int | None:
    direct = usage.get("reasoning_tokens")
    if direct is not None:
        return int(direct)
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
        return int(details["reasoning_tokens"])
    return None


def _sum_complete(records: tuple[ModelCallUsage, ...], field: str) -> int | None:
    if not records:
        return None
    values = [getattr(record, field) for record in records]
    if any(value is None for value in values):
        return None
    return sum(values)


def _sum_known(records: tuple[ModelCallUsage, ...], field: str) -> int | None:
    values = [getattr(record, field) for record in records]
    known = [value for value in values if value is not None]
    return sum(known) if known else None
