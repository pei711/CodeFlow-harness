from __future__ import annotations

import contextlib
import contextvars
import fcntl
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import tiktoken

from codeflow.providers.base import (
    ErrorClassification,
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    StreamDelta,
)

from .canonical import canonical_digest, to_primitive


class ProviderBudgetError(RuntimeError):
    pass


class ProviderRequestNotDispatchedError(ProviderBudgetError):
    pass


_INPUT_COUNT_METHOD = "max(tiktoken/cl100k_base*1.2,utf8-bytes)+64/canonical-request-json/v1"
_INPUT_TOKEN_MARGIN_RATIO = 1.2
_INPUT_TOKEN_FIXED_MARGIN = 64


@dataclass
class _TrialCallState:
    trial_id: str
    max_logical_calls: int
    max_attempts_per_call: int
    max_input_tokens_per_call: int | None
    max_output_tokens_per_call: int | None


_TRIAL_STATE = contextvars.ContextVar[_TrialCallState | None]("codeflowbench_provider_trial_state", default=None)


@contextlib.contextmanager
def provider_call_budget_scope(
    *,
    trial_id: str,
    max_logical_calls: int,
    max_attempts_per_call: int,
    max_input_tokens_per_call: int | None = None,
    max_output_tokens_per_call: int | None = None,
) -> Iterator[None]:
    if (
        max_logical_calls < 0
        or max_attempts_per_call < 1
        or (max_input_tokens_per_call is not None and max_input_tokens_per_call < 1)
        or (max_output_tokens_per_call is not None and max_output_tokens_per_call < 1)
    ):
        raise ValueError("Provider call limits are invalid")
    token = _TRIAL_STATE.set(
        _TrialCallState(
            trial_id=trial_id,
            max_logical_calls=max_logical_calls,
            max_attempts_per_call=max_attempts_per_call,
            max_input_tokens_per_call=max_input_tokens_per_call,
            max_output_tokens_per_call=max_output_tokens_per_call,
        )
    )
    try:
        yield
    finally:
        _TRIAL_STATE.reset(token)


@dataclass(frozen=True)
class ProviderBudgetConfig:
    hard_cap_cny: float
    external_service_reserve_cny: float
    max_total_request_attempts: int
    max_input_tokens_per_call: int
    max_output_tokens_per_call: int
    input_cache_miss_usd_per_million: float
    output_usd_per_million: float
    conservative_usd_to_cny_multiplier: float
    max_additional_request_attempts: int | None = None
    request_attempt_baseline: int = 0
    approval_digest: str | None = None
    ledger_prefix_event_count: int = 0
    ledger_prefix_digest: str | None = None
    ledger_prefix_charged_cny: float = 0.0

    def __post_init__(self) -> None:
        if self.request_attempt_baseline < 0 or self.max_total_request_attempts < self.request_attempt_baseline:
            raise ValueError("Provider request-attempt limits are invalid")
        if self.max_additional_request_attempts is not None and self.max_additional_request_attempts < 0:
            raise ValueError(
                "additional Provider request-attempt limit is invalid",
            )
        if self.ledger_prefix_event_count < 0 or self.ledger_prefix_charged_cny < 0:
            raise ValueError("Provider ledger prefix is invalid")
        if self.ledger_prefix_event_count and self.ledger_prefix_digest is None:
            raise ValueError("Provider ledger prefix digest is required")

    @property
    def maximum_request_cny(self) -> float:
        return self.cost_cny(
            self.max_input_tokens_per_call,
            self.max_output_tokens_per_call,
        )

    def cost_cny(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        usd = (
            input_tokens / 1_000_000 * self.input_cache_miss_usd_per_million
            + output_tokens / 1_000_000 * self.output_usd_per_million
        )
        return usd * self.conservative_usd_to_cny_multiplier


@dataclass(frozen=True)
class ProviderBudgetSnapshot:
    ledger_path: str
    ledger_event_count: int
    request_attempts: int
    additional_request_attempts: int
    additional_request_attempt_ceiling: int | None
    provider_charged_cny: float
    external_service_reserve_cny: float
    total_committed_cny: float
    hard_cap_cny: float
    open_reservations: int
    accounting_complete: bool
    ledger_digest: str
    request_attempt_baseline: int
    request_attempt_lifetime_ceiling: int
    approval_digest: str | None
    high_water_path: str
    high_water_digest: str


class ProviderBudgetLedger:
    def __init__(
        self,
        path: Path,
        config: ProviderBudgetConfig,
    ) -> None:
        self.path = Path(path)
        self.high_water_path = self.path.with_suffix(
            ".high-water.json",
        )
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(
        self,
        *,
        trial_id: str,
        request_digest: str,
        model: str | None,
        estimated_input_tokens: int,
        maximum_input_tokens: int | None = None,
        maximum_output_tokens: int | None = None,
        max_logical_calls: int | None = None,
        max_attempts_per_call: int = 1,
    ) -> str:
        input_ceiling = (
            maximum_input_tokens if maximum_input_tokens is not None else self.config.max_input_tokens_per_call
        )
        if input_ceiling < 1 or input_ceiling > self.config.max_input_tokens_per_call:
            raise ProviderBudgetError(
                "request input ceiling exceeds the campaign ceiling",
            )
        if estimated_input_tokens > input_ceiling:
            raise ProviderBudgetError(
                "estimated input tokens exceed the per-call ceiling",
            )
        output_ceiling = (
            maximum_output_tokens if maximum_output_tokens is not None else self.config.max_output_tokens_per_call
        )
        if output_ceiling < 1 or output_ceiling > self.config.max_output_tokens_per_call:
            raise ProviderBudgetError(
                "request output ceiling exceeds the campaign ceiling",
            )
        if (max_logical_calls is not None and max_logical_calls < 0) or max_attempts_per_call < 1:
            raise ProviderBudgetError(
                "request retry limits are invalid",
            )
        maximum_request_cny = self.config.cost_cny(
            input_ceiling,
            output_ceiling,
        )
        request_id = uuid.uuid4().hex
        with self._locked() as handle:
            events = _read_events(handle)
            state = _ledger_state(events)
            if state.accounting_blocked:
                raise ProviderBudgetError(
                    "Provider budget accounting is incomplete",
                )
            if state.request_attempts >= self.config.max_total_request_attempts:
                raise ProviderBudgetError(
                    "Provider request-attempt ceiling reached",
                )
            scoped = _scoped_reservation_state(
                events,
                baseline=self.config.request_attempt_baseline,
            )
            logical_call_key = canonical_digest(
                {
                    "trial_id": trial_id,
                    "request_digest": request_digest,
                }
            )
            attempt_ordinal = (
                scoped.attempts_by_logical_call.get(
                    logical_call_key,
                    0,
                )
                + 1
            )
            if (
                logical_call_key not in scoped.attempts_by_logical_call
                and max_logical_calls is not None
                and scoped.logical_calls_by_trial.get(trial_id, 0) >= max_logical_calls
            ):
                raise ProviderBudgetError(
                    "per-Trial logical Provider-call ceiling reached",
                )
            if attempt_ordinal > max_attempts_per_call:
                raise ProviderBudgetError(
                    "per-call Provider retry ceiling reached",
                )
            if (
                attempt_ordinal > 1
                and self.config.max_additional_request_attempts is not None
                and scoped.additional_request_attempts >= self.config.max_additional_request_attempts
            ):
                raise ProviderBudgetError(
                    "campaign additional Provider-attempt ceiling reached",
                )
            committed = state.provider_charged_cny + self.config.external_service_reserve_cny + maximum_request_cny
            if committed > self.config.hard_cap_cny:
                raise ProviderBudgetError(
                    "Provider request would cross the campaign CNY hard cap",
                )
            self._append_event(
                handle,
                {
                    "schema": "codeflow.codeflowbench.provider-budget.v1",
                    "kind": "reserved",
                    "request_id": request_id,
                    "trial_id": trial_id,
                    "request_digest": request_digest,
                    "logical_call_key": logical_call_key,
                    "attempt_ordinal": attempt_ordinal,
                    "is_additional_attempt": attempt_ordinal > 1,
                    "approval_digest": self.config.approval_digest,
                    "model": model,
                    "estimated_input_tokens": estimated_input_tokens,
                    "input_token_counter": _INPUT_COUNT_METHOD,
                    "maximum_input_tokens": input_ceiling,
                    "maximum_output_tokens": output_ceiling,
                    "reserved_cny": maximum_request_cny,
                    "timestamp_ns": time.time_ns(),
                },
            )
        return request_id

    def settle(
        self,
        request_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if input_tokens < 0 or output_tokens < 0:
            self.fail(request_id, reason="usage_token_count_negative")
            raise ProviderBudgetError("Provider usage must not be negative")
        charged = self.config.cost_cny(
            input_tokens,
            output_tokens,
        )
        with self._locked() as handle:
            events = _read_events(handle)
            reservation = _require_open_reservation(
                events,
                request_id,
            )
            violations = []
            if input_tokens > int(
                reservation["maximum_input_tokens"],
            ):
                violations.append("input_token_ceiling_exceeded")
            if output_tokens > int(
                reservation["maximum_output_tokens"],
            ):
                violations.append("output_token_ceiling_exceeded")
            self._append_event(
                handle,
                {
                    "schema": "codeflow.codeflowbench.provider-budget.v1",
                    "kind": "settled",
                    "request_id": request_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "charged_cny": charged,
                    "policy_violations": violations,
                    "timestamp_ns": time.time_ns(),
                },
            )
        if violations:
            raise ProviderBudgetError(
                "Provider usage exceeded the per-call token ceiling",
            )

    def fail(self, request_id: str, *, reason: str) -> None:
        with self._locked() as handle:
            events = _read_events(handle)
            if not _is_open(events, request_id):
                return
            reservation = _reservation_event(events, request_id)
            self._append_event(
                handle,
                {
                    "schema": "codeflow.codeflowbench.provider-budget.v1",
                    "kind": "failed",
                    "request_id": request_id,
                    "reason": reason,
                    "charged_cny": float(
                        reservation["reserved_cny"],
                    ),
                    "timestamp_ns": time.time_ns(),
                },
            )

    def snapshot(self) -> ProviderBudgetSnapshot:
        with self._locked() as handle:
            events = _read_events(handle)
        state = _ledger_state(events)
        scoped = _scoped_reservation_state(
            events,
            baseline=self.config.request_attempt_baseline,
        )
        total = state.provider_charged_cny + self.config.external_service_reserve_cny
        high_water = self._high_water_payload(events)
        return ProviderBudgetSnapshot(
            ledger_path=str(self.path),
            ledger_event_count=len(events),
            request_attempts=state.request_attempts,
            additional_request_attempts=(scoped.additional_request_attempts),
            additional_request_attempt_ceiling=(self.config.max_additional_request_attempts),
            provider_charged_cny=state.provider_charged_cny,
            external_service_reserve_cny=(self.config.external_service_reserve_cny),
            total_committed_cny=total,
            hard_cap_cny=self.config.hard_cap_cny,
            open_reservations=state.open_reservations,
            accounting_complete=(state.open_reservations == 0 and not state.accounting_blocked),
            ledger_digest=canonical_digest(events),
            request_attempt_baseline=(self.config.request_attempt_baseline),
            request_attempt_lifetime_ceiling=(self.config.max_total_request_attempts),
            approval_digest=self.config.approval_digest,
            high_water_path=str(self.high_water_path),
            high_water_digest=canonical_digest(high_water),
        )

    @contextlib.contextmanager
    def _locked(self):
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                events = _read_events(handle)
                self._validate_prefix(events)
                self._validate_or_bootstrap_high_water(events)
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append_event(
        self,
        handle,
        event: dict[str, Any],
    ) -> None:
        _append_event(handle, event)
        self._write_high_water(_read_events(handle))

    def _validate_prefix(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        count = self.config.ledger_prefix_event_count
        digest = self.config.ledger_prefix_digest
        if digest is None:
            return
        if len(events) < count:
            raise ProviderBudgetError(
                "Provider budget ledger was truncated below its approval prefix",
            )
        prefix = events[:count]
        if canonical_digest(prefix) != digest or not math.isclose(
            _ledger_state(prefix).provider_charged_cny,
            self.config.ledger_prefix_charged_cny,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ProviderBudgetError(
                "Provider budget ledger approval prefix does not match",
            )

    def _validate_or_bootstrap_high_water(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        expected = self._high_water_payload(events)
        if self.high_water_path.exists():
            try:
                actual = json.loads(
                    self.high_water_path.read_text(encoding="utf-8"),
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ProviderBudgetError(
                    "Provider budget high-water record is corrupt",
                ) from exc
            if actual != expected:
                event_count = actual.get("event_count")
                if (
                    isinstance(event_count, int)
                    and not isinstance(event_count, bool)
                    and 0 <= event_count < len(events)
                    and actual == self._high_water_payload(events[:event_count])
                ):
                    self._write_high_water(events)
                    return
                raise ProviderBudgetError(
                    "Provider budget ledger does not match its high-water record",
                )
            return
        if self.config.approval_digest is not None:
            raise ProviderBudgetError(
                "Provider budget high-water record is missing",
            )
        can_bootstrap = (
            not events
            or (
                self.config.ledger_prefix_digest is not None
                and len(events) == self.config.ledger_prefix_event_count
                and canonical_digest(events) == self.config.ledger_prefix_digest
            )
            or (self.config.approval_digest is None and self.config.ledger_prefix_digest is None)
        )
        if not can_bootstrap:
            raise ProviderBudgetError(
                "Provider budget high-water record is missing",
            )
        self._write_high_water(events)

    def _high_water_payload(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state = _ledger_state(events)
        return {
            "schema": "codeflow.codeflowbench.provider-budget-high-water.v1",
            "ledger_path": str(self.path),
            "event_count": len(events),
            "ledger_digest": canonical_digest(events),
            "provider_charged_cny": state.provider_charged_cny,
        }

    def _write_high_water(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        payload = self._high_water_payload(events)
        self.high_water_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.high_water_path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                to_primitive(payload),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, self.high_water_path)
            directory_fd = os.open(
                self.high_water_path.parent,
                os.O_RDONLY,
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _LedgerState:
    request_attempts: int
    provider_charged_cny: float
    open_reservations: int
    accounting_blocked: bool


@dataclass(frozen=True)
class _ScopedReservationState:
    attempts_by_logical_call: dict[str, int]
    logical_calls_by_trial: dict[str, int]
    additional_request_attempts: int


class BudgetGuardedProvider(LLMProvider):
    _CHAT_RETRY_DELAYS = (0,)

    def __init__(
        self,
        delegate: LLMProvider,
        *,
        ledger: ProviderBudgetLedger,
    ) -> None:
        super().__init__(
            api_key=getattr(delegate, "api_key", None),
            api_base=getattr(delegate, "api_base", None),
        )
        self._delegate = delegate
        self.ledger = ledger
        self.generation = getattr(
            delegate,
            "generation",
            GenerationSettings(),
        )
        configure_transport_retries = getattr(
            delegate,
            "set_transport_num_retries",
            None,
        )
        if callable(configure_transport_retries):
            configure_transport_retries(0)
        elif delegate.__class__.__module__.startswith("codeflow.providers."):
            raise ProviderBudgetError(
                "Provider does not expose transport retry control",
            )

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
        request_id = self._reserve_before_dispatch(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
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
        except BaseException as exc:
            self.ledger.fail(
                request_id,
                reason=f"provider_exception:{type(exc).__name__}",
            )
            raise
        self._settle_response(request_id, response)
        return response

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
        request_id = self._reserve_before_dispatch(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        final_usage: dict[str, Any] | None = None
        provider_error = False
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
                    final_usage = dict(delta.usage)
                if delta.finish_reason == "error" or delta.error_classification is not None:
                    provider_error = True
                yield delta
        except BaseException as exc:
            self.ledger.fail(
                request_id,
                reason=f"provider_exception:{type(exc).__name__}",
            )
            raise
        self._settle_usage(
            request_id,
            final_usage or {},
            provider_error=provider_error,
        )

    def get_default_model(self) -> str:
        return self._delegate.get_default_model()

    @classmethod
    def classify_error(
        cls,
        exc: BaseException | None = None,
        content: str | None = None,
    ) -> ErrorClassification:
        current = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, ProviderBudgetError) and str(current) in {
                "per-Trial logical Provider-call ceiling reached",
                "estimated input tokens exceed the per-call ceiling",
            }:
                return ErrorClassification("task_budget_exhausted")
            current = current.__cause__ or current.__context__
        return super().classify_error(exc, content)

    def _reserve(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> str:
        state = _TRIAL_STATE.get()
        if state is None:
            raise ProviderBudgetError(
                "paid Provider call occurred outside a budget scope",
            )
        output_ceiling = (
            state.max_output_tokens_per_call
            if state.max_output_tokens_per_call is not None
            else self.ledger.config.max_output_tokens_per_call
        )
        if max_tokens > output_ceiling:
            raise ProviderBudgetError(
                "requested output tokens exceed the per-call ceiling",
            )
        request_digest = canonical_digest(
            {
                "messages": messages,
                "tools": tools or [],
                "model": model or self.get_default_model(),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "tool_choice": tool_choice,
            }
        )
        estimated_input_tokens = _conservative_serialized_input_tokens(
            messages,
            tools,
            model=model or self.get_default_model(),
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        return self.ledger.reserve(
            trial_id=state.trial_id,
            request_digest=request_digest,
            model=model or self.get_default_model(),
            estimated_input_tokens=estimated_input_tokens,
            maximum_input_tokens=(
                state.max_input_tokens_per_call
                if state.max_input_tokens_per_call is not None
                else self.ledger.config.max_input_tokens_per_call
            ),
            maximum_output_tokens=output_ceiling,
            max_logical_calls=state.max_logical_calls,
            max_attempts_per_call=state.max_attempts_per_call,
        )

    def _reserve_before_dispatch(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> str:
        try:
            return self._reserve(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
        except ProviderRequestNotDispatchedError:
            raise
        except Exception as exc:
            raise ProviderRequestNotDispatchedError(str(exc)) from exc

    def _settle_response(
        self,
        request_id: str,
        response: LLMResponse,
    ) -> None:
        self._settle_usage(
            request_id,
            response.usage,
            provider_error=(response.finish_reason == "error" or response.error_classification is not None),
        )

    def _settle_usage(
        self,
        request_id: str,
        usage: dict[str, Any],
        *,
        provider_error: bool,
    ) -> None:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
        ):
            if provider_error:
                self.ledger.fail(
                    request_id,
                    reason="provider_error_usage_missing",
                )
                return
            self.ledger.fail(
                request_id,
                reason="required_usage_fields_missing",
            )
            raise ProviderBudgetError(
                "Provider response is missing required usage accounting",
            )
        if total_tokens != prompt_tokens + completion_tokens:
            if provider_error:
                self.ledger.fail(
                    request_id,
                    reason="provider_error_usage_total_mismatch",
                )
                return
            self.ledger.fail(
                request_id,
                reason="usage_total_mismatch",
            )
            raise ProviderBudgetError(
                "Provider usage accounting is internally inconsistent",
            )
        self.ledger.settle(
            request_id,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )


def _conservative_serialized_input_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
    tool_choice: str | dict[str, Any] | None,
) -> int:
    payload = json.dumps(
        to_primitive(
            {
                "messages": messages,
                "tools": tools or [],
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "tool_choice": tool_choice,
            }
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        encoded = tiktoken.get_encoding("cl100k_base").encode(payload)
    except Exception as exc:
        raise ProviderBudgetError(
            "frozen input tokenizer is unavailable",
        ) from exc
    payload_bytes = len(payload.encode("utf-8"))
    return (
        max(
            1,
            math.ceil(len(encoded) * _INPUT_TOKEN_MARGIN_RATIO),
            payload_bytes,
        )
        + _INPUT_TOKEN_FIXED_MARGIN
    )


def _read_events(handle) -> list[dict[str, Any]]:
    handle.seek(0)
    events = []
    for line in handle:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProviderBudgetError(
                "Provider budget ledger is corrupt",
            ) from exc
        if not isinstance(event, dict):
            raise ProviderBudgetError(
                "Provider budget ledger event is invalid",
            )
        events.append(event)
    return events


def _append_event(handle, event: dict[str, Any]) -> None:
    handle.seek(0, os.SEEK_END)
    handle.write(
        json.dumps(
            to_primitive(event),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def _ledger_state(events: list[dict[str, Any]]) -> _LedgerState:
    reservations: dict[str, float] = {}
    terminal: dict[str, float] = {}
    for event in events:
        request_id = str(event.get("request_id", ""))
        kind = event.get("kind")
        if kind == "reserved":
            if request_id in reservations:
                raise ProviderBudgetError(
                    "duplicate Provider budget reservation",
                )
            reservations[request_id] = float(event["reserved_cny"])
        elif kind in {"settled", "failed"}:
            if request_id not in reservations:
                raise ProviderBudgetError(
                    "Provider budget terminal event has no reservation",
                )
            if request_id in terminal:
                raise ProviderBudgetError(
                    "duplicate Provider budget terminal event",
                )
            terminal[request_id] = float(event["charged_cny"])
        else:
            raise ProviderBudgetError(
                "unknown Provider budget ledger event",
            )
    charged = sum(terminal.get(request_id, reserved) for request_id, reserved in reservations.items())
    accounting_blocked = any(
        (
            event.get("kind") == "failed"
            and event.get("reason")
            in {
                "required_usage_fields_missing",
                "usage_total_mismatch",
                "usage_token_count_negative",
            }
        )
        or (event.get("kind") == "settled" and bool(event.get("policy_violations")))
        for event in events
    )
    return _LedgerState(
        request_attempts=len(reservations),
        provider_charged_cny=charged,
        open_reservations=len(reservations.keys() - terminal.keys()),
        accounting_blocked=accounting_blocked,
    )


def _scoped_reservation_state(
    events: list[dict[str, Any]],
    *,
    baseline: int,
) -> _ScopedReservationState:
    reservations = [event for event in events if event.get("kind") == "reserved"]
    if baseline < 0 or baseline > len(reservations):
        raise ProviderBudgetError(
            "Provider budget approval baseline is outside the ledger",
        )
    attempts_by_logical_call: dict[str, int] = {}
    logical_calls_by_trial: dict[str, set[str]] = {}
    additional = 0
    for event in reservations[baseline:]:
        trial_id = str(event.get("trial_id", ""))
        request_digest = str(event.get("request_digest", ""))
        if not trial_id or not request_digest:
            raise ProviderBudgetError(
                "Provider budget reservation identity is invalid",
            )
        logical_call_key = canonical_digest(
            {
                "trial_id": trial_id,
                "request_digest": request_digest,
            }
        )
        ordinal = attempts_by_logical_call.get(logical_call_key, 0) + 1
        if (
            ("logical_call_key" in event and event.get("logical_call_key") != logical_call_key)
            or ("attempt_ordinal" in event and event.get("attempt_ordinal") != ordinal)
            or ("is_additional_attempt" in event and event.get("is_additional_attempt") is not (ordinal > 1))
        ):
            raise ProviderBudgetError(
                "Provider budget reservation retry identity is invalid",
            )
        attempts_by_logical_call[logical_call_key] = ordinal
        logical_calls_by_trial.setdefault(trial_id, set()).add(
            logical_call_key,
        )
        if ordinal > 1:
            additional += 1
    return _ScopedReservationState(
        attempts_by_logical_call=attempts_by_logical_call,
        logical_calls_by_trial={trial_id: len(keys) for trial_id, keys in logical_calls_by_trial.items()},
        additional_request_attempts=additional,
    )


def _is_open(
    events: list[dict[str, Any]],
    request_id: str,
) -> bool:
    reserved = any(event.get("kind") == "reserved" and event.get("request_id") == request_id for event in events)
    terminal = any(
        event.get("kind") in {"settled", "failed"} and event.get("request_id") == request_id for event in events
    )
    return reserved and not terminal


def _require_open_reservation(
    events: list[dict[str, Any]],
    request_id: str,
) -> dict[str, Any]:
    if not _is_open(events, request_id):
        raise ProviderBudgetError(
            "Provider budget reservation is not open",
        )
    return _reservation_event(events, request_id)


def _reservation_event(
    events: list[dict[str, Any]],
    request_id: str,
) -> dict[str, Any]:
    return next(event for event in events if event.get("kind") == "reserved" and event.get("request_id") == request_id)


__all__ = [
    "BudgetGuardedProvider",
    "ProviderBudgetConfig",
    "ProviderBudgetError",
    "ProviderBudgetLedger",
    "ProviderBudgetSnapshot",
    "ProviderRequestNotDispatchedError",
    "provider_call_budget_scope",
]
