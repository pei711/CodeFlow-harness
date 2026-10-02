from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.codeflowbench.budget import (
    BudgetGuardedProvider,
    ProviderBudgetConfig,
    ProviderBudgetError,
    ProviderBudgetLedger,
    provider_call_budget_scope,
)
from benchmarks.codeflowbench.canonical import canonical_digest
from codeflow.providers.base import (
    ErrorClassification,
    LLMProvider,
    LLMResponse,
)
from codeflow.providers.litellm_provider import LiteLLMProvider


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls = 0

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        del messages, tools, model, max_tokens, temperature
        del reasoning_effort, tool_choice
        self.calls += 1
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return "test/budget"


def _config(
    *,
    max_input_tokens_per_call: int = 16_000,
    max_output_tokens_per_call: int = 1_500,
    max_additional_request_attempts: int | None = None,
) -> ProviderBudgetConfig:
    return ProviderBudgetConfig(
        hard_cap_cny=100,
        external_service_reserve_cny=5,
        max_total_request_attempts=8,
        max_input_tokens_per_call=max_input_tokens_per_call,
        max_output_tokens_per_call=max_output_tokens_per_call,
        input_cache_miss_usd_per_million=0.14,
        output_usd_per_million=0.28,
        conservative_usd_to_cny_multiplier=7.5,
        max_additional_request_attempts=(max_additional_request_attempts),
    )


def _ledger(
    tmp_path,
    *,
    max_input_tokens_per_call: int = 16_000,
    max_output_tokens_per_call: int = 1_500,
    max_additional_request_attempts: int | None = None,
) -> ProviderBudgetLedger:
    return ProviderBudgetLedger(
        tmp_path / "provider-budget.jsonl",
        _config(
            max_input_tokens_per_call=max_input_tokens_per_call,
            max_output_tokens_per_call=max_output_tokens_per_call,
            max_additional_request_attempts=(max_additional_request_attempts),
        ),
    )


def _success_usage(
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 2,
) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def test_budget_guard_classifies_local_ceiling_as_task_budget() -> None:
    classification = BudgetGuardedProvider.classify_error(
        ProviderBudgetError(
            "per-Trial logical Provider-call ceiling reached",
        ),
    )

    assert classification == ErrorClassification(
        "task_budget_exhausted",
    )
    assert (
        BudgetGuardedProvider.classify_error(
            ProviderBudgetError(
                "Provider budget accounting is incomplete",
            ),
        ).category
        == "unknown"
    )


def test_settlement_charges_observed_usage_before_blocking_on_ceiling(
    tmp_path,
) -> None:
    ledger = _ledger(
        tmp_path,
        max_input_tokens_per_call=10,
        max_output_tokens_per_call=5,
    )
    request_id = ledger.reserve(
        trial_id="over-ceiling",
        request_digest="a" * 64,
        model="test/budget",
        estimated_input_tokens=10,
    )

    with pytest.raises(ProviderBudgetError, match="usage exceeded"):
        ledger.settle(
            request_id,
            input_tokens=20,
            output_tokens=7,
        )

    snapshot = ledger.snapshot()
    assert snapshot.request_attempts == 1
    assert snapshot.provider_charged_cny == pytest.approx(
        ledger.config.cost_cny(20, 7),
    )
    assert snapshot.open_reservations == 0
    with pytest.raises(ProviderBudgetError, match="accounting is incomplete"):
        ledger.reserve(
            trial_id="blocked",
            request_digest="b" * 64,
            model="test/budget",
            estimated_input_tokens=1,
        )


@pytest.mark.asyncio
async def test_transient_error_without_usage_is_charged_and_outer_retry_is_ledgered(
    tmp_path,
) -> None:
    transient = LLMResponse(
        content="temporary outage",
        finish_reason="error",
        error_classification=ErrorClassification(
            "server",
            retryable=True,
        ),
    )
    delegate = _ScriptedProvider(
        [
            transient,
            LLMResponse(
                content="ok",
                usage=_success_usage(),
            ),
        ],
    )
    ledger = _ledger(tmp_path)
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)

    with provider_call_budget_scope(
        trial_id="retry",
        max_logical_calls=1,
        max_attempts_per_call=2,
    ):
        response = await guarded.chat_with_retry(
            [{"role": "user", "content": "retry me"}],
            max_tokens=10,
        )

    assert response.content == "ok"
    assert delegate.calls == 2
    snapshot = ledger.snapshot()
    assert snapshot.request_attempts == 2
    assert snapshot.provider_charged_cny == pytest.approx(
        ledger.config.maximum_request_cny + ledger.config.cost_cny(10, 2),
    )
    assert snapshot.accounting_complete is True


@pytest.mark.asyncio
async def test_per_call_retry_limit_survives_fresh_budget_scopes(
    tmp_path,
) -> None:
    delegate = _ScriptedProvider(
        [
            LLMResponse(content="first", usage=_success_usage()),
            LLMResponse(content="not reached", usage=_success_usage()),
        ],
    )
    ledger = _ledger(tmp_path)
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)
    messages = [{"role": "user", "content": "same request"}]

    with provider_call_budget_scope(
        trial_id="durable-retry",
        max_logical_calls=1,
        max_attempts_per_call=1,
    ):
        await guarded.chat(messages, max_tokens=10)
    with provider_call_budget_scope(
        trial_id="durable-retry",
        max_logical_calls=1,
        max_attempts_per_call=1,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="per-call Provider retry ceiling",
        ):
            await guarded.chat(messages, max_tokens=10)

    assert delegate.calls == 1
    assert ledger.snapshot().additional_request_attempts == 0


@pytest.mark.asyncio
async def test_global_additional_attempt_limit_survives_trial_scopes(
    tmp_path,
) -> None:
    delegate = _ScriptedProvider(
        [
            LLMResponse(content="a1", usage=_success_usage()),
            LLMResponse(content="a2", usage=_success_usage()),
            LLMResponse(content="b1", usage=_success_usage()),
            LLMResponse(content="not reached", usage=_success_usage()),
        ],
    )
    ledger = _ledger(
        tmp_path,
        max_additional_request_attempts=1,
    )
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)

    for _ in range(2):
        with provider_call_budget_scope(
            trial_id="trial-a",
            max_logical_calls=1,
            max_attempts_per_call=2,
        ):
            await guarded.chat(
                [{"role": "user", "content": "request a"}],
                max_tokens=10,
            )
    with provider_call_budget_scope(
        trial_id="trial-b",
        max_logical_calls=1,
        max_attempts_per_call=2,
    ):
        await guarded.chat(
            [{"role": "user", "content": "request b"}],
            max_tokens=10,
        )
    with provider_call_budget_scope(
        trial_id="trial-b",
        max_logical_calls=1,
        max_attempts_per_call=2,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="additional Provider-attempt ceiling",
        ):
            await guarded.chat(
                [{"role": "user", "content": "request b"}],
                max_tokens=10,
            )

    assert delegate.calls == 3
    snapshot = ledger.snapshot()
    assert snapshot.additional_request_attempts == 1
    assert snapshot.additional_request_attempt_ceiling == 1


def test_ledger_truncation_is_rejected_by_high_water_record(
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)
    request_id = ledger.reserve(
        trial_id="paid-call",
        request_digest="f" * 64,
        model="test/budget",
        estimated_input_tokens=100,
    )
    ledger.settle(
        request_id,
        input_tokens=100,
        output_tokens=10,
    )
    assert ledger.snapshot().ledger_event_count == 2

    ledger.path.write_text("", encoding="utf-8")

    with pytest.raises(
        ProviderBudgetError,
        match="high-water",
    ):
        ledger.snapshot()


def test_approved_ledger_and_high_water_rollback_is_rejected(
    tmp_path,
) -> None:
    path = tmp_path / "provider-budget.jsonl"
    ProviderBudgetLedger(path, _config()).snapshot()
    config = ProviderBudgetConfig(
        **{
            **_config().__dict__,
            "approval_digest": "a" * 64,
            "ledger_prefix_event_count": 0,
            "ledger_prefix_digest": canonical_digest([]),
        }
    )
    ledger = ProviderBudgetLedger(path, config)
    request_id = ledger.reserve(
        trial_id="paid-call",
        request_digest="f" * 64,
        model="test/budget",
        estimated_input_tokens=100,
    )
    ledger.settle(
        request_id,
        input_tokens=100,
        output_tokens=10,
    )
    assert ledger.snapshot().request_attempts == 1

    path.write_text("", encoding="utf-8")
    ledger.high_water_path.unlink()

    with pytest.raises(
        ProviderBudgetError,
        match="high-water record is missing",
    ):
        ProviderBudgetLedger(path, config).snapshot()


def test_ledger_repairs_high_water_after_durable_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.snapshot().ledger_event_count == 0
    original = ledger._write_high_water

    def interrupt_after_ledger_append(events) -> None:
        del events
        raise RuntimeError("simulated high-water interruption")

    monkeypatch.setattr(
        ledger,
        "_write_high_water",
        interrupt_after_ledger_append,
    )
    with pytest.raises(
        RuntimeError,
        match="high-water interruption",
    ):
        ledger.reserve(
            trial_id="paid-call",
            request_digest="f" * 64,
            model="test/budget",
            estimated_input_tokens=100,
        )

    monkeypatch.setattr(ledger, "_write_high_water", original)
    snapshot = ledger.snapshot()

    assert snapshot.ledger_event_count == 1
    assert snapshot.request_attempts == 1
    assert snapshot.open_reservations == 1
    assert snapshot.accounting_complete is False


@pytest.mark.asyncio
async def test_serialized_request_ceiling_blocks_before_delegate_transport(
    tmp_path,
) -> None:
    delegate = _ScriptedProvider(
        [
            LLMResponse(
                content="not reached",
                usage=_success_usage(),
            ),
        ],
    )
    ledger = _ledger(
        tmp_path,
        max_input_tokens_per_call=50,
    )
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)
    empty_messages = [{"role": "user", "content": ""} for _ in range(100)]

    with provider_call_budget_scope(
        trial_id="serialized-ceiling",
        max_logical_calls=1,
        max_attempts_per_call=1,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="estimated input tokens",
        ):
            await guarded.chat(
                empty_messages,
                max_tokens=10,
            )

    assert delegate.calls == 0
    assert ledger.snapshot().request_attempts == 0


@pytest.mark.asyncio
async def test_trial_scope_input_ceiling_is_stricter_than_campaign_ceiling(
    tmp_path,
) -> None:
    delegate = _ScriptedProvider(
        [
            LLMResponse(
                content="not reached",
                usage=_success_usage(),
            ),
        ],
    )
    ledger = _ledger(
        tmp_path,
        max_input_tokens_per_call=42_000,
    )
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)

    with provider_call_budget_scope(
        trial_id="memory-input-ceiling",
        max_logical_calls=4,
        max_attempts_per_call=1,
        max_input_tokens_per_call=15_000,
        max_output_tokens_per_call=1_500,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="estimated input tokens",
        ):
            await guarded.chat(
                [{"role": "user", "content": "x" * 20_000}],
                max_tokens=10,
            )

    assert delegate.calls == 0
    assert ledger.snapshot().request_attempts == 0


def test_failed_request_charges_its_trial_scope_reservation(
    tmp_path,
) -> None:
    ledger = _ledger(
        tmp_path,
        max_input_tokens_per_call=42_000,
    )
    request_id = ledger.reserve(
        trial_id="memory-failed-request",
        request_digest="c" * 64,
        model="test/budget",
        estimated_input_tokens=100,
        maximum_input_tokens=15_000,
        maximum_output_tokens=1_500,
    )

    ledger.fail(request_id, reason="provider_exception:RuntimeError")

    assert ledger.snapshot().provider_charged_cny == pytest.approx(
        ledger.config.cost_cny(15_000, 1_500),
    )


@pytest.mark.asyncio
async def test_trial_scope_output_ceiling_blocks_before_delegate_transport(
    tmp_path,
) -> None:
    delegate = _ScriptedProvider(
        [
            LLMResponse(
                content="not reached",
                usage=_success_usage(),
            ),
        ],
    )
    guarded = BudgetGuardedProvider(
        delegate,
        ledger=_ledger(
            tmp_path,
            max_input_tokens_per_call=42_000,
        ),
    )

    with provider_call_budget_scope(
        trial_id="context-output-ceiling",
        max_logical_calls=8,
        max_attempts_per_call=1,
        max_input_tokens_per_call=42_000,
        max_output_tokens_per_call=1_200,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="requested output tokens",
        ):
            await guarded.chat(
                [{"role": "user", "content": "small"}],
                max_tokens=1_201,
            )

    assert delegate.calls == 0


@pytest.mark.asyncio
async def test_utf8_byte_bound_blocks_cjk_payload_before_transport(
    tmp_path,
) -> None:
    delegate = _ScriptedProvider(
        [
            LLMResponse(
                content="not reached",
                usage=_success_usage(),
            ),
        ],
    )
    ledger = _ledger(
        tmp_path,
        max_input_tokens_per_call=250,
    )
    guarded = BudgetGuardedProvider(delegate, ledger=ledger)

    with provider_call_budget_scope(
        trial_id="cjk-ceiling",
        max_logical_calls=1,
        max_attempts_per_call=1,
    ):
        with pytest.raises(
            ProviderBudgetError,
            match="estimated input tokens",
        ):
            await guarded.chat(
                [{"role": "user", "content": "的" * 100}],
                max_tokens=10,
            )

    assert delegate.calls == 0
    assert ledger.snapshot().request_attempts == 0


def _litellm_response() -> SimpleNamespace:
    message = SimpleNamespace(
        content="ok",
        tool_calls=[],
        reasoning_content=None,
        thinking_blocks=None,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason="stop",
    )
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        prompt_tokens_details=None,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
    )


@pytest.mark.asyncio
async def test_budget_guard_forces_litellm_and_openai_transport_retries_to_zero(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    async def fake_acompletion(**kwargs):
        captured.append(dict(kwargs))
        return _litellm_response()

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )
    delegate = LiteLLMProvider(
        default_model="deepseek/deepseek-v4-flash",
        provider_name="deepseek",
    )
    guarded = BudgetGuardedProvider(
        delegate,
        ledger=_ledger(tmp_path),
    )

    with provider_call_budget_scope(
        trial_id="transport-retries",
        max_logical_calls=1,
        max_attempts_per_call=1,
    ):
        await guarded.chat(
            [{"role": "user", "content": "hello"}],
            max_tokens=10,
        )

    assert len(captured) == 1
    assert captured[0]["num_retries"] == 0


@pytest.mark.asyncio
async def test_litellm_runtime_default_does_not_override_transport_retries(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    async def fake_acompletion(**kwargs):
        captured.append(dict(kwargs))
        return _litellm_response()

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )
    provider = LiteLLMProvider(
        default_model="deepseek/deepseek-v4-flash",
        provider_name="deepseek",
    )

    await provider.chat(
        [{"role": "user", "content": "hello"}],
        max_tokens=10,
    )

    assert len(captured) == 1
    assert "num_retries" not in captured[0]
