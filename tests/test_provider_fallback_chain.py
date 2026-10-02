"""Fallback-chain behavior for ``LLMProvider.chat_with_retry``.

Covers:
- no fallbacks → single-model retry behavior is unchanged
- exhausted transient error on primary → switches to next model
- fallback-worthy fatal error (billing/availability) → switches model
- non-fallback fatal error (invalid request / context length) → no switch
- a later model succeeding stops the chain
- chain exhausted → last error surfaces
"""

from __future__ import annotations

import pytest

from codeflow.providers.base import LLMProvider, LLMResponse


class _ScriptedProvider(LLMProvider):
    """Provider returning queued responses keyed by model.

    ``script`` maps a model id (or None) to a list of LLMResponse to return
    on successive calls for that model. Records the order models were called.
    """

    def __init__(self, script: dict[str | None, list[LLMResponse]]):
        super().__init__(api_key="test")
        self._script = script
        self.calls: list[str | None] = []

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ):
        self.calls.append(model)
        queue = self._script.get(model, [])
        if queue:
            return queue.pop(0)
        return LLMResponse(content="ok", finish_reason="stop")

    def get_default_model(self) -> str:
        return "default-model"


@pytest.mark.asyncio
async def test_no_fallbacks_preserves_single_model_behavior():
    provider = _ScriptedProvider({"m1": [LLMResponse(content="hello", finish_reason="stop")]})
    resp = await provider.chat_with_retry(messages=[], model="m1")
    assert resp.content == "hello"
    assert provider.calls == ["m1"]


@pytest.mark.asyncio
async def test_exhausted_transient_falls_back_to_next_model():
    transient = LLMResponse(content="429 rate limit", finish_reason="error")
    provider = _ScriptedProvider(
        {
            "primary": [transient] * 4,
            "backup": [LLMResponse(content="recovered", finish_reason="stop")],
        }
    )

    provider._CHAT_RETRY_DELAYS = (0, 0, 0)
    resp = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
    )
    assert resp.content == "recovered"
    assert provider.calls == ["primary"] * 4 + ["backup"]


@pytest.mark.asyncio
async def test_billing_error_now_falls_back():

    provider = _ScriptedProvider(
        {
            "primary": [LLMResponse(content="insufficient credit / billing", finish_reason="error")],
            "backup": [LLMResponse(content="ok", finish_reason="stop")],
        }
    )
    resp = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
    )
    assert resp.content == "ok"

    assert provider.calls == ["primary", "backup"]


@pytest.mark.asyncio
async def test_auth_error_does_not_fall_back():

    provider = _ScriptedProvider(
        {
            "primary": [LLMResponse(content="401 unauthorized: invalid api key", finish_reason="error")],
            "backup": [LLMResponse(content="ok", finish_reason="stop")],
        }
    )
    resp = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
    )
    assert resp.finish_reason == "error"
    assert provider.calls == ["primary"]


@pytest.mark.asyncio
async def test_invalid_request_does_not_fall_back():
    provider = _ScriptedProvider(
        {
            "primary": [LLMResponse(content="400 invalid request: bad schema", finish_reason="error")],
            "backup": [LLMResponse(content="ok", finish_reason="stop")],
        }
    )
    resp = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
    )
    assert resp.finish_reason == "error"
    assert provider.calls == ["primary"]


@pytest.mark.asyncio
async def test_context_length_overflow_does_not_fall_back():
    provider = _ScriptedProvider(
        {
            "primary": [
                LLMResponse(
                    content="This model's maximum context length is 8192 tokens",
                    finish_reason="error",
                )
            ],
            "backup": [LLMResponse(content="ok", finish_reason="stop")],
        }
    )
    resp = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
    )
    assert resp.finish_reason == "error"
    assert provider.calls == ["primary"]


@pytest.mark.asyncio
async def test_chain_exhausted_returns_last_error():
    err = LLMResponse(content="503 overloaded", finish_reason="error")
    provider = _ScriptedProvider(
        {
            "primary": [err] * 4,
            "backup": [err] * 4,
        }
    )
    provider._CHAT_RETRY_DELAYS = (0, 0, 0)
    resp = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
    )
    assert resp.finish_reason == "error"

    assert provider.calls == ["primary"] * 4 + ["backup"] * 4


@pytest.mark.asyncio
async def test_attempt_hooks_replan_and_observe_every_retry_and_fallback():
    transient = LLMResponse(content="503 overloaded", finish_reason="error")
    provider = _ScriptedProvider(
        {
            "prepared-primary": [transient, transient],
            "prepared-backup": [LLMResponse(content="recovered", finish_reason="stop")],
        }
    )
    provider._CHAT_RETRY_DELAYS = (0,)
    prepared: list[str | None] = []
    observed: list[tuple[str | None, str | None]] = []

    def transform(messages, tools, model):
        prepared.append(model)
        return messages, tools, f"prepared-{model}"

    async def observe(response, model):
        observed.append((model, response.model))

    response = await provider.chat_with_retry(
        messages=[],
        model="primary",
        fallback_models=["backup"],
        request_transform=transform,
        response_observer=observe,
    )

    assert response.content == "recovered"
    assert prepared == ["primary", "backup"]
    assert observed == [
        ("prepared-primary", "prepared-primary"),
        ("prepared-primary", "prepared-primary"),
        ("prepared-backup", "prepared-backup"),
    ]


@pytest.mark.asyncio
async def test_should_fallback_classification():

    assert LLMProvider._should_fallback("429 rate limit") is True
    assert LLMProvider._should_fallback("503 overloaded") is True
    assert LLMProvider._should_fallback("connection reset") is True
    assert LLMProvider._should_fallback("insufficient credit / billing") is True
    assert LLMProvider._should_fallback("model not found") is True
    assert LLMProvider._should_fallback("invalid request") is False
    assert LLMProvider._should_fallback("401 unauthorized") is False
    assert LLMProvider._should_fallback("maximum context length exceeded") is False
    assert LLMProvider._should_fallback("ok") is False
    assert LLMProvider._should_fallback(None) is False
