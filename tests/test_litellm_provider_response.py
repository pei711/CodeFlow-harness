from types import SimpleNamespace

import pytest

from codeflow.providers.litellm_provider import LiteLLMProvider


def test_provider_does_not_own_cache_control_by_default() -> None:
    provider = LiteLLMProvider(api_key="test", default_model="anthropic/claude-sonnet-4-5")

    assert provider.disable_auto_cache_control is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("disable_auto", "expected_markers"), [(True, 0), (False, 2)])
async def test_chat_payload_respects_cache_control_owner(
    monkeypatch: pytest.MonkeyPatch,
    disable_auto: bool,
    expected_markers: int,
) -> None:
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model="anthropic/claude-sonnet-4-5",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=[],
                        reasoning_content=None,
                        thinking_blocks=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                prompt_tokens_details=None,
            ),
        )

    monkeypatch.setattr("codeflow.providers.litellm_provider.acompletion", fake_acompletion)
    provider = LiteLLMProvider(
        api_key="test",
        default_model="anthropic/claude-sonnet-4-5",
        disable_auto_cache_control=disable_auto,
    )
    tools = [{"type": "function", "function": {"name": "read"}}]

    await provider.chat(
        messages=[{"role": "system", "content": "stable"}],
        tools=tools,
    )

    payload = {"messages": captured["messages"], "tools": captured["tools"]}
    assert _cache_markers(payload) == expected_markers


def _cache_markers(value) -> int:
    if isinstance(value, dict):
        return int("cache_control" in value) + sum(_cache_markers(item) for item in value.values())
    if isinstance(value, list):
        return sum(_cache_markers(item) for item in value)
    return 0


def test_parse_response_preserves_actual_model_identity() -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    response = SimpleNamespace(
        model="deepseek-v4-flash",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ready",
                    tool_calls=[],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=None,
        ),
    )

    parsed = provider._parse_response(response)

    assert parsed.model == "deepseek-v4-flash"


def test_parse_response_normalizes_deepseek_cache_usage() -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    response = SimpleNamespace(
        model="deepseek-v4-flash",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ready",
                    tool_calls=[],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=4090,
            completion_tokens=8,
            total_tokens=4098,
            prompt_cache_hit_tokens=3968,
            prompt_cache_miss_tokens=122,
            prompt_tokens_details=None,
        ),
    )

    parsed = provider._parse_response(response)

    assert parsed.usage == {
        "prompt_tokens": 4090,
        "completion_tokens": 8,
        "total_tokens": 4098,
        "cache_read_input_tokens": 3968,
        "cache_miss_input_tokens": 122,
    }


def test_parse_response_preserves_explicit_zero_cache_counts() -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    response = SimpleNamespace(
        model="anthropic/claude-sonnet-4-5",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="ready",
                    tool_calls=[],
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            cache_read_input_tokens=0,
            _cache_read_input_tokens=9,
            cache_creation_input_tokens=0,
            _cache_creation_input_tokens=7,
            prompt_tokens_details=None,
        ),
    )

    parsed = provider._parse_response(response)

    assert parsed.usage["cache_read_input_tokens"] == 0
    assert parsed.usage["cache_creation_input_tokens"] == 0
