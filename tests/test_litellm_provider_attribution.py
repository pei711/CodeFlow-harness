"""Tests for OpenRouter app attribution headers injected by LiteLLMProvider."""

from __future__ import annotations

from unittest.mock import patch

from codeflow.providers.litellm_provider import _ANTHROPIC_EXTRA_KEYS, LiteLLMProvider


def _make_provider(provider_name: str, extra_headers: dict | None = None) -> LiteLLMProvider:
    with (
        patch("codeflow.providers.litellm_provider.litellm"),
        patch("codeflow.providers.litellm_provider.LiteLLMProvider._setup_env"),
    ):
        return LiteLLMProvider(
            api_key="sk-test",
            provider_name=provider_name,
            extra_headers=extra_headers,
        )


def test_openrouter_injects_all_attribution_headers():
    provider = _make_provider("openrouter")
    assert provider.extra_headers["HTTP-Referer"] == "https://github.com/pei711/codeflow-harness"
    assert provider.extra_headers["X-Title"] == "CodeFlow Agent Harness"
    assert provider.extra_headers["X-OpenRouter-Title"] == "CodeFlow Agent Harness"
    assert provider.extra_headers["X-OpenRouter-Categories"] == "cli-agent,personal-agent"


def test_openrouter_user_headers_override_defaults():
    provider = _make_provider("openrouter", extra_headers={"X-OpenRouter-Title": "Custom"})
    assert provider.extra_headers["X-OpenRouter-Title"] == "Custom"
    assert provider.extra_headers["HTTP-Referer"] == "https://github.com/pei711/codeflow-harness"


def test_non_openrouter_provider_has_no_attribution():
    provider = _make_provider("anthropic")
    assert "X-OpenRouter-Title" not in provider.extra_headers
    assert "HTTP-Referer" not in provider.extra_headers
    assert "X-OpenRouter-Categories" not in provider.extra_headers


def test_extra_msg_keys_anthropic_spec_preserves_thinking_blocks():
    keys = LiteLLMProvider._extra_msg_keys("anthropic/claude-opus-4-5", "anthropic/claude-opus-4-5")
    assert keys == _ANTHROPIC_EXTRA_KEYS
    assert keys == frozenset({"thinking_blocks"})


def test_extra_msg_keys_matches_on_claude_in_original_model():
    assert LiteLLMProvider._extra_msg_keys("claude-3", "openai/claude-3") == _ANTHROPIC_EXTRA_KEYS


def test_extra_msg_keys_matches_on_resolved_anthropic_prefix():
    assert LiteLLMProvider._extra_msg_keys("some-alias", "anthropic/foo") == _ANTHROPIC_EXTRA_KEYS


def test_extra_msg_keys_non_anthropic_preserves_nothing():
    assert LiteLLMProvider._extra_msg_keys("gpt-4o", "gpt-4o") == frozenset()


def test_anthropic_sanitizer_preserves_signed_thinking_blocks():
    thinking_blocks = [
        {"type": "thinking", "thinking": "Inspect the repository.", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]

    messages = LiteLLMProvider._sanitize_messages(
        [{"role": "assistant", "content": None, "thinking_blocks": thinking_blocks}],
        extra_keys=_ANTHROPIC_EXTRA_KEYS,
    )

    assert messages[0]["thinking_blocks"] == thinking_blocks
