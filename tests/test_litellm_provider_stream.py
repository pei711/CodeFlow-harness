"""Streaming tests for `LiteLLMProvider.chat_stream`.

Covers:
- happy-path: chat_stream yields StreamDelta sequence matching mock chunks
- _normalize_stream_chunk default OpenAI shape extraction
- None-content chunks (e.g. final stop chunk) are skipped (return None → no yield)
- signature parity with chat() (messages/tools/model/max_tokens/temperature/
  reasoning_effort/tool_choice all accepted; stream=True forwarded to acompletion)

Mocks patch `codeflow.providers.litellm_provider.acompletion` because the
provider module imports `from litellm import acompletion` at top level, so
patching `litellm.acompletion` after import would not be picked up.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from litellm.types.utils import Usage

from codeflow.call_efficiency import CallEfficiency
from codeflow.providers.base import LLMResponse, StreamDelta
from codeflow.providers.litellm_provider import LiteLLMProvider


@dataclass
class _FakeDelta:
    content: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta
    finish_reason: str | None = None
    index: int = 0


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice]
    usage: Any | None = None
    model: str | None = None


def _chunk(content: str | None) -> _FakeChunk:
    return _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=content))])


async def _fake_stream(chunks: list[_FakeChunk]):
    """Async generator standing in for litellm's streamed response."""
    for ch in chunks:
        yield ch


def _make_provider() -> LiteLLMProvider:

    return LiteLLMProvider(api_key="test-key", default_model="openai/gpt-4o")


@pytest.mark.asyncio
async def test_chat_stream_yields_stream_deltas_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat_stream yields StreamDelta sequence matching mock OpenAI-shape chunks."""
    chunks = [_chunk("Hel"), _chunk("lo"), _chunk(" world")]

    captured_kwargs: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured_kwargs.update(kwargs)
        return _fake_stream(chunks)

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = _make_provider()
    out: list[StreamDelta] = []
    async for delta in provider.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/gpt-4o",
    ):
        out.append(delta)

    assert [d.content for d in out] == ["Hel", "lo", " world"]
    assert all(isinstance(d, StreamDelta) for d in out)

    assert captured_kwargs.get("stream") is True

    assert captured_kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_chat_stream_default_payload_has_no_provider_cache_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr("codeflow.providers.litellm_provider.acompletion", fake_acompletion)
    provider = LiteLLMProvider(api_key="test", default_model="anthropic/claude-sonnet-4-5")
    messages = [{"role": "system", "content": "stable"}]
    tools = [{"type": "function", "function": {"name": "read"}}]

    _ = [delta async for delta in provider.chat_stream(messages=messages, tools=tools)]

    assert "cache_control" not in str({"messages": captured["messages"], "tools": captured["tools"]})


def test_normalize_stream_chunk_openai_shape_default() -> None:
    """_normalize_stream_chunk default path extracts OpenAI-shape content."""
    provider = _make_provider()
    chunk = _chunk("token")
    delta = provider._normalize_stream_chunk(chunk)
    assert delta is not None
    assert delta.content == "token"
    assert delta.tool_call_delta is None
    assert delta.usage is None


def test_normalize_stream_chunk_returns_none_for_empty_payload() -> None:
    """Chunks with no content/tool_calls/usage return None — chat_stream skips them."""
    provider = _make_provider()

    chunk = _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=None), finish_reason="stop")])
    assert provider._normalize_stream_chunk(chunk) is None


def test_normalize_stream_chunk_preserves_usage_only_terminal_chunk() -> None:
    provider = _make_provider()
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    chunk = _FakeChunk(choices=[], usage=usage, model="openai/gpt-4o")

    delta = provider._normalize_stream_chunk(chunk)

    assert delta is not None
    assert delta.content is None
    assert delta.model == "openai/gpt-4o"
    assert delta.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }


@pytest.mark.parametrize(
    ("model", "usage", "expected_fresh", "expected_read", "expected_write"),
    [
        (
            "deepseek/deepseek-v4-flash",
            Usage(
                prompt_tokens=100,
                completion_tokens=5,
                total_tokens=105,
                prompt_cache_hit_tokens=80,
                prompt_cache_miss_tokens=20,
            ),
            20,
            80,
            0,
        ),
        (
            "openai/gpt-4o",
            Usage(
                prompt_tokens=100,
                completion_tokens=5,
                total_tokens=105,
                prompt_tokens_details={"cached_tokens": 80},
            ),
            20,
            80,
            0,
        ),
        (
            "anthropic/claude-sonnet-4-5",
            Usage(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=115,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=10,
            ),
            20,
            80,
            10,
        ),
    ],
)
def test_stream_terminal_usage_uses_canonical_cache_fields(
    tmp_path,
    model: str,
    usage: Usage,
    expected_fresh: int,
    expected_read: int,
    expected_write: int,
) -> None:
    provider = _make_provider()
    delta = provider._normalize_stream_chunk(_FakeChunk(choices=[], usage=usage, model=model))

    assert delta is not None
    controller = CallEfficiency(mode="observe", telemetry_dir=tmp_path, persist=False)
    record = controller.record(
        LLMResponse(content="ok", usage=delta.usage or {}, model=model),
        requested_model=model,
        session_key="session-1",
    )

    assert record.usage.input_tokens == expected_fresh
    assert record.usage.cache_read_tokens == expected_read
    assert record.usage.cache_write_tokens == expected_write
    assert record.usage.complete is True


@pytest.mark.asyncio
async def test_chat_stream_skips_none_content_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mixed sequence with a None-content chunk: normalizer returns None → no yield."""
    chunks = [
        _chunk("a"),
        _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content=None), finish_reason=None)]),
        _chunk("b"),
    ]

    async def fake_acompletion(**_kwargs: Any):
        return _fake_stream(chunks)

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = _make_provider()
    out = [d async for d in provider.chat_stream(messages=[{"role": "user", "content": "hi"}])]

    assert [d.content for d in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_chat_stream_signature_parity_with_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat_stream accepts every chat() kwarg without raising.

    Smoke check: pass the full chat() parameter set and verify kwargs hit
    acompletion (stream=True, model/messages/tools/tool_choice present;
    reasoning_effort forwarded; max_tokens/temperature forwarded).
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = _make_provider()
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    out: list[StreamDelta] = []
    async for delta in provider.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        model="openai/gpt-4o-mini",
        max_tokens=128,
        temperature=0.3,
        reasoning_effort="medium",
        tool_choice="auto",
    ):
        out.append(delta)

    assert [d.content for d in out] == ["ok"]
    assert captured["stream"] is True
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 0.3
    assert captured["reasoning_effort"] == "medium"
    assert captured["tool_choice"] == "auto"
    assert captured["tools"] == tools

    assert "gpt-4o-mini" in captured["model"]


@pytest.mark.asyncio
async def test_deepseek_replays_reasoning_field_for_assistant_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = LiteLLMProvider(api_key="test-key", default_model="deepseek/deepseek-v4-flash")
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call12345",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "content": "contents", "tool_call_id": "call12345"},
    ]

    _ = [delta async for delta in provider.chat_stream(messages=messages)]

    assert captured["messages"][1]["reasoning_content"] == ""


@pytest.mark.asyncio
async def test_openai_does_not_receive_deepseek_reasoning_replay_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any):
        captured.update(kwargs)
        return _fake_stream([_chunk("ok")])

    monkeypatch.setattr(
        "codeflow.providers.litellm_provider.acompletion",
        fake_acompletion,
    )

    provider = LiteLLMProvider(api_key="test-key", default_model="openai/gpt-4o")
    messages = [
        {"role": "user", "content": "inspect the repository"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call12345",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
    ]

    _ = [delta async for delta in provider.chat_stream(messages=messages)]

    assert "reasoning_content" not in captured["messages"][1]
