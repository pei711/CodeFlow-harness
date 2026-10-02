"""Tests for the usage_sink populated by AgentLoop and surfaced to the TUI.

Pins the wire shape that ``turn.send`` relays as ``message.complete.payload.usage``:
per-turn token counts plus the live context-window gauge (used / max / percent)
and the estimated cost. Before this, only the token counts were populated, so the
TUI context bar stayed frozen at 0% and never showed cost.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.call_efficiency import CallEfficiency, CallEfficiencyProvider
from codeflow.providers.base import LLMProvider, LLMResponse, StreamDelta
from codeflow.spine.message import ChatType, Source
from codeflow.spine.turn import Origin, TurnRequest
from codeflow.token_wise import pricing
from codeflow.token_wise.base import TokenStrategy
from codeflow.token_wise.registry import StrategyRegistry

_REAL_FETCH = pricing._fetch_openrouter_models


class UsageProvider(LLMProvider):
    """Returns a fixed reply with a known usage snapshot. No tool calls."""

    def __init__(self, model: str, prompt_tokens: int, completion_tokens: int):
        super().__init__(api_key="test")
        self._model = model
        self._usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

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
        return LLMResponse(content="ok", finish_reason="stop", usage=self._usage)

    def get_default_model(self) -> str:
        return self._model


class StreamingUsageProvider(UsageProvider):
    async def chat_stream(self, messages, tools=None, model=None, **kwargs):
        yield StreamDelta(content="ok", model=model or self._model)
        yield StreamDelta(
            content=None,
            model=model or self._model,
            usage=self._usage,
            finish_reason="stop",
        )


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture(autouse=True)
def _reset_openrouter_cache():
    pricing._OPENROUTER_CACHE.clear()
    yield
    pricing._OPENROUTER_CACHE.clear()


def _make_agent(workspace: Path, provider: LLMProvider, model: str, window: int) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=2,
        context_window_tokens=window,
        restrict_to_workspace=True,
    )


class _UsageObserver(TokenStrategy):
    name = "usage_observer"

    def __init__(self) -> None:
        self.snapshots = []

    async def after_llm_call(self, response, usage) -> None:
        self.snapshots.append(usage)


@pytest.mark.asyncio
async def test_active_call_efficiency_records_agent_loop_attempt_and_projects_legacy_usage(
    workspace,
) -> None:
    model = "deepseek/deepseek-v4-flash"
    provider = UsageProvider(model, prompt_tokens=100, completion_tokens=5)
    controller = CallEfficiency(mode="observe", telemetry_dir=workspace, persist=False)
    observer = _UsageObserver()
    agent = AgentLoop(
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=2,
        context_window_tokens=40000,
        restrict_to_workspace=True,
        call_efficiency=controller,
        strategies=StrategyRegistry([observer]),
    )

    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
    )

    assert len(controller.records) == 1
    assert controller.records[0].requested_model == model
    assert controller.records[0].attempted_model == model
    assert controller.records[0].session_key == "s1"
    assert observer.snapshots[0].input_tokens == 100
    assert observer.snapshots[0].estimated_cost_usd == controller.records[0].estimated_cost_usd


@pytest.mark.asyncio
async def test_streaming_runtime_records_exactly_one_provider_attempt(workspace) -> None:
    model = "deepseek/deepseek-v4-flash"
    delegate = StreamingUsageProvider(model, prompt_tokens=100, completion_tokens=5)
    controller = CallEfficiency(mode="observe", telemetry_dir=workspace, persist=False)
    provider = CallEfficiencyProvider(delegate, controller)
    agent = AgentLoop(
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=2,
        context_window_tokens=40000,
        restrict_to_workspace=True,
        call_efficiency=controller,
    )

    streamed: list[str] = []

    async def on_delta(text: str) -> None:
        streamed.append(text)

    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
        on_token_delta=on_delta,
    )

    assert streamed == ["ok"]
    assert len(controller.records) == 1
    assert controller.records[0].actual_model == model


@pytest.mark.asyncio
async def test_usage_sink_omits_unknown_cost_but_keeps_context_gauge(workspace):
    """An unpriced model omits cost without losing its context-window usage."""
    provider = UsageProvider("stub", prompt_tokens=6000, completion_tokens=2000)
    agent = _make_agent(workspace, provider, model="stub", window=40000)
    sink: dict = {}

    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
        usage_sink=sink,
    )

    assert sink["context_max"] == 40000
    assert sink["context_used"] == 8000
    assert sink["context_percent"] == 20
    assert "cost_usd" not in sink


@pytest.mark.asyncio
async def test_usage_sink_keeps_known_zero_cost(workspace):
    model = "deepseek/deepseek-v4-flash"
    provider = UsageProvider(model, prompt_tokens=0, completion_tokens=0)
    agent = _make_agent(workspace, provider, model=model, window=40000)
    sink: dict = {}

    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
        usage_sink=sink,
    )

    assert sink["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_usage_sink_clears_known_cost_when_next_model_is_unpriced(workspace):
    model = "deepseek/deepseek-v4-flash"
    provider = UsageProvider(model, prompt_tokens=100, completion_tokens=50)
    agent = _make_agent(workspace, provider, model=model, window=40000)
    sink: dict = {}

    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="priced", sender_id="user", chat_type=ChatType.DM),
            text="priced",
        ),
        session_key="priced",
        usage_sink=sink,
    )
    assert sink["cost_usd"] > 0

    agent.model = "unpriced/model"
    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="unpriced", sender_id="user", chat_type=ChatType.DM),
            text="unpriced",
        ),
        session_key="unpriced",
        usage_sink=sink,
    )

    assert "cost_usd" not in sink


@pytest.mark.asyncio
async def test_usage_sink_context_max_from_live_openrouter(workspace, monkeypatch):
    """An OpenRouter model LiteLLM lags on gets its real window from /models."""
    models = [
        {
            "id": "deepseek/deepseek-v4-pro",
            "context_length": 163840,
            "pricing": {"prompt": "0.0000005", "completion": "0.0000015"},
        }
    ]

    def handler(_req):
        return httpx.Response(200, content=json.dumps({"data": models}))

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(pricing, "_fetch_openrouter_models", _REAL_FETCH)
    monkeypatch.setattr(pricing, "_ALLOW_NETWORK_CATALOG", True)
    monkeypatch.setattr(pricing.httpx, "Client", client_factory)
    monkeypatch.setattr(pricing, "_OPENROUTER_CACHE_TIME", 0.0)

    provider = UsageProvider("openrouter/deepseek/deepseek-v4-pro", 1000, 500)
    agent = _make_agent(
        workspace,
        provider,
        model="openrouter/deepseek/deepseek-v4-pro",
        window=8192,
    )
    sink: dict = {}

    await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
        usage_sink=sink,
    )

    assert sink["context_max"] == 163840
    assert sink["context_used"] == 1500
