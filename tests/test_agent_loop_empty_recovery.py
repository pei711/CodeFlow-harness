"""Empty / thinking-only response recovery.

A turn that ends with no visible text is usually a weak-model dud, not a real
"done". The loop recovers the turn (bounded per turn) before falling back to the
canned reply, and the synthetic scaffolding never reaches persisted history.

Decision logic lives in ``codeflow.agent.loop.recovery`` as a pure function and
is unit-tested in isolation; the loop tests cover the side effects (re-feeding
reasoning, injecting nudges, stripping synthetic messages).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.agent.loop.recovery import (
    RecoveryAction,
    RecoveryLimits,
    classify_empty_response,
    has_inline_thinking,
    has_thinking,
    limits_from_defaults,
)
from codeflow.providers.base import LLMProvider, LLMResponse
from codeflow.spine.message import ChatType, Source
from codeflow.spine.turn import Origin, TurnRequest


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _make_agent(workspace: Path, provider: LLMProvider, limits: RecoveryLimits | None = None) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        workspace=workspace,
        model="stub",
        max_iterations=10,
        restrict_to_workspace=True,
        empty_recovery=limits,
    )


def _classify(
    response, visible, *, prev_had_tool_calls=False, nudges_done=0, prefill_retries=0, empty_retries=0, limits=None
):
    return classify_empty_response(
        response,
        visible,
        prev_had_tool_calls=prev_had_tool_calls,
        nudges_done=nudges_done,
        prefill_retries=prefill_retries,
        empty_retries=empty_retries,
        limits=limits or RecoveryLimits(),
    )


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #


def test_has_inline_thinking():
    assert has_inline_thinking("<think>...</think>") is True
    assert has_inline_thinking("<THINKING>x") is True
    assert has_inline_thinking("plain text") is False
    assert has_inline_thinking(None) is False
    assert has_inline_thinking("") is False


def test_has_thinking():
    assert has_thinking(LLMResponse(content=None, reasoning_content="hmm")) is True
    assert has_thinking(LLMResponse(content=None, thinking_blocks=[{"x": 1}])) is True
    assert has_thinking(LLMResponse(content="<think>...</think>")) is True
    assert has_thinking(LLMResponse(content="real answer")) is False
    assert has_thinking(LLMResponse(content=None)) is False


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #


def test_classify_visible_text_completes():
    assert _classify(LLMResponse(content="answer"), "answer") is RecoveryAction.COMPLETE


def test_classify_disabled_completes():
    limits = RecoveryLimits(enabled=False)
    assert _classify(LLMResponse(content=None), "", limits=limits) is RecoveryAction.COMPLETE


def test_classify_thinking_only_prefills_until_budget():
    resp = LLMResponse(content=None, reasoning_content="hmm")
    assert _classify(resp, "", prefill_retries=0) is RecoveryAction.PREFILL
    assert _classify(resp, "", prefill_retries=1) is RecoveryAction.PREFILL

    assert _classify(resp, "", prefill_retries=2) is RecoveryAction.RETRY


def test_classify_post_tool_empty_nudges():
    resp = LLMResponse(content=None)
    assert _classify(resp, "", prev_had_tool_calls=True, nudges_done=0) is RecoveryAction.NUDGE

    assert _classify(resp, "", prev_had_tool_calls=True, nudges_done=1) is RecoveryAction.RETRY


def test_classify_thinking_takes_priority_over_nudge():

    resp = LLMResponse(content=None, reasoning_content="hmm")
    assert _classify(resp, "", prev_had_tool_calls=True) is RecoveryAction.PREFILL


def test_classify_plain_empty_retries_until_budget():
    resp = LLMResponse(content=None)
    assert _classify(resp, "", empty_retries=2) is RecoveryAction.RETRY
    assert _classify(resp, "", empty_retries=3) is RecoveryAction.COMPLETE


def test_classify_always_reasoning_model_still_retries_after_prefill():

    resp = LLMResponse(content=None, reasoning_content="hmm")
    assert _classify(resp, "", prefill_retries=2, empty_retries=0) is RecoveryAction.RETRY
    assert _classify(resp, "", prefill_retries=2, empty_retries=3) is RecoveryAction.COMPLETE


def test_limits_from_defaults_maps_fields():
    class _D:
        empty_recovery_enabled = False
        post_tool_empty_max_nudges = 5
        thinking_prefill_max_retries = 6
        empty_content_max_retries = 7

    limits = limits_from_defaults(_D())
    assert limits == RecoveryLimits(
        enabled=False,
        post_tool_empty_max_nudges=5,
        thinking_prefill_max_retries=6,
        empty_content_max_retries=7,
    )


def test_limits_from_defaults_uses_defaults_for_missing_attrs():
    assert limits_from_defaults(object()) == RecoveryLimits()


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #


class _EmptyThenAnswerProvider(LLMProvider):
    def __init__(self, empties: int = 2):
        super().__init__(api_key="test")
        self._empties = empties
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
    ):
        self.calls += 1
        if self.calls <= self._empties:
            return LLMResponse(content="", finish_reason="stop")
        return LLMResponse(content="real answer", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_empty_then_recovers_and_does_not_persist_scaffolding(workspace):
    provider = _EmptyThenAnswerProvider(empties=2)
    agent = _make_agent(workspace, provider)

    out = await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
    )

    assert out is not None
    assert out[0] == "real answer"
    assert provider.calls == 3

    session = agent.sessions.get_or_create("s1")
    for m in session.messages:
        assert not m.get("_recovery_synthetic")


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #


class _ThinkingThenAnswerProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key="test")
        self.calls = 0
        self.saw_reasoning_replay = False

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
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(content="", reasoning_content="let me think", finish_reason="stop")

        if any(m.get("_recovery_synthetic") for m in messages):
            self.saw_reasoning_replay = True
        return LLMResponse(content="final answer", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_thinking_only_recovers_via_prefill(workspace):
    provider = _ThinkingThenAnswerProvider()
    agent = _make_agent(workspace, provider)

    out = await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
    )

    assert out is not None
    assert out[0] == "final answer"
    assert provider.saw_reasoning_replay is True
    session = agent.sessions.get_or_create("s1")
    for m in session.messages:
        assert not m.get("_recovery_synthetic")


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #


class _AlwaysEmptyProvider(LLMProvider):
    def __init__(self):
        super().__init__(api_key="test")
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
    ):
        self.calls += 1
        return LLMResponse(content="", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_persistently_empty_is_bounded_then_falls_back(workspace):
    limits = RecoveryLimits()
    provider = _AlwaysEmptyProvider()
    agent = _make_agent(workspace, provider, limits=limits)

    out = await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
    )

    assert out is not None

    assert provider.calls == 1 + limits.empty_content_max_retries
    assert "no response" in out[0].lower()


@pytest.mark.asyncio
async def test_recovery_disabled_falls_back_immediately(workspace):
    provider = _AlwaysEmptyProvider()
    agent = _make_agent(workspace, provider, limits=RecoveryLimits(enabled=False))

    out = await agent._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="test", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
            text="hi",
        ),
        session_key="s1",
    )

    assert out is not None
    assert provider.calls == 1
    assert "no response" in out[0].lower()
