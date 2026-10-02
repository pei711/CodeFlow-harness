"""Persisted session messages carry a wall-clock timestamp.

The real agent loop persists turns through ``AgentLoop._save_turn`` which
appends raw dicts via the ``Session.record`` choke point. These tests drive
the real loop path (stubbed LLM) and assert the JSONL lines on disk carry a
``timestamp`` and no longer carry the dropped per-message ``received_at`` /
``turn_id`` — pinning the simplified stamping contract at the level that
reproduces a real TUI/CLI turn.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.providers.base import LLMProvider, LLMResponse
from codeflow.spine.message import ChatType, Source
from codeflow.spine.turn import Origin, TurnRequest


class StubProvider(LLMProvider):
    """Always returns a fixed assistant message. No tool calls."""

    def __init__(self, content: str = "stub response"):
        super().__init__(api_key="test")
        self._content = content

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
        return LLMResponse(content=self._content, finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _make_agent(workspace: Path) -> AgentLoop:
    return AgentLoop(
        provider=StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
    )


def _make_msg(content: str = "hello") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="tui",
            chat_id="chat1",
            sender_id="user",
            chat_type=ChatType.DM,
        ),
        text=content,
    )


def _persisted_messages(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / "sessions" / "tui" / "chat1.jsonl"
    assert path.exists(), "session file was not persisted"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in records if r.get("_type") != "metadata"]


@pytest.mark.asyncio
async def test_persisted_messages_carry_timestamp_not_turn_fields(workspace):
    agent = _make_agent(workspace)
    out = await agent._process_message(_make_msg("hello"))
    assert out is not None

    msgs = _persisted_messages(workspace)
    roles = [m.get("role") for m in msgs]
    assert "user" in roles and "assistant" in roles

    for m in msgs:
        assert m.get("timestamp"), f"missing timestamp: {m}"
        assert "received_at" not in m, f"received_at should be dropped: {m}"
        assert "turn_id" not in m, f"turn_id should be dropped: {m}"


@pytest.mark.asyncio
async def test_subagent_session_persists_only_the_user_visible_summary(workspace):
    agent = _make_agent(workspace)
    request = _make_msg("[Subagent result]\nUntrusted internal payload")
    request = TurnRequest(
        origin=Origin.SUBAGENT,
        source=request.source,
        text=request.text,
    )

    out = await agent._process_message(request, origin=Origin.SUBAGENT)

    assert out is not None
    messages = _persisted_messages(workspace)
    assert [message["role"] for message in messages] == ["assistant"]
    assert messages[0]["content"] == "stub response"
    assert "Subagent result" not in json.dumps(messages)


def test_save_turn_sanitizes_nested_image_data_without_mutating_input(workspace):
    agent = _make_agent(workspace)
    session = agent.sessions.get_or_create("tui:sanitize")
    data_uri = "data:image/png;base64,QUJD"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"inspect {data_uri}"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
        {"role": "assistant", "content": f"assistant {data_uri}"},
        {"role": "tool", "content": f"tool {data_uri}"},
    ]
    original = copy.deepcopy(messages)

    agent._save_turn(session, messages, 0, Origin.USER)

    serialized = json.dumps(session.messages)
    assert "data:image/" not in serialized
    assert "[image]" in serialized
    assert messages == original
