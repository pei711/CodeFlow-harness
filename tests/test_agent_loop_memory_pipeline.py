"""Deterministic full-turn coverage of the channel-to-memory pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.memory_engine.backend import Memory
from codeflow.providers.base import LLMProvider, LLMResponse
from codeflow.spine.message import ChatType, Source
from codeflow.spine.turn import Origin, TurnRequest

_USER_MEMO = "MEMO_user_prefers_terse_answers"


class _StubProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.seen_messages: list[dict] = []

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
        self.seen_messages = messages
        return LLMResponse(content="ok", finish_reason="stop")

    def get_default_model(self) -> str:
        return "stub"

    def prompt_text(self) -> str:
        return "\n".join(str(message.get("content")) for message in self.seen_messages)


class _FakeBackend:
    def __init__(self) -> None:
        self.recall_calls: list[dict[str, Any]] = []
        self.store_calls: list[dict[str, Any]] = []
        self.feedback_calls: list[dict[str, Any]] = []
        self.store_raises: Exception | None = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        self.recall_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id,
                "top_k": top_k,
            }
        )
        if user_id is not None:
            return [Memory(text=_USER_MEMO, score=1.0)]
        return []

    async def store(self, session_id, messages):
        self.store_calls.append({"session_id": session_id, "messages": messages})
        if self.store_raises is not None:
            raise self.store_raises

    async def feedback(self, signals):
        self.feedback_calls.append(signals)


def _make_agent(workspace: Path, *, backend=None) -> AgentLoop:
    return AgentLoop(
        provider=_StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        backend=backend,
    )


def _msg(content: str = "how do I back up a config file safely?") -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="mock", chat_id="c1", sender_id="user", chat_type=ChatType.DM),
        text=content,
    )


async def test_full_turn_recalls_injects_and_stores(tmp_path: Path) -> None:
    backend = _FakeBackend()
    agent = _make_agent(tmp_path, backend=backend)

    out = await agent._process_message(_msg())
    assert out is not None
    assert backend.recall_calls == [
        {
            "query": "how do I back up a config file safely?",
            "user_id": "default",
            "agent_id": None,
            "top_k": 5,
        }
    ]
    assert _USER_MEMO in agent.provider.prompt_text()
    assert len(backend.store_calls) == 1
    call = backend.store_calls[0]
    assert call["session_id"] == "mock:c1"
    persisted = agent.sessions.peek("mock:c1")
    assert persisted is not None
    assert call["messages"] == persisted.messages
    assert backend.feedback_calls == []


async def test_no_backend_turn_completes_without_memory_calls(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, backend=None)
    agent.context.memory.get_memory_context = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("disabled Memory must not read host state")
    )

    out = await agent._process_message(_msg())
    assert out is not None
    assert agent.memory_enabled is False
    agent.configure_personalization(True)
    assert agent.enable_personalization is False


async def test_store_failure_surfaces_after_session_is_saved(tmp_path: Path) -> None:
    backend = _FakeBackend()
    backend.store_raises = RuntimeError("memory unavailable")
    agent = _make_agent(tmp_path, backend=backend)

    with pytest.raises(RuntimeError, match="memory unavailable"):
        await agent._process_message(_msg())

    assert len(backend.store_calls) == 1
    persisted = agent.sessions.peek("mock:c1")
    assert persisted is not None
    assert [message["role"] for message in persisted.messages] == ["user", "assistant"]
