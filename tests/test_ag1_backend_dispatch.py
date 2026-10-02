"""AG-1 — AgentLoop ``backend`` wiring + ``_dispatch_backend_store``.

The two after-turn callsites (system-message path + REPL path) now call
:meth:`AgentLoop._dispatch_backend_store` as the third peer step in the
after-turn pipeline (alongside ``context_engine.after_turn`` and
``maybe_consolidate``). This file exercises the dispatcher in isolation
— the full end-to-end "AgentLoop processes a turn and the backend
ultimately sees it" path is left to integration tests that wire a real
LLM provider; here we keep things small + focused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.providers.base import LLMResponse
from codeflow.spine.message import ChatType, Media, Source
from codeflow.spine.turn import Origin, TurnRequest

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class _StubProvider:
    api_key = "test"

    def get_default_model(self) -> str:
        return "stub"

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not invoked in this dispatcher smoke test")

    async def chat_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("not invoked in this dispatcher smoke test")


class _ReplyProvider(_StubProvider):
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def chat_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs["messages"])
        return LLMResponse(content="done")


class _FakeBackend:
    def __init__(self) -> None:
        self.store_calls: list[dict[str, Any]] = []
        self.store_raises: Exception | None = None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def feedback(self, signals):
        pass

    async def recall(self, query, *, user_id=None, agent_id=None, top_k):
        return []

    async def store(self, session_id, messages):
        self.store_calls.append(
            {
                "session_id": session_id,
                "messages": messages,
            }
        )
        if self.store_raises is not None:
            raise self.store_raises


def _make_loop(workspace: Path, *, backend=None, provider=None) -> AgentLoop:
    return AgentLoop(
        provider=provider or _StubProvider(),
        workspace=workspace,
        model="stub",
        max_iterations=2,
        restrict_to_workspace=True,
        backend=backend,
    )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestConstructorWiring:
    def test_default_backend_is_none(self, tmp_path: Path) -> None:
        agent = _make_loop(tmp_path)
        assert agent.backend is None

    def test_explicit_backend_stored(self, tmp_path: Path) -> None:
        b = _FakeBackend()
        agent = _make_loop(tmp_path, backend=b)
        assert agent.backend is b


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestDispatcher:
    async def test_no_backend_is_noop(self, tmp_path: Path) -> None:
        agent = _make_loop(tmp_path, backend=None)

        await agent._dispatch_backend_store(
            "session-1",
            [{"role": "user", "content": "hi"}],
        )

    async def test_empty_messages_skips_backend(
        self,
        tmp_path: Path,
    ) -> None:
        b = _FakeBackend()
        agent = _make_loop(tmp_path, backend=b)
        await agent._dispatch_backend_store("session-1", [])

        assert b.store_calls == []

    async def test_calls_backend_store_with_full_slice(
        self,
        tmp_path: Path,
    ) -> None:
        b = _FakeBackend()
        agent = _make_loop(tmp_path, backend=b)
        slice_ = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"},
        ]
        await agent._dispatch_backend_store("session-key-x", slice_)
        assert len(b.store_calls) == 1
        call = b.store_calls[0]
        assert call["session_id"] == "session-key-x"
        assert call["messages"] == slice_

    async def test_omits_inline_image_data_without_mutating_live_messages(
        self,
        tmp_path: Path,
    ) -> None:
        data_uri = "data:image/png;base64,QUJDRA=="
        messages = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
            }
        ]
        backend = _FakeBackend()
        agent = _make_loop(tmp_path, backend=backend)

        await agent._dispatch_backend_store("session-1", messages)

        assert messages[0]["content"][0]["image_url"]["url"] == data_uri
        assert backend.store_calls[0]["messages"][0]["content"][0]["image_url"]["url"] == "[image data omitted]"

    async def test_backend_exception_propagates(
        self,
        tmp_path: Path,
    ) -> None:
        b = _FakeBackend()
        b.store_raises = RuntimeError("evermem down")
        agent = _make_loop(tmp_path, backend=b)
        with pytest.raises(RuntimeError, match="evermem down"):
            await agent._dispatch_backend_store(
                "s",
                [{"role": "user", "content": "x"}],
            )
        assert len(b.store_calls) == 1


async def test_subagent_announce_is_not_forwarded_to_persisted_turn_artifacts(
    tmp_path: Path,
) -> None:
    backend = _FakeBackend()
    agent = _make_loop(tmp_path, backend=backend, provider=_ReplyProvider())
    after_turn_calls: list[dict[str, Any]] = []

    async def record_after_turn(session_key: str, outcome: dict[str, Any]) -> None:
        after_turn_calls.append({"session_key": session_key, "outcome": outcome})

    agent.context_engine.after_turn = record_after_turn
    request = TurnRequest(
        origin=Origin.SUBAGENT,
        source=Source(
            channel="test",
            chat_id="chat",
            sender_id="subagent",
            chat_type=ChatType.DM,
        ),
        text="internal subagent announce",
    )

    await agent._process_message(request, origin=Origin.SUBAGENT)

    persisted = agent.sessions.get_or_create("test:chat").messages
    assert [message["role"] for message in persisted] == ["assistant"]
    assert after_turn_calls[0]["outcome"]["messages"] == persisted
    assert backend.store_calls[0]["messages"] == persisted


async def test_turn_artifacts_are_sanitized_after_live_multimodal_model_input(
    tmp_path: Path,
) -> None:
    provider = _ReplyProvider()
    backend = _FakeBackend()
    agent = _make_loop(tmp_path, backend=backend, provider=provider)
    after_turn_calls: list[dict[str, Any]] = []

    async def record_after_turn(session_key: str, outcome: dict[str, Any]) -> None:
        after_turn_calls.append({"session_key": session_key, "outcome": outcome})

    agent.context_engine.after_turn = record_after_turn
    request = TurnRequest(
        origin=Origin.USER,
        source=Source(
            channel="test",
            chat_id="media",
            sender_id="user",
            chat_type=ChatType.DM,
        ),
        text="inspect",
        media=(Media(path="snapshot.png", mime="image/png", kind="image", content=b"ABCD"),),
    )

    await agent._process_message(request, origin=Origin.USER)

    live_user = next(message for message in provider.calls[0] if message["role"] == "user")
    live_image = next(item for item in live_user["content"] if item.get("type") == "image_url")
    assert live_image["image_url"]["url"] == "data:image/png;base64,QUJDRA=="

    persisted = agent.sessions.get_or_create("test:media").messages
    assert after_turn_calls[0]["outcome"]["messages"] == persisted
    assert backend.store_calls[0]["messages"] == persisted
    persisted_user = next(message for message in persisted if message["role"] == "user")
    assert persisted_user["content"] == [
        {"type": "text", "text": "[image]"},
        {"type": "text", "text": "inspect"},
    ]


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


class TestLegacyCompat:
    def test_construction_without_backend_unchanged(
        self,
        tmp_path: Path,
    ) -> None:
        """Pre-AG-1 construction (no ``backend=`` keyword) still works
        end-to-end. After Phase B-3 the ``self.memory`` facade is gone;
        we now assert against the direct subsystem fields AgentLoop
        holds (``memory_consolidator`` + ``context.skills``)."""
        from codeflow.memory_engine.consolidate.consolidator import (
            MemoryConsolidator,
        )

        agent = _make_loop(tmp_path)
        assert isinstance(agent.memory_consolidator, MemoryConsolidator)
        assert agent.context.skills is not None
        assert agent.backend is None
