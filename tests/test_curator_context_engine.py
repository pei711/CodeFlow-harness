from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from codeflow.agent.loop import AgentLoop
from codeflow.config import ContextConfig
from codeflow.context_engine import ContextAssembler, TurnContext
from codeflow.context_engine.curator import CuratorArchiveStore
from codeflow.context_engine.segments.curator import CuratorSegmentBuilder
from codeflow.memory_engine.base import TokenBudget
from codeflow.providers.base import ErrorClassification, LLMProvider, LLMResponse, ToolCallRequest
from codeflow.spine.message import ChatType, Media, Source
from codeflow.spine.turn import Origin, TurnRequest


class CuratorScriptProvider(LLMProvider):
    def __init__(self, *, curator_mode: str = "slow"):
        super().__init__(api_key="test")
        self.curator_mode = curator_mode
        self.curator_calls = 0
        self.main_calls = 0
        self.curator_tool_names: set[str] = set()

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
        tool_names = {tool.get("function", {}).get("name") for tool in (tools or []) if isinstance(tool, dict)}
        if "curator_build_context" in tool_names:
            self.curator_tool_names = tool_names
            self.curator_calls += 1
            if self.curator_mode == "fallback":
                return LLMResponse(content="I cannot decide.", tool_calls=[])
            if self.curator_mode == "error":
                return LLMResponse(
                    content="provider unavailable",
                    finish_reason="error",
                    error_classification=ErrorClassification(category="rate_limit"),
                )
            if self.curator_calls == 1:
                return LLMResponse(
                    content=None,
                    reasoning_content="curator internal reasoning",
                    thinking_blocks=[{"thinking": "curator private thinking"}],
                    tool_calls=[
                        ToolCallRequest(
                            id="curator_archive_1",
                            name="curator_archive_messages",
                            arguments={
                                "message_ids": [0, 1],
                                "reason": "old context",
                                "tags": ["old"],
                                "summary": "old setup",
                            },
                        )
                    ],
                )
            assert messages[-2]["reasoning_content"] == "curator internal reasoning"
            assert messages[-2]["thinking_blocks"] == [{"thinking": "curator private thinking"}]
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="curator_build_1",
                        name="curator_build_context",
                        arguments={
                            "include_message_ids": [0, 1, 2, 3],
                            "working_state_injection": "Keep project setup and latest request.",
                            "notes": "test plan",
                        },
                    )
                ],
            )

        self.main_calls += 1
        return LLMResponse(
            content="main done",
            reasoning_content="main agent private reasoning",
            thinking_blocks=[{"thinking": "main agent private thinking"}],
        )

    def get_default_model(self) -> str:
        return "fake-main"


class ConcurrentCuratorProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.first_turn_started = asyncio.Event()
        self.release_first_turn = asyncio.Event()

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
        payload = json.loads(messages[1]["content"])
        current_message = payload["current_user_message"]
        if current_message == "turn-a":
            self.first_turn_started.set()
            await self.release_first_turn.wait()
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id=f"build-{current_message}",
                    name="curator_build_context",
                    arguments={"include_message_ids": []},
                )
            ],
        )

    def get_default_model(self) -> str:
        return "fake-main"


class HangingCuratorProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__(api_key="test")
        self.never = asyncio.Event()

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
        await self.never.wait()

    def get_default_model(self) -> str:
        return "fake-main"


def _session_messages() -> list[dict]:
    return [
        {"role": "user", "content": "Initial project rule: preserve exact config.", "timestamp": "2026-05-12T10:00:00"},
        {"role": "assistant", "content": "Noted the config preservation rule.", "timestamp": "2026-05-12T10:01:00"},
        {"role": "user", "content": "Now design curator context management.", "timestamp": "2026-05-12T11:00:00"},
        {
            "role": "assistant",
            "content": "We should use manifest plus selective retrieval.",
            "timestamp": "2026-05-12T11:01:00",
        },
    ]


def _budget() -> TokenBudget:
    return TokenBudget(
        context_length=4096,
        reserved_output=512,
        reserved_tools=100,
        reserved_system=500,
        available_history=2984,
    )


def test_curator_persisted_artifacts_omit_inline_image_data(tmp_path: Path):
    data_uri = "data:image/png;base64,QUJDRA=="
    store = CuratorArchiveStore(tmp_path, ContextConfig())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": f"embedded {data_uri}"},
            ],
        }
    ]

    manifest = store.build_manifest("tui:media", messages)
    archived = store.archive_messages("tui:media", manifest, messages, [0])
    store.write_working_state("tui:media", {"decisions": [f"embedded {data_uri}"]})
    store.append_trace("tui:media", "turn-1", "main_agent_result", {"messages": messages})

    archive_path = tmp_path / archived["archive_refs"][0].split("#", 1)[0]
    persisted_paths = [
        store.manifest_path("tui:media"),
        archive_path,
        store.state_path("tui:media"),
        store.trace_path("tui:media", "turn-1"),
    ]
    for path in persisted_paths:
        persisted = path.read_text(encoding="utf-8")
        assert data_uri not in persisted
        assert "QUJDRA==" not in persisted
    assert messages[0]["content"][0]["image_url"]["url"] == data_uri


def test_agentloop_uses_curator_and_keeps_internal_tools_private(tmp_path: Path):
    loop = AgentLoop(
        provider=CuratorScriptProvider(),
        workspace=tmp_path,
        context_config=ContextConfig(engine="curator", fast_path_threshold=0.0),
    )

    assert loop.context_engine.name == "context_assembler"
    assert loop.context_engine.owns_compaction is True
    assert isinstance(loop.context_engine, ContextAssembler)
    assert not any(name.startswith("curator_") for name in loop.tools.tool_names)


@pytest.mark.asyncio
async def test_curator_slow_path_archives_and_writes_trace(tmp_path: Path):
    provider = CuratorScriptProvider(curator_mode="slow")
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path,
        context_config=ContextConfig(engine="curator", fast_path_threshold=0.0),
    )

    assembled = await loop.context_engine.assemble(
        "cli:curator-test",
        _session_messages(),
        _budget(),
        turn=TurnContext(current_message="Please continue the curator design.", channel="cli", chat_id="curator-test"),
    )

    assert assembled.metadata["path"] == "slow"
    assert provider.curator_calls == 2
    assert "Curator Working State" in assembled.messages[0]["content"]
    trace_path = Path(assembled.metadata["trace_path"])
    assert trace_path.exists()
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "curator_archive_messages" in trace_text
    assert "slow_path_accepted" in trace_text

    manifest = json.loads((tmp_path / "memory/.curator/manifest/cli_curator-test.json").read_text(encoding="utf-8"))
    archived = [item for item in manifest["items"] if item["archived"]]
    assert [item["id"] for item in archived] == [0, 1]
    assert list((tmp_path / "memory/.curator/archive").glob("**/*.jsonl"))


@pytest.mark.asyncio
async def test_curator_fallback_when_internal_agent_does_not_finish(tmp_path: Path):
    provider = CuratorScriptProvider(curator_mode="fallback")
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path,
        context_config=ContextConfig(engine="curator", fast_path_threshold=0.0),
    )

    assembled = await loop.context_engine.assemble(
        "cli:fallback-test",
        _session_messages(),
        _budget(),
        turn=TurnContext(current_message="Continue.", channel="cli", chat_id="fallback-test"),
    )

    assert assembled.metadata["path"] == "fallback"
    assert assembled.metadata["fallback_reason"] == "invalid_plan"
    assert assembled.messages[0]["role"] == "system"
    assert assembled.messages[-1]["role"] == "user"
    assert "Continue." in assembled.messages[-1]["content"]
    assert "curator_read_memory" not in provider.curator_tool_names


@pytest.mark.asyncio
async def test_curator_timeout_records_distinct_fallback_reason(tmp_path: Path):
    loop = AgentLoop(
        provider=HangingCuratorProvider(),
        workspace=tmp_path,
        context_config=ContextConfig(
            engine="curator",
            fast_path_threshold=0.0,
            curator_timeout_seconds=0.01,
        ),
    )

    assembled = await loop.context_engine.assemble(
        "cli:timeout-test",
        _session_messages(),
        _budget(),
        turn=TurnContext(current_message="Continue.", channel="cli", chat_id="timeout-test"),
    )

    assert assembled.metadata["path"] == "fallback"
    assert assembled.metadata["fallback_reason"] == "timeout"


@pytest.mark.asyncio
async def test_curator_provider_error_records_category(tmp_path: Path):
    loop = AgentLoop(
        provider=CuratorScriptProvider(curator_mode="error"),
        workspace=tmp_path,
        context_config=ContextConfig(engine="curator", fast_path_threshold=0.0),
    )

    assembled = await loop.context_engine.assemble(
        "cli:provider-error-test",
        _session_messages(),
        _budget(),
        turn=TurnContext(current_message="Continue.", channel="cli", chat_id="provider-error-test"),
    )

    assert assembled.metadata["path"] == "fallback"
    assert assembled.metadata["fallback_reason"] == "provider_failure"
    assert assembled.metadata["provider_error_category"] == "rate_limit"


@pytest.mark.asyncio
async def test_concurrent_curator_builds_keep_multimodal_prefixes_turn_local(tmp_path: Path):
    provider = ConcurrentCuratorProvider()
    config = ContextConfig(engine="curator", fast_path_threshold=0.0)
    curator = CuratorSegmentBuilder(
        tmp_path,
        config,
        provider,
        "fake-main",
        4096,
        lambda: [],
        max_steps=1,
    )
    assembler = ContextAssembler([curator], lambda: [])
    turn_a = TurnContext(
        current_message="turn-a",
        media=[Media(path="a.png", mime="image/png", kind="image", content=b"A")],
    )
    turn_b = TurnContext(
        current_message="turn-b",
        media=[Media(path="b.png", mime="image/png", kind="image", content=b"B" * 20_000)],
    )

    first = asyncio.create_task(assembler.assemble("tui:a", [], _budget(), turn=turn_a))
    await asyncio.wait_for(provider.first_turn_started.wait(), timeout=1)
    try:
        second = await assembler.assemble("tui:b", [], _budget(), turn=turn_b)
    finally:
        provider.release_first_turn.set()
    first_result = await first

    def image_bytes(assembled) -> bytes:
        url = next(
            item["image_url"]["url"] for item in assembled.messages[-1]["content"] if item.get("type") == "image_url"
        )
        return base64.b64decode(url.split(",", 1)[1])

    assert first_result.metadata["path"] == "slow"
    assert image_bytes(first_result) == b"A"
    assert image_bytes(second) == b"B" * 20_000


@pytest.mark.asyncio
async def test_process_message_records_main_and_curator_trajectories(tmp_path: Path):
    provider = CuratorScriptProvider(curator_mode="slow")
    loop = AgentLoop(
        provider=provider,
        workspace=tmp_path,
        context_config=ContextConfig(engine="curator", fast_path_threshold=0.0),
    )
    session = loop.sessions.get_or_create("cli:trace-test")
    session.messages.extend(_session_messages())
    loop.sessions.save(session)

    response = await loop._process_message(
        TurnRequest(
            origin=Origin.USER,
            source=Source(
                channel="cli",
                chat_id="trace-test",
                sender_id="user",
                chat_type=ChatType.DM,
            ),
            text="Use the curator trajectory and answer.",
        )
    )

    assert response is not None
    assert response[0] == "main done"
    traces = list((tmp_path / "memory/.curator/traces/cli_trace-test").glob("*.jsonl"))
    assert len(traces) == 1
    trace_text = traces[0].read_text(encoding="utf-8")
    assert "curator_llm_request" in trace_text
    assert "main_agent_result" in trace_text
    assert "curator internal reasoning" not in trace_text
    assert "curator private thinking" not in trace_text
    assert "main agent private reasoning" not in trace_text
    assert "main agent private thinking" not in trace_text
    assert '"reasoning_content"' not in trace_text
    assert '"thinking_blocks"' not in trace_text
    request_records = [
        record for line in trace_text.splitlines() if (record := json.loads(line))["event"] == "curator_llm_request"
    ]
    assert all(
        "reasoning_content" not in message and "thinking_blocks" not in message
        for record in request_records
        for message in record["payload"]["messages"]
    )


def test_history_from_messages_preserves_reasoning_fields():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "chain of thought",
            "thinking_blocks": [{"thinking": "block"}],
        },
    ]

    history = CuratorSegmentBuilder._history_from_messages(messages)

    assert history[1]["reasoning_content"] == "chain of thought"
    assert history[1]["thinking_blocks"] == [{"thinking": "block"}]


def test_curator_prompt_keeps_only_the_active_decision() -> None:
    prompt = CuratorSegmentBuilder._system_prompt()

    assert "latest explicit user decision active" in prompt
    assert "only the active version of a decision" in prompt
