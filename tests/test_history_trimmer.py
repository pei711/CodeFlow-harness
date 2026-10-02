from __future__ import annotations

from codeflow.context_engine.history_trimmer import HistoryTrimmer


class _CountingProvider:
    def estimate_prompt_tokens(self, messages, tools, model):
        return len(messages) * 100, "test"


def test_history_from_ids_preserves_reasoning_fields():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "chain of thought",
            "thinking_blocks": [{"thinking": "block"}],
        },
    ]

    history = HistoryTrimmer.history_from_ids(messages, [0, 1])

    assert history[1]["reasoning_content"] == "chain of thought"
    assert history[1]["thinking_blocks"] == [{"thinking": "block"}]


def test_history_from_ids_drops_non_provider_keys():
    messages = [
        {"role": "user", "content": "hi", "timestamp": "2026-07-08T00:00:00"},
    ]

    history = HistoryTrimmer.history_from_ids(messages, [0])

    assert history == [{"role": "user", "content": "hi"}]


def test_trim_drops_complete_turns_by_relevance_not_oldest_message():
    messages = [
        {"role": "user", "content": "old relevant"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "low relevance"},
        {"role": "assistant", "content": "low answer"},
        {"role": "user", "content": "recent relevant"},
        {"role": "assistant", "content": "recent answer"},
    ]
    trimmer = HistoryTrimmer(_CountingProvider(), "test", lambda: [], 450)

    built, outcome = trimmer.trim(
        session_messages=messages,
        ids=list(range(len(messages))),
        protected_ids=set(),
        priority_scores={0: 0.9, 1: 0.9, 2: 0.1, 3: 0.1, 4: 0.8, 5: 0.8},
        reserved_output=50,
        build_messages=lambda history: [
            {"role": "system", "content": "system"},
            *history,
            {"role": "user", "content": "current"},
        ],
    )

    assert outcome.ok is True
    assert outcome.included_ids == [0, 1]
    assert [message["content"] for message in built[1:-1]] == ["old relevant", "old answer"]
    assert all("dropped turn messages" in warning for warning in outcome.warnings)


def test_trim_never_orphans_tool_results():
    messages = [
        {"role": "user", "content": "use tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "read", "content": "result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"},
    ]
    trimmer = HistoryTrimmer(_CountingProvider(), "test", lambda: [], 450)

    built, outcome = trimmer.trim(
        session_messages=messages,
        ids=list(range(len(messages))),
        protected_ids=set(),
        priority_scores={mid: (0.1 if mid < 4 else 0.9) for mid in range(len(messages))},
        reserved_output=50,
        build_messages=lambda history: [
            {"role": "system", "content": "system"},
            *history,
            {"role": "user", "content": "current"},
        ],
    )

    assert outcome.included_ids == [4, 5]
    assert trimmer.structural_errors(built) == []
