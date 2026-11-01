from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from codeflow.config.codeflow import ContextConfig
from codeflow.context_engine.base import AssembledPrefix, AssemblyContext
from codeflow.context_engine.history_trimmer import HistoryTrimmer
from codeflow.context_engine.segments.codeflow_history import CodeFlowHistorySegmentBuilder
from codeflow.memory_engine import TokenBudget
from codeflow.providers.base import LLMProvider, LLMResponse
from codeflow.utils.helpers import estimate_message_tokens


class _SummaryProvider(LLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="summary from scripted provider")

    def get_default_model(self) -> str:
        return "scripted/summary"


def _context() -> AssemblyContext:
    history = [
        {"role": "user", "content": "Keep the release region fixed to ap-east-1."},
        {"role": "assistant", "content": "Acknowledged."},
    ]
    for index in range(18):
        history.extend(
            [
                {"role": "user", "content": f"Unrelated deterministic filler {index}: " + ("x " * 120)},
                {"role": "assistant", "content": f"Filler response {index}: " + ("y " * 100)},
            ]
        )
    history.extend(
        [
            {"role": "user", "content": "Final request; retain ap-east-1."},
            {"role": "assistant", "content": "Ready."},
        ]
    )
    return AssemblyContext(
        session_key="threshold-test",
        current_message="Create the release report.",
        media=None,
        channel="test",
        chat_id="threshold-test",
        session_messages=history,
        # Same effective history budget for every threshold arm.
        budget=TokenBudget(8_192, 500, 300, 892, 6_500),
        prefix=AssembledPrefix(
            system_prefix="You are a test agent.",
            user_message={"role": "user", "content": "Create the release report."},
            tool_defs=[],
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threshold", "expected_compacted"),
    [(0.50, True), (0.80, False), (0.93, False)],
)
async def test_structured_compact_threshold_changes_trigger_without_changing_budget(
    tmp_path: Path,
    threshold: float,
    expected_compacted: bool,
) -> None:
    provider = _SummaryProvider()
    config = ContextConfig(structured_compact_threshold=threshold)
    builder = CodeFlowHistorySegmentBuilder(
        state_root=tmp_path,
        config=config,
        provider=provider,
        model="scripted/summary",
        context_window_tokens=8_192,
        get_tool_definitions=lambda: [],
    )

    segment = await builder.build(_context())

    assert segment is not None
    assert segment.meta["context_compacted"] is expected_compacted
    assert segment.meta["pressure_tier"] == ("tier3_summary" if expected_compacted else "tier1_snip")
    assert provider.calls == int(expected_compacted)
    assert segment.meta["prompt_budget_tokens"] == 8_192 - 500


def test_structured_compact_threshold_must_be_a_ratio() -> None:
    with pytest.raises(ValueError):
        ContextConfig(structured_compact_threshold=0)


def _builder(tmp_path, provider, **settings):
    return CodeFlowHistorySegmentBuilder(
        state_root=tmp_path,
        config=ContextConfig(protect_first_n=1, **settings),
        provider=provider,
        model="scripted/summary",
        context_window_tokens=8192,
        get_tool_definitions=lambda: [],
    )


def _pressure_context(pressure):
    ctx = _context()
    # The final request deliberately does not repeat the initial constraint.
    ctx.session_messages[-2] = {"role": "user", "content": "Prepare the final report now."}
    tokens = sum(estimate_message_tokens(message) for message in ctx.session_messages)
    return replace(ctx, budget=replace(ctx.budget, available_history=int(tokens / pressure)))


@pytest.mark.asyncio
@pytest.mark.parametrize("pressure", [0.85, 0.97])
async def test_original_user_constraint_survives_pruning_and_lossy_summary(tmp_path, pressure):
    ctx = _pressure_context(pressure)
    original = deepcopy(ctx.session_messages)
    builder = _builder(tmp_path, _SummaryProvider())
    segment = await builder.build(ctx)

    assert segment.history[0] == original[0]
    assert segment.meta["protected_user_message_ids"] == [0]
    assert HistoryTrimmer.structural_errors(segment.history) == []
    assert ctx.session_messages == original
    if pressure > 0.95:
        assert segment.meta["summary_boundary"] > 0
        assert original[0]["content"] not in "summary from scripted provider"


@pytest.mark.asyncio
async def test_truncated_summary_is_rejected_and_fallback_retains_initial_requirement(tmp_path):
    class Truncated(_SummaryProvider):
        async def chat(self, **kwargs):
            self.calls += 1
            self.requested_output = kwargs["max_tokens"]
            return LLMResponse(content="incomplete and misleading summary", finish_reason="length")

    provider = Truncated()
    builder = _builder(tmp_path, provider)
    ctx = _pressure_context(0.97)
    segment = await builder.build(ctx)

    assert provider.requested_output == 2048
    assert segment.meta["summary_mode"] == "deterministic_fallback"
    assert segment.meta["summary_failure_reason"] == "output_truncated"
    assert "incomplete and misleading summary" not in segment.text
    assert ctx.session_messages[0]["content"] in segment.text
    assert segment.history[0] == ctx.session_messages[0]


@pytest.mark.asyncio
async def test_summary_timeout_uses_fallback_without_losing_raw_constraint(tmp_path):
    import asyncio

    class Stalled(_SummaryProvider):
        async def chat(self, **kwargs):
            await asyncio.Event().wait()

    builder = _builder(tmp_path, Stalled(), compact_timeout_seconds=0.01)
    ctx = _pressure_context(0.97)
    segment = await builder.build(ctx)
    assert segment.meta["summary_mode"] == "deterministic_fallback"
    assert segment.meta["summary_failure_reason"] == "TimeoutError"
    assert segment.history[0] == ctx.session_messages[0]


@pytest.mark.asyncio
async def test_persisted_boundary_reuses_summary_only_for_matching_source(tmp_path):
    provider = _SummaryProvider()
    ctx = _pressure_context(0.97)
    first = await _builder(tmp_path, provider).build(ctx)
    restored = await _builder(tmp_path, provider).build(ctx)
    assert provider.calls == 1
    assert restored.history[0] == ctx.session_messages[0]
    assert restored.meta["summary_boundary"] == first.meta["summary_boundary"]

    # Replacing history with the same length must invalidate the old summary.
    ctx.session_messages[0] = {"role": "user", "content": "Keep region fixed to eu-west-2."}
    replaced = await _builder(tmp_path, provider).build(ctx)
    assert provider.calls == 2
    assert replaced.history[0] == ctx.session_messages[0]
    assert "ap-east-1" not in str(replaced.history) + replaced.text


@pytest.mark.asyncio
async def test_error_response_text_is_never_saved_as_summary(tmp_path):
    class ErrorText(_SummaryProvider):
        async def chat(self, **kwargs):
            return LLMResponse(content="provider internal error", finish_reason="error")

    segment = await _builder(tmp_path, ErrorText()).build(_pressure_context(0.97))
    assert segment.meta["summary_failure_reason"] == "provider_error"
    assert "provider internal error" not in segment.text
