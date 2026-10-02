import dataclasses

import pytest

from codeflow.spine import (
    ChatType,
    Emit,
    Origin,
    Source,
    Text,
    ToolEvent,
    ToolPhase,
    TurnOutcome,
    TurnRequest,
    TurnRunner,
    Usage,
)


def _req():
    src = Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    return TurnRequest(origin=Origin.USER, source=src, text="hi")


def test_turn_outcome_is_frozen_with_usage_reply_and_tool_evidence():
    o = TurnOutcome(
        usage=Usage(1, 2, 3),
        explicit_reply=True,
        memory_hits=2,
        injected_skill_ids=("local/example",),
        context_path="slow",
        context_fallback_reason=None,
        skill_source_failures=("remote",),
    )
    assert o.usage == Usage(1, 2, 3)
    assert o.explicit_reply is True
    assert o.tool_calls == 0
    assert o.tool_failures == 0
    assert o.memory_hits == 2
    assert o.injected_skill_ids == ("local/example",)
    assert o.context_path == "slow"
    assert o.context_fallback_reason is None
    assert o.skill_source_failures == ("remote",)
    fields = {f.name for f in dataclasses.fields(TurnOutcome)}
    assert fields == {
        "usage",
        "explicit_reply",
        "tool_calls",
        "tool_failures",
        "memory_hits",
        "injected_skill_ids",
        "context_path",
        "context_fallback_reason",
        "skill_source_failures",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.explicit_reply = False


def test_turn_runner_protocol_is_a_weak_has_run_check():

    class HasRun:
        async def run(self, req, emit):
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    class NoRun:
        pass

    assert isinstance(HasRun(), TurnRunner)
    assert not isinstance(NoRun(), TurnRunner)


async def test_runner_emits_runner_events_and_returns_outcome():
    emitted: list = []

    async def emit(ev) -> None:
        emitted.append(ev)

    class Fake:
        async def run(self, req: TurnRequest, emit: Emit) -> TurnOutcome:
            await emit(Text(content="hi"))
            await emit(ToolEvent(phase=ToolPhase.START, tool_call_id="t1", name="grep"))
            return TurnOutcome(usage=Usage(10, 20, 30), explicit_reply=True)

    fake = Fake()
    assert isinstance(fake, TurnRunner)
    outcome = await fake.run(_req(), emit)
    assert [type(e) for e in emitted] == [Text, ToolEvent]
    assert outcome == TurnOutcome(usage=Usage(10, 20, 30), explicit_reply=True)


async def test_runner_emitting_nothing_is_a_legal_turn():
    async def emit(ev) -> None:
        raise AssertionError("a silent turn must not emit")

    class Silent:
        async def run(self, req: TurnRequest, emit: Emit) -> TurnOutcome:
            return TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)

    outcome = await Silent().run(_req(), emit)
    assert outcome == TurnOutcome(usage=Usage(0, 0, 0), explicit_reply=False)


def test_emit_is_a_callable_type_alias():
    from codeflow.spine.runner import Emit as EmitAlias

    assert EmitAlias is Emit
