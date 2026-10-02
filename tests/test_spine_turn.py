import dataclasses

import pytest

from codeflow.spine import BusyPolicy, ChatType, Origin, Source, TurnRequest


def _src():
    return Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)


def test_origin_is_closed_three_value_enum_without_system():
    assert {o.value for o in Origin} == {"user", "cron", "subagent"}
    assert not hasattr(Origin, "SYSTEM")
    with pytest.raises(ValueError):
        Origin("system")


def test_busy_policy_is_closed_three_value_enum():
    assert {b.value for b in BusyPolicy} == {"append", "inject", "interrupt"}
    with pytest.raises(ValueError):
        BusyPolicy("drop")


def test_enum_str_renders_as_value():

    assert str(Origin.USER) == "user"
    assert str(BusyPolicy.APPEND) == "append"


def test_message_id_and_conversation_are_independent_axes():

    base = dict(origin=Origin.USER, source=_src(), text="x")
    a = TurnRequest(**base, message_id="557")
    b = TurnRequest(**base, message_id="558")
    assert a.message_id != b.message_id
    assert a.conversation is None and b.conversation is None


def test_turn_request_defaults():
    r = TurnRequest(origin=Origin.USER, source=_src(), text="hi")
    assert r.media == ()
    assert r.message_id is None
    assert r.conversation is None
    assert r.busy is BusyPolicy.APPEND


def test_turn_request_carries_message_id_not_reply_to():
    fields = {f.name for f in dataclasses.fields(TurnRequest)}
    assert "message_id" in fields
    assert "reply_to" not in fields


def test_turn_request_is_frozen():
    r = TurnRequest(origin=Origin.CRON, source=_src(), text="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.text = "y"


def test_turn_request_is_not_hashable_via_source_extras():

    r = TurnRequest(origin=Origin.USER, source=_src(), text="x")
    with pytest.raises(TypeError):
        hash(r)


def test_turn_request_has_no_conversation_id_derivation():

    assert not hasattr(TurnRequest, "conversation_id")
    fields = {f.name for f in dataclasses.fields(TurnRequest)}
    assert "conversation" in fields
    assert "conversation_id" not in fields
