import dataclasses

import pytest

from codeflow.spine import ChatType, Media, Source


def test_chat_type_is_closed_two_value_str_enum():
    assert {c.value for c in ChatType} == {"dm", "group"}
    assert ChatType.DM == "dm"


def test_chat_type_str_renders_as_value():

    assert str(ChatType.DM) == "dm"
    assert str(ChatType.GROUP) == "group"


def test_source_defaults_extras():
    s = Source(channel="telegram", chat_id="42", sender_id="7", chat_type=ChatType.DM)
    assert s.extras == {}


def test_source_is_frozen():
    s = Source(channel="cli", chat_id="c", sender_id="u", chat_type=ChatType.GROUP)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.chat_id = "other"


def test_chat_type_value_lookup_and_rejects_unknown():
    assert ChatType("dm") is ChatType.DM
    assert ChatType("group") is ChatType.GROUP
    with pytest.raises(ValueError):
        ChatType("channel")


def test_source_and_media_are_value_objects():
    base = dict(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    assert Source(**base) == Source(**base)
    assert Source(**base) != Source(**{**base, "chat_id": "other"})
    assert Media("p", "m", "k") == Media("p", "m", "k")


def test_source_extras_are_independent_per_instance():
    a = Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    b = Source(channel="t", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    a.extras["x"] = 1
    assert b.extras == {}


def test_source_is_intentionally_not_hashable():

    s = Source(channel="cli", chat_id="c", sender_id="u", chat_type=ChatType.DM)
    with pytest.raises(TypeError):
        hash(s)


def test_media_carries_path_mime_kind_and_is_frozen():
    m = Media(path="/tmp/a.jpg", mime="image/jpeg", kind="image", content=b"snapshot")
    assert (m.path, m.mime, m.kind, m.content) == ("/tmp/a.jpg", "image/jpeg", "image", b"snapshot")
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.path = "/tmp/b.jpg"
