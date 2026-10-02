"""Tests for the extracted channel services: Intake (inbound gate + spine
submit) and transcribe_audio."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from codeflow.channels.intake import Intake
from codeflow.channels.transcribe import transcribe_audio


def test_intake_is_allowed():
    assert Intake("tg", SimpleNamespace(allow_from=["*"])).is_allowed("u1") is True
    assert Intake("tg", SimpleNamespace(allow_from=["u1"])).is_allowed("u1") is True
    assert Intake("tg", SimpleNamespace(allow_from=["u9"])).is_allowed("u1") is False
    assert Intake("tg", SimpleNamespace(allow_from=[])).is_allowed("u1") is False


def test_intake_custom_allow_check_overrides():

    intake = Intake("tg", SimpleNamespace(allow_from=[]), allow_check=lambda s: s == "ok")
    assert intake.is_allowed("ok") is True
    assert intake.is_allowed("nope") is False


def test_intake_custom_allow_check_gates_publish():
    submit = AsyncMock()
    intake = Intake("tg", SimpleNamespace(allow_from=["*"]), allow_check=lambda s: False)
    intake.set_submit(submit)
    asyncio.run(intake.publish(sender_id="u", chat_id="c", content="x"))
    submit.assert_not_awaited()


def test_intake_submit_path_builds_turnrequest():
    from codeflow.spine import ChatType, Origin

    submit = AsyncMock()
    intake = Intake("tg", SimpleNamespace(allow_from=["*"]))
    intake.set_submit(submit)
    asyncio.run(
        intake.publish(
            sender_id=123,
            chat_id=456,
            content="hi",
            media=["/m.jpg"],
            metadata={"chat_type": "group", "message_id": "9"},
            session_key="s1",
        )
    )
    submit.assert_awaited_once()
    req = submit.await_args.args[0]
    assert req.origin is Origin.USER
    assert (req.source.channel, req.source.chat_id, req.source.sender_id) == ("tg", "456", "123")
    assert req.source.chat_type is ChatType.GROUP
    assert req.source.extras == {"chat_type": "group", "message_id": "9"}
    assert req.text == "hi"
    assert [m.path for m in req.media] == ["/m.jpg"]
    assert req.conversation == "s1"


def test_intake_submit_path_dm_default_and_no_conversation():
    from codeflow.spine import ChatType

    submit = AsyncMock()
    intake = Intake("tg", SimpleNamespace(allow_from=["*"]))
    intake.set_submit(submit)
    asyncio.run(intake.publish(sender_id="u", chat_id="c", content="x"))
    req = submit.await_args.args[0]
    assert req.source.chat_type is ChatType.DM
    assert req.conversation is None
    assert req.source.extras == {}


def test_intake_submit_path_denied_does_not_submit():
    submit = AsyncMock()
    intake = Intake("tg", SimpleNamespace(allow_from=[]))
    intake.set_submit(submit)
    asyncio.run(intake.publish(sender_id="u", chat_id="c", content="x"))
    submit.assert_not_awaited()


def test_intake_no_submit_wired_drops(monkeypatch):

    intake = Intake("tg", SimpleNamespace(allow_from=["*"]))
    asyncio.run(intake.publish(sender_id="u", chat_id="c", content="x"))


async def test_sealed_intake_drops_old_handler_publish_after_transport_stop():
    submit = AsyncMock()
    intake = Intake("tg", SimpleNamespace(allow_from=["*"]))
    intake.set_submit(submit)
    handler_started = asyncio.Event()
    resume_handler = asyncio.Event()

    async def old_handler() -> None:
        handler_started.set()
        await resume_handler.wait()
        await intake.publish(sender_id="u", chat_id="c", content="late")

    handler = asyncio.create_task(old_handler())
    await handler_started.wait()
    intake.seal()
    resume_handler.set()
    await handler

    submit.assert_not_awaited()


async def test_wait_idle_blocks_until_publish_already_in_submit_finishes():
    submit_started = asyncio.Event()
    release_submit = asyncio.Event()

    async def submit(req) -> None:
        submit_started.set()
        await release_submit.wait()

    intake = Intake("tg", SimpleNamespace(allow_from=["*"]))
    intake.set_submit(submit)
    publish = asyncio.create_task(intake.publish(sender_id="u", chat_id="c", content="in flight"))
    await submit_started.wait()
    intake.seal()

    try:
        drain = asyncio.create_task(intake.wait_idle())
        await asyncio.sleep(0)
        assert not drain.done()
        release_submit.set()
        await publish
        await drain
    finally:
        release_submit.set()
        await publish


def test_transcribe_audio_delegates(monkeypatch):
    class _FakeProvider:
        def __init__(self, api_key=None):
            pass

        async def transcribe(self, path):
            return "hello world"

    monkeypatch.setattr("codeflow.providers.transcription.GroqTranscriptionProvider", _FakeProvider)
    assert asyncio.run(transcribe_audio("/a.ogg", api_key="k")) == "hello world"


def test_transcribe_audio_swallows_errors(monkeypatch):
    class _Boom:
        def __init__(self, api_key=None):
            raise RuntimeError("nope")

    monkeypatch.setattr("codeflow.providers.transcription.GroqTranscriptionProvider", _Boom)
    assert asyncio.run(transcribe_audio("/a.ogg")) == ""


def test_transcribe_audio_empty_key_becomes_none(monkeypatch):
    seen = {}

    class _Rec:
        def __init__(self, api_key=None):
            seen["key"] = api_key

        async def transcribe(self, path):
            return ""

    monkeypatch.setattr("codeflow.providers.transcription.GroqTranscriptionProvider", _Rec)
    asyncio.run(transcribe_audio("/a.ogg", api_key=""))
    assert seen["key"] is None
    asyncio.run(transcribe_audio("/a.ogg", api_key="k"))
    assert seen["key"] == "k"
