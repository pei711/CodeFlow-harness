"""Tests for codeflow.channels.adapters.wecom — frame/body parsing, per-type
content extraction, and inbound dedup. Pure surface; no live SDK."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from codeflow.channels.adapters.wecom.channel import WecomChannel


def _channel(welcome_message="", allow_from=("*",), bot_id="b", secret="s"):
    cfg = SimpleNamespace(bot_id=bot_id, secret=secret, welcome_message=welcome_message, allow_from=list(allow_from))
    ch = WecomChannel(cfg)
    ch.intake.publish = AsyncMock()
    return ch


def _text_frame(msg_id="m1", sender="u1", chat_id="c1", text="hi"):
    return SimpleNamespace(
        body={
            "msgid": msg_id,
            "from": {"userid": sender},
            "chattype": "single",
            "chatid": chat_id,
            "text": {"content": text},
        }
    )


@contextmanager
def _receipts(level="INFO"):
    """Collect adapter log lines so a receipt can be pinned by its wording."""
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level=level)
    try:
        yield lines
    finally:
        logger.remove(sink_id)


def test_body_from_frame_attr():
    assert WecomChannel._body(SimpleNamespace(body={"x": 1})) == {"x": 1}


def test_body_from_dict():
    assert WecomChannel._body({"body": {"y": 2}}) == {"y": 2}
    assert WecomChannel._body({"y": 2}) == {"y": 2}


def test_body_other_type():
    assert WecomChannel._body(123) == {}


def test_extract_text():
    ch = _channel()
    assert asyncio.run(ch._extract({"text": {"content": "hello"}}, "text")) == "hello"


def test_extract_voice_uses_platform_transcription():
    ch = _channel()
    body = {"voice": {"content": "transcribed words"}}
    assert asyncio.run(ch._extract(body, "voice")) == "[voice] transcribed words"


def test_extract_voice_without_content():
    ch = _channel()
    assert asyncio.run(ch._extract({"voice": {}}, "voice")) == "[voice]"


def test_extract_mixed():
    ch = _channel()
    body = {
        "mixed": {
            "item": [
                {"type": "text", "text": {"content": "hi there"}},
                {"type": "image"},
            ]
        }
    }
    out = asyncio.run(ch._extract(body, "mixed"))
    assert "hi there" in out
    assert "[image]" in out


def test_extract_image_downloads_and_labels(monkeypatch):
    import codeflow.channels.adapters.wecom.channel as wecom_mod

    monkeypatch.setattr(wecom_mod, "save_media_bytes", lambda channel, data, name: Path("/m/abcd_pic.jpg"))
    ch = _channel()
    ch._client = AsyncMock()
    ch._client.download_file = AsyncMock(return_value=(b"\x89PNG", "pic.jpg"))
    out = asyncio.run(ch._extract({"image": {"url": "u", "aeskey": "k"}}, "image"))
    assert "[image: pic.jpg]" in out
    assert "[Image: source: /m/abcd_pic.jpg]" in out


def test_extract_file_uses_provided_name_over_server_name(monkeypatch):
    import codeflow.channels.adapters.wecom.channel as wecom_mod

    monkeypatch.setattr(wecom_mod, "save_media_bytes", lambda channel, data, name: Path("/m/h_doc.pdf"))
    ch = _channel()
    ch._client = AsyncMock()
    ch._client.download_file = AsyncMock(return_value=(b"%PDF", "server.bin"))
    out = asyncio.run(ch._extract({"file": {"url": "u", "aeskey": "k", "name": "doc.pdf"}}, "file"))
    assert "[file: doc.pdf]" in out


def test_extract_image_missing_keys_marks_failed():
    ch = _channel()
    assert asyncio.run(ch._extract({"image": {}}, "image")) == "[image: image: download failed]"


def test_extract_image_marks_failed_when_download_returns_no_data():
    ch = _channel()
    ch._client = AsyncMock()
    ch._client.download_file = AsyncMock(return_value=(None, "pic.jpg"))
    out = asyncio.run(ch._extract({"image": {"url": "u", "aeskey": "k"}}, "image"))
    assert out == "[image: image: download failed]"


def test_process_dedup_skips_repeated_msgid():
    ch = _channel()
    frame = SimpleNamespace(
        body={
            "msgid": "m1",
            "from": {"userid": "u1"},
            "chattype": "single",
            "text": {"content": "hi"},
        }
    )
    asyncio.run(ch._process(frame, "text"))
    asyncio.run(ch._process(frame, "text"))
    assert ch.intake.publish.await_count == 1


def test_frames_are_lru_capped(monkeypatch):
    import codeflow.channels.adapters.wecom.channel as wecom_mod

    monkeypatch.setattr(wecom_mod, "_FRAMES_CAP", 2)
    ch = _channel()
    for i in range(3):
        frame = SimpleNamespace(
            body={
                "msgid": f"m{i}",
                "from": {"userid": f"u{i}"},
                "chattype": "single",
                "chatid": f"c{i}",
                "text": {"content": "hi"},
            }
        )
        asyncio.run(ch._process(frame, "text"))
    assert len(ch._frames) == 2
    assert "c0" not in ch._frames
    assert "c2" in ch._frames


def test_send_noop_without_client():
    ch = _channel()
    ch._client = None
    asyncio.run(ch.send("c1", "hi"))


def test_send_skips_empty_content():
    ch = _channel()
    ch._client = AsyncMock()
    asyncio.run(ch.send("c1", "   "))
    ch._client.reply_stream.assert_not_awaited()


def test_send_skips_when_no_frame():
    ch = _channel()
    ch._client = AsyncMock()
    asyncio.run(ch.send("c1", "hi"))
    ch._client.reply_stream.assert_not_awaited()


def test_send_media_surfaced_as_notice():
    """reply_stream is text-only — dropped attachments become a visible
    notice instead of vanishing."""
    ch = _channel()
    ch._client = AsyncMock()
    ch._frames["c1"] = SimpleNamespace(body={})
    asyncio.run(ch.send("c1", "hi", media=["/m/report.pdf"]))
    sent_text = ch._client.reply_stream.await_args.args[2]
    assert "hi" in sent_text and "[Attachment not sent: report.pdf]" in sent_text


def test_send_replies_with_cached_frame():
    ch = _channel()
    ch._client = AsyncMock()
    ch._frames["c1"] = SimpleNamespace(body={})
    asyncio.run(ch.send("c1", "hi"))
    ch._client.reply_stream.assert_awaited_once()
    args, kwargs = ch._client.reply_stream.await_args
    assert args[0] is ch._frames["c1"]
    assert args[1].startswith("stream_")
    assert args[2] == "hi"
    assert kwargs["finish"] is True


@pytest.mark.parametrize(
    "error",
    [TimeoutError("ws ack timeout"), ConnectionError("socket reset")],
    ids=["timeout", "connection"],
)
def test_send_reraises_transient_for_manager_retry(error):
    """A ws drop/timeout propagates (the inbound frame is still cached, so the
    manager retry can succeed); business errors stay swallowed."""
    ch = _channel()
    ch._client = AsyncMock()
    ch._client.reply_stream = AsyncMock(side_effect=error)
    ch._frames["c1"] = SimpleNamespace(body={})
    with pytest.raises(type(error)):
        asyncio.run(ch.send("c1", "hi"))


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), ValueError("bad payload")],
    ids=["runtime", "value"],
)
def test_send_swallows_reply_error(error):
    ch = _channel()
    ch._client = AsyncMock()
    ch._client.reply_stream = AsyncMock(side_effect=error)
    ch._frames["c1"] = SimpleNamespace(body={})
    asyncio.run(ch.send("c1", "hi"))


def test_send_logs_sent_receipt():
    ch = _channel()
    ch._client = AsyncMock()
    ch._frames["c1"] = SimpleNamespace(body={})
    with _receipts() as lines:
        asyncio.run(ch.send("c1", "hi"))
    assert "WeCom message sent: chat_id=c1" in "".join(lines)


def test_send_logs_no_receipt_when_reply_fails():
    """A swallowed reply error must not leave a 'sent' receipt behind."""
    ch = _channel()
    ch._client = AsyncMock()
    ch._client.reply_stream = AsyncMock(side_effect=RuntimeError("boom"))
    ch._frames["c1"] = SimpleNamespace(body={})
    with _receipts() as lines:
        asyncio.run(ch.send("c1", "hi"))
    assert "WeCom message sent" not in "".join(lines)


def test_on_enter_chat_sends_welcome_when_configured():
    ch = _channel(welcome_message="hello there")
    ch._client = AsyncMock()
    frame = SimpleNamespace(body={"chatid": "c1"})
    asyncio.run(ch._on_enter_chat(frame))
    ch._client.reply_welcome.assert_awaited_once()
    args, _ = ch._client.reply_welcome.await_args
    assert args[0] is frame
    assert args[1] == {"msgtype": "text", "text": {"content": "hello there"}}


def test_on_enter_chat_noop_without_welcome():
    ch = _channel(welcome_message="")
    ch._client = AsyncMock()
    asyncio.run(ch._on_enter_chat(SimpleNamespace(body={"chatid": "c1"})))
    ch._client.reply_welcome.assert_not_awaited()


def test_process_disallowed_sender_skips_download_and_publish():
    """Denied sender is rejected before _extract (which downloads media) and
    before publishing — not merely dropped at the central intake."""
    ch = _channel(allow_from=[])
    ch._extract = AsyncMock()
    frame = SimpleNamespace(
        body={
            "msgid": "m1",
            "from": {"userid": "u1"},
            "chattype": "single",
            "image": {"url": "u", "aeskey": "k"},
        }
    )
    with _receipts("WARNING") as lines:
        asyncio.run(ch._process(frame, "image"))
    ch._extract.assert_not_awaited()
    ch.intake.publish.assert_not_awaited()
    assert "WeCom inbound rejected by allowlist: sender=u1" in "".join(lines)


def test_process_logs_accept_receipt():
    ch = _channel()
    with _receipts() as lines:
        asyncio.run(ch._process(_text_frame(), "text"))
    assert "WeCom inbound accepted: message_id=m1 msg_type=text chat_type=single" in "".join(lines)


def test_process_logs_duplicate_receipt():
    ch = _channel()
    frame = _text_frame(msg_id="dup")
    with _receipts() as lines:
        asyncio.run(ch._process(frame, "text"))
        asyncio.run(ch._process(frame, "text"))
    assert "WeCom duplicate event suppressed: message_id=dup" in "".join(lines)


def test_process_logs_empty_content_drop():
    ch = _channel()
    with _receipts() as lines:
        asyncio.run(ch._process(_text_frame(text=""), "text"))
    ch.intake.publish.assert_not_awaited()
    assert "WeCom inbound dropped: message_id=m1 has no extractable content" in "".join(lines)


def test_process_non_dict_body_is_dropped():
    ch = _channel()
    with _receipts("WARNING") as lines:
        asyncio.run(ch._process(SimpleNamespace(body=["not", "a", "dict"]), "text"))
    ch.intake.publish.assert_not_awaited()
    assert "WeCom inbound dropped: invalid body type" in "".join(lines)


def test_process_missing_sender_falls_back_to_unknown():
    """A frame without a usable 'from' block still routes, tagged unknown."""
    ch = _channel()
    frame = SimpleNamespace(body={"msgid": "m1", "from": "not-a-dict", "text": {"content": "hi"}})
    asyncio.run(ch._process(frame, "text"))
    assert ch.intake.publish.await_args.kwargs["sender_id"] == "unknown"


def test_process_without_msgid_derives_a_dedup_key():
    ch = _channel()
    frame = SimpleNamespace(body={"chatid": "c1", "sendertime": "42", "text": {"content": "hi"}})
    asyncio.run(ch._process(frame, "text"))
    assert ch.intake.publish.await_args.kwargs["metadata"]["message_id"] == "c1_42"


def test_process_unhandled_error_logs_and_does_not_raise():
    """A publish-time failure must drop the event, never kill the SDK loop."""
    ch = _channel()
    ch.intake.publish = AsyncMock(side_effect=RuntimeError("spine down"))
    with _receipts("ERROR") as lines:
        asyncio.run(ch._process(_text_frame(), "text"))
    assert "WeCom inbound dropped: event handling failed" in "".join(lines)


@pytest.mark.parametrize(
    "bot_id, secret",
    [("", "s"), ("b", ""), ("", "")],
    ids=["no-bot-id", "no-secret", "neither"],
)
def test_start_bails_out_without_credentials(bot_id, secret):
    """Missing credentials must stop before any SDK client is constructed."""
    ch = _channel(bot_id=bot_id, secret=secret)
    with _receipts("ERROR") as lines:
        asyncio.run(ch.start())
    assert ch._client is None
    assert ch.is_running is False
    assert "WeCom bot_id and secret not configured" in "".join(lines)


def test_wecom_satisfies_channel_contract():
    from codeflow.channels import Channel
    from codeflow.channels.contract import capability_violations

    ch = _channel()
    assert isinstance(ch, Channel)
    assert capability_violations(ch) == []


def test_wecom_spec_declares_beta_maturity():
    from codeflow.channels.adapters.wecom.spec import SPEC

    assert SPEC.maturity == "beta"


def test_wecom_spec_factory_raises_import_error_without_sdk(monkeypatch):
    """A missing extra must surface as ImportError so the manager can disable
    just this channel."""
    import sys

    monkeypatch.delitem(sys.modules, "codeflow.channels.adapters.wecom.channel", raising=False)
    monkeypatch.setitem(sys.modules, "wecom_aibot_sdk", None)
    from codeflow.channels.adapters.wecom.spec import SPEC

    with pytest.raises(ImportError):
        SPEC.factory(SimpleNamespace(bot_id="b", secret="s", allow_from=["*"]))


def test_wecom_spec_import_is_cheap():
    """Importing wecom.spec must NOT pull in wecom_aibot_sdk (the heavy import is
    deferred into SPEC.factory)."""
    import subprocess
    import sys

    code = (
        "import sys, codeflow.channels.adapters.wecom.spec as s;"
        "assert 'wecom_aibot_sdk' not in sys.modules, 'spec import pulled in wecom_aibot_sdk';"
        "assert callable(s.SPEC.factory) and s.SPEC.display_name == 'WeCom'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
