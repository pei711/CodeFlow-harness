"""Tests for the qq adapter package.

parsing.py — pure route/content resolution from a botpy message.
channel.py — inbound dedup/dispatch and SDK send routing.

Real botpy WebSocket connection / API are live flows left to integration/manual
testing.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from loguru import logger

from codeflow.channels.adapters.qq import parsing as qp
from codeflow.channels.adapters.qq.channel import QQChannel


def _channel(allow_from=("*",)):
    ch = QQChannel(SimpleNamespace(app_id="a", secret="s", allow_from=list(allow_from)))
    ch.intake.publish = AsyncMock()
    return ch


def _group_msg(mid="m1", content="hello"):
    return SimpleNamespace(id=mid, content=content, group_openid="g1", author=SimpleNamespace(member_openid="u1"))


@contextmanager
def _receipts(level="INFO"):
    """Collect adapter log lines so a receipt can be pinned by its wording."""
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level=level)
    try:
        yield lines
    finally:
        logger.remove(sink_id)


def test_clean_content():
    assert qp.clean_content(SimpleNamespace(content="  hi  ")) == "hi"
    assert qp.clean_content(SimpleNamespace(content="")) == ""
    assert qp.clean_content(SimpleNamespace(content=None)) == ""


def test_resolve_route_group():
    assert qp.resolve_route(_group_msg(), is_group=True) == ("g1", "u1", "group")


def test_resolve_route_c2c_by_id():
    data = SimpleNamespace(author=SimpleNamespace(id="u2"))
    assert qp.resolve_route(data, is_group=False) == ("u2", "u2", "c2c")


def test_resolve_route_c2c_user_openid_fallback():
    data = SimpleNamespace(author=SimpleNamespace(user_openid="u3"))
    assert qp.resolve_route(data, is_group=False) == ("u3", "u3", "c2c")


def test_resolve_route_c2c_unknown():
    data = SimpleNamespace(author=SimpleNamespace())
    assert qp.resolve_route(data, is_group=False) == ("unknown", "unknown", "c2c")


def test_resolve_route_guild_dm():
    """A botpy DirectMessage carries guild_id (the DM session id) — replies
    must route through post_dms, not the C2C endpoint."""
    data = SimpleNamespace(guild_id="gld9", author=SimpleNamespace(id="u7"))
    assert qp.resolve_route(data, is_group=False) == ("gld9", "u7", "guild_dm")


def test_on_message_group_dispatch():
    ch = _channel()
    asyncio.run(ch._on_message(_group_msg(), is_group=True))
    kw = ch.intake.publish.await_args.kwargs
    assert (kw["sender_id"], kw["chat_id"], kw["content"]) == ("u1", "g1", "hello")
    assert kw["metadata"] == {"message_id": "m1", "chat_type": "group"}
    assert ch._chat_type_cache["g1"] == "group"


def test_on_message_c2c_dispatch():
    ch = _channel()
    data = SimpleNamespace(id="m2", content="yo", author=SimpleNamespace(id="u2"))
    asyncio.run(ch._on_message(data, is_group=False))
    kw = ch.intake.publish.await_args.kwargs
    assert (kw["sender_id"], kw["chat_id"], kw["content"]) == ("u2", "u2", "yo")
    assert ch._chat_type_cache["u2"] == "c2c"


def test_on_message_dedup():
    ch = _channel()
    asyncio.run(ch._on_message(_group_msg(mid="dup"), is_group=True))
    asyncio.run(ch._on_message(_group_msg(mid="dup"), is_group=True))
    assert ch.intake.publish.await_count == 1


def test_on_message_empty_content_skipped():
    ch = _channel()
    asyncio.run(ch._on_message(_group_msg(content="   "), is_group=True))
    ch.intake.publish.assert_not_awaited()


def _attachment(content_type="image/png", filename="pic.png"):
    return SimpleNamespace(content_type=content_type, filename=filename)


@pytest.mark.parametrize(
    "content_type, filename, expected",
    [
        ("image/png", "pic.png", "[image: pic.png]"),
        ("video/mp4", "clip.mp4", "[video: clip.mp4]"),
        ("audio/amr", "note.amr", "[voice: note.amr]"),
        ("application/pdf", "doc.pdf", "[file: doc.pdf]"),
        (None, "blob.bin", "[file: blob.bin]"),
        ("image/png", None, "[image]"),
        ("image/png", "../../etc/passwd", "[image: passwd]"),
    ],
    ids=["image", "video", "audio", "file", "no-type", "no-name", "traversal"],
)
def test_attachment_labels_normalize_sdk_metadata(content_type, filename, expected):
    """botpy exposes content_type/filename, so labelling needs no download."""
    data = SimpleNamespace(attachments=[_attachment(content_type, filename)])
    assert qp.attachment_labels(data) == [expected]


def test_attachment_labels_absent_or_empty():
    assert qp.attachment_labels(SimpleNamespace()) == []
    assert qp.attachment_labels(SimpleNamespace(attachments=[])) == []
    assert qp.attachment_labels(SimpleNamespace(attachments=None)) == []


def test_compose_content_joins_text_and_labels():
    data = SimpleNamespace(content=" hi ", attachments=[_attachment()])
    assert qp.compose_content(data) == "hi\n[image: pic.png]"


def test_on_message_attachment_only_still_dispatches():
    """An attachment-only message carries no text but must not be dropped."""
    ch = _channel()
    data = SimpleNamespace(id="m9", content="", author=SimpleNamespace(id="u2"), attachments=[_attachment()])
    asyncio.run(ch._on_message(data, is_group=False))
    assert ch.intake.publish.await_args.kwargs["content"] == "[image: pic.png]"


def test_on_message_disallowed_sender_rejected_before_side_effects():
    """A denied sender must not reach the route cache or the intake."""
    ch = _channel(allow_from=[])
    with _receipts("WARNING") as lines:
        asyncio.run(ch._on_message(_group_msg(), is_group=True))
    ch.intake.publish.assert_not_awaited()
    assert ch._chat_type_cache == {}
    assert "QQ inbound rejected by allowlist: sender=u1" in "".join(lines)


def test_on_message_logs_accept_receipt():
    ch = _channel()
    with _receipts() as lines:
        asyncio.run(ch._on_message(_group_msg(), is_group=True))
    assert "QQ inbound accepted: message_id=m1 chat_type=group" in "".join(lines)


def test_on_message_logs_duplicate_receipt():
    ch = _channel()
    with _receipts() as lines:
        asyncio.run(ch._on_message(_group_msg(mid="dup"), is_group=True))
        asyncio.run(ch._on_message(_group_msg(mid="dup"), is_group=True))
    assert "QQ duplicate event suppressed: message_id=dup" in "".join(lines)


def test_on_message_without_id_is_dropped():
    ch = _channel()
    with _receipts("WARNING") as lines:
        asyncio.run(ch._on_message(SimpleNamespace(content="hi"), is_group=False))
    ch.intake.publish.assert_not_awaited()
    assert "QQ inbound dropped: event carries no message id" in "".join(lines)


def test_on_message_malformed_event_logs_and_does_not_raise():
    """A missing author field must drop the event, never kill the SDK loop."""
    ch = _channel()
    with _receipts("ERROR") as lines:
        asyncio.run(ch._on_message(SimpleNamespace(id="m4", content="hi"), is_group=False))
    ch.intake.publish.assert_not_awaited()
    assert "QQ inbound dropped: event handling failed" in "".join(lines)


def _client():
    client = MagicMock()
    client.api.post_group_message = AsyncMock()
    client.api.post_c2c_message = AsyncMock()
    return client


def test_send_group_routes_to_group_api():
    ch = _channel()
    ch._client = _client()
    ch._chat_type_cache["g1"] = "group"
    asyncio.run(ch.send("g1", "reply"))
    ch._client.api.post_group_message.assert_awaited_once()
    ch._client.api.post_c2c_message.assert_not_called()
    kw = ch._client.api.post_group_message.await_args.kwargs
    assert kw["group_openid"] == "g1" and kw["markdown"] == {"content": "reply"}
    assert kw["msg_id"] is None


def test_send_c2c_default_route():
    ch = _channel()
    ch._client = _client()
    asyncio.run(ch.send("u2", "hi"))
    ch._client.api.post_c2c_message.assert_awaited_once()
    kw = ch._client.api.post_c2c_message.await_args.kwargs
    assert kw["openid"] == "u2"
    assert kw["msg_id"] is None


def test_send_increments_msg_seq():
    ch = _channel()
    ch._client = _client()
    before = ch._msg_seq
    asyncio.run(ch.send("u2", "a"))
    asyncio.run(ch.send("u2", "b"))
    assert ch._client.api.post_c2c_message.await_args_list[0].kwargs["msg_seq"] == before + 1
    assert ch._client.api.post_c2c_message.await_args_list[1].kwargs["msg_seq"] == before + 2


def test_send_guild_dm_routes_to_post_dms():
    ch = _channel()
    ch._client = _client()
    ch._client.api.post_dms = AsyncMock()
    dm = SimpleNamespace(id="m3", content="hi bot", guild_id="gld9", author=SimpleNamespace(id="u7"))
    asyncio.run(ch._on_message(dm, is_group=False))
    assert ch._chat_type_cache["gld9"] == "guild_dm"

    asyncio.run(ch.send("gld9", "reply"))
    ch._client.api.post_dms.assert_awaited_once_with(guild_id="gld9", content="reply", msg_id=None)
    ch._client.api.post_c2c_message.assert_not_called()
    ch._client.api.post_group_message.assert_not_called()


def test_send_media_surfaced_as_notice():
    """The reply endpoints are text-only — dropped attachments become a visible
    notice instead of vanishing."""
    ch = _channel()
    ch._client = _client()
    with _receipts("WARNING") as lines:
        asyncio.run(ch.send("u2", "here", media=["/m/report.pdf"]))
    sent = ch._client.api.post_c2c_message.await_args.kwargs["markdown"]["content"]
    assert "here" in sent and "[Attachment not sent: report.pdf]" in sent
    assert "QQ reply is text-only; 1 attachment(s) not sent" in "".join(lines)


def test_send_media_notice_survives_empty_text():
    ch = _channel()
    ch._client = _client()
    asyncio.run(ch.send("u2", "", media=["/tmp/pic.png"]))
    kw = ch._client.api.post_c2c_message.await_args.kwargs
    assert kw["openid"] == "u2"
    assert kw["markdown"] == {"content": "[Attachment not sent: pic.png]"}


def test_send_logs_sent_receipt():
    ch = _channel()
    ch._client = _client()
    with _receipts() as lines:
        asyncio.run(ch.send("u2", "hi"))
    assert "QQ message sent: chat_id=u2 chat_type=c2c" in "".join(lines)


def test_send_no_client_is_noop():
    ch = _channel()
    ch._client = None
    asyncio.run(ch.send("u2", "x"))


def test_send_reraises_transient_for_manager_retry():
    """5xx / network errors propagate so manager._send_with_retry can back off;
    other errors stay swallowed (see test_send_swallows_api_error)."""
    from botpy.errors import ServerError

    ch = _channel()
    ch._client = _client()
    ch._client.api.post_c2c_message = AsyncMock(side_effect=ServerError("502"))
    with pytest.raises(ServerError):
        asyncio.run(ch.send("u2", "x"))


@pytest.mark.parametrize(
    "error",
    [TimeoutError("read timeout"), ConnectionError("reset")],
    ids=["timeout", "connection"],
)
def test_send_reraises_network_errors(error):
    ch = _channel()
    ch._client = _client()
    ch._client.api.post_c2c_message = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        asyncio.run(ch.send("u2", "x"))


@pytest.mark.parametrize(
    "error",
    [RuntimeError("boom"), ValueError("bad payload")],
    ids=["runtime", "value"],
)
def test_send_swallows_api_error(error):
    ch = _channel()
    ch._client = _client()
    ch._client.api.post_c2c_message = AsyncMock(side_effect=error)
    asyncio.run(ch.send("u2", "x"))


@pytest.mark.parametrize(
    "app_id, secret",
    [("", "s"), ("a", ""), ("", "")],
    ids=["no-app-id", "no-secret", "neither"],
)
def test_start_bails_out_without_credentials(app_id, secret):
    """Missing credentials must stop before any SDK client is constructed."""
    ch = QQChannel(SimpleNamespace(app_id=app_id, secret=secret, allow_from=["*"]))
    with _receipts("ERROR") as lines:
        asyncio.run(ch.start())
    assert ch._client is None
    assert ch.is_running is False
    assert "QQ app_id and secret not configured" in "".join(lines)


def test_qq_satisfies_channel_contract():
    from codeflow.channels import Channel
    from codeflow.channels.contract import capability_violations

    ch = QQChannel(SimpleNamespace(app_id="a", secret="s"))
    assert isinstance(ch, Channel)
    assert capability_violations(ch) == []


def test_qq_spec_declares_beta_maturity():
    from codeflow.channels.adapters.qq.spec import SPEC

    assert SPEC.maturity == "beta"


def test_qq_spec_factory_raises_import_error_without_sdk(monkeypatch):
    """A missing extra must surface as ImportError so the manager can disable
    just this channel."""
    import sys

    monkeypatch.delitem(sys.modules, "codeflow.channels.adapters.qq.channel", raising=False)
    monkeypatch.setitem(sys.modules, "botpy", None)
    from codeflow.channels.adapters.qq.spec import SPEC

    with pytest.raises(ImportError):
        SPEC.factory(SimpleNamespace(app_id="a", secret="s", allow_from=["*"]))


def test_qq_spec_import_is_cheap():
    """Importing qq.spec must NOT pull in the botpy SDK (the heavy import is
    deferred into SPEC.factory)."""
    import subprocess
    import sys

    code = (
        "import sys, codeflow.channels.adapters.qq.spec as s;"
        "assert 'botpy' not in sys.modules, 'spec import pulled in the botpy SDK';"
        "assert callable(s.SPEC.factory) and s.SPEC.display_name == 'QQ'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
