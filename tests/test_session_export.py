"""Tests for the session transcript exporter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from codeflow.session.export import (
    default_export_path,
    render_transcript,
    verify_export,
    write_portable_export,
    write_transcript,
)
from codeflow.session.manager import Session


def _session(key: str = "tui:20260622_120000_abcdef", **kw) -> Session:
    return Session(key=key, **kw)


def test_header_reflects_metadata():
    s = _session(
        created_at=datetime(2026, 6, 22, 12, 0, 0),
        updated_at=datetime(2026, 6, 22, 12, 5, 0),
        metadata={"title": "My chat"},
    )
    s.add_message("user", "hello")
    s.add_message("assistant", "hi there")
    out = render_transcript(s)
    assert s.key in out
    assert "My chat" in out
    assert "2026-06-22" in out
    assert "2" in out


def test_user_and_assistant_headings():
    s = _session()
    s.add_message("user", "question?")
    s.add_message("assistant", "answer.")
    out = render_transcript(s)
    assert "question?" in out
    assert "answer." in out

    lower = out.lower()
    assert "user" in lower and "assistant" in lower


def test_reasoning_block_present_when_set():
    s = _session()
    s.add_message("user", "q")
    s.record(
        {
            "role": "assistant",
            "content": "final answer",
            "reasoning_content": "let me think step by step",
        }
    )
    out = render_transcript(s)
    assert "let me think step by step" in out
    assert "final answer" in out


def test_reasoning_block_absent_when_unset():
    s = _session()
    s.add_message("user", "q")
    s.add_message("assistant", "plain answer")
    out = render_transcript(s)
    assert "plain answer" in out

    assert "think" not in out.lower()


def test_tool_calls_and_results_rendered():
    s = _session()
    s.add_message("user", "read it")
    s.record(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x.txt"}'},
                }
            ],
        }
    )
    s.record(
        {
            "role": "tool",
            "content": "the file body",
            "tool_call_id": "c1",
            "name": "read_file",
        }
    )
    out = render_transcript(s)
    assert "read_file" in out
    assert "x.txt" in out
    assert "the file body" in out


def test_empty_session_renders_header_only():
    s = _session()
    out = render_transcript(s)
    assert s.key in out
    assert isinstance(out, str) and out.strip()


def test_default_export_path_is_workspace_relative(tmp_path: Path):
    p = default_export_path(tmp_path, "tui:20260622_120000_abcdef")
    assert p.parent == tmp_path / "exports"
    assert p.name.endswith(".codeflow-session.json")

    assert ":" not in p.name


def test_write_transcript_writes_and_returns_abs_path(tmp_path: Path):
    s = _session()
    s.add_message("user", "hello")
    dest = tmp_path / "transcript.md"
    written = write_transcript(s, dest)
    assert written.is_absolute()
    assert written.exists()
    assert "hello" in written.read_text(encoding="utf-8")


def test_portable_export_preserves_raw_messages_and_verifies(tmp_path: Path):
    s = _session(metadata={"parent_session_id": "tui:parent"})
    s.record(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "original"},
                {"type": "image_url", "image_url": {"url": "file:///tmp/example.png"}},
            ],
        }
    )
    dest = default_export_path(tmp_path, s.key)
    write_portable_export(s, dest)

    import json

    envelope = json.loads(dest.read_text(encoding="utf-8"))
    assert envelope["payload"]["metadata"]["parent_session_id"] == "tui:parent"
    assert envelope["payload"]["messages"] == s.messages
    assert verify_export(dest) is True


def test_portable_export_verification_rejects_modified_body(tmp_path: Path):
    s = _session()
    s.add_message("user", "original")
    dest = default_export_path(tmp_path, s.key)
    write_portable_export(s, dest)

    dest.write_text(dest.read_text(encoding="utf-8").replace("original", "tampered"), encoding="utf-8")

    assert verify_export(dest) is False


def test_write_transcript_overwrites(tmp_path: Path):
    s = _session()
    s.add_message("user", "first")
    dest = default_export_path(tmp_path, s.key)
    write_transcript(s, dest)
    s.add_message("assistant", "second")
    write_transcript(s, dest)
    body = dest.read_text(encoding="utf-8")
    assert "first" in body and "second" in body

    assert body.count("first") == 1


def test_write_transcript_creates_parent_dir(tmp_path: Path):
    s = _session()
    dest = tmp_path / "nested" / "deep" / "out.md"
    written = write_transcript(s, dest)
    assert written.exists()
