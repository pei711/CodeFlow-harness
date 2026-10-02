"""Tests for SessionManager."""

import json
import multiprocessing
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from codeflow.session.manager import Session, SessionManager, new_chat_id
from codeflow.utils.atomic_io import StorageCorruptionError


def _turn_worker(workspace_str: str, key: str, writer_id: int) -> None:
    mgr = SessionManager(Path(workspace_str))
    session = mgr.get_or_create(key)
    session.add_message("user", f"q-{writer_id}")
    session.add_message("assistant", f"a-{writer_id}")
    mgr.save(session)


def _late_save_worker(
    workspace_str: str,
    key: str,
    ready: Any,
    proceed: Any,
    result: Any,
) -> None:
    mgr = SessionManager(Path(workspace_str))
    session = mgr.get_or_create(key)
    ready.set()
    if not proceed.wait(timeout=30):
        result.put("timeout")
        return
    session.add_message("assistant", "late response")
    try:
        mgr.save(session)
    except FileNotFoundError:
        result.put("deleted")
    else:
        result.put("recreated")


def test_new_chat_id_shape():
    """A minted chat_id matches the opaque sortable form YYYYMMDD_HHMMSS_xxxxxx."""
    cid = new_chat_id()
    assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", cid), cid


def test_new_chat_id_sortable_by_time():
    """Lexicographic order of chat_ids matches chronological mint order."""
    early = new_chat_id(now=datetime(2026, 6, 10, 14, 30, 52))
    late = new_chat_id(now=datetime(2026, 6, 10, 14, 30, 53))
    assert early < late


def test_new_chat_id_unique_same_second():
    """Two chat_ids minted in the same second still differ (uuid suffix)."""
    now = datetime(2026, 6, 10, 14, 30, 52)
    assert new_chat_id(now=now) != new_chat_id(now=now)


def test_save_writes_nested_channel_path(tmp_path: Path):
    """A saved session lands at sessions/{channel}/{chat_id}.jsonl."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:20260610_143052_a1b2c3")
    session.add_message("user", "hi")
    mgr.save(session)
    assert (tmp_path / "sessions" / "tui" / "20260610_143052_a1b2c3.jsonl").exists()


def test_key_from_path_reverses_nested_encoding(tmp_path: Path):
    """key_from_path maps sessions/{channel}/{chat_id}.jsonl back to
    channel:chat_id; a chat_id containing an underscore is preserved verbatim."""
    path = tmp_path / "sessions" / "telegram" / "user_42.jsonl"
    assert SessionManager.key_from_path(path) == "telegram:user_42"


def test_deterministic_chat_id_maps_uniformly(tmp_path: Path):
    """Deterministic chat_ids (cron:x) use the same nested rule as minted ones."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("cron:morning_brief")
    session.add_message("user", "ping")
    mgr.save(session)
    assert (tmp_path / "sessions" / "cron" / "morning_brief.jsonl").exists()


def test_roundtrip_load_from_nested_path(tmp_path: Path):
    """A fresh manager loads a saved session back from the nested path."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:abc123")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    mgr.save(session)

    loaded = SessionManager(tmp_path).get_or_create("tui:abc123")
    assert [m["content"] for m in loaded.messages] == ["hello", "world"]


def test_record_stamps_timestamp_only(tmp_path: Path):
    """record() stamps a per-message timestamp and carries neither the
    dropped per-message received_at nor turn_id."""
    session = Session(key="tui:t1")
    session.add_message("user", "q1")
    session.add_message("assistant", "a1")
    session.add_message("tool", "r1")

    for m in session.messages:
        assert m["timestamp"]
        assert "received_at" not in m
        assert "turn_id" not in m


def test_get_history_preserves_reasoning_content_for_provider_replay():
    session = Session(key="tui:t1")
    session.record({"role": "user", "content": "inspect the repository"})
    session.record(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call12345"}],
            "reasoning_content": "I should inspect the repository first.",
        }
    )

    assert session.get_history()[1]["reasoning_content"] == "I should inspect the repository first."


def test_get_history_preserves_thinking_blocks_for_provider_replay():
    session = Session(key="tui:t1")
    session.record({"role": "user", "content": "inspect the repository"})
    thinking_blocks = [
        {"type": "thinking", "thinking": "I should inspect the repository first.", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]
    session.record(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "toolu_123"}],
            "thinking_blocks": thinking_blocks,
        }
    )

    assert session.get_history()[1]["thinking_blocks"] == thinking_blocks


def test_save_reserves_metadata_keys(tmp_path: Path):
    """Metadata reserves source/channel/chat_id/title/parent_session_id."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:meta01")
    session.add_message("user", "x")
    mgr.save(session)

    first_line = (tmp_path / "sessions" / "tui" / "meta01.jsonl").read_text(encoding="utf-8").splitlines()[0]
    meta = json.loads(first_line)["metadata"]
    assert meta["channel"] == "tui"
    assert meta["chat_id"] == "meta01"
    assert meta["parent_session_id"] is None
    assert "source" in meta
    assert "title" in meta


def test_load_preserves_on_disk_message_order(tmp_path: Path):
    """Messages keep file order on load even when received_at is out of order."""
    session_dir = tmp_path / "sessions" / "tui"
    session_dir.mkdir(parents=True)
    lines = [
        {"_type": "metadata", "key": "tui:order01", "metadata": {}},
        {"role": "user", "content": "late", "received_at": "2026-06-10T10:00:05"},
        {"role": "user", "content": "early", "received_at": "2026-06-10T10:00:01"},
    ]
    (session_dir / "order01.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    loaded = SessionManager(tmp_path).get_or_create("tui:order01")
    assert [m["content"] for m in loaded.messages] == ["late", "early"]


def test_created_session_is_lazy_until_first_save(tmp_path: Path):
    """get_or_create materializes no file; the session is absent from list."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:lazy01")
    assert not (tmp_path / "sessions" / "tui" / "lazy01.jsonl").exists()
    assert mgr.list_sessions() == []

    session.add_message("user", "first")
    mgr.save(session)
    assert (tmp_path / "sessions" / "tui" / "lazy01.jsonl").exists()
    assert [info["key"] for info in mgr.list_sessions()] == ["tui:lazy01"]


def test_list_sessions_sees_nested_layout(tmp_path: Path):
    """list_sessions enumerates nested per-channel files."""
    mgr = SessionManager(tmp_path)
    for key in ("tui:s1", "cli:s2"):
        session = mgr.get_or_create(key)
        session.add_message("user", "x")
        mgr.save(session)

    keys = {info["key"] for info in mgr.list_sessions()}
    assert keys == {"tui:s1", "cli:s2"}


def _seed_nested(tmp_path: Path, channel: str, chat_id: str, updated_at: str) -> Path:
    channel_dir = tmp_path / "sessions" / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    path = channel_dir / f"{chat_id}.jsonl"
    meta = {
        "_type": "metadata",
        "key": f"{channel}:{chat_id}",
        "updated_at": updated_at,
        "metadata": {},
    }
    path.write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return path


def test_find_most_recent_chat_id_nested_by_updated_at(tmp_path: Path):
    """Returns the chat_id with the newest updated_at on the channel."""
    _seed_nested(tmp_path, "tui", "older", "2026-06-10T10:00:00")
    _seed_nested(tmp_path, "tui", "newer", "2026-06-10T11:00:00")
    _seed_nested(tmp_path, "cli", "distractor", "2026-06-10T12:00:00")

    mgr = SessionManager(tmp_path)
    assert mgr.find_most_recent_chat_id("tui") == "newer"
    assert mgr.find_most_recent_chat_id("cli") == "distractor"
    assert mgr.find_most_recent_chat_id("feishu") is None


def test_session_discovery_skips_malformed_metadata_without_poisoning_valid_sessions(tmp_path: Path):
    _seed_nested(tmp_path, "tui", "valid", "2026-06-10T11:00:00")
    channel_dir = tmp_path / "sessions" / "tui"
    malformed_records = {
        "bad_updated": {
            "_type": "metadata",
            "key": "tui:bad_updated",
            "updated_at": 7,
            "metadata": {},
        },
        "bad_key": {
            "_type": "metadata",
            "key": 7,
            "updated_at": "2099-01-01T00:00:00",
            "metadata": {},
        },
        "wrong_identity": {
            "_type": "metadata",
            "key": "tui:other_identity",
            "updated_at": "2099-01-01T00:00:00",
            "metadata": {},
        },
    }
    for filename, record in malformed_records.items():
        (channel_dir / f"{filename}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    manager = SessionManager(tmp_path)

    assert [item["key"] for item in manager.list_sessions()] == ["tui:valid"]
    assert manager.find_most_recent_chat_id("tui") == "valid"
    assert manager.resolve_key("bad").status == "not_found"


def test_find_most_recent_ignores_old_flat_files(tmp_path: Path):
    """Pre-refactor flat files are ignored for lookup but never deleted."""
    _seed_nested(tmp_path, "tui", "nested01", "2026-06-10T10:00:00")
    flat = tmp_path / "sessions" / "tui_flat01.jsonl"
    flat.write_text(
        json.dumps(
            {
                "_type": "metadata",
                "key": "tui:flat01",
                "updated_at": "2026-06-10T23:59:59",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mgr = SessionManager(tmp_path)
    assert mgr.find_most_recent_chat_id("tui") == "nested01"
    assert flat.exists()
    assert "tui:flat01" not in {info["key"] for info in mgr.list_sessions()}


def test_save_appends_instead_of_rewriting(tmp_path: Path):
    """A later save appends the new turn; earlier bytes stay untouched."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:app01")
    session.add_message("user", "q1")
    mgr.save(session)
    path = tmp_path / "sessions" / "tui" / "app01.jsonl"
    first_save = path.read_text(encoding="utf-8")

    session.add_message("assistant", "a1")
    mgr.save(session)
    assert path.read_text(encoding="utf-8").startswith(first_save)

    loaded = SessionManager(tmp_path).get_or_create("tui:app01")
    assert [m["content"] for m in loaded.messages] == ["q1", "a1"]


def test_no_lock_sidecar_beside_session_jsonl(tmp_path: Path):
    """After a save, the channel dir holds the transcript only; the flock
    sidecar lives in a hidden .lock/ subdir and never clutters the listing."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:lock01")
    session.add_message("user", "x")
    mgr.save(session)

    channel_dir = tmp_path / "sessions" / "tui"
    beside = [p.name for p in channel_dir.iterdir() if p.is_file() and p.name.endswith(".lock")]
    assert beside == []
    assert (channel_dir / ".lock" / "lock01.jsonl.lock").exists()
    assert [info["key"] for info in mgr.list_sessions()] == ["tui:lock01"]


def test_clear_rewrites_file(tmp_path: Path):
    """clear() + save truncates the transcript on disk (atomic replace)."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:clr01")
    session.add_message("user", "q1")
    mgr.save(session)

    session.clear()
    mgr.save(session)
    loaded = SessionManager(tmp_path).get_or_create("tui:clr01")
    assert loaded.messages == []

    session.add_message("user", "q2")
    mgr.save(session)
    loaded = SessionManager(tmp_path).get_or_create("tui:clr01")
    assert [m["content"] for m in loaded.messages] == ["q2"]


def test_clear_advances_epoch_and_rejects_stale_manager(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:clear_epoch")
    session.add_message("user", "before clear")
    owner.save(session)

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)

    session.clear()
    owner.save(session)
    assert session._storage_epoch == 1

    stale.add_message("assistant", "late stale response")
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        stale_manager.save(stale)

    loaded = SessionManager(tmp_path).get_or_create(session.key)
    assert loaded.messages == []


def test_lazy_history_rewrite_fences_stale_manager_without_materializing_file(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:lazy_rewrite")
    session.add_message("user", "owner history")

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)
    stale.add_message("user", "stale history")

    session.clear()
    owner.commit_history_rewrite(session)

    path = tmp_path / "sessions" / "tui" / "lazy_rewrite.jsonl"
    assert not path.exists()
    assert session._storage_epoch == 1
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        stale_manager.save(stale)


def test_persisted_empty_history_rewrite_fences_stale_manager(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:empty_rewrite")
    owner.save(session)

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)
    stale.add_message("user", "stale history")

    session.clear()
    owner.commit_history_rewrite(session)

    assert session._storage_epoch == 1
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        stale_manager.save(stale)
    assert SessionManager(tmp_path).get_or_create(session.key).messages == []


def test_history_rewrite_rejects_messages_appended_after_snapshot(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:rewrite_cas")
    session.add_message("user", "q1")
    session.add_message("assistant", "a1")
    owner.save(session)

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)
    peer_manager = SessionManager(tmp_path)
    peer = peer_manager.get_or_create(session.key)
    peer.add_message("user", "q2")
    peer.add_message("assistant", "a2")
    peer_manager.save(peer)

    stale.undo_last_turn()
    with pytest.raises(FileNotFoundError, match="changed before rewrite"):
        stale_manager.commit_history_rewrite(stale)

    loaded = SessionManager(tmp_path).get_or_create(session.key)
    assert [message["content"] for message in loaded.messages] == ["q1", "a1", "q2", "a2"]


def test_concurrent_writers_lose_no_turns(tmp_path: Path):
    """Two processes saving the same session: both turn blocks land,
    each block's messages contiguous (tool_call/result adjacency)."""
    key = "tui:race01"
    procs = [multiprocessing.Process(target=_turn_worker, args=(str(tmp_path), key, w)) for w in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    loaded = SessionManager(tmp_path).get_or_create(key)
    contents = [m["content"] for m in loaded.messages]
    assert sorted(contents) == ["a-0", "a-1", "q-0", "q-1"]
    for writer_id in (0, 1):
        q_idx = contents.index(f"q-{writer_id}")
        assert contents[q_idx + 1] == f"a-{writer_id}"


def test_find_most_recent_reflects_latest_append(tmp_path: Path):
    """Recency follows the LAST metadata record, not the first line."""
    mgr = SessionManager(tmp_path)
    first = mgr.get_or_create("tui:first")
    first.add_message("user", "x")
    mgr.save(first)
    second = mgr.get_or_create("tui:second")
    second.add_message("user", "y")
    mgr.save(second)

    first.add_message("user", "z")
    mgr.save(first)
    assert mgr.find_most_recent_chat_id("tui") == "first"


def test_loader_skips_partial_trailing_line(tmp_path: Path):
    """A crash mid-append leaves a partial trailing line; loader skips it."""
    session_dir = tmp_path / "sessions" / "tui"
    session_dir.mkdir(parents=True)
    full = json.dumps({"role": "user", "content": "full"})
    (session_dir / "crash01.jsonl").write_text(
        json.dumps({"_type": "metadata", "key": "tui:crash01", "metadata": {}})
        + "\n"
        + full
        + "\n"
        + '{"role": "assistant", "content": "tru',
        encoding="utf-8",
    )

    loaded = SessionManager(tmp_path).get_or_create("tui:crash01")
    assert [m["content"] for m in loaded.messages] == ["full"]


def test_save_after_partial_trailing_line_rewrites_clean_transcript(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:partial_repair")
    session.add_message("user", "q1")
    manager.save(session)
    path = tmp_path / "sessions" / "tui" / "partial_repair.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write('{"role":"assistant","content":"partial')

    reloaded_manager = SessionManager(tmp_path)
    reloaded = reloaded_manager.get_or_create(session.key)
    reloaded.add_message("assistant", "a1")
    reloaded_manager.save(reloaded)

    durable = SessionManager(tmp_path).get_or_create(session.key)
    assert [message["content"] for message in durable.messages] == ["q1", "a1"]
    assert '"content":"partial' not in path.read_text(encoding="utf-8")


def test_stale_append_rejects_recoverable_partial_trailing_line(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:partial_stale_append")
    session.add_message("user", "q1")
    owner.save(session)

    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)
    path = tmp_path / "sessions" / "tui" / "partial_stale_append.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write('{"role":"assistant","content":"partial')
    damaged_bytes = path.read_bytes()

    stale.add_message("assistant", "a1")
    with pytest.raises(FileNotFoundError, match="partial trailing record"):
        stale_manager.save(stale)

    assert path.read_bytes() == damaged_bytes
    repair_manager = SessionManager(tmp_path)
    repair = repair_manager.get_or_create(session.key)
    assert [message["content"] for message in repair.messages] == ["q1"]
    repair.add_message("assistant", "a1")
    repair_manager.save(repair)
    assert [message["content"] for message in SessionManager(tmp_path).get_or_create(session.key).messages] == [
        "q1",
        "a1",
    ]


def test_incomplete_multibyte_tail_is_loaded_and_repaired(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:partial_multibyte")
    session.add_message("user", "q1")
    owner.save(session)
    path = tmp_path / "sessions" / "tui" / "partial_multibyte.jsonl"
    with path.open("ab") as file:
        file.write(b'{"role":"assistant","content":"\xe4')

    repair_manager = SessionManager(tmp_path)
    repair = repair_manager.get_or_create(session.key)
    assert [message["content"] for message in repair.messages] == ["q1"]
    repair.add_message("assistant", "a1")
    repair_manager.save(repair)

    assert [message["content"] for message in SessionManager(tmp_path).get_or_create(session.key).messages] == [
        "q1",
        "a1",
    ]
    path.read_text(encoding="utf-8")


def test_stale_append_rejects_incomplete_multibyte_tail(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:partial_multibyte_stale")
    session.add_message("user", "q1")
    owner.save(session)
    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)
    path = tmp_path / "sessions" / "tui" / "partial_multibyte_stale.jsonl"
    with path.open("ab") as file:
        file.write(b'{"role":"assistant","content":"\xe4')
    damaged_bytes = path.read_bytes()

    stale.add_message("assistant", "a1")
    with pytest.raises(FileNotFoundError, match="partial trailing record"):
        stale_manager.save(stale)

    assert path.read_bytes() == damaged_bytes
    assert [message["content"] for message in SessionManager(tmp_path).get_or_create(session.key).messages] == ["q1"]


def test_invalid_utf8_before_eof_fails_closed(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:invalid_utf8")
    session.add_message("user", "must not disappear")
    owner.save(session)
    path = tmp_path / "sessions" / "tui" / "invalid_utf8.jsonl"
    with path.open("ab") as file:
        file.write(b"\xff\n")

    with pytest.raises(StorageCorruptionError, match="failed to load session"):
        SessionManager(tmp_path).get_or_create(session.key)


def test_corrupt_storage_epoch_is_not_minted_as_blank_session(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:corrupt_epoch")
    session.add_message("user", "must not disappear")
    manager.save(session)
    epoch_path = tmp_path / "sessions" / "tui" / ".generation" / "corrupt_epoch.jsonl.epoch"
    epoch_path.write_text("broken", encoding="ascii")

    with pytest.raises(StorageCorruptionError, match="invalid deletion epoch"):
        SessionManager(tmp_path).get_or_create(session.key)


def test_non_object_jsonl_record_is_not_minted_as_blank_session(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:corrupt_record")
    session.add_message("user", "must not disappear")
    manager.save(session)
    path = tmp_path / "sessions" / "tui" / "corrupt_record.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write("[]\n")

    with pytest.raises(StorageCorruptionError, match="record must be an object"):
        SessionManager(tmp_path).get_or_create(session.key)


def test_metadata_key_collision_fails_closed(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:a/b")
    session.add_message("user", "private history")
    manager.save(session)

    with pytest.raises(StorageCorruptionError, match="does not match requested key"):
        SessionManager(tmp_path).get_or_create("tui:a:b")


def test_metadata_less_session_only_loads_through_canonical_path_key(tmp_path: Path):
    path = tmp_path / "sessions" / "tui" / "a_b.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"role":"user","content":"legacy history"}\n', encoding="utf-8")

    canonical = SessionManager(tmp_path).get_or_create("tui:a_b")
    assert [message["content"] for message in canonical.messages] == ["legacy history"]
    for alias in ("tui:a/b", "tui:a:b"):
        with pytest.raises(StorageCorruptionError, match="does not match canonical path key"):
            SessionManager(tmp_path).get_or_create(alias)


def test_identityless_metadata_record_fails_closed(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:identityless_metadata")
    session.add_message("user", "private history")
    manager.save(session)
    path = tmp_path / "sessions" / "tui" / "identityless_metadata.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"_type": "metadata", "metadata": {"title": "override"}}) + "\n")

    with pytest.raises(StorageCorruptionError, match="metadata key must be a string"):
        SessionManager(tmp_path).get_or_create(session.key)


def test_corrupt_identity_line_cannot_fall_back_to_canonical_alias(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:a/b")
    session.add_message("user", "private history")
    manager.save(session)
    path = tmp_path / "sessions" / "tui" / "a_b.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        '{"_type":"metadata","key":"tui:a/b"\n' + "\n".join(lines[1:]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StorageCorruptionError, match="invalid JSON record"):
        SessionManager(tmp_path).get_or_create("tui:a_b")


def test_colliding_lazy_key_is_rejected_before_first_append(tmp_path: Path):
    first_manager = SessionManager(tmp_path)
    second_manager = SessionManager(tmp_path)
    first = first_manager.get_or_create("tui:a/b")
    second = second_manager.get_or_create("tui:a:b")
    first.add_message("user", "first identity")
    second.add_message("user", "colliding identity")

    first_manager.save(first)
    path = tmp_path / "sessions" / "tui" / "a_b.jsonl"
    original = path.read_bytes()

    with pytest.raises(StorageCorruptionError, match="does not match requested key"):
        second_manager.save(second)

    assert path.read_bytes() == original
    loaded = SessionManager(tmp_path).get_or_create(first.key)
    assert [message["content"] for message in loaded.messages] == ["first identity"]


def test_stale_append_does_not_overwrite_pending_clarification(tmp_path: Path):
    owner = SessionManager(tmp_path)
    session = owner.get_or_create("tui:metadata_cas")
    session.add_message("user", "deploy")
    owner.save(session)

    setter_manager = SessionManager(tmp_path)
    setter = setter_manager.get_or_create(session.key)
    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(session.key)
    pending = {
        "original_message": "deploy",
        "question": "which environment?",
        "domain": "deployment",
    }
    setter.pending_clarification = pending
    setter_manager.save(setter)

    stale.add_message("assistant", "stale response")
    with pytest.raises(FileNotFoundError, match="metadata changed before append"):
        stale_manager.save(stale)

    loaded = SessionManager(tmp_path).get_or_create(session.key)
    assert loaded.pending_clarification == pending
    assert [message["content"] for message in loaded.messages] == ["deploy"]


def test_session_read_error_is_not_minted_as_blank_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:read_error")
    session.add_message("user", "must not disappear")
    manager.save(session)

    def _raise_read_error(_path: Path):
        raise OSError("device unavailable")

    monkeypatch.setattr("codeflow.session.manager.locked_read", _raise_read_error)
    with pytest.raises(StorageCorruptionError, match="failed to load session"):
        SessionManager(tmp_path).get_or_create(session.key)


def test_missing_epoch_zero_primary_file_fails_closed(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:missing_primary")
    session.add_message("user", "must not disappear")
    manager.save(session)
    path = tmp_path / "sessions" / "tui" / "missing_primary.jsonl"
    path.unlink()

    with pytest.raises(StorageCorruptionError, match="known generation"):
        SessionManager(tmp_path).get_or_create(session.key)


def test_stale_lazy_writer_cannot_recreate_missing_epoch_zero_primary(tmp_path: Path):
    key = "tui:missing_primary_stale_writer"
    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(key)

    owner_manager = SessionManager(tmp_path)
    owner = owner_manager.get_or_create(key)
    owner.add_message("user", "must not disappear")
    owner_manager.save(owner)
    path = tmp_path / "sessions" / "tui" / "missing_primary_stale_writer.jsonl"
    path.unlink()

    stale.add_message("user", "must not replace lost history")
    with pytest.raises(StorageCorruptionError, match="known generation is missing primary file"):
        stale_manager.save(stale)

    assert not path.exists()


def test_lazy_session_generation_comes_from_locked_load_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from codeflow.session import manager as manager_module

    monkeypatch.setattr(manager_module, "locked_read", lambda _path: (None, 4, True))

    def fail_naked_epoch_read(_path: Path) -> int:
        raise AssertionError("generation must come from locked_read")

    monkeypatch.setattr(manager_module, "read_epoch", fail_naked_epoch_read)

    session = SessionManager(tmp_path).get_or_create("tui:deleted_generation")

    assert session._storage_epoch == 4


def test_lazy_undo_persists_retained_metadata_when_all_messages_are_removed(tmp_path: Path):
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:lazy_undo_metadata")
    session.metadata["title"] = "Retained title"
    session.add_message("user", "remove me")
    session.add_message("assistant", "remove me too")

    assert session.undo_last_turn() == 2
    manager.commit_history_rewrite(session)

    loaded = SessionManager(tmp_path).peek(session.key)
    assert loaded is not None
    assert loaded.messages == []
    assert loaded.metadata["title"] == "Retained title"


def test_legacy_global_sessions_shim_removed(tmp_path: Path, monkeypatch):
    """Files outside the configured Session root are not consulted."""
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_file = legacy / "tui_x.jsonl"
    legacy_file.write_text(
        json.dumps({"_type": "metadata", "key": "tui:x"})
        + "\n"
        + json.dumps({"role": "user", "content": "old"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "codeflow.session.manager.get_legacy_sessions_dir",
        lambda: legacy,
        raising=False,
    )

    session = SessionManager(tmp_path / "ws").get_or_create("tui:x")
    assert session.messages == []
    assert legacy_file.exists()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_delete_removes_file_and_returns_true(tmp_path: Path):
    """delete() removes the JSONL file and returns True."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:del01")
    session.add_message("user", "hi")
    mgr.save(session)
    path = tmp_path / "sessions" / "tui" / "del01.jsonl"
    assert path.exists()

    result = mgr.delete("tui:del01")
    assert result is True
    assert not path.exists()


def test_delete_invalidates_cache(tmp_path: Path):
    """delete() removes the key from the in-memory cache."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:del02")
    session.add_message("user", "x")
    mgr.save(session)
    assert "tui:del02" in mgr._cache

    mgr.delete("tui:del02")
    assert "tui:del02" not in mgr._cache


def test_delete_unknown_key_returns_false(tmp_path: Path):
    """delete() on a key with no file returns False without error."""
    mgr = SessionManager(tmp_path)
    result = mgr.delete("tui:nonexistent_del")
    assert result is False
    assert not (tmp_path / "sessions" / "tui" / ".generation" / "nonexistent_del.jsonl.epoch").exists()


def test_delete_cached_lazy_session_fences_stale_reference(tmp_path: Path):
    """Deleting a known lazy Session prevents an external reference from saving it."""
    mgr = SessionManager(tmp_path)
    stale = mgr.get_or_create("tui:lazy_deleted")

    assert mgr.delete(stale.key) is False

    stale.add_message("user", "late")
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        mgr.save(stale)


def test_delete_returns_false_when_unlink_fails(tmp_path: Path, monkeypatch):
    """delete() returns False when removal raises — True only if a file was removed."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:del04")
    session.add_message("user", "x")
    mgr.save(session)

    def _boom_unlink(self, missing_ok=False):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _boom_unlink)
    assert mgr.delete("tui:del04") is False


def test_failed_delete_retains_cached_unsaved_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:delete_failure_tail")
    session.add_message("user", "persisted")
    mgr.save(session)
    session.add_message("assistant", "unsaved tail")

    def _boom_unlink(self, missing_ok=False):
        raise OSError("permission denied")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", _boom_unlink)
        assert mgr.delete(session.key) is False

    assert mgr.peek(session.key) is session
    assert [message["content"] for message in session.messages] == [
        "persisted",
        "unsaved tail",
    ]
    assert mgr.flush(session.key) is True
    loaded = SessionManager(tmp_path).get_or_create(session.key)
    assert [message["content"] for message in loaded.messages] == [
        "persisted",
        "unsaved tail",
    ]


def test_delete_does_not_touch_other_sessions(tmp_path: Path):
    """delete() only removes the targeted session file."""
    mgr = SessionManager(tmp_path)
    for key in ("tui:keep01", "tui:del03"):
        s = mgr.get_or_create(key)
        s.add_message("user", "y")
        mgr.save(s)

    mgr.delete("tui:del03")
    assert (tmp_path / "sessions" / "tui" / "keep01.jsonl").exists()
    assert not (tmp_path / "sessions" / "tui" / "del03.jsonl").exists()


def test_saved_session_cannot_recreate_file_after_other_manager_deletes_it(tmp_path: Path):
    """A stale persisted Session cannot resurrect a transcript after deletion."""
    writer = SessionManager(tmp_path)
    session = writer.get_or_create("tui:deleted_then_saved")
    session.add_message("user", "before deletion")
    writer.save(session)
    path = tmp_path / "sessions" / "tui" / "deleted_then_saved.jsonl"

    assert SessionManager(tmp_path).delete(session.key) is True
    session.add_message("assistant", "late response")

    with pytest.raises(FileNotFoundError, match="deleted"):
        writer.save(session)
    assert not path.exists()


def test_empty_saved_session_cannot_recreate_file_after_delete(tmp_path: Path):
    """Persistence state is distinct from the persisted message count."""
    writer = SessionManager(tmp_path)
    session = writer.get_or_create("tui:empty_deleted")
    writer.save(session)
    path = tmp_path / "sessions" / "tui" / "empty_deleted.jsonl"

    assert SessionManager(tmp_path).delete(session.key) is True

    with pytest.raises(FileNotFoundError, match="deleted"):
        writer.save(session)
    assert not path.exists()


def test_cleared_saved_session_cannot_replace_file_after_delete(tmp_path: Path):
    """The rewrite path obeys the same deletion barrier as append."""
    writer = SessionManager(tmp_path)
    session = writer.get_or_create("tui:cleared_deleted")
    session.add_message("user", "before deletion")
    writer.save(session)
    path = tmp_path / "sessions" / "tui" / "cleared_deleted.jsonl"

    assert SessionManager(tmp_path).delete(session.key) is True
    session.clear()

    with pytest.raises(FileNotFoundError, match="deleted"):
        writer.save(session)
    assert not path.exists()


def test_cross_process_late_save_cannot_recreate_deleted_session(tmp_path: Path):
    """A process holding a loaded Session cannot save it after peer deletion."""
    key = "tui:cross_process_delete"
    seed = SessionManager(tmp_path)
    session = seed.get_or_create(key)
    session.add_message("user", "before deletion")
    seed.save(session)
    path = tmp_path / "sessions" / "tui" / "cross_process_delete.jsonl"

    ready = multiprocessing.Event()
    proceed = multiprocessing.Event()
    result = multiprocessing.Queue()
    writer = multiprocessing.Process(
        target=_late_save_worker,
        args=(str(tmp_path), key, ready, proceed, result),
    )
    writer.start()
    try:
        assert ready.wait(timeout=30)
        assert SessionManager(tmp_path).delete(key) is True
        proceed.set()
        writer.join(timeout=30)

        assert writer.exitcode == 0
        assert result.get(timeout=5) == "deleted"
        assert not path.exists()
    finally:
        proceed.set()
        if writer.is_alive():
            writer.terminate()
        writer.join(timeout=5)
        result.close()
        result.join_thread()


def test_fresh_session_can_reuse_key_after_prior_session_is_deleted(tmp_path: Path):
    """The deletion barrier belongs to the stale object, not the session key."""
    first = SessionManager(tmp_path)
    old = first.get_or_create("tui:reuse_after_delete")
    old.add_message("user", "old")
    first.save(old)
    assert first.delete(old.key) is True

    second = SessionManager(tmp_path)
    new = second.get_or_create(old.key)
    new.add_message("user", "new")
    second.save(new)

    loaded = SessionManager(tmp_path).get_or_create(old.key)
    assert [message["content"] for message in loaded.messages] == ["new"]


def test_stale_session_cannot_append_after_deleted_key_is_reused(tmp_path: Path):
    """A recreated key has a new incarnation that rejects the old Session."""
    old_manager = SessionManager(tmp_path)
    old = old_manager.get_or_create("tui:reuse_with_stale_writer")
    old.add_message("user", "old")
    old_manager.save(old)

    assert SessionManager(tmp_path).delete(old.key) is True

    new_manager = SessionManager(tmp_path)
    new = new_manager.get_or_create(old.key)
    new.add_message("user", "new")
    new_manager.save(new)

    old.add_message("assistant", "late old assistant")
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        old_manager.save(old)

    loaded = SessionManager(tmp_path).get_or_create(old.key)
    assert [message["content"] for message in loaded.messages] == ["new"]


def test_lazy_session_cannot_append_after_key_is_created_deleted_and_reused(tmp_path: Path):
    """A never-saved Session still observes deletion epochs."""
    key = "tui:lazy_reuse_with_stale_writer"
    stale_manager = SessionManager(tmp_path)
    stale = stale_manager.get_or_create(key)

    first_manager = SessionManager(tmp_path)
    first = first_manager.get_or_create(key)
    first.add_message("user", "first")
    first_manager.save(first)
    assert first_manager.delete(key) is True

    new_manager = SessionManager(tmp_path)
    new = new_manager.get_or_create(key)
    new.add_message("user", "new")
    new_manager.save(new)

    stale.add_message("assistant", "late unsaved writer")
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        stale_manager.save(stale)

    loaded = SessionManager(tmp_path).get_or_create(key)
    assert [message["content"] for message in loaded.messages] == ["new"]


def test_read_only_session_directory_can_still_be_loaded(tmp_path: Path):
    """Read-only access does not require creating lock or epoch sidecars."""
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:readonly")
    session.add_message("user", "read me")
    manager.save(session)
    channel_dir = tmp_path / "sessions" / "tui"

    for hidden in (channel_dir / ".lock", channel_dir / ".generation"):
        if hidden.exists():
            for child in hidden.iterdir():
                child.unlink()
            hidden.rmdir()
    channel_dir.chmod(0o555)

    try:
        loaded = SessionManager(tmp_path).peek("tui:readonly")
    finally:
        channel_dir.chmod(0o755)

    assert loaded is not None
    assert [message["content"] for message in loaded.messages] == ["read me"]


def test_read_only_load_retries_when_deletion_epoch_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lockless read-only fallback never pairs bytes with a different epoch."""
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("tui:readonly_epoch")
    session.add_message("user", "read consistently")
    manager.save(session)

    from codeflow.session import manager as manager_module

    def deny_lock(_path: Path):
        raise PermissionError("read-only directory")

    epochs = iter((0, 1, 1, 1))
    monkeypatch.setattr(manager_module, "locked_read", deny_lock)
    monkeypatch.setattr(manager_module, "read_epoch", lambda _path: next(epochs))

    loaded = SessionManager(tmp_path).peek("tui:readonly_epoch")

    assert loaded is not None
    assert loaded._storage_epoch == 1
    assert [message["content"] for message in loaded.messages] == ["read consistently"]


def test_peek_returns_cached_session_without_extra_load(tmp_path: Path):
    """peek() returns the cached Session when already in memory."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:peek01")
    session.add_message("user", "peek test")
    mgr.save(session)

    peeked = mgr.peek("tui:peek01")
    assert peeked is session


def test_peek_loads_from_disk_without_caching(tmp_path: Path):
    """peek() loads from disk for unknown keys but does not add to cache."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:peek02")
    session.add_message("user", "disk message")
    mgr.save(session)

    fresh_mgr = SessionManager(tmp_path)
    peeked = fresh_mgr.peek("tui:peek02")
    assert peeked is not None
    assert peeked.messages[0]["content"] == "disk message"
    assert "tui:peek02" not in fresh_mgr._cache


def test_peek_returns_none_for_unknown_key(tmp_path: Path):
    """peek() returns None for a key that has no file and is not cached."""
    mgr = SessionManager(tmp_path)
    assert mgr.peek("tui:ghost") is None


def test_flush_saves_dirty_session(tmp_path: Path):
    """flush() persists a session with unpersisted messages and returns True."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:flush01")
    session.add_message("user", "first")
    mgr.save(session)
    session.add_message("assistant", "second")
    assert session._persisted_count == 1

    assert mgr.flush("tui:flush01") is True

    path = tmp_path / "sessions" / "tui" / "flush01.jsonl"
    lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    msg_lines = [ln for ln in lines if ln.get("_type") != "metadata"]
    assert len(msg_lines) == 2


def test_flush_persists_shrunk_session(tmp_path: Path):
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:flush_shrink")
    session.add_message("user", "first turn")
    session.add_message("assistant", "first answer")
    session.add_message("user", "second turn")
    session.add_message("assistant", "second answer")
    mgr.save(session)

    assert session.undo_last_turn() == 2
    assert mgr.flush(session.key) is True

    loaded = SessionManager(tmp_path).get_or_create(session.key)
    assert [message["content"] for message in loaded.messages] == [
        "first turn",
        "first answer",
    ]


def test_flush_persists_equal_length_message_mutation(tmp_path: Path):
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:flush_mutation")
    session.add_message("user", "before")
    mgr.save(session)

    session.messages[0]["content"] = "after"
    assert mgr.flush(session.key) is True

    loaded = SessionManager(tmp_path).get_or_create(session.key)
    assert [message["content"] for message in loaded.messages] == ["after"]


def test_flush_skips_clean_session(tmp_path: Path):
    """flush() does not rewrite a clean session and returns True."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:flush02")
    session.add_message("user", "saved")
    mgr.save(session)
    path = tmp_path / "sessions" / "tui" / "flush02.jsonl"
    before = path.read_text()

    assert mgr.flush("tui:flush02") is True
    assert path.read_text() == before


def test_flush_does_nothing_for_uncached_key(tmp_path: Path):
    """flush() is a no-op for an uncached key and returns True."""
    mgr = SessionManager(tmp_path)
    assert mgr.flush("tui:not_in_cache") is True


def test_flush_returns_false_when_save_fails(tmp_path: Path, monkeypatch):
    """flush() swallows a save failure and returns False."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:flush03")
    session.add_message("user", "dirty")

    def _boom_save(s) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(mgr, "save", _boom_save)
    assert mgr.flush("tui:flush03") is False


def test_exists_true_for_saved_session(tmp_path: Path):
    """exists() is True once the session file is on disk."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:ex01")
    session.add_message("user", "x")
    mgr.save(session)
    assert mgr.exists("tui:ex01") is True


def test_exists_false_for_lazy_or_unknown_session(tmp_path: Path):
    """exists() is False for a lazy (never-saved) or unknown key."""
    mgr = SessionManager(tmp_path)
    mgr.get_or_create("tui:ex02")
    assert mgr.exists("tui:ex02") is False
    assert mgr.exists("tui:ghost") is False


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------


def test_list_sessions_channel_filter(tmp_path: Path):
    """list_sessions(channel='tui') returns only tui sessions."""
    mgr = SessionManager(tmp_path)
    for key in ("tui:ch01", "cli:ch02", "tui:ch03"):
        s = mgr.get_or_create(key)
        s.add_message("user", "x")
        mgr.save(s)

    tui_sessions = mgr.list_sessions(channel="tui")
    keys = {info["key"] for info in tui_sessions}
    assert keys == {"tui:ch01", "tui:ch03"}


def test_list_sessions_no_channel_returns_all(tmp_path: Path):
    """list_sessions() with no filter returns all channels (backward compat)."""
    mgr = SessionManager(tmp_path)
    for key in ("tui:all01", "cli:all02"):
        s = mgr.get_or_create(key)
        s.add_message("user", "x")
        mgr.save(s)

    assert len(mgr.list_sessions()) == 2


def test_list_sessions_includes_message_count(tmp_path: Path):
    """list_sessions entries include message_count matching the stored messages."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:mc01")
    for i in range(3):
        session.add_message("user", f"msg{i}")
    mgr.save(session)

    entries = mgr.list_sessions()
    assert len(entries) == 1
    assert entries[0]["message_count"] == 3


def test_list_sessions_message_count_excludes_metadata_lines(tmp_path: Path):
    """message_count counts only message lines, not metadata records."""
    mgr = SessionManager(tmp_path)
    session = mgr.get_or_create("tui:mc02")
    session.add_message("user", "one")
    mgr.save(session)
    session.add_message("assistant", "two")
    mgr.save(session)

    entries = mgr.list_sessions()
    assert entries[0]["message_count"] == 2


def _msg(role, content):
    return {"role": role, "content": content}


def test_undo_last_turn_drops_last_user_block():
    s = Session(key="tui:t1")
    s.messages = [
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
        _msg("assistant", "a2"),
        _msg("tool", "t2"),
    ]
    removed = s.undo_last_turn()
    assert removed == 3
    assert [m["content"] for m in s.messages] == ["q1", "a1"]


def test_undo_last_turn_no_user_returns_zero():
    s = Session(key="tui:t1")
    s.messages = [_msg("assistant", "a1"), _msg("tool", "t1")]
    assert s.undo_last_turn() == 0
    assert len(s.messages) == 2


def test_undo_last_turn_empty_session_returns_zero():
    s = Session(key="tui:t1")
    assert s.undo_last_turn() == 0


def test_undo_last_turn_never_crosses_last_consolidated():
    s = Session(key="tui:t1")
    s.messages = [
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
        _msg("assistant", "a2"),
    ]
    s.last_consolidated = 2
    removed = s.undo_last_turn()
    assert removed == 2
    assert [m["content"] for m in s.messages] == ["q1", "a1"]
    assert s.undo_last_turn() == 0
    assert len(s.messages) == 2


def test_undo_last_turn_n_clamps_to_tail_first_user():
    s = Session(key="tui:t1")
    s.messages = [
        _msg("user", "q1"),
        _msg("assistant", "a1"),
        _msg("user", "q2"),
        _msg("assistant", "a2"),
    ]
    removed = s.undo_last_turn(n=5)
    assert removed == 4
    assert s.messages == []


def test_clear_then_save_truncates_file_on_disk(tmp_path):
    from codeflow.session.manager import SessionManager

    mgr = SessionManager(tmp_path)
    s = mgr.get_or_create("tui:keepme")
    s.record({"role": "user", "content": "q1"})
    s.record({"role": "assistant", "content": "a1"})
    mgr.save(s)
    assert mgr.exists("tui:keepme")

    s.clear()
    mgr.save(s)

    fresh = SessionManager(tmp_path)
    reloaded = fresh.get_or_create("tui:keepme")
    assert reloaded.messages == []
    assert reloaded.key == "tui:keepme"


def test_undo_then_save_truncates_file_on_disk(tmp_path):
    from codeflow.session.manager import SessionManager

    mgr = SessionManager(tmp_path)
    s = mgr.get_or_create("tui:undome")
    for role, content in [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2")]:
        s.record({"role": role, "content": content})
    mgr.save(s)

    removed = s.undo_last_turn()
    assert removed == 2
    mgr.save(s)

    fresh = SessionManager(tmp_path)
    reloaded = fresh.get_or_create("tui:undome")
    assert [m["content"] for m in reloaded.messages] == ["q1", "a1"]
    assert reloaded.key == "tui:undome"


def _seed(mgr: SessionManager, key: str, *turns: tuple[str, str]) -> Session:
    session = mgr.get_or_create(key)
    for role, content in turns:
        session.add_message(role, content)
    mgr.save(session)
    return session


def test_fork_copies_history_to_new_same_channel_session(tmp_path: Path):
    """fork mints a fresh same-channel chat_id holding a verbatim message copy."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src01", ("user", "q1"), ("assistant", "a1"))

    child = mgr.fork("cli:src01")

    assert child is not None
    assert child.key.startswith("cli:")
    assert child.key != "cli:src01"
    assert [m["content"] for m in child.messages] == ["q1", "a1"]


def test_fork_sets_parent_session_id_to_full_source_key(tmp_path: Path):
    """The child's parent_session_id is the source's full session key (composite)."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src02", ("user", "x"))

    child = mgr.fork("cli:src02")

    assert child.metadata["parent_session_id"] == "cli:src02"


def test_fork_leaves_source_unchanged(tmp_path: Path):
    """Forking does not mutate the source session on disk."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src03", ("user", "x"))

    mgr.fork("cli:src03")

    reloaded = SessionManager(tmp_path).get_or_create("cli:src03")
    assert reloaded.metadata.get("parent_session_id") is None
    assert [m["content"] for m in reloaded.messages] == ["x"]


def test_fork_child_is_persisted_immediately(tmp_path: Path):
    """fork is never lazy — the child file exists right after fork."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src04", ("user", "x"))

    child = mgr.fork("cli:src04")

    assert mgr.exists(child.key)
    loaded = SessionManager(tmp_path).get_or_create(child.key)
    assert [m["content"] for m in loaded.messages] == ["x"]


def test_fork_returns_none_when_source_flush_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = SessionManager(tmp_path)
    source = _seed(mgr, "cli:fork_flush_failure", ("user", "persisted"))
    source.add_message("assistant", "unsaved tail")

    def _boom_save(_session: Session) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(mgr, "save", _boom_save)

    assert mgr.fork(source.key) is None
    assert mgr.peek(source.key) is source
    assert [message["content"] for message in source.messages] == [
        "persisted",
        "unsaved tail",
    ]


def test_fork_child_independent_after_parent_delete(tmp_path: Path):
    """Deleting the parent leaves the child's copied history intact."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src05", ("user", "q1"), ("assistant", "a1"))
    child = mgr.fork("cli:src05")

    mgr.delete("cli:src05")

    loaded = SessionManager(tmp_path).get_or_create(child.key)
    assert [m["content"] for m in loaded.messages] == ["q1", "a1"]


def test_fork_inherits_last_consolidated(tmp_path: Path):
    """The child inherits the source's last_consolidated boundary."""
    mgr = SessionManager(tmp_path)
    src = _seed(mgr, "cli:src06", ("user", "a"), ("assistant", "b"))
    src.last_consolidated = 1
    mgr.save(src)

    child = mgr.fork("cli:src06")

    assert child.last_consolidated == 1


def test_fork_resets_pending_clarification(tmp_path: Path):
    """The child does not carry the source's clarification wait-state."""
    mgr = SessionManager(tmp_path)
    src = _seed(mgr, "cli:src07", ("user", "a"))
    src.pending_clarification = {"original_message": "a", "question": "?", "domain": "d"}
    mgr.save(src)

    child = mgr.fork("cli:src07")

    assert child.pending_clarification is None


def test_fork_refuses_missing_source(tmp_path: Path):
    """Forking a source that does not exist returns None and creates nothing."""
    mgr = SessionManager(tmp_path)
    assert mgr.fork("cli:nope") is None


def test_fork_refuses_empty_source(tmp_path: Path):
    """Forking a zero-message source (e.g. titled-only) is refused."""
    mgr = SessionManager(tmp_path)
    titled = mgr.get_or_create("cli:src08")
    titled.metadata["title"] = "empty"
    mgr.save(titled)

    assert mgr.fork("cli:src08") is None


def test_fork_deepcopies_messages(tmp_path: Path):
    """Child messages are a deepcopy — mutating the source's nested content
    block after fork does not leak into the child."""
    mgr = SessionManager(tmp_path)
    src = mgr.get_or_create("cli:src09")
    src.record({"role": "user", "content": [{"type": "text", "text": "hi"}]})
    mgr.save(src)

    child = mgr.fork("cli:src09")
    src.messages[0]["content"].append({"type": "text", "text": "MUTATED"})

    assert child.messages[0]["content"] == [{"type": "text", "text": "hi"}]


def test_fork_default_title_appends_fork_suffix(tmp_path: Path):
    """Without an explicit title, a titled parent yields '<title> (fork)'."""
    mgr = SessionManager(tmp_path)
    src = _seed(mgr, "cli:src10", ("user", "x"))
    src.metadata["title"] = "My chat"
    mgr.save(src)

    child = mgr.fork("cli:src10")

    assert child.metadata["title"] == "My chat (fork)"


def test_fork_untitled_parent_yields_no_title(tmp_path: Path):
    """An untitled parent yields a child with no title (no bare '(fork)')."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src11", ("user", "x"))

    child = mgr.fork("cli:src11")

    assert child.metadata.get("title") is None


def test_fork_explicit_title_overrides(tmp_path: Path):
    """An explicit title is used verbatim."""
    mgr = SessionManager(tmp_path)
    _seed(mgr, "cli:src12", ("user", "x"))

    child = mgr.fork("cli:src12", title="Custom")

    assert child.metadata["title"] == "Custom"


def test_resolve_key_full_key_passthrough(tmp_path: Path):
    """A value carrying ':' is treated as a full key, no lookup."""
    mgr = SessionManager(tmp_path)
    res = mgr.resolve_key("feishu:abc123")
    assert res.status == "resolved"
    assert res.key == "feishu:abc123"


def test_resolve_key_bare_exact_cross_channel(tmp_path: Path):
    """A bare chat_id resolves to its full key on whatever channel holds it."""
    mgr = SessionManager(tmp_path)
    cid = "20990101_000000_aaaaaa"
    _seed(mgr, f"tui:{cid}", ("user", "hi"))
    res = mgr.resolve_key(cid)
    assert res.status == "resolved"
    assert res.key == f"tui:{cid}"


def test_resolve_key_bare_prefix_unique(tmp_path: Path):
    """A unique prefix resolves to the single matching key."""
    mgr = SessionManager(tmp_path)
    cid = "20990101_000000_cccccc"
    _seed(mgr, f"cli:{cid}", ("user", "hi"))
    res = mgr.resolve_key(cid[:20])
    assert res.status == "resolved"
    assert res.key == f"cli:{cid}"


def test_resolve_key_ambiguous_returns_candidates(tmp_path: Path):
    """The same bare id on two channels is ambiguous; both keys surface."""
    mgr = SessionManager(tmp_path)
    cid = "20990101_000000_dddddd"
    _seed(mgr, f"cli:{cid}", ("user", "hi"))
    _seed(mgr, f"tui:{cid}", ("user", "hi"))
    res = mgr.resolve_key(cid)
    assert res.status == "ambiguous"
    assert set(res.candidates) == {f"cli:{cid}", f"tui:{cid}"}


def test_resolve_key_not_found(tmp_path: Path):
    """No match anywhere yields not_found (no minting, no fallback)."""
    mgr = SessionManager(tmp_path)
    res = mgr.resolve_key("nope000")
    assert res.status == "not_found"
    assert res.key is None
