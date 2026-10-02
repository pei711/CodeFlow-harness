"""Unit tests for MemoryStore locking and active profile CAS writes."""

from __future__ import annotations

import sys
import time
from multiprocessing import Process, Value
from pathlib import Path

import pytest

from codeflow.memory_engine.consolidate.consolidator import MemoryStore


def test_locked_yields_without_throwing(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.write_long_term("hello\n")
    with store.locked():
        assert store.read_long_term() == "hello\n"


def test_lock_path_is_sibling(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    assert store.memory_lock_path.name == store.memory_file.name + ".lock"
    assert store.memory_lock_path.parent == store.memory_file.parent


def test_splice_write_returns_true_when_unchanged(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    original = "## Profile\n\nbefore\n"
    store.write_long_term(original)

    ok = store._splice_section_and_write(
        "## Profile",
        "after",
        expected_prev=original,
    )

    assert ok is True
    assert store.read_long_term() == "## Profile\n\nafter\n"


def test_splice_write_skips_when_concurrent_modification(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    original = "## Profile\n\noriginal\n"
    changed_by_other = "## Profile\n\nchanged-by-other\n"
    store.write_long_term(original)
    store.memory_file.write_text(changed_by_other, encoding="utf-8")

    ok = store._splice_section_and_write(
        "## Profile",
        "our-update",
        expected_prev=original,
    )

    assert ok is False
    assert store.read_long_term() == changed_by_other


# ---------------------------------------------------------------------------


def _worker_acquire_and_hold(lock_dir: str, hold_secs: float, ready: Value, done: Value) -> None:
    """Helper run in subprocess: grab the same fcntl lock and hold it."""
    store = MemoryStore(Path(lock_dir))
    with store.locked():
        ready.value = 1
        time.sleep(hold_secs)
        done.value = 1


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl POSIX-only")
def test_lock_serializes_across_processes(tmp_path: Path) -> None:
    """Two processes locking the same MemoryStore must serialize:
    process B can only acquire after process A releases.
    """
    store = MemoryStore(tmp_path)
    store.write_long_term("init\n")

    ready = Value("i", 0)
    done = Value("i", 0)
    p = Process(
        target=_worker_acquire_and_hold,
        args=(str(tmp_path), 0.3, ready, done),
    )
    p.start()

    deadline = time.monotonic() + 5
    while ready.value == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.value == 1, "worker never acquired the lock"

    t0 = time.monotonic()
    with store.locked():
        elapsed = time.monotonic() - t0

        assert elapsed >= 0.15, (
            f"main process didn't wait for worker to release "
            f"(elapsed={elapsed:.3f}s); fcntl lock not enforced cross-process"
        )

    p.join(timeout=5)
    assert done.value == 1
