"""Tests for codeflow.utils.atomic_io."""

import multiprocessing
from pathlib import Path

import pytest

from codeflow.utils.atomic_io import atomic_replace, locked_append, locked_delete

WRITERS = 2
CALLS_PER_WRITER = 50
LINES_PER_CALL = 5


def _append_worker(path_str: str, writer_id: int) -> None:
    for call_idx in range(CALLS_PER_WRITER):
        block = [f"{writer_id}:{call_idx}:{line_idx}" for line_idx in range(LINES_PER_CALL)]
        locked_append(Path(path_str), block)


def test_locked_append_appends_lines(tmp_path: Path):
    """Sequential calls accumulate lines in order."""
    path = tmp_path / "s.jsonl"
    locked_append(path, ["a", "b"])
    locked_append(path, ["c"])
    assert path.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_lock_lives_in_hidden_lock_subdir(tmp_path: Path):
    """The advisory lock sidecar lives in a hidden ``.lock/`` dir derived from
    the target's own parent — never beside the target file."""
    path = tmp_path / "s.jsonl"
    locked_append(path, ["a"])
    beside = [p.name for p in tmp_path.iterdir() if p.is_file() and p.name.endswith(".lock")]
    assert beside == []
    assert (tmp_path / ".lock" / "s.jsonl.lock").exists()


def test_locked_append_concurrent_writers_lose_nothing(tmp_path: Path):
    """Two processes appending concurrently: every line lands, and the
    lines of one locked_append call stay contiguous (turn-block invariant)."""
    path = tmp_path / "s.jsonl"
    procs = [multiprocessing.Process(target=_append_worker, args=(str(path), w)) for w in range(WRITERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == WRITERS * CALLS_PER_WRITER * LINES_PER_CALL
    assert len(set(lines)) == len(lines)

    block_positions: dict[tuple[str, str], list[int]] = {}
    for pos, line in enumerate(lines):
        writer_id, call_idx, _ = line.split(":")
        block_positions.setdefault((writer_id, call_idx), []).append(pos)
    for positions in block_positions.values():
        assert positions == list(range(positions[0], positions[0] + LINES_PER_CALL))


def test_locked_append_repairs_missing_trailing_newline(tmp_path: Path):
    """Appending after a crashed partial line starts on a fresh line, so
    the new record is not merged into the partial one."""
    path = tmp_path / "s.jsonl"
    path.write_text('{"partial": "tru', encoding="utf-8")
    locked_append(path, ["next"])
    assert path.read_text(encoding="utf-8") == '{"partial": "tru\nnext\n'


def test_atomic_replace_swaps_content(tmp_path: Path):
    """atomic_replace replaces the whole file and leaves no temp residue."""
    path = tmp_path / "s.jsonl"
    path.write_text("old\n", encoding="utf-8")
    atomic_replace(path, "new1\nnew2\n")
    assert path.read_text(encoding="utf-8") == "new1\nnew2\n"
    residue = [p.name for p in tmp_path.iterdir() if p.name not in ("s.jsonl", ".lock", ".generation")]
    assert residue == []
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_replace_creates_missing_file(tmp_path: Path):
    path = tmp_path / "fresh.jsonl"
    atomic_replace(path, "data\n")
    assert path.read_text(encoding="utf-8") == "data\n"


def test_locked_append_does_not_recreate_required_file_after_delete(tmp_path: Path):
    path = tmp_path / "deleted.jsonl"
    locked_append(path, ["first"])

    assert locked_delete(path) is True
    with pytest.raises(FileNotFoundError, match="deleted"):
        locked_append(path, ["late"], require_existing=True)
    assert not path.exists()


def test_atomic_replace_does_not_recreate_required_file_after_delete(tmp_path: Path):
    path = tmp_path / "deleted.jsonl"
    atomic_replace(path, "first\n")

    assert locked_delete(path) is True
    with pytest.raises(FileNotFoundError, match="deleted"):
        atomic_replace(path, "late\n", require_existing=True)
    assert not path.exists()


def test_locked_delete_unknown_file_returns_false(tmp_path: Path):
    assert locked_delete(tmp_path / "unknown.jsonl") is False


def test_locked_delete_unknown_file_does_not_create_epoch(tmp_path: Path):
    path = tmp_path / "unknown.jsonl"

    assert locked_delete(path, increment_epoch=True) is False
    assert not (tmp_path / ".generation" / "unknown.jsonl.epoch").exists()


def test_locked_delete_can_fence_a_known_missing_file(tmp_path: Path):
    path = tmp_path / "lazy.jsonl"

    assert locked_delete(path, increment_epoch=True, fence_missing=True) is False
    assert (tmp_path / ".generation" / "lazy.jsonl.epoch").read_text(encoding="ascii") == "1"


def test_generation_rejects_stale_append_after_same_path_is_recreated(tmp_path: Path):
    path = tmp_path / "reused.jsonl"
    first_epoch = locked_append(path, ["old"], expected_epoch=0)

    assert locked_delete(path, increment_epoch=True) is True
    second_epoch = locked_append(path, ["new"], expected_epoch=1)

    assert first_epoch == 0
    assert second_epoch == 1
    with pytest.raises(FileNotFoundError, match="deleted or replaced"):
        locked_append(path, ["late"], expected_epoch=first_epoch)
    assert path.read_text(encoding="utf-8") == "new\n"


@pytest.mark.parametrize("corrupt_epoch", [b"not-an-int", b"\xff"])
def test_generation_corruption_fails_closed_against_stale_append(tmp_path: Path, corrupt_epoch: bytes):
    path = tmp_path / "reused.jsonl"
    first_epoch = locked_append(path, ["old"], expected_epoch=0)
    assert locked_delete(path, increment_epoch=True) is True
    locked_append(path, ["new"], expected_epoch=1)
    (tmp_path / ".generation" / "reused.jsonl.epoch").write_bytes(corrupt_epoch)

    with pytest.raises(ValueError, match="invalid deletion epoch"):
        locked_append(path, ["late"], expected_epoch=first_epoch)

    assert path.read_text(encoding="utf-8") == "new\n"


def test_missing_epoch_after_generation_change_rejects_stale_epoch_zero_writer(
    tmp_path: Path,
):
    path = tmp_path / "reused.jsonl"
    locked_append(path, ["old"], expected_epoch=0)
    assert (
        atomic_replace(
            path,
            "new\n",
            expected_epoch=0,
            require_existing=True,
            increment_epoch=True,
        )
        == 1
    )

    (tmp_path / ".generation" / "reused.jsonl.epoch").unlink()

    with pytest.raises(ValueError, match="missing deletion epoch"):
        locked_append(path, ["late"], expected_epoch=0)
    assert path.read_text(encoding="utf-8") == "new\n"


def test_brand_new_path_accepts_epoch_zero_writer(tmp_path: Path):
    path = tmp_path / "brand-new.jsonl"

    assert locked_append(path, ["first"], expected_epoch=0) == 0
    assert path.read_text(encoding="utf-8") == "first\n"


def test_generation_read_propagates_non_missing_os_error(tmp_path: Path):
    path = tmp_path / "blocked.jsonl"
    epoch_path = tmp_path / ".generation" / "blocked.jsonl.epoch"
    epoch_path.mkdir(parents=True)

    with pytest.raises(IsADirectoryError):
        locked_append(path, ["late"], expected_epoch=0)

    assert not path.exists()


def test_helpers_work_cross_platform(tmp_path: Path):
    """The write helpers work on every platform. Locking is cross-platform via
    portalocker (POSIX fcntl + Windows LockFileEx) — there is no longer an
    fcntl-absent 'degrade to unlocked' path."""
    path = tmp_path / "s.jsonl"
    locked_append(path, ["x"])
    atomic_replace(path, "y\n")
    assert path.read_text(encoding="utf-8") == "y\n"
