"""Deterministic concurrency tests; no live model, network, or timing speedup claims."""

import asyncio
import hashlib
import threading
from collections import Counter

import pytest

from codeflow.agent.loop.checkpoint import CheckpointService
from codeflow.agent.tools.filesystem import EditFileTool, ReadFileTool, WriteFileTool
from codeflow.agent.tools.registry import ToolRegistry
from codeflow.agent.tools.resource_locks import ResourceLockManager
from codeflow.spine import ChatType, Origin, OriginPools, Scheduler, Source, TurnOutcome, TurnRequest, Usage
from codeflow.utils.atomic_file import atomic_write_bytes


async def settle(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0)


def request(conversation, text, origin=Origin.USER):
    return TurnRequest(origin, Source("test", conversation, "u", ChatType.DM), text, conversation=conversation)


async def noop(_event):
    pass


async def test_lane_fifo_and_cross_conversation_progress():
    release = asyncio.Event()
    entered = []
    active = set()

    class Runner:
        async def run(self, req, emit, drain):
            assert req.conversation not in active
            active.add(req.conversation)
            entered.append((req.conversation, req.text))
            try:
                await release.wait()
                return TurnOutcome(Usage(0, 0, 0), False)
            finally:
                active.remove(req.conversation)

    scheduler = Scheduler(Runner(), OriginPools(4, 2), noop)
    handles = [scheduler.submit(request("a", str(i))) for i in range(5)]
    handles.append(scheduler.submit(request("b", "0")))
    try:
        await settle(lambda: len(entered) == 2)
        assert set(active) == {"a", "b"}
        release.set()
        await asyncio.wait_for(asyncio.gather(*(h.result() for h in handles)), 5)
        assert [text for lane, text in entered if lane == "a"] == list(map(str, range(5)))
    finally:
        release.set()
        await scheduler.shutdown(0)


async def test_four_user_two_system_slots_released_on_every_terminal_path():
    release = asyncio.Event()
    counts = Counter()
    peaks = Counter()
    started = set()

    class Runner:
        async def run(self, req, emit, drain):
            key = "user" if req.origin is Origin.USER else "system"
            counts[key] += 1
            peaks[key] = max(peaks[key], counts[key])
            started.add(req.text)
            try:
                await release.wait()
                if req.text.endswith("fail"):
                    raise ValueError("expected failure")
                return TurnOutcome(Usage(0, 0, 0), False)
            finally:
                counts[key] -= 1

    scheduler = Scheduler(Runner(), OriginPools(4, 2), noop)
    handles = []
    completed = Counter()
    try:
        for origin in (Origin.USER, Origin.CRON, Origin.SUBAGENT):
            for index in range(8):
                label = f"{origin}-{index}" + ("fail" if index == 1 else "")
                handle = scheduler.submit(request(label, label, origin))
                handle._fut.add_done_callback(lambda _, label=label: completed.update([label]))
                handles.append(handle)
        await settle(lambda: counts == {"user": 4, "system": 2})
        handles[0].cancel()  # running
        handles[7].cancel()  # queued / waiting for a pool slot
        handles[0].cancel()  # idempotent
        await settle(lambda: "user-4" in started)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*(h.result() for h in handles)), 5)
        await asyncio.sleep(0)
        assert any(result is None for result in results)
        assert peaks == {"user": 4, "system": 2}
        assert counts == {"user": 0, "system": 0}
        assert len(completed) == len(handles)
        assert set(completed.values()) == {1}
        assert await scheduler.submit(request("later", "later")).result() is not None
    finally:
        release.set()
        await scheduler.shutdown(0)


@pytest.mark.parametrize("user,system", [(0, 2), (4, 0), (-1, 2)])
def test_empty_pools_are_rejected(user, system):
    with pytest.raises(ValueError, match="positive"):
        OriginPools(user, system)


def registry(workspace, locks):
    tools = ToolRegistry(resource_locks=locks)
    tools.register(ReadFileTool(workspace, workspace))
    tools.register(WriteFileTool(workspace, workspace))
    return tools


async def test_real_file_reads_overlap_across_agent_registries(tmp_path, monkeypatch):
    (tmp_path / "same.txt").write_text("shared", encoding="utf-8")
    locks = ResourceLockManager()
    agents = [registry(tmp_path, locks) for _ in range(2)]
    barrier = threading.Barrier(2, timeout=5)
    original = ReadFileTool._execute_sync

    def read(self, *args):
        # Breaks if either the event loop or same-file reads are serialized.
        barrier.wait()
        return original(self, *args)

    monkeypatch.setattr(ReadFileTool, "_execute_sync", read)
    results = await asyncio.gather(*(agent.execute("read_file", {"path": "same.txt"}) for agent in agents))
    assert all(not result.failed and "shared" in result for result in results)
    assert not locks._active and not locks._waiting


async def test_same_file_stale_write_is_rejected(tmp_path):
    path = tmp_path / "same.txt"
    path.write_bytes(b"base")
    locks = ResourceLockManager()
    agents = [registry(tmp_path, locks) for _ in range(2)]
    version = hashlib.sha256(b"base").hexdigest()
    results = await asyncio.gather(
        *(
            agent.execute("write_file", {"path": "same.txt", "content": value, "expected_sha256": version})
            for agent, value in zip(agents, ("first", "second"))
        )
    )
    assert sum(result.failed for result in results) == 1
    assert "write conflict" in next(result for result in results if result.failed)
    assert path.read_bytes() in (b"first", b"second")


async def test_independent_edits_preserve_both_updates(tmp_path):
    path = tmp_path / "same.txt"
    path.write_bytes(b"first=old\r\nsecond=old\r\n")
    locks = ResourceLockManager()
    agents = [registry(tmp_path, locks) for _ in range(2)]
    for agent in agents:
        agent.register(EditFileTool(tmp_path, tmp_path))
    results = await asyncio.gather(
        *(
            agent.execute("edit_file", {"path": "same.txt", "old_text": f"{key}=old", "new_text": f"{key}=new"})
            for agent, key in zip(agents, ("first", "second"))
        )
    )
    assert all(not result.failed for result in results)
    assert path.read_bytes() == b"first=new\r\nsecond=new\r\n"
    result = await agents[0].execute(
        "edit_file",
        {"path": "same.txt", "old_text": "first=new", "new_text": "first=stale", "expected_sha256": "0" * 64},
    )
    assert result.failed and "write conflict" in result


async def test_create_if_missing_and_hash_from_read(tmp_path):
    tools = registry(tmp_path, ResourceLockManager())
    params = {"path": "new.txt", "content": "中文", "expected_sha256": "missing"}
    assert not (await tools.execute("write_file", params)).failed
    assert (await tools.execute("write_file", params)).failed
    result = await tools.execute("read_file", {"path": "new.txt"})
    assert hashlib.sha256("中文".encode()).hexdigest() in result


def test_atomic_replace_failure_keeps_old_file_and_cleans_temporary(tmp_path, monkeypatch):
    import codeflow.utils.atomic_file as atomic

    path = tmp_path / "same.txt"
    path.write_bytes(b"old")

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(atomic.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_bytes(path, b"new")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.glob(".*.tmp")) == []


async def test_cancellation_waits_for_thread_before_releasing_file_lock(tmp_path, monkeypatch):
    (tmp_path / "same.txt").write_bytes(b"base")
    locks = ResourceLockManager()
    writer, reader = registry(tmp_path, locks), registry(tmp_path, locks)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    original = WriteFileTool._execute_sync

    def delayed(self, *args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original(self, *args)

    monkeypatch.setattr(WriteFileTool, "_execute_sync", delayed)
    writing = asyncio.create_task(writer.execute("write_file", {"path": "same.txt", "content": "complete"}))
    reading = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        writing.cancel()
        await asyncio.sleep(0)
        writing.cancel()
        reading = asyncio.create_task(reader.execute("read_file", {"path": "same.txt"}))
        await settle(lambda: bool(locks._waiting))
        assert not writing.done() and not reading.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await writing
        assert "complete" in await asyncio.wait_for(reading, 5)
        assert not locks._active and not locks._waiting
    finally:
        release.set()
        await asyncio.gather(*(t for t in (writing, reading) if t), return_exceptions=True)


async def test_lock_fairness_overlap_unrelated_paths_and_cancelled_waiter(tmp_path):
    locks = ResourceLockManager()
    parent = tmp_path / "tree"
    events = []

    async def hold(resource, write, label):
        async with locks.write(resource) if write else locks.read(resource):
            events.append(label)

    async with locks.read(parent):
        writer = asyncio.create_task(hold(parent / "child", True, "writer"))
        await settle(lambda: len(locks._waiting) == 1)
        reader = asyncio.create_task(hold(parent, False, "reader"))
        unrelated = asyncio.create_task(hold(tmp_path / "other", True, "other"))
        await unrelated
        assert events == ["other"]
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        await reader  # earlier cancelled writer no longer fences compatible reads
    assert events == ["other", "reader"]
    assert not locks._active and not locks._waiting


async def test_repeated_cancel_during_lock_release_cannot_leak_writer(tmp_path):
    locks = ResourceLockManager()
    entered = asyncio.Event()

    async def writer():
        async with locks.write(tmp_path / "file"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(writer())
    await entered.wait()
    async with locks._condition:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not locks._active and not locks._waiting


async def test_checkpoint_transaction_cannot_interleave(tmp_path, monkeypatch):
    service = CheckpointService(tmp_path)
    calls = []

    async def git(*args):
        calls.append((asyncio.current_task().get_name(), args[0]))
        await asyncio.sleep(0)
        return 0, "file.txt" if args[0] == "diff" else "abc", ""

    monkeypatch.setattr(service, "_git", git)
    service._ready = True
    await asyncio.gather(
        asyncio.create_task(service.commit_turn("a"), name="a"),
        asyncio.create_task(service.commit_turn("b"), name="b"),
    )
    owners = [owner for owner, _ in calls]
    assert owners == sorted(owners)
    assert [command for _, command in calls].count("add") == 2
