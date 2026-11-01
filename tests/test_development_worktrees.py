"""Real local Git integration tests, using disposable repositories only."""

import asyncio
import json
import shutil
import sys

import pytest

from codeflow.agent.subagent import manager as manager_module
from codeflow.agent.subagent.manager import SubagentManager
from codeflow.agent.subagent.worktree import GitWorktreeManager, WorktreeError
from codeflow.providers.base import LLMResponse, ToolCallRequest
from codeflow.sandbox.interfaces import ExecResult


@pytest.fixture
async def manager(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    worktrees = GitWorktreeManager(repo, tmp_path / "state")
    await worktrees._git(repo, "init", "-b", "main")
    (repo / ".gitignore").write_text(".worktrees/\n.codeflow/\n__pycache__/\n", encoding="utf-8")
    (repo / "same.txt").write_text("base\n", encoding="utf-8")
    await worktrees._git(repo, "add", ".gitignore", "same.txt")
    await worktrees._git(repo, "commit", "-m", "initial")
    await worktrees._git(repo, "branch", "delivery")
    return worktrees


async def commit(manager, record, filename, content):
    (record.path / filename).write_text(content, encoding="utf-8")
    await manager._git(record.path, "add", "--", filename)
    await manager._git(record.path, "commit", "-m", "task change")


async def accept(_path):
    return True


async def test_independent_worktrees_isolate_same_file_and_retain_dirty_state(manager):
    (manager.repository / "same.txt").write_text("parent uncommitted\n", encoding="utf-8")
    first, second = await asyncio.gather(manager.create("first"), manager.create("second"))
    assert first.branch != second.branch
    assert (first.path / "same.txt").read_text() == "base\n"
    (first.path / "same.txt").write_text("first\n", encoding="utf-8")
    (second.path / "same.txt").write_text("second\n", encoding="utf-8")
    for record, status in ((first, "completed"), (second, "cancelled")):
        final = await manager.finish(record.task_id, status)
        assert final.dirty is True
        saved = json.loads((manager.state_dir / f"{record.task_id}.json").read_text(encoding="utf-8"))
        assert saved["status"] == status and saved["dirty"] is True
        assert record.path.exists()
    assert (manager.repository / "same.txt").read_text() == "parent uncommitted\n"
    with pytest.raises(WorktreeError, match="committed"):
        await manager.review(first.task_id)


async def test_conflict_is_detected_and_preserved_without_target_update(manager):
    first, second = await asyncio.gather(manager.create("first"), manager.create("second"))
    await commit(manager, first, "same.txt", "first\n")
    await commit(manager, second, "same.txt", "second\n")
    reviews = [await manager.review(record.task_id) for record in (first, second)]
    before = await manager._git(manager.repository, "rev-parse", "main")
    with pytest.raises(WorktreeError):
        await manager.integrate(reviews, target="main", verifier=accept)
    candidate = next(record for key, record in manager.records.items() if key.startswith("integration-"))
    assert candidate.status == "failed" and candidate.dirty is True
    assert candidate.path.exists()
    assert await manager._git(candidate.path, "diff", "--name-only", "--diff-filter=U") == "same.txt"
    assert await manager._git(manager.repository, "rev-parse", "main") == before


async def test_failed_real_test_command_never_advances_target(manager):
    record = await manager.create("broken")
    await commit(manager, record, "test_feature.py", "assert 1 == 2\n")
    review = await manager.review(record.task_id)
    before = await manager._git(manager.repository, "rev-parse", "delivery")
    verified_in = []

    async def verifier(path):
        verified_in.append(path)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "test_feature.py",
            cwd=path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        return process.returncode == 0

    with pytest.raises(WorktreeError, match="verifier rejected"):
        await manager.integrate([review], target="delivery", verifier=verifier)
    assert len(verified_in) == 1 and verified_in[0] != record.path
    assert await manager._git(manager.repository, "rev-parse", "delivery") == before


async def test_reviewed_verified_candidate_promotes_only_with_explicit_cas(manager):
    record = await manager.create("feature")
    await commit(manager, record, "feature.py", "value = 42\nassert value == 42\n")
    review = await manager.review(record.task_id)
    assert "+value = 42" in review.diff
    before = await manager._git(manager.repository, "rev-parse", "delivery")

    async def verifier(path):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "py_compile",
            "feature.py",
            cwd=path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        if process.returncode != 0:
            return False
        process = await asyncio.create_subprocess_exec(sys.executable, "feature.py", cwd=path)
        return await process.wait() == 0

    candidate = await manager.integrate([review], target="delivery", verifier=verifier)
    assert await manager._git(manager.repository, "rev-parse", "delivery") == before
    await manager.promote(candidate)
    assert await manager._git(manager.repository, "rev-parse", "delivery") == candidate.head
    assert await manager._git(manager.repository, "rev-parse", "main") == before
    assert manager.records[candidate.task_id].status == "promoted"


async def test_review_and_verification_are_invalidated_by_new_changes(manager):
    record = await manager.create("feature")
    await commit(manager, record, "feature.txt", "first\n")
    review = await manager.review(record.task_id)
    await commit(manager, record, "feature.txt", "second\n")
    with pytest.raises(WorktreeError, match="changed after review"):
        await manager.integrate([review], target="delivery", verifier=accept)
    review = await manager.review(record.task_id)
    candidate = await manager.integrate([review], target="delivery", verifier=accept)
    candidate_record = manager.records[candidate.task_id]
    (candidate_record.path / "unreviewed.txt").write_text("pending", encoding="utf-8")
    with pytest.raises(WorktreeError, match="changed since verification"):
        await manager.promote(candidate)
    assert await manager._git(manager.repository, "rev-parse", "delivery") == candidate.target_base


async def test_target_drift_and_checked_out_target_are_rejected(manager):
    record = await manager.create("feature")
    await commit(manager, record, "feature.txt", "feature\n")
    review = await manager.review(record.task_id)
    candidate = await manager.integrate([review], target="main", verifier=accept)
    with pytest.raises(WorktreeError, match="checked out"):
        await manager.promote(candidate)
    candidate = await manager.integrate([review], target="delivery", verifier=accept)
    await manager._git(manager.repository, "update-ref", "refs/heads/delivery", review.head)
    with pytest.raises(WorktreeError):
        await manager.promote(candidate)
    assert await manager._git(manager.repository, "rev-parse", "delivery") == review.head


async def test_cancelled_integration_retains_workspace_and_status(manager):
    record = await manager.create("feature")
    await commit(manager, record, "feature.txt", "feature\n")
    review = await manager.review(record.task_id)
    entered = asyncio.Event()

    async def verifier(_path):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(manager.integrate([review], target="delivery", verifier=verifier))
    await asyncio.wait_for(entered.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    candidate_record = next(item for key, item in manager.records.items() if key.startswith("integration-"))
    assert candidate_record.status == "cancelled" and candidate_record.path.exists()
    assert await manager._git(manager.repository, "rev-parse", "delivery") == record.base


async def test_plain_spawn_creates_no_worktree_and_development_binds_all_tools(manager, monkeypatch):
    class Provider:
        def get_default_model(self):
            return "stub"

        async def chat_with_retry(self, *, messages, **kwargs):
            await asyncio.sleep(0)
            if len(messages) > 2:
                return LLMResponse("done")
            label = messages[1]["content"]
            return LLMResponse(
                None,
                [
                    ToolCallRequest("w", "write_file", {"path": "same.txt", "content": label}),
                    ToolCallRequest("r", "read_file", {"path": "same.txt"}),
                    ToolCallRequest("e", "exec", {"command": "echo test"}),
                ],
            )

    calls = []
    bound_workspaces = []

    class Executor:
        is_sandboxed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def exec(self, command, cwd, **kwargs):
            calls.append(cwd)
            return ExecResult("ok", "", 0)

    def build(_config, workspace, _ids):
        bound_workspaces.append(workspace)
        return Executor()

    monkeypatch.setattr(manager_module, "build_executor", build)
    subagents = SubagentManager(Provider(), manager.repository)
    announced = []
    subagents.set_submit(announced.append)
    await subagents.spawn("plain")
    await asyncio.gather(*subagents._running_tasks.values())
    assert not subagents.worktrees.records
    # The existing plain mode preserves its old workspace behavior.
    assert (manager.repository / "same.txt").read_text() == "plain"
    await subagents.spawn("first", development=True, session_key="a")
    await subagents.spawn("second", development=True, session_key="b")
    outcomes = await asyncio.gather(*subagents._running_tasks.values())
    records = list(subagents.worktrees.records.values())
    assert len(records) == 2
    assert {record.path for record in records} == set(bound_workspaces[1:])
    assert {str(record.path) for record in records} == set(calls[1:])
    assert {(record.path / "same.txt").read_text() for record in records} == {"first", "second"}
    assert (manager.repository / "same.txt").read_text() == "plain"
    assert all("Retained for review" in outcome.result for outcome in outcomes)
    assert all(record.status == "completed" and record.dirty for record in records)
    assert {request.conversation for request in announced[1:]} == {"a", "b"}


async def test_cancelled_subagent_retains_dirty_workspace(manager, monkeypatch):
    class Provider:
        def get_default_model(self):
            return "stub"

    class Executor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    subagents = SubagentManager(Provider(), manager.repository)
    entered = asyncio.Event()

    async def inner(*args):
        workspace = subagents._task_workspace.get()
        (workspace / "pending.txt").write_text("retain", encoding="utf-8")
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager_module, "build_executor", lambda *args: Executor())
    monkeypatch.setattr(subagents, "_run_subagent_inner", inner)
    subagents.set_submit(lambda request: None)
    await subagents.spawn("long", development=True, session_key="a")
    await asyncio.wait_for(entered.wait(), 10)
    assert await subagents.cancel_by_session("a") == 1
    record = next(iter(subagents.worktrees.records.values()))
    assert record.status == "cancelled" and record.dirty is True
    assert (record.path / "pending.txt").read_text() == "retain"


@pytest.mark.parametrize("task_id", ["../escape", "a/b", "", "--option"])
async def test_worktree_task_ids_cannot_escape(manager, task_id):
    with pytest.raises(ValueError):
        await manager.create(task_id)
