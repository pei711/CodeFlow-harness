"""Unit tests for the grep and find search tools.

Tests exercise both the ripgrep-backed path (when rg is on PATH) and the
pure-Python fallback (forced by patching shutil.which to return None).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from codeflow.agent.tools.file_search import FindTool, GrepTool


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\ndef hello():\n    return 'world'\n")
    (tmp_path / "src" / "util.py").write_text("def helper():\n    return 42\n")
    (tmp_path / "README.md").write_text("# Title\nhello there\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("def hello():\n    pass\n")
    return tmp_path


async def test_grep_content_finds_match(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern=r"def hello")
    assert "app.py" in out
    assert "node_modules" not in out


async def test_grep_no_match(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern=r"zzz_nonexistent")
    assert out == "No matches found."


async def test_grep_glob_filter(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="hello", glob="*.md")
    assert "README.md" in out
    assert "app.py" not in out


async def test_grep_files_with_matches(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern=r"def hello", output_mode="files_with_matches")
    assert "app.py" in out
    assert ":" not in out.split("\n")[0]


async def test_grep_count(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="def", output_mode="count")
    assert "app.py:1" in out
    assert "util.py:1" in out


async def test_grep_case_insensitive(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="HELLO", case_insensitive=True)
    assert "app.py" in out


async def test_grep_invalid_regex(tree: Path):
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="(unclosed")
    assert "invalid regular expression" in out


async def test_grep_python_fallback(tree: Path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern=r"def hello")
    assert "app.py" in out
    assert "node_modules" not in out


async def test_grep_fallback_skips_binary(tree: Path, monkeypatch):
    (tree / "blob.bin").write_bytes(b"def hello\x00\x01binary")
    monkeypatch.setattr(shutil, "which", lambda _: None)
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="hello")
    assert "blob.bin" not in out


async def test_grep_context(tree: Path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="return 'world'", context=1)
    assert "def hello" in out


async def test_grep_outside_allowed_dir(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool = GrepTool(workspace=workspace, allowed_dir=workspace)
    out = await tool.execute(pattern="x", path="/etc")
    assert "Error" in out


async def test_grep_cancellation_reaps_ripgrep_process(tree: Path, monkeypatch):
    started = asyncio.Event()

    class _Process:
        def __init__(self) -> None:
            self.killed = False
            self.waited = False

        async def communicate(self):
            started.set()
            await asyncio.Event().wait()

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.waited = True
            return -9

    process = _Process()

    async def _create_process(*args, **kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _create_process)
    tool = GrepTool(workspace=tree, allowed_dir=tree)
    task = asyncio.create_task(tool._run_rg("rg", "hello", tree, None, "content", False, 0, 100))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True
    assert process.waited is True


async def test_find_basename_recursive(tree: Path):
    tool = FindTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="*.py")
    assert "src/app.py" in out
    assert "src/util.py" in out
    assert "node_modules" not in out


async def test_find_path_pattern(tree: Path):
    tool = FindTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="src/*.py")
    assert "src/app.py" in out
    assert "README.md" not in out


async def test_find_no_match(tree: Path):
    tool = FindTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="*.rs")
    assert out == "No files found matching pattern."


async def test_find_limit(tree: Path):
    tool = FindTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="*.py", limit=1)
    assert "showing first 1 of 2" in out


async def test_find_sorted_by_recency(tree: Path):

    import os
    import time

    os.utime(tree / "src" / "util.py", (time.time() + 100, time.time() + 100))
    tool = FindTool(workspace=tree, allowed_dir=tree)
    out = await tool.execute(pattern="*.py")
    lines = out.splitlines()
    assert lines[0].endswith("util.py")
