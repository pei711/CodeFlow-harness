import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from codeflow.agent.loop.checkpoint import CheckpointService
from codeflow.sandbox.direct_executor import DirectExecutor
from codeflow.sandbox.srt_executor import SrtExecutor


@pytest.mark.parametrize("kind", ["direct", "srt", "checkpoint"])
async def test_cancellation_kills_and_reaps_owned_process(kind, tmp_path, monkeypatch):
    process = MagicMock()
    process.wait = AsyncMock(return_value=0)
    entered = asyncio.Event()

    async def communicate():
        entered.set()
        await asyncio.Event().wait()

    process.communicate = communicate
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if kind == "direct":
        work = DirectExecutor().exec("long command", cwd=str(tmp_path))
    elif kind == "checkpoint":
        work = CheckpointService(tmp_path)._git("add", "-A")
    else:
        executor = SrtExecutor(tmp_path)
        monkeypatch.setattr(executor, "_runtime", lambda env: ("node", {}))
        work = executor.exec("long command")
    task = asyncio.create_task(work)
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


async def test_real_shell_cancellation_stops_child_before_delayed_write(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "(root / 'ready').write_text('ready')\n"
        "time.sleep(1)\n"
        "(root / 'late').write_text('must not run after cancellation')\n",
        encoding="utf-8",
    )
    task = asyncio.create_task(DirectExecutor().exec(f'"{sys.executable}" "{script}" "{tmp_path}"'))
    try:
        async with asyncio.timeout(5):
            while not (tmp_path / "ready").exists():
                await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(1.1)
        assert not (tmp_path / "late").exists()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
