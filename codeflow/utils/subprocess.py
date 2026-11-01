"""Cancellation-safe cleanup of a directly owned asyncio subprocess."""

import asyncio
import contextlib
import os
import signal
import subprocess

from codeflow.spine._barrier import finish_barrier


def owned_process_options() -> dict:
    """Give owned commands a killable process group on either host platform."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def stop_process(process: asyncio.subprocess.Process, *, tree: bool = False) -> None:
    async def stop() -> None:
        if tree and isinstance(process.pid, int):
            if os.name == "nt":
                # CREATE_NEW_PROCESS_GROUP alone does not kill descendants on
                # Windows. taskkill is an OS utility, called without a shell.
                try:
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await killer.wait()
                except OSError:
                    # Still kill/reap the direct child if the OS utility is
                    # unavailable. Detached descendants need executor isolation.
                    pass
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        # Reap the owned process before the surrounding resource lock is freed.
        await process.wait()

    await finish_barrier(asyncio.create_task(stop()))
