"""Ready-first lifecycle for the TUI's Runtime Assembly."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from codeflow.spine._barrier import finish_barrier

if TYPE_CHECKING:
    from codeflow.cli._runtime_assembly import RuntimeAssembly


class TuiRuntimeHost:
    """Build the Runtime Assembly off the TUI-RPC event loop."""

    def __init__(self, build: "Callable[[], RuntimeAssembly]") -> None:
        self._build = build
        self._task: asyncio.Task[RuntimeAssembly] | None = None
        self._runtime: RuntimeAssembly | None = None
        self._closed = False
        self._close_lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._build_runtime())

    async def acquire(self) -> "RuntimeAssembly":
        self.start()
        task = self._task
        if task is None:
            raise RuntimeError("TUI Runtime Host failed to start")
        return await asyncio.shield(task)

    def get_now(self) -> "RuntimeAssembly | None":
        return self._runtime

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            task = self._task
            if task is None:
                self._closed = True
                return
            runtime = None
            if task.done():
                try:
                    runtime = task.result()
                except (asyncio.CancelledError, Exception):
                    self._closed = True
                    return
                runtime.begin_close()
            close_task = asyncio.create_task(self._close_runtime(task, runtime))
            await finish_barrier(close_task)

    async def _close_runtime(
        self,
        task: "asyncio.Task[RuntimeAssembly]",
        runtime: "RuntimeAssembly | None",
    ) -> None:
        if runtime is None:
            try:
                runtime = await task
            except (asyncio.CancelledError, Exception):
                self._closed = True
                return
            runtime.begin_close()
        await runtime.close()
        self._closed = True

    async def _build_runtime(self) -> "RuntimeAssembly":
        runtime = await asyncio.to_thread(self._build)
        self._runtime = runtime
        return runtime


__all__ = ["TuiRuntimeHost"]
