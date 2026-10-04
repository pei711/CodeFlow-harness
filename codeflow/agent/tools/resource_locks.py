"""Coordinate concurrent Tool access to overlapping filesystem resources.

Readers may share a resource. A writer excludes readers and writers for the same
path, and directory resources overlap their descendants so a directory scan does
not race a file mutation below it. The manager is process-local and intended to
be shared by the host AgentLoop and its Subagents.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator


@dataclass(eq=False)
class _Request:
    resource: str
    write: bool


class ResourceLockManager:
    """Fair shared/exclusive locks for normalized path resources."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active: list[_Request] = []
        self._waiting: list[_Request] = []

    @staticmethod
    def normalize(resource: str | os.PathLike[str]) -> str:
        """Return a stable absolute key, including platform case normalization."""
        return os.path.normcase(os.path.normpath(str(Path(resource).resolve())))

    @staticmethod
    def _overlaps(left: str, right: str) -> bool:
        left_path = Path(left)
        right_path = Path(right)
        return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents

    def _can_enter(self, request: _Request) -> bool:
        for active in self._active:
            if self._overlaps(request.resource, active.resource) and (request.write or active.write):
                return False

        # Do not let later readers starve an earlier writer. Compatible readers
        # for the same resource may still enter together.
        for waiting in self._waiting:
            if waiting is request:
                break
            if self._overlaps(request.resource, waiting.resource) and (request.write or waiting.write):
                return False
        return True

    @asynccontextmanager
    async def _hold(self, resource: str | os.PathLike[str], *, write: bool) -> AsyncIterator[None]:
        request = _Request(self.normalize(resource), write)
        async with self._condition:
            self._waiting.append(request)
            try:
                await self._condition.wait_for(lambda: self._can_enter(request))
            except BaseException:
                self._waiting.remove(request)
                self._condition.notify_all()
                raise
            self._waiting.remove(request)
            self._active.append(request)
            self._condition.notify_all()

        try:
            yield
        finally:
            async with self._condition:
                self._active.remove(request)
                self._condition.notify_all()

    def read(self, resource: str | os.PathLike[str]):
        """Acquire a shared lock; compatible readers may proceed together."""
        return self._hold(resource, write=False)

    def write(self, resource: str | os.PathLike[str]):
        """Acquire an exclusive lock for this resource and overlapping paths."""
        return self._hold(resource, write=True)


__all__ = ["ResourceLockManager"]
