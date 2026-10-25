"""Offload blocking I/O without releasing its caller's locks before it stops."""

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from codeflow.spine._barrier import finish_barrier

_P = ParamSpec("_P")
_T = TypeVar("_T")


async def run_blocking_io(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    # A Python thread cannot be killed. Shield and join it on cancellation so
    # Registry's resource lock still covers the complete read/check/write.
    result: list[_T] = []

    async def run() -> None:
        result.append(await asyncio.to_thread(function, *args, **kwargs))

    await finish_barrier(asyncio.create_task(run()))
    return result[0]
