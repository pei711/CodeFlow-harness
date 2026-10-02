"""Sandbox Debug Server 与 CLI 共享的 Internal Async Helpers。

模块集中处理 Teardown/Race Path 中 Task Cancellation 的细微语义，避免各调用方只 ``cancel()`` 却忘记
Await，最终产生未回收异常 Warning。它不拥有 Sandbox 生命周期，只提供可复用的单 Task 收尾动作。
"""

from __future__ import annotations

import asyncio


async def cancel_and_collect(task: asyncio.Task) -> None:
    """Cancel 单个 Task，并 Await 后吸收其 Result。

    Convention：在 Teardown / Race Path 处理 Single Task 时使用。Handler 结束时若有 Multiple Tasks，
    更适合先全部 ``cancel()``，再调用 ``asyncio.gather(
    *tasks, return_exceptions=True)``；对单任务两者
    Functionally Equivalent，只是 List Fan-out 时 Gather 更方便。

    若只 Cancel 不 Await，Task 随后抛出非 `CancelledError` 时会在 Garbage Collection 阶段出现
    ``Task exception was never retrieved`` Warning。此 Helper 通过统一回收，使底层 Coroutine 的结果
    不污染 Teardown Logs。它覆盖 Pending、Completed、Failed、Cancelled 所有状态：Done Task 的
    `cancel()` 是 No-op，``await task`` 会立即返回或抛出。

    Await 得到 `CancelledError` 有两种含义：Task 因本次 Cancel 结束时应 Swallow；Parent Caller 自己被
    Cancel、正在向下传播时必须 Re-raise，让 Shutdown 继续流动。两者通过 ``task.cancelled()`` 区分，
    只有前者为 `True`。其他 Task Exception 会被吸收，因为此函数只负责清理而非重新解释业务失败。
    """
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
    except Exception:
        pass
