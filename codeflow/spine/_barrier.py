"""保证已经启动的 Spine 清理屏障在调用方取消下仍能完成。

Scheduler shutdown 与 DeliveryHub close 都可能在等待期间收到 caller cancellation。若直接
await，取消会穿透并中止 cleanup，留下 Worker、队列或资源半关闭。本模块用 `finish_barrier`
shield 同一个既有 Task，先收集 cleanup 失败和取消事实，再按优先级向外传播；它不创建或
重启清理任务，任务所有权仍属于调用方。
"""

import asyncio


async def finish_barrier(
    task: asyncio.Task[None],
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> None:
    """等待 ``task`` 真正结束，并让 caller cancellation 优先于 barrier failure。

    ``task`` 必须是已经启动的 cleanup Task。循环使用 `asyncio.shield`，因此等待者被取消时只
    记录第一个 `CancelledError`，不会取消 barrier；Task 完成后再读取其最终异常。调用方也可
    通过 ``cancellation`` 传入进入本函数前已经捕获的取消事实。

    若取消与 cleanup failure 同时存在，重新抛出取消并把 failure 作为 cause；只有 failure 时
    原样抛出，二者都没有则正常返回。这样上层既看到最重要的取消语义，又不会丢失清理失败
    证据。
    """
    failure: BaseException | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            failure = exc
            break
    if failure is None:
        try:
            await task
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            failure = exc
    if cancellation is not None:
        if failure is not None:
            raise cancellation from failure
        raise cancellation
    if failure is not None:
        raise failure
