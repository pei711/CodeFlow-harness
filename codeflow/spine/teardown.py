"""为 Scheduler 与 DeliveryHub 这一对 Spine 所有者提供确定性的联合 teardown。

执行侧和投递侧拥有彼此独立的 barrier，任一失败都不能成为跳过另一侧清理的理由。本模块
先尝试 Scheduler shutdown，再尝试 DeliveryHub close，记录各自失败后统一决定向外抛什么；
这样 Host 只调用一个入口，也不会因第一个异常留下另一半 Worker 或队列。
"""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from codeflow.spine.delivery import DeliveryHub
from codeflow.spine.scheduler import Scheduler


async def teardown_spine(scheduler: Scheduler, delivery: DeliveryHub, *, grace: float) -> None:
    """依次尝试两个 shutdown barrier，最后抛出优先级最高的失败。

    Scheduler 先以 ``grace`` 完成 seal、drain 和运行中 Turn 收尾，DeliveryHub 随后执行
    `aclose`；内部 `attempt` 捕获并记录每一步，让第一步失败也不阻止第二步。多个普通异常只
    保存第一个，便于保持确定结果；caller cancellation 单独保存。

    两步都尝试后，若曾取消则优先重新抛出 `CancelledError`，否则抛第一个普通 failure，均无
    问题才返回。这里的优先级与单个 `finish_barrier` 一致：清理必须尽量完成，但调用方取消
    事实不能被较低优先级异常覆盖。
    """
    first_error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None

    async def attempt(name: str, cleanup: Callable[[], Awaitable[None]]) -> None:
        nonlocal cancellation, first_error
        try:
            await cleanup()
        except asyncio.CancelledError as exc:
            logger.opt(exception=exc).error("Spine teardown step {} was cancelled", name)
            if cancellation is None:
                cancellation = exc
        except BaseException as exc:
            logger.opt(exception=exc).error("Spine teardown step {} failed", name)
            if first_error is None:
                first_error = exc

    await attempt("scheduler", lambda: scheduler.shutdown(grace=grace))
    await attempt("delivery", delivery.aclose)
    if cancellation is not None:
        raise cancellation
    if first_error is not None:
        raise first_error
