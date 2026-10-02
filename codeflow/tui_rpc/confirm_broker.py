"""管理 TUI-RPC 中一次确认请求从发出到回答的完整往返。

当 Runtime 的危险或有分支行为需要用户确认时，``ConfirmBroker`` 发送
``confirm.request`` notification，并用 ``request_id`` 保存等待中的 ``Future``；TUI
随后调用 ``confirm.respond``，对应 method 再通过 ``resolve()`` 唤醒原任务。它与
:class:`SubscriptionEmitter` 的所有权相同：由 RPC server 创建，``send_frame`` 绑定到
``RpcServer.send_frame``，再传给
``register_confirm_methods(dispatcher, confirm_broker=...)``。

传输层必须 fail-safe：hard-limit timeout、connection EOF（调用 :meth:`cancel_all`）或
internal error 都回退到 prompt 的 ``default``；外部 Task cancellation 则继续传播，
不能被误装成用户选择。返回布尔值只表示本次确认决策，不代表后续操作成功、任务完成或
结果已交付。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from loguru import logger

# 单次确认在后端保持挂起的硬上限，与前端可见的 30 秒倒计时无关；
# 35 = 30 + 5 秒网络余量。超时后安全回退到提示框的默认选项。
_CONFIRM_HARD_LIMIT_S = 35.0

SendFrame = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class _PendingConfirm:
    future: asyncio.Future
    default: bool


class ConfirmBroker:
    """发送 ``confirm.request`` 并等待匹配的 ``confirm.respond``。

    实例生命周期与 ``RpcServer`` 一致，拥有所有尚未完成的 ``request_id`` 到 Future 的
    映射。每项 pending state 同时保存 prompt 的默认布尔值，以便连接断开时安全收敛。
    ``Dispatcher`` 的 confirm methods 只负责把前端回答交回本对象；真正等待确认的
    Runtime coroutine 由本对象唤醒。

    Broker 不持久化回答，也不判断确认后的业务动作是否执行成功。Server 关闭前应调用
    ``cancel_all()``，否则等待方可能永久悬挂。
    """

    def __init__(self, send_frame: SendFrame) -> None:
        self._send_frame = send_frame
        self._pending: dict[str, _PendingConfirm] = {}

    async def await_confirm(self, prompt: str, *, default: bool) -> bool:
        """发出 ``confirm.request``，等待同一 ``request_id`` 的回答。

        ``prompt`` 是展示给用户的文字，``default`` 是无回答时的安全选择。方法先注册
        Future，再发送 JSON-RPC notification，随后最多等待 ``_CONFIRM_HARD_LIMIT_S``。

        正常返回 TUI 提交的布尔值；hard-limit timeout、EOF（:meth:`cancel_all`）或任何
        internal error 均返回 ``default``。外部 task cancellation 不会在此转成默认值，
        而是向上游传播。无论哪种退出路径，pending 映射都会在 ``finally`` 中清理。
        """
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = _PendingConfirm(future=future, default=default)
        try:
            await self._send_frame(
                {
                    "jsonrpc": "2.0",
                    "method": "confirm.request",
                    "params": {
                        "request_id": request_id,
                        "prompt": prompt,
                        "default": default,
                    },
                }
            )
            return await asyncio.wait_for(future, _CONFIRM_HARD_LIMIT_S)
        except asyncio.TimeoutError:
            return default
        except Exception:  # noqa: BLE001 — 失效保护：工作线程必须得到布尔值
            logger.exception("confirm_broker: await_confirm failed for {}", request_id)
            return default
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, answer: bool) -> bool:
        """用 ``answer`` 完成指定 ``request_id`` 的 pending confirm。

        找到尚未完成的 Future 时设置结果并返回 ``True``。未知、已超时、已取消或已经回答
        的 ID 返回 ``False``，且不修改任何状态，因此重复 ``confirm.respond`` 是幂等的。
        返回值只说明 Broker 是否接纳本次回答，不代表被确认的业务动作已经执行。
        """
        pending = self._pending.get(request_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(answer)
        return True

    def cancel_all(self) -> None:
        """把所有等待中的确认安全收敛到各自默认值。

        ``RpcServer`` 在 connection EOF 或关闭路径调用本方法，使每个 Future 都能结束，
        避免 Runtime coroutine 永久等待。已完成的 Future 保持不变；pending 映射仍由各自
        ``await_confirm()`` 的 ``finally`` 清理。这里的 cancel 是协议降级，不抛出
        ``CancelledError``。
        """
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_result(pending.default)


__all__ = ["ConfirmBroker", "SendFrame", "_CONFIRM_HARD_LIMIT_S"]
