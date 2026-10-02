"""管理 per-session subscription，并用 16ms coalesce loop 合并 token event。

registry 维护 ``session_key → [Subscription]``。每个 subscription 拥有 capacity 512 的
bounded ``asyncio.Queue`` 和独立 consumer task；consumer 每 16ms 形成 batch，只合并连续
``token.delta``，其他 event 原样通过并保持顺序。Queue overflow 时只影响该 subscription：
发送 ``error(code=-32016)`` 后关闭，其他 subscriber 继续工作。

``SubscriptionEmitter`` 由 RPC server 拥有，并传给
``register_turn_methods(dispatcher, emitter)``。``emit()`` 返回只表示 event 已入 queue 或
overflow 已处理；真正写入 transport 发生在后台 coalesce task，仍不等于 frontend 已渲染。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from loguru import logger

COALESCE_WINDOW_S = 0.016
QUEUE_CAPACITY = 512


@dataclass
class Subscription:
    sub_id: str
    session_key: str
    queue: asyncio.Queue
    coalesce_task: asyncio.Task | None = None
    closed: bool = False


SendFrame = Callable[[dict[str, Any]], Awaitable[None]]


class SubscriptionEmitter:
    """把 TurnEvent notification 路由到对应 Session 的 subscribers。

    实例生命周期与 ``RpcServer`` 一致，拥有 session/id 两套 index。每个 ``register()``
    创建独立 queue 与 coalesce task；``unregister()`` 关闭单项，``close_session()`` 关闭一个
    Session 的全部订阅。subscription 是 Session-scoped，Turn 结束或取消不会自动关闭。

    本对象只保证进程内 FIFO/coalesce 规则，不持久化 event，也没有 transport ack。
    """

    def __init__(self, send_frame: SendFrame) -> None:
        self._send_frame = send_frame
        self._by_session: dict[str, list[Subscription]] = {}
        self._by_id: dict[str, Subscription] = {}

    async def register(self, session_key: str) -> str:
        """为 ``session_key`` 创建 subscription，启动 coalesce loop，并返回 ``sub_id``。

        ID 使用随机 UUID hex。queue capacity 固定为 ``QUEUE_CAPACITY``；task 创建后立即放入
        两套 index。返回成功表示订阅已可接收后续 event，不重放注册前历史。
        """
        sub = Subscription(
            sub_id=uuid4().hex,
            session_key=session_key,
            queue=asyncio.Queue(maxsize=QUEUE_CAPACITY),
        )
        sub.coalesce_task = asyncio.create_task(self._coalesce_loop(sub))
        self._by_session.setdefault(session_key, []).append(sub)
        self._by_id[sub.sub_id] = sub
        return sub.sub_id

    async def unregister(self, sub_id: str) -> bool:
        """关闭存在且仍 open 的 subscription，操作幂等。

        首次关闭会取消 coalesce task、移除 indexes 并返回 ``True``；unknown 或已关闭 ID
        返回 ``False``。队列中尚未发送的 event 会被丢弃。
        """
        sub = self._by_id.get(sub_id)
        if sub is None or sub.closed:
            self._by_id.pop(sub_id, None)
            return False
        self._mark_closed(sub)
        return True

    async def emit(self, session_key: str, event: dict[str, Any]) -> None:
        """把 ``event`` 推入 ``session_key`` 的每个 open subscriber queue。

        无 subscriber 时 no-op。每个 queue 独立使用 ``put_nowait``；overflow 时向该
        subscriber 直接发送 error event 并关闭它，其他 subscriber 不受影响。函数不验证
        event schema，调用方必须传入 OpenRPC 允许的 TurnEvent shape。
        """
        for sub in list(self._by_session.get(session_key, [])):
            if sub.closed:
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                await self._close_overflow(sub)

    async def close_session(self, session_key: str) -> None:
        """关闭属于 ``session_key`` 的全部 subscription。

        对当前 bucket snapshot 逐项调用幂等 ``unregister``。函数完成后该 Session 不再接收
        event，但不会取消对应 Turn。
        """
        for sub in list(self._by_session.get(session_key, [])):
            await self.unregister(sub.sub_id)

    # ------------------------------------------------------------------
    # 内部状态
    # ------------------------------------------------------------------

    def _mark_closed(self, sub: Subscription) -> None:
        """标记 subscription closed，取消 consumer，并从两套 index 删除。

        最后一个 session subscription 删除后同时移除空 bucket。此函数不 await 被取消 task，
        也不发送 terminal notification。
        """
        sub.closed = True
        if sub.coalesce_task is not None and not sub.coalesce_task.done():
            sub.coalesce_task.cancel()
        self._by_id.pop(sub.sub_id, None)
        bucket = self._by_session.get(sub.session_key)
        if bucket is not None:
            bucket[:] = [s for s in bucket if s.sub_id != sub.sub_id]
            if not bucket:
                del self._by_session[sub.session_key]

    async def _coalesce_loop(self, sub: Subscription) -> None:
        """运行单个 subscription 的 16ms window coalesce loop。

        每轮先阻塞等待首个 event，再 sleep 16ms 收集 batch，随后 non-blocking drain queue；
        ``_merge_consecutive_token_deltas`` 只合并连续 ``token.delta``，其他 event 保序；最后把每个
        merged event 包装成 method ``event`` 的 JSON-RPC notification 写出。

        ``CancelledError`` 是正常关闭并被吞掉；其他异常记录日志，避免传播到 server read
        pump。异常退出不会自动重启 consumer。
        """
        try:
            while not sub.closed:
                first = await sub.queue.get()
                batch: list[dict[str, Any]] = [first]
                await asyncio.sleep(COALESCE_WINDOW_S)
                while not sub.queue.empty():
                    batch.append(sub.queue.get_nowait())
                merged = _merge_consecutive_token_deltas(batch)
                for event in merged:
                    if sub.closed:
                        return
                    await self._send_frame(
                        {
                            "jsonrpc": "2.0",
                            "method": "event",
                            "params": {
                                "subscription_id": sub.sub_id,
                                "event": event,
                            },
                        }
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("subscription coalesce loop crashed sub_id={}", sub.sub_id)

    async def _close_overflow(self, sub: Subscription) -> None:
        """发送 ``-32016`` overflow notification，再关闭 subscription。

        code ``-32016`` 来自 extension range ``-32016..-32049``。early draft 曾使用
        ``-32010``，但它与 live ``ConfigFieldReadonlyError`` 冲突。即使 notification 写出
        失败，``finally`` 仍会关闭 subscription，防止继续无界积压。
        """
        if sub.closed:
            return
        try:
            await self._send_frame(
                {
                    "jsonrpc": "2.0",
                    "method": "event",
                    "params": {
                        "subscription_id": sub.sub_id,
                        "event": {
                            "type": "error",
                            "payload": {
                                "code": -32016,
                                "message": "subscription_capacity_exceeded",
                                "reason": "internal",
                            },
                        },
                    },
                }
            )
        except Exception:
            logger.exception(
                "failed to send overflow notification sub_id={}",
                sub.sub_id,
            )
        finally:
            self._mark_closed(sub)


def _merge_consecutive_token_deltas(
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把连续 ``token.delta`` run 折叠为单个 merged frame。

    delta 的 ``payload.text`` 按原顺序拼接；任何 non-delta event 都终止当前 run，并原样加入
    结果，因此 overall event ordering 保持不变。空 batch 返回空 list。函数假定 delta
    payload 已含 text，不执行 schema validation。
    """
    result: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for event in batch:
        if event.get("type") == "token.delta":
            if pending is None:
                pending = {
                    "type": "token.delta",
                    "payload": {"text": event["payload"]["text"]},
                }
            else:
                pending["payload"]["text"] += event["payload"]["text"]
        else:
            if pending is not None:
                result.append(pending)
                pending = None
            result.append(event)
    if pending is not None:
        result.append(pending)
    return result


__all__ = [
    "COALESCE_WINDOW_S",
    "QUEUE_CAPACITY",
    "SendFrame",
    "Subscription",
    "SubscriptionEmitter",
]
