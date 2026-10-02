"""管理以 ``conversation_id`` 为键的 sync-to-async ask-user 往返。

``QuestionBroker`` 把 :class:`ConfirmBroker` 的等待模式扩展到 ``ask_user`` Tool。二者的 key
不同：confirm 自己生成 ``request_id``，调用者也只持有它；ask_user 的回答还可能来自入站
Channel message，唯一稳定 handle 是 ``conversation_id``。因此 pending registry 以
``conversation_id`` 为键，同时仍生成 internal ``request_id`` 放入 notification，允许偏好
request_id 的 frontend 回答。

与 ConfirmBroker 一样，本对象 transport-agnostic：构造参数是 notification emit callable，
production 绑定到 ``RpcServer.send_frame``。timeout、internal error、connection EOF（通过
:meth:`cancel_all`）都 fail-safe 为 prompt ``default``；Task cancellation 继续传播，避免
取消 Turn 后用 fabricated answer 恢复 Agent。得到 answer 只恢复暂停的 Tool call，不表示
整个任务完成。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from loguru import logger

SendFrame = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class _PendingQuestion:
    future: asyncio.Future
    request_id: str
    default: str


class QuestionBroker:
    """发送 ``clarify.request`` notification，并等待对应 answer。

    实例由 ``RpcServer`` 拥有，维护 ``conversation_id -> pending`` 与反向
    ``request_id -> conversation_id`` 两张表。每个 conversation 最多一个 pending question，
    因为单个 Turn 是 serial，不能并发 ask 两次；若出现 overlap，旧问题先用自身 default
    收敛，再由新问题替换。

    Broker 不持久化问题或答案。Server 关闭时应调用 ``cancel_all()``，否则 Tool coroutine
    可能永久等待。
    """

    def __init__(self, send_frame: SendFrame) -> None:
        self._send_frame = send_frame
        self._pending: dict[str, _PendingQuestion] = {}
        # 建立 request_id -> conversation_id 反向索引，使 :meth:`reply` 能接受任一句柄。
        self._by_request: dict[str, str] = {}

    async def await_question(
        self,
        conversation_id: str,
        *,
        prompt: str,
        choices: list[str] | None = None,
        default: str = "",
        timeout_s: float = 600.0,
    ) -> str:
        """发出 ``clarify.request``，等待匹配的 answer string。

        ``conversation_id`` 标识 Turn 对话，``prompt`` 是问题，``choices`` 可为空，
        ``default`` 是失败降级答案，``timeout_s`` 默认 600 秒。同一 conversation 已有问题时
        视为 programming error：记录 overlap，让 stale Future 以自己的 default 完成，再
        替换为新问题。

        notification 同时携带 ``conversation_id``、新生成的 ``request_id``、``question`` 和
        ``choices``。timeout、EOF（:meth:`cancel_all`）或 internal error 返回 ``default``；
        external task cancellation 传播。``finally`` 只清理当前 request 自己的记录，避免
        overlap 中旧 coroutine 删除新 pending state。
        """
        existing = self._pending.get(conversation_id)
        if existing is not None:
            logger.error(
                "question_broker: overlapping question for conversation {}; fail-safing the stale one",
                conversation_id,
            )
            self._by_request.pop(existing.request_id, None)
            if not existing.future.done():
                existing.future.set_result(existing.default)

        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[conversation_id] = _PendingQuestion(future=future, request_id=request_id, default=default)
        self._by_request[request_id] = conversation_id
        try:
            # ``clarify.request`` 是 ui-tui 前端已有的多选提示协议：
            # {question, choices, request_id} -> ClarifyPrompt -> clarify.respond。代理器复用它；
            # conversation_id 供网关通道路由使用，前端会忽略额外键。
            await self._send_frame(
                {
                    "jsonrpc": "2.0",
                    "method": "clarify.request",
                    "params": {
                        "conversation_id": conversation_id,
                        "request_id": request_id,
                        "question": prompt,
                        "choices": choices or [],
                    },
                }
            )
            return await asyncio.wait_for(future, timeout_s)
        except asyncio.TimeoutError:
            return default
        except Exception:  # noqa: BLE001 — 失效保护：循环必须得到字符串
            logger.exception("question_broker: await_question failed for {}", conversation_id)
            return default
        finally:
            # 只撤回当前请求自己的记录；重叠问题可能已在同一 conversation_id 下替换了它。
            current = self._pending.get(conversation_id)
            if current is not None and current.request_id == request_id:
                self._pending.pop(conversation_id, None)
            self._by_request.pop(request_id, None)

    def pending_req(self, conversation_id: str) -> str | None:
        """返回 conversation 当前 pending ``request_id``，不存在时返回 ``None``。

        该值用于 Channel ingress 或测试定位等待项；读取不会延长 timeout，也不会消费问题。
        """
        pending = self._pending.get(conversation_id)
        return pending.request_id if pending is not None else None

    def reply(self, key: str, answer: str) -> bool:
        """使用 ``conversation_id`` OR ``request_id`` 完成 pending question。

        找到未完成 Future 时写入 ``answer`` 并返回 ``True``。unknown key、已超时或已经回答
        返回 ``False``，所以重复 reply 幂等。返回值只表示 Broker 接纳答案。
        """
        conversation_id = key if key in self._pending else self._by_request.get(key)
        if conversation_id is None:
            return False
        pending = self._pending.get(conversation_id)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(answer)
        return True

    def cancel_all(self) -> None:
        """让所有 pending question 以各自 default 安全完成。

        connection EOF 或 server shutdown 时调用。已完成 Future 保持不变；registry 由各自
        ``await_question()`` 的 ``finally`` 清理。此处不会抛出 ``CancelledError``。
        """
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_result(pending.default)


__all__ = ["QuestionBroker", "SendFrame"]
