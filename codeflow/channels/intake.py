"""Inbound Intake：把 Channel Raw Inbound 转成 Permission-checked Spine ``TurnRequest``。

`Intake` 作为 Framework Service Inject 到 Adapter，采用 Composition 而非 Inheritance。数据流是先检查
Sender Policy，再把 Platform Chat/Sender/Media/Metadata 规范成 `Source`/`TurnRequest`，最后交给 Gateway
Wired Spine Dispatch 与 Scheduler。

被 Allow 只通过当前名单；Submit Return 表示 Dispatch Call 返回，不等于 Turn 完成或 Reply Delivered。
Shutdown 时 Seal、Cancel、Wait-idle 共同封住新 Inbound 并收敛已接受工作。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from codeflow.auth.allowlist import is_allowed
from codeflow.spine import ChatType, Media, Origin, Source, TurnRequest


class Intake:
    """Per-channel Inbound Gate + Submitter，每个 Channel 一个实例。

    ``allow_check`` 可为平台 Bespoke Permission Rule Override Default Allowlist。Permitted Message 通过
    ``set_submit`` Wiring 的 Spine Dispatch 发送为 ``TurnRequest``，Channel Metadata 放在 ``Source.extras``。
    实例拥有 Seal Flag、Inflight Counter/Tasks 与 Idle Event，支撑 Gateway Shutdown Barrier。
    """

    def __init__(
        self,
        channel_name: str,
        config: Any,
        allow_check: Callable[[str], bool] | None = None,
    ):
        self.channel_name = channel_name
        self.config = config
        self._allow_check = allow_check
        self._submit: Callable[[TurnRequest], Awaitable[None]] | None = None
        self._sealed = False
        self._inflight = 0
        self._inflight_tasks: set[asyncio.Task[Any]] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    def set_submit(self, submit: Callable[[TurnRequest], Awaitable[None]]) -> None:
        """Wire Gateway 的 Spine Inbound Dispatch。

        ``publish`` 通过它提交 `TurnRequest`。Dispatch Control-aware，会拦截 ``/stop`` / ``/restart``；Intake
        保持 Dumb Gate + Builder，Control/Cancel Logic 位于拥有 Scheduler/Agent 的 Gateway。设置只保存
        Callable，不探活。
        """
        self._submit = submit

    def is_allowed(self, sender_id: str) -> bool:
        """执行 Deny-by-default Allowlist Check，Custom ``allow_check`` 优先。

        ``*`` 表示 All，Empty 表示 Deny All。返回 True 只通过 Sender Gate，不验证消息签名。
        """
        if self._allow_check is not None:
            return self._allow_check(sender_id)
        return is_allowed(self.channel_name, sender_id, getattr(self.config, "allow_from", None))

    def seal(self) -> None:
        """Gateway Shutdown 期间 Permanently Reject 新 Inbound Publishes。

        Seal 不 Cancel 已 Inflight Task，也不可 Unseal；已接受工作由 `wait_idle`/`cancel_inflight` 管理。
        """
        self._sealed = True

    async def wait_idle(self) -> None:
        """等待 ``seal`` 前已经 Admitted 的所有 Publish 返回。

        Event 在 Inflight Count 降到零时 Set；方法不判断这些 Publish 的 Turn/Delivery Outcome。
        """
        await self._idle.wait()

    def cancel_inflight(self) -> int:
        """Cancel Shutdown 前 Admitted 的 Inflight Publish Tasks，并返回 Count。

        只发出 Cancellation，不 Await Completion；Caller 随后应 `wait_idle`。没有 Current Task Reference 的
        Publish 仍由 Counter 跟踪但不在返回集合。
        """
        tasks = tuple(self._inflight_tasks)
        for task in tasks:
            task.cancel()
        return len(tasks)

    async def publish(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """检查 Permission，并把 Message 作为 `TurnRequest` 提交给 Spine Dispatch。

        Sealed、Denied 或未 Wire Dispatch 时 Log + Drop，不 Raise。Accepted Path 把 IDs Stringify、Metadata
        写入 Extras、推断 Group/DM Shape、Media Path 包成 Generic File，并可用 `session_key` Override
        Conversation。`finally` 无论 Success/Exception/Cancel 都释放 Inflight State。
        """
        if self._sealed:
            logger.info(
                "Intake for channel {} is sealed; dropping inbound from {}",
                self.channel_name,
                sender_id,
            )
            return
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender {} on channel {}. Add them to allowFrom list in config to grant access.",
                sender_id,
                self.channel_name,
            )
            return

        if self._submit is None:
            logger.error(
                "Intake for channel {} has no spine dispatch wired; dropping inbound from {}",
                self.channel_name,
                sender_id,
            )
            return

        self._inflight += 1
        task = asyncio.current_task()
        if task is not None:
            self._inflight_tasks.add(task)
        self._idle.clear()
        try:
            meta = metadata or {}
            await self._submit(
                TurnRequest(
                    origin=Origin.USER,
                    source=Source(
                        channel=self.channel_name,
                        chat_id=str(chat_id),
                        sender_id=str(sender_id),
                        # 该值不承载处理语义（channel 从 extras 携带的 metadata
                        # 读取真实 chat_type），这里只为 Spine 尽力提供形状：
                        # channel 声明为 group 时使用 group，否则使用 DM。
                        chat_type=ChatType.GROUP if meta.get("chat_type") == "group" else ChatType.DM,
                        extras=meta,
                    ),
                    text=content,
                    media=tuple(Media(path=p, mime="application/octet-stream", kind="file") for p in (media or [])),
                    # session_key_override -> conversation：run_turn 的 cid
                    # 使用 `conversation or channel:chat_id`。
                    conversation=session_key,
                )
            )
        finally:
            if task is not None:
                self._inflight_tasks.discard(task)
            self._inflight -= 1
            if self._inflight == 0:
                self._idle.set()
