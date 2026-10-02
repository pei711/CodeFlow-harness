"""使用 botpy SDK 的 WebSocket 接入 QQ C2C、group 与 direct messages。

本模块解决的是 QQ 平台事件如何进入 CodeFlow Runtime、Runtime 文本回复又如何回到原会话。
``_make_bot_class()`` 创建的 botpy ``Client`` 子类只负责把 SDK 回调转交给
``QQChannel``；纯粹的寻址和内容归一化位于 :mod:`.parsing`，发送则调用 botpy SDK API。
实例会缓存入站消息 ID 以去重，并保存 ``chat_id`` 对应的 ``chat_type``，使出站回复能
选择 ``post_group_message``、``post_dms`` 或 ``post_c2c_message``。

媒体在两个方向都有明确边界：入站附件只根据 botpy 暴露的 metadata 转成确定性文本
label，不下载字节，因为 botpy 没有为短期 attachment URL 提供 download helper；出站
reply endpoint 只承载 markdown text，也不会上传附件，而是为每个被舍弃的文件加入明确
notice line。SDK 接受发送请求只表示平台调用成功，不证明用户已读、Agent 任务完成或
交付结果可用于正向结论。本模块不负责 Agent 推理、Session 持久化或媒体存储。
"""

from __future__ import annotations

import asyncio
from collections import deque

import botpy
from botpy.errors import ServerError
from botpy.message import C2CMessage, GroupMessage
from loguru import logger

from codeflow.channels.adapters.qq import parsing
from codeflow.channels.base import ChannelBase
from codeflow.channels.errors import transient_network
from codeflow.channels.media import safe_name
from codeflow.config.schema import QQConfig

_RECONNECT_DELAY_S = 5
_DEDUP_CAP = 1000


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """构造把 botpy 事件转发给 ``channel`` 的 ``Client`` 子类。

    返回的类型在 ``QQChannel.start()`` 中实例化，并订阅 public messages、direct
    message、C2C 与 group at-message 回调。每个回调只把原始 message 和
    ``is_group`` 交给 ``QQChannel._on_message()``，不自行做 allowlist、去重或发布。
    ``ext_handlers=False`` 还会关闭 botpy 默认文件日志，避免只读文件系统写入失败；
    CodeFlow 的运行日志统一交给 loguru。
    """
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # 禁用 botpy 文件日志（默认 botpy.log 在只读文件系统上会失败）；
            # CodeFlow 统一通过 loguru 记录日志。
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: {}", self.robot.name)

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message, is_group=False)

        async def on_group_at_message_create(self, message: "GroupMessage"):
            await channel._on_message(message, is_group=True)

        async def on_direct_message_create(self, message):
            await channel._on_message(message, is_group=False)

    return _Bot


class QQChannel(ChannelBase):
    """使用 botpy SDK over WebSocket 的 QQ Channel 适配器。

    实例由 ``ChannelManager`` 启停，生命周期内拥有 botpy client、最近 1000 个消息 ID
    的去重队列、递增 ``msg_seq``，以及 ``chat_id -> chat_type`` 路由缓存。入站事件经
    :mod:`.parsing` 归一化后发布到 ``ChannelIntake``；出站 ``send()`` 根据缓存选择
    group、guild DM 或 C2C endpoint。

    这些缓存都只存在于当前进程，重启后不会恢复；缺失路由缓存时出站默认按 ``c2c``
    处理。实例不拥有 Turn、Agent 或 Delivery 的最终状态。入站发布成功不代表任务完成，
    平台 API 调用成功也不代表用户已经收到或阅读回复。
    """

    config: QQConfig
    name = "qq"
    display_name = "QQ"

    def __init__(self, config: QQConfig):
        super().__init__(config)
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque[str] = deque(maxlen=_DEDUP_CAP)
        self._msg_seq: int = 1
        self._chat_type_cache: dict[str, str] = {}

    # ── 生命周期 ───────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return
        self._running = True
        self._client = _make_bot_class(self)()
        logger.info("QQ bot started (C2C & Group supported)")
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                logger.warning("QQ bot error: {}", e)
            if self._running:
                logger.info("Reconnecting QQ bot in {}s...", _RECONNECT_DELAY_S)
                await asyncio.sleep(_RECONNECT_DELAY_S)

    async def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        logger.info("QQ bot stopped")

    # ── 入站 ──────────────────────────────────────────────────────

    async def _on_message(self, data: "C2CMessage | GroupMessage", is_group: bool = False) -> None:
        try:
            message_id = getattr(data, "id", None)
            if not message_id:
                logger.warning("QQ inbound dropped: event carries no message id")
                return
            if message_id in self._processed_ids:
                logger.info("QQ duplicate event suppressed: message_id={}", message_id)
                return
            self._processed_ids.append(message_id)

            chat_id, user_id, chat_type = parsing.resolve_route(data, is_group)
            if not self.is_allowed(user_id):  # 在缓存路由状态或发布前拒绝
                logger.warning("QQ inbound rejected by allowlist: sender={}", user_id)
                return

            content = parsing.compose_content(data)
            if not content:
                logger.info("QQ inbound dropped: message_id={} has no text or attachment", message_id)
                return

            self._chat_type_cache[chat_id] = chat_type
            await self.intake.publish(
                sender_id=user_id,
                chat_id=chat_id,
                content=content,
                metadata={"message_id": message_id, "chat_type": chat_type},
            )
            logger.info("QQ inbound accepted: message_id={} chat_type={}", message_id, chat_type)
        except Exception:
            logger.exception("QQ inbound dropped: event handling failed")

    # ── 出站 ──────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str, media: list[str] | None = None) -> None:
        if not self._client:
            logger.warning("QQ client not initialized")
            return
        media = media or []
        if media:
            # 此处使用的回复端点只支持文本；应向用户说明附件被丢弃，
            # 而不是静默丢失。
            logger.warning("QQ reply is text-only; {} attachment(s) not sent", len(media))
            notes = "\n".join(
                f"[Attachment not sent: {safe_name(m)}]" for m in media if isinstance(m, str) and m.strip()
            )
            content = f"{content}\n{notes}".strip()
        # 递增每条消息的序号，避免 QQ API 对回复去重。
        self._msg_seq += 1
        chat_type = self._chat_type_cache.get(chat_id, "c2c")
        try:
            if chat_type == "group":
                await self._client.api.post_group_message(
                    group_openid=chat_id,
                    msg_type=2,
                    markdown={"content": content},
                    msg_id=None,
                    msg_seq=self._msg_seq,
                )
            elif chat_type == "guild_dm":
                # Guild 私信通过 DM session（post_dms）回复；C2C 端点会拒绝
                # guild user id。post_dms 没有 msg_seq。
                await self._client.api.post_dms(
                    guild_id=chat_id,
                    content=content,
                    msg_id=None,
                )
            else:
                await self._client.api.post_c2c_message(
                    openid=chat_id,
                    msg_type=2,
                    markdown={"content": content},
                    msg_id=None,
                    msg_seq=self._msg_seq,
                )
            logger.info("QQ message sent: chat_id={} chat_type={}", chat_id, chat_type)
        except Exception as e:
            if isinstance(e, ServerError) or transient_network(e):
                raise  # 5xx 或网络中断：交给 manager._send_with_retry 退避
            logger.error("Error sending QQ message: {}", e)
