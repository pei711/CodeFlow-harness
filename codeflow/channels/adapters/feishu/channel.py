"""把 Feishu/Lark 消息接入 CodeFlow Runtime，并把 Runtime 回复交付回会话。

没有 Agent 开发经验的读者可以把本模块理解为一座双向桥：入站方向由
``lark-oapi`` 的 WebSocket long connection 接收事件，完成去重、群聊寻址、
allowlist 校验、媒体下载和语音转写后，通过 ``ChannelIntake`` 发布给 Runtime；
出站方向则把文本、卡片、图片或文件转换为 lark Open API 请求。

关键入口是 ``FeishuChannel``。它持有 SDK client、WebSocket 线程、主事件循环引用
和本进程内的消息去重窗口。收到事件只表示平台已把消息交给适配器；发布到
``ChannelIntake`` 也不等于 Agent 任务完成。相应地，API 返回成功只证明 Feishu
接受了发送请求，不证明用户已阅读回复，更不能单独作为任务完成或正向结论可用的证据。
本模块不负责 Agent 推理、Session 持久化、调度策略，也不负责配置凭证或授予
``speech_to_text:speech`` 权限。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import lark_oapi as lark
from loguru import logger

from codeflow.channels.adapters.feishu import cards, content
from codeflow.channels.base import ChannelBase
from codeflow.channels.errors import transient_network
from codeflow.channels.media import save_media_bytes
from codeflow.channels.transcribe import transcribe_audio
from codeflow.config.schema import FeishuConfig
from codeflow.spine.delivery import TerminalDeliveryError

_MSG_TYPE_LABEL = {"image": "[image]", "audio": "[audio]", "file": "[file]", "sticker": "[sticker]"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif"}
_PLAYABLE_EXTS = {".opus", ".mp4", ".mov", ".avi"}
_FILE_TYPE = {
    ".opus": "opus",
    ".mp4": "mp4",
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "doc",
    ".xls": "xls",
    ".xlsx": "xls",
    ".ppt": "ppt",
    ".pptx": "ppt",
}
_DEDUP_CAP = 1000


class FeishuChannel(ChannelBase):
    """通过 WebSocket long connection 运行的 Feishu bot 适配器。

    实例随 Channel Runtime 启动并在 ``stop()`` 时停止接收入站消息，不需要 public
    IP 或 webhook。对象拥有 lark Open API client、WebSocket 专用线程、回投主
    ``asyncio`` loop 的引用、最近消息去重表以及本次生命周期内的 native STT
    禁用状态；这些状态都不会跨进程持久化。

    入站消息最终交给基类注入的 ``ChannelIntake``，出站交付由 ``send()`` 完成。
    ``ChannelManager`` 管理实例生命周期，``codeflow.spine`` 消费已发布的入站消息。
    适配器只确认平台调用是否成功，不拥有 Agent 任务状态、最终交付判定或用户已读状态。
    """

    name = "feishu"
    display_name = "Feishu"

    config: FeishuConfig

    def __init__(self, config: FeishuConfig):
        super().__init__(config)
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._native_stt_disabled = False

    # ── 生命周期 ───────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.config.app_id or not self.config.app_secret:
            logger.error("Feishu app_id and app_secret not configured")
            return
        self._running = True
        self._loop = asyncio.get_running_loop()

        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        builder = lark.EventDispatcherHandler.builder(
            self.config.encrypt_key or "",
            self.config.verification_token or "",
        ).register_p2_im_message_receive_v1(self._on_message_sync)
        for method in (
            "register_p2_im_message_reaction_created_v1",
            "register_p2_im_message_message_read_v1",
            "register_p2_im_chat_access_event_bot_p2p_chat_entered_v1",
        ):
            register = getattr(builder, method, None)
            if callable(register):
                builder = register(self._ignore_event)

        self._ws_client = lark.ws.Client(
            self.config.app_id,
            self.config.app_secret,
            event_handler=builder.build(),
            log_level=lark.LogLevel.INFO,
        )
        self._ws_thread = threading.Thread(target=self._run_ws_supervised, daemon=True)
        self._ws_thread.start()

        logger.info("Feishu bot started (WebSocket long connection, no public IP needed)")
        while self._running:
            await asyncio.sleep(1)

    def _run_ws_supervised(self) -> None:
        """在专用线程和事件循环中监督 lark WebSocket client。

        ``lark_oapi`` 会在模块级保存 ``loop = asyncio.get_event_loop()``。如果直接
        复用 CodeFlow 已运行的主事件循环，SDK 自己驱动 loop 时会发生冲突。因此本方法为
        WebSocket 线程建立一个空闲 loop，并把 ``lark_oapi.ws.client.loop`` 指向它。

        ``start()`` 抛出的异常会记录为警告；只要 channel 仍处于 running 状态，就固定
        等待 5 秒后重连。退出循环后关闭线程自己的 loop，但不会关闭 CodeFlow 主 loop。
        """
        import lark_oapi.ws.client as lark_ws

        ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(ws_loop)
        lark_ws.loop = ws_loop
        try:
            while self._running:
                try:
                    self._ws_client.start()
                except Exception as e:
                    logger.warning("Feishu WebSocket error: {}", e)
                if self._running:
                    time.sleep(5)
        finally:
            ws_loop.close()

    async def stop(self) -> None:
        # lark.ws.Client 没有 stop()；丢弃引用并退出即可关闭。
        self._running = False
        logger.info("Feishu bot stopped")

    # ── 群组寻址 ──────────────────────────────────────────────────

    def _is_bot_mentioned(self, message: Any) -> bool:
        if "@_all" in (message.content or ""):
            return True
        for mention in getattr(message, "mentions", None) or []:
            mid = getattr(mention, "id", None)
            if mid and not getattr(mid, "user_id", None) and (getattr(mid, "open_id", None) or "").startswith("ou_"):
                return True
        return False

    def _addressed_to_bot(self, message: Any) -> bool:
        return self.config.group_policy == "open" or self._is_bot_mentioned(message)

    # ── 反应（确认交互）───────────────────────────────────────────

    async def _react(self, message_id: str, emoji_type: str) -> None:
        if not self._client:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._react_sync, message_id, emoji_type)

    def _react_sync(self, message_id: str, emoji_type: str) -> None:
        from lark_oapi.api.im.v1 import (
            CreateMessageReactionRequest,
            CreateMessageReactionRequestBody,
            Emoji,
        )

        try:
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(Emoji.builder().emoji_type(emoji_type).build())
                    .build()
                )
                .build()
            )
            response = self._client.im.v1.message_reaction.create(request)
            if not response.success():
                logger.warning("Failed to add reaction: code={}, msg={}", response.code, response.msg)
        except Exception as e:
            logger.warning("Error adding reaction: {}", e)

    # ── 出站 ──────────────────────────────────────────────────────

    async def send(self, chat_id: str, content: str, media: list[str] | None = None) -> None:
        if not self._client:
            raise TerminalDeliveryError("Feishu client not initialized")
        receive_id_type = "chat_id" if chat_id.startswith("oc_") else "open_id"
        loop = asyncio.get_running_loop()
        try:
            for path in media or []:
                await self._send_one_media(loop, receive_id_type, chat_id, path)
            if content and content.strip():
                await self._send_text(loop, receive_id_type, chat_id, content)
        except TerminalDeliveryError:
            raise
        except Exception as e:
            if transient_network(e):
                raise
            raise TerminalDeliveryError(f"Feishu send failed: {e}") from e

    async def _send_one_media(self, loop, receive_id_type, chat_id, path) -> None:
        if not os.path.isfile(path):
            logger.warning("Media file not found: {}", path)
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in _IMAGE_EXTS:
            key = await loop.run_in_executor(None, self._upload_image_sync, path)
            if key:
                await self._post(loop, receive_id_type, chat_id, "image", {"image_key": key})
        else:
            key = await loop.run_in_executor(None, self._upload_file_sync, path)
            if key:
                kind = "media" if ext in _PLAYABLE_EXTS else "file"
                await self._post(loop, receive_id_type, chat_id, kind, {"file_key": key})

    async def _send_text(self, loop, receive_id_type, chat_id, text) -> None:
        fmt = cards.detect_format(text)
        if fmt == "text":
            await self._post_raw(loop, receive_id_type, chat_id, "text", cards.text_payload(text))
        elif fmt == "post":
            await self._post_raw(loop, receive_id_type, chat_id, "post", cards.post_payload(text))
        else:
            for payload in cards.card_payloads(text):
                await self._post_raw(loop, receive_id_type, chat_id, "interactive", payload)

    async def _post(self, loop, receive_id_type, chat_id, msg_type, body: dict) -> None:
        await self._post_raw(loop, receive_id_type, chat_id, msg_type, json.dumps(body, ensure_ascii=False))

    async def _post_raw(self, loop, receive_id_type, chat_id, msg_type, content_json: str) -> None:
        await loop.run_in_executor(None, self._send_message_sync, receive_id_type, chat_id, msg_type, content_json)

    def _send_message_sync(self, receive_id_type: str, receive_id: str, msg_type: str, content_json: str) -> None:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        request = (
            CreateMessageRequest.builder()
            .receive_id_type(receive_id_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(receive_id)
                .msg_type(msg_type)
                .content(content_json)
                .build()
            )
            .build()
        )
        response = self._client.im.v1.message.create(request)
        if not response.success():
            raise TerminalDeliveryError(
                f"Feishu {msg_type} message rejected: code={response.code}, "
                f"msg={response.msg}, log_id={response.get_log_id()}"
            )
        sent_id = getattr(getattr(response, "data", None), "message_id", None)
        logger.info("Feishu message sent: msg_type={} message_id={}", msg_type, sent_id)

    # ── 媒体上传/下载（逐适配器使用 Lark SDK）────────────────────

    def _upload_image_sync(self, path: str) -> str | None:
        from lark_oapi.api.im.v1 import CreateImageRequest, CreateImageRequestBody

        try:
            with open(path, "rb") as f:
                request = (
                    CreateImageRequest.builder()
                    .request_body(CreateImageRequestBody.builder().image_type("message").image(f).build())
                    .build()
                )
                response = self._client.im.v1.image.create(request)
                if response.success():
                    return response.data.image_key
                logger.error("Failed to upload image: code={}, msg={}", response.code, response.msg)
        except Exception as e:
            logger.error("Error uploading image {}: {}", path, e)
        return None

    def _upload_file_sync(self, path: str) -> str | None:
        from lark_oapi.api.im.v1 import CreateFileRequest, CreateFileRequestBody

        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "rb") as f:
                request = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type(_FILE_TYPE.get(ext, "stream"))
                        .file_name(os.path.basename(path))
                        .file(f)
                        .build()
                    )
                    .build()
                )
                response = self._client.im.v1.file.create(request)
                if response.success():
                    return response.data.file_key
                logger.error("Failed to upload file: code={}, msg={}", response.code, response.msg)
        except Exception as e:
            logger.error("Error uploading file {}: {}", path, e)
        return None

    def _download_resource_sync(
        self, message_id: str, file_key: str, resource_type: str
    ) -> tuple[bytes | None, str | None]:
        from lark_oapi.api.im.v1 import GetMessageResourceRequest

        # 飞书只接受 'image' 或 'file'；音频通过 file 端点发送。
        api_type = "file" if resource_type == "audio" else resource_type
        try:
            request = (
                GetMessageResourceRequest.builder().message_id(message_id).file_key(file_key).type(api_type).build()
            )
            response = self._client.im.v1.message_resource.get(request)
            if response.success():
                data = response.file
                return (data.read() if hasattr(data, "read") else data), response.file_name
            logger.error("Failed to download {}: code={}, msg={}", resource_type, response.code, response.msg)
        except Exception:
            logger.exception("Error downloading {} {}", resource_type, file_key)
        return None, None

    async def _download_media(
        self, msg_type: str, content_json: dict, message_id: str | None
    ) -> tuple[str | None, str]:
        loop = asyncio.get_running_loop()
        data = filename = None
        if msg_type == "image":
            key = content_json.get("image_key")
            if key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_resource_sync, message_id, key, "image"
                )
                filename = filename or f"{key[:16]}.jpg"
        elif msg_type in ("audio", "file", "media"):
            key = content_json.get("file_key")
            if key and message_id:
                data, filename = await loop.run_in_executor(
                    None, self._download_resource_sync, message_id, key, msg_type
                )
                filename = filename or key[:16]
                if msg_type == "audio" and not filename.endswith(".opus"):
                    filename = f"{filename}.opus"
        if data and filename:
            path = save_media_bytes("feishu", data, filename)
            return str(path), f"[{msg_type}: {path.name}]"
        return None, f"[{msg_type}: download failed]"

    # ── 转写 ──────────────────────────────────────────────────────

    async def _transcribe(self, path: str) -> str:
        """把语音文件转成文本，优先使用 Feishu native STT，失败时回退到 Whisper。

        ``path`` 是已下载到本地的语音文件路径。Feishu speech-to-text 不需要额外的
        provider key；当它返回非空文本时直接作为结果。若 native STT 在本次 channel
        生命周期内已被判定为权限不足或租户套餐不可用，后续消息会跳过该调用，避免每次
        都承担已知无效的网络延迟。其余失败由 ``transcribe_audio()`` 使用基类配置的
        Whisper provider 处理。

        返回值是转写文本；下游仍需把它作为用户输入参与 Agent 执行。转写成功只证明
        获得了文字，不证明语义正确，也不代表任务完成或回复已经交付。
        """
        if not self._native_stt_disabled:
            loop = asyncio.get_running_loop()
            native = await loop.run_in_executor(None, self._lark_stt_sync, path)
            if native:
                return native
        return await transcribe_audio(path, self.transcription_api_key, channel=self.name)

    def _lark_stt_sync(self, path: str) -> str | None:
        """同步调用 Feishu ``file_recognize`` API 识别一个语音文件。

        Feishu 语音消息采用 ``opus``，API 可以直接接收，因此这里只做 base64 编码，
        不进行转码。请求固定使用 ``format="opus"`` 和 ``engine_type="16k_auto"``，
        并为每次识别生成新的 ``file_id``。错误码 ``99991400`` 最多退避重试两次；
        其他确定性失败会调用 ``_disable_native_stt()``，避免后续重复尝试。

        成功时返回 ``recognition_text``；任何 API 拒绝、空结果或异常均返回 ``None``，
        由调用方回退到 Whisper。``None`` 是降级信号，不表示音频内容为空。
        """
        from lark_oapi.api.speech_to_text.v1 import (
            FileConfig,
            FileRecognizeSpeechRequest,
            FileRecognizeSpeechRequestBody,
            Speech,
        )

        try:
            audio_b64 = base64.b64encode(Path(path).read_bytes()).decode()
            request = (
                FileRecognizeSpeechRequest.builder()
                .request_body(
                    FileRecognizeSpeechRequestBody.builder()
                    .speech(Speech.builder().speech(audio_b64).build())
                    .config(
                        FileConfig.builder().format("opus").engine_type("16k_auto").file_id(uuid.uuid4().hex).build()
                    )
                    .build()
                )
                .build()
            )
            # file_recognize 有 QPS 限制（错误码 99991400）；该限制是暂时的，
            # 因此先退避重试数次，再放弃并回退到 Whisper。
            for attempt in range(3):
                response = self._client.speech_to_text.v1.speech.file_recognize(request)
                if response.success() and response.data and response.data.recognition_text:
                    return response.data.recognition_text
                code = getattr(response, "code", None)
                if code == 99991400 and attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                self._disable_native_stt(code, getattr(response, "msg", ""))
                break
        except Exception as e:
            # 可能是暂时性网络错误：本次回退，但仍保持原生转写启用。
            logger.warning("Feishu native STT failed, falling back to Whisper: {}", e)
        return None

    def _disable_native_stt(self, code: int | None, msg: str) -> None:
        """在确定性失败后停用本次生命周期内的 native STT，并记录修复提示。

        ``code`` 和 ``msg`` 来自 Feishu API。``99991672`` 表示应用缺少
        ``speech_to_text:speech`` 权限；``99991400`` 表示 ``file_recognize`` 被限流
        或当前 tenant plan 不可用；其他错误保留原始 code 和 msg 供排查。方法只设置
        内存中的 ``_native_stt_disabled``，不会修改配置或持久化状态。

        日志帮助 operator 判断应授予 scope、升级 Feishu plan，还是继续依赖 Whisper
        key。停用 native STT 不等于语音消息处理失败，因为调用链仍可以走 Whisper 回退。
        """
        self._native_stt_disabled = True
        if code == 99991672:  # 应用缺少 speech_to_text:speech 权限
            logger.warning(
                "Feishu native STT disabled: the app lacks the 'speech_to_text:speech' "
                "permission. Grant it in the Feishu developer console (then publish a "
                "new app version) for key-free transcription; using Whisper meanwhile. ({})",
                msg,
            )
        elif code == 99991400:  # 频率/可用性限制
            logger.warning(
                "Feishu native STT disabled: file_recognize is rate-limited or "
                "unavailable on this tenant's plan (the API requires a paid Feishu "
                "plan). Using Whisper instead — set providers.groq.api_key to enable "
                "it, or upgrade the Feishu plan for key-free transcription. ({})",
                msg,
            )
        else:
            logger.warning("Feishu native STT disabled (code={}, msg={}); using Whisper.", code, msg)

    # ── 入站 ──────────────────────────────────────────────────────

    def _on_message_sync(self, data: Any) -> None:
        """把 lark WebSocket（WS）线程收到的事件安全地投递到 CodeFlow 主事件循环。

        ``lark.ws.Client`` 没有 ``stop()``，底层 socket 可能在 channel 停止后继续回调；
        因此 ``_running`` 为假时必须丢弃事件，避免旧实例向已重启的 Runtime 再次发布。
        主 loop 存在且仍运行时，使用 ``asyncio.run_coroutine_threadsafe()`` 调度
        ``_on_message()``。本方法只完成跨线程调度，不代表入站消息已通过校验或被处理。
        """
        if not self._running:
            # lark.ws.Client 没有 stop()：socket 可能比 stop() 活得更久并继续投递。
            # 此处丢弃僵尸投递，确保已停止（或已重启）的实例不会再次发布。
            return
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: Any) -> None:
        try:
            message = data.event.message
            sender = data.event.sender
            message_id = message.message_id
            if message_id in self._seen:
                logger.info("Feishu duplicate event suppressed: message_id={}", message_id)
                return
            self._seen[message_id] = None
            while len(self._seen) > _DEDUP_CAP:
                self._seen.popitem(last=False)
            if sender.sender_type == "bot":
                return

            sender_id = sender.sender_id.open_id if sender.sender_id else "unknown"
            chat_type = message.chat_type
            msg_type = message.message_type
            if chat_type == "group" and not self._addressed_to_bot(message):
                return
            if not self.is_allowed(sender_id):  # 在表态或下载媒体前拒绝
                logger.info("Feishu inbound rejected by allowlist: sender={}", sender_id)
                return

            await self._react(message_id, self.config.react_emoji)

            content_text, media_paths = await self._extract(msg_type, message, message_id)
            if not content_text and not media_paths:
                return

            logger.info(
                "Feishu inbound accepted: message_id={} chat_type={} msg_type={}",
                message_id,
                chat_type,
                msg_type,
            )
            reply_to = message.chat_id if chat_type == "group" else sender_id
            await self.intake.publish(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content_text,
                media=media_paths,
                metadata={"message_id": message_id, "chat_type": chat_type, "msg_type": msg_type},
            )
        except Exception as e:
            logger.error("Error processing Feishu message: {}", e)

    async def _extract(self, msg_type: str, message: Any, message_id: str) -> tuple[str, list[str]]:
        try:
            payload = json.loads(message.content) if message.content else {}
        except json.JSONDecodeError:
            payload = {}
        parts: list[str] = []
        media: list[str] = []

        if msg_type == "text":
            if text := payload.get("text"):
                parts.append(text)
        elif msg_type == "post":
            text, image_keys = content.extract_post(payload)
            if text:
                parts.append(text)
            for key in image_keys:
                path, label = await self._download_media("image", {"image_key": key}, message_id)
                if path:
                    media.append(path)
                parts.append(label)
        elif msg_type in ("image", "audio", "file", "media"):
            path, label = await self._download_media(msg_type, payload, message_id)
            if path:
                media.append(path)
            if msg_type == "audio" and path:
                if transcription := await self._transcribe(path):
                    label = f"[transcription: {transcription}]"
            parts.append(label)
        elif msg_type in ("share_chat", "share_user", "interactive", "share_calendar_event", "system", "merge_forward"):
            if text := content.extract_share_card(payload, msg_type):
                parts.append(text)
        else:
            parts.append(_MSG_TYPE_LABEL.get(msg_type, f"[{msg_type}]"))

        return "\n".join(parts), media

    @staticmethod
    def _ignore_event(_data: Any) -> None:
        """消费但不处理 reaction、read 与 p2p-enter 事件，避免 SDK 输出无关噪声。

        这些事件会被 dispatcher 正常接收，但当前 Runtime 不把它们转换为 Agent 输入，
        也不据此推断用户已读、任务完成或交付成功。参数 ``_data`` 被有意忽略。
        """
