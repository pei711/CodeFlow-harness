"""实现 Agent 主动向当前或指定 Channel 用户发送消息的 Message Tool。

同一个 Tool 实例会被多个 Lane 共享，因此 Channel、Chat、message_id、发送 callback 和“本轮
是否已回复”都放在 ContextVar 的冻结 `_MsgTurn` 中。`run_turn` 可把 callback 路由到统一
Stream/Text 出口；若 Tool 已发送主回复，AgentLoop 会用 `sent_in_turn` 抑制第二次交付。
"""

from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable

from codeflow.agent.tools.base import Tool


@dataclass(frozen=True)
class _MsgTurn:
    """保存单 Turn 的发送目标、路由 callback 与已发送状态，并按 asyncio Task 隔离。

    MessageTool 是共享单例，但每个 Turn 在自己的 Lane Task 中运行。ContextVar 让 set_context、
    set_send_callback 和 ``sent`` flag 都保持 turn-local，USER Turn 与并发 System Turn 不会互相
    覆盖回复地址。

    对象使用 frozen + copy-on-write：每个 mutator 都以 `replace` 创建新值并重新绑定 ContextVar，
    不原地修改共享引用。因此 Child Task 即使继承 Parent 当前值，也无法通过同一对象把变化写
    回 Parent Context。
    """

    channel: str
    chat_id: str
    message_id: str | None
    send_callback: Callable[[str, list[str]], Awaitable[None]] | None
    sent: bool = False


class MessageTool(Tool):
    """通过 Turn-local callback 向 Chat Channel 用户发送文本和附件。

    Tool 默认从 `_MsgTurn` 获取 Channel/Chat，也允许调用参数覆盖目标。缺目标或 callback 时
    返回 Error；发送成功后仅当目标等于本 Turn 自身地址才设置 `sent_in_turn`，供 AgentLoop
    抑制重复回复。异常会转换为错误字符串，实际重试与平台能力仍由 callback/Outlet 所有。
    """

    def __init__(
        self,
        send_callback: Callable[[str, list[str]], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        # 构造参数是兜底基线；每个 Turn 任务首次访问时，会将其复制到自己的 ContextVar 槽位。
        self._default = _MsgTurn(
            channel=default_channel,
            chat_id=default_chat_id,
            message_id=default_message_id,
            send_callback=send_callback,
        )
        self._turn: ContextVar[_MsgTurn] = ContextVar("message_tool_turn")

    def _cur(self) -> _MsgTurn:
        return self._turn.get(None) or self._default

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """设置本 Turn 默认发送 Channel、Chat 与可选 message_id。

        方法以 copy-on-write 更新 ContextVar，不改变 constructor default。`execute` 未显式传
        channel/chat_id 时使用这里的目标；message_id 当前随路由状态保存，供需要回复锚点的
        上层接线。设置只对当前 asyncio Context 生效。
        """
        self._turn.set(replace(self._cur(), channel=channel, chat_id=chat_id, message_id=message_id))

    def set_send_callback(
        self,
        callback: Callable[[str, list[str]], Awaitable[None]] | None,
    ) -> None:
        """设置当前 Turn 的异步 send callback，并允许用 ``None`` 显式禁用。

        callback 接收最终 content 与 media path 列表，通常由 `run_turn` 接到 Stream/Text/Media
        的单一 emit 边界。``None`` 是合法值且与 constructor default 一致；此时 `execute` 返回
        ``not configured`` Error，不会调用空引用。更新使用冻结状态 copy-on-write，其他并发
        Turn 保持自己的 callback。
        """
        self._turn.set(replace(self._cur(), send_callback=callback))

    def start_turn(self) -> None:
        """在新 Turn 开始时把当前 Context 的 ``sent`` 状态重置为假。

        Channel、Chat、message_id 与 callback 保持不变，只清除“已经发送主回复”的观察事实。
        若不重置，共享 Task Context 的下一 Turn 可能错误抑制正常 Text 交付。方法不发送消息。
        """
        self._turn.set(replace(self._cur(), sent=False))

    @property
    def sent_in_turn(self) -> bool:
        """返回本 Turn 是否已向自己的默认目标成功发送回复。

        只有 `execute` 成功且最终 channel/chat_id 与 `_MsgTurn` 默认目标一致时才为 ``True``；向
        另一个显式目标发送不算当前 Turn 主回复。AgentLoop 用该值避免 message Tool 与普通
        return path 重复交付，它不表示对方平台已经展示消息。
        """
        return self._cur().sent

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The message content to send"},
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (feishu, qq, or wecom)",
                },
                "chat_id": {"type": "string", "description": "Optional: target chat/user ID"},
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        st = self._cur()
        channel = channel or st.channel
        chat_id = chat_id or st.chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not st.send_callback:
            return "Error: Message sending not configured"

        try:
            await st.send_callback(content, media or [])
            if channel == st.channel and chat_id == st.chat_id:
                self._turn.set(replace(st, sent=True))
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
