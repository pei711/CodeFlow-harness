"""为 QQ 适配器提供无 I/O 的寻址与内容归一化函数。

初学者可以把这里看成 QQ 平台数据到 CodeFlow 入站字段之间的纯转换层：从 botpy message
解析 ``chat_id``、``user_id`` 和 ``chat_type``，清理文本，并把附件 metadata 变成
稳定的文字 label。函数不访问网络、不下载媒体，也不修改 Channel 状态，因此可以直接
用确定性单元测试覆盖；botpy SDK 的连接、重连、发布和发送编排位于 :mod:`.channel`。

这些函数只证明路由字段和文本转换符合当前规则，不证明 QQ 已交付消息、Agent 已完成
任务或结果已经持久化。
"""

from __future__ import annotations

from typing import Any

from codeflow.channels.media import safe_name

_ATTACHMENT_KINDS = (("image/", "image"), ("video/", "video"), ("audio/", "voice"))


def clean_content(data: Any) -> str:
    return (getattr(data, "content", None) or "").strip()


def attachment_labels(data: Any) -> list[str]:
    """把 botpy 入站附件 metadata 转换为文字 label。

    ``data`` 通常是 botpy ``BaseMessage``。botpy 会暴露 ``content_type`` 和
    ``filename``，因此 label 的生成是确定性的，不需要 I/O。函数按 MIME 前缀区分
    ``image``、``video``、``voice``，其他类型归为 ``file``；服务端 filename 必须经过
    ``safe_name``，防止构造的名称把路径片段带入 Turn 文本。

    返回值是一组 ``[kind: filename]`` 或 ``[kind]`` 字符串。附件字节位于短期 platform
    URL，而 botpy 没有 download helper，所以 QQ 入站媒体不会被获取或持久化；这与 SDK
    支持下载的 Feishu 和 WeCom SDKs 不同。label 只表示平台报告了附件，不表示内容已验证。
    """
    labels: list[str] = []
    for item in getattr(data, "attachments", None) or []:
        content_type = (getattr(item, "content_type", None) or "").lower()
        kind = next((k for prefix, k in _ATTACHMENT_KINDS if content_type.startswith(prefix)), "file")
        filename = getattr(item, "filename", None)
        labels.append(f"[{kind}: {safe_name(filename)}]" if filename else f"[{kind}]")
    return labels


def compose_content(data: Any) -> str:
    """生成 QQ 消息交给 Turn 的归一化入站文本。

    ``data`` 的正文先由 ``clean_content()`` 去除首尾空白，再追加
    ``attachment_labels()`` 生成的每条附件 label，各部分用换行连接。消息既无正文又无
    附件时返回空字符串，调用方会据此丢弃事件。返回文本不包含附件字节，也不代表消息
    已经发布到 Runtime 或进入 Session。
    """
    parts = [text] if (text := clean_content(data)) else []
    parts.extend(attachment_labels(data))
    return "\n".join(parts)


def resolve_route(data: Any, is_group: bool) -> tuple[str, str, str]:
    """解析入站消息的 ``(chat_id, user_id, chat_type)``。

    ``data`` 是 botpy message，``is_group`` 由 SDK 回调类型给出。Group message 使用
    group openid 作为 ``chat_id``、member openid 作为发送者，并返回 ``"group"``；
    guild direct message（botpy ``DirectMessage``）携带 ``guild_id``，它是回复必须通过
    ``post_dms`` 返回的 DM session id，与 QQ C2C 不同，因此返回 ``"guild_dm"``；
    普通 C2C 使用 author id，并在缺失时回退到 ``user_openid``，类型为 ``"c2c"``。

    返回的第三项是协议路由值，调用方会把它缓存在当前进程中。函数不验证 allowlist，
    也不保证这些平台标识仍然有效。
    """
    if is_group:
        return data.group_openid, data.author.member_openid, "group"
    user_id = str(getattr(data.author, "id", None) or getattr(data.author, "user_openid", "unknown"))
    if guild_id := getattr(data, "guild_id", None):
        return str(guild_id), user_id, "guild_dm"
    return user_id, user_id, "c2c"
