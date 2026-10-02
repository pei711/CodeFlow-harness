"""写入 Durable Artifacts 前清理 Ephemeral Multimodal Data。

Agent 消息可能内联 ``data:image/...;base64,...``，直接持久化会把体积很大、可能敏感的原始图片复制
到 Session、Trace 或其他长期文件。本模块把这类 Data URI 替换为稳定的
``[image data omitted]`` Marker，同时递归保留 Payload 的其余结构。它只针对内联 Image Data，不是
通用 Secret Redactor，也不会删除普通网络图片 URL。
"""

from __future__ import annotations

import re
from typing import Any

IMAGE_DATA_OMITTED = "[image data omitted]"

_IMAGE_DATA_URI = re.compile(
    r"data:image/[a-z0-9.+-]+"
    r"(?:;[a-z0-9!#$&^_.+-]+=[^;,\s]+)*"
    r";base64,[a-z0-9+/=_-]*",
    re.IGNORECASE,
)


def sanitize_persisted_text(value: str) -> str:
    return _IMAGE_DATA_URI.sub(IMAGE_DATA_OMITTED, value)


def sanitize_persisted_payload(value: Any) -> Any:
    """返回 Non-mutating Copy，并移除其中的 Inline Image Data。

    字符串通过 `sanitize_persisted_text` 替换 Data URI；List、Tuple、Dict、Set 与 Frozenset 会递归
    重建，Dict Keys 也会清理；其他标量原样返回。原输入容器不被修改，但无法识别的自定义对象不会被
    深拷贝或遍历。返回结构适合写入 Durable Artifact，Marker 明确表示此处曾有图片而不是原内容为空。
    """
    if isinstance(value, str):
        return sanitize_persisted_text(value)
    if isinstance(value, list):
        return [sanitize_persisted_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_persisted_payload(item) for item in value)
    if isinstance(value, dict):
        return {sanitize_persisted_payload(key): sanitize_persisted_payload(item) for key, item in value.items()}
    if isinstance(value, set):
        return {sanitize_persisted_payload(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(sanitize_persisted_payload(item) for item in value)
    return value


__all__ = [
    "IMAGE_DATA_OMITTED",
    "sanitize_persisted_payload",
    "sanitize_persisted_text",
]
