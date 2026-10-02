"""Feishu Inbound Message 的 Rich-content Extraction。

Feishu 会发送 Post、Interactive Card、Share Card 等 Payload，需要 Flatten 成 Agent 可消费的 Plain String。
Helpers 遍历各 Shape，提取 Human-readable Text；Post 还返回 Embedded Image Keys。它们理解 Feishu
Card/Post JSON Schema，因此属于 Adapter，不放 Shared Layer。

Extraction Best-effort：Unknown Shape 生成 Type Label 或 Empty，而不是把解析成功误当成附件已下载。
"""

from __future__ import annotations

import json

_SHARE_LABELS = {
    "share_chat": lambda c: f"[shared chat: {c.get('chat_id', '')}]",
    "share_user": lambda c: f"[shared user: {c.get('user_id', '')}]",
    "share_calendar_event": lambda c: f"[shared calendar event: {c.get('event_key', '')}]",
    "system": lambda c: "[system message]",
    "merge_forward": lambda c: "[merged forward messages]",
}


def extract_share_card(content: dict, msg_type: str) -> str:
    """把 Share-card / Interactive / System Payload Flatten 为 Text。

    Interactive 递归提取 Elements；Known Share Types 生成带 ID Label，Unknown 返回 ``[msg_type]``。
    """
    if msg_type == "interactive":
        parts = extract_interactive(content)
        return "\n".join(parts) if parts else f"[{msg_type}]"
    label = _SHARE_LABELS.get(msg_type)
    return label(content) if label else f"[{msg_type}]"


def extract_interactive(content) -> list[str]:
    """递归提取 Interactive Card 的 Title + Element Text。

    String 先 JSON Decode，Invalid String 作为 Plain Text 保留；Dict 遍历 Elements、Nested Card 与 Header。
    Non-dict 返回 Empty List。输出按遍历顺序，不保留完整视觉布局。
    """
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return [content] if content.strip() else []
    if not isinstance(content, dict):
        return []

    parts: list[str] = []
    parts += _title_text(content.get("title"))
    elements = content.get("elements")
    if isinstance(elements, list):
        for group in elements:
            for element in group if isinstance(group, list) else [group]:
                parts += extract_element(element)
    if card := content.get("card"):
        parts += extract_interactive(card)
    header = content.get("header") or {}
    parts += _title_text(header.get("title"))
    return parts


def _title_text(title) -> list[str]:
    if isinstance(title, dict):
        text = title.get("content") or title.get("text")
        return [f"title: {text}"] if text else []
    if isinstance(title, str) and title:
        return [f"title: {title}"]
    return []


def extract_element(element: dict) -> list[str]:
    """从单个 Card Element 提取 Text/Links，并递归 Containers。

    支持 Markdown/Plain、Div Fields、Anchor、Button URL、Image Alt 以及 Unknown Note/Column Containers。
    不支持的字段被忽略；Image 只保留 Alt，不下载 Bytes。
    """
    if not isinstance(element, dict):
        return []
    tag = element.get("tag", "")
    parts: list[str] = []

    if tag in ("markdown", "lark_md", "plain_text"):
        if content := element.get("content"):
            parts.append(content)
    elif tag == "div":
        parts += _text_field(element.get("text"))
        for field in element.get("fields", []):
            if isinstance(field, dict):
                parts += _text_field(field.get("text"))
    elif tag == "a":
        if href := element.get("href"):
            parts.append(f"link: {href}")
        if text := element.get("text"):
            parts.append(text)
    elif tag == "button":
        parts += _text_field(element.get("text"))
        url = element.get("url") or (element.get("multi_url") or {}).get("url")
        if url:
            parts.append(f"link: {url}")
    elif tag == "img":
        alt = element.get("alt")
        parts.append(alt.get("content", "[image]") if isinstance(alt, dict) else "[image]")
    else:  # note / column_set / 未知容器
        for nested in element.get("elements", []):
            parts += extract_element(nested)
        for col in element.get("columns", []):
            for nested in col.get("elements", []):
                parts += extract_element(nested)
    return parts


def _text_field(text) -> list[str]:
    if isinstance(text, dict):
        content = text.get("content") or text.get("text")
        return [content] if content else []
    if isinstance(text, str) and text:
        return [text]
    return []


def extract_post(content: dict) -> tuple[str, list[str]]:
    """从 Feishu Rich-text Post 提取 ``(text, image_keys)``。

    接受 Direct ``{title, content}``、Localized ``{zh_cn: {...}}`` 与 ``{post: {...}}`` Envelope，也尝试其他
    Dict Locale。没有可解析 Block 返回 ``("", [])``。Image Keys 只是后续 SDK Download Handles。
    """
    root = content
    if isinstance(root, dict) and isinstance(root.get("post"), dict):
        root = root["post"]
    if not isinstance(root, dict):
        return "", []

    if "content" in root:
        text, images = _parse_post_block(root)
        if text or images:
            return text, images
    for locale in ("zh_cn", "en_us", "ja_jp"):
        if locale in root:
            text, images = _parse_post_block(root[locale])
            if text or images:
                return text, images
    for value in root.values():
        if isinstance(value, dict):
            text, images = _parse_post_block(value)
            if text or images:
                return text, images
    return "", []


def _parse_post_block(block: dict) -> tuple[str, list[str]]:
    if not isinstance(block, dict) or not isinstance(block.get("content"), list):
        return "", []
    texts, images = [], []
    if title := block.get("title"):
        texts.append(title)
    for row in block["content"]:
        if not isinstance(row, list):
            continue
        for el in row:
            if not isinstance(el, dict):
                continue
            tag = el.get("tag")
            if tag in ("text", "a"):
                texts.append(el.get("text", ""))
            elif tag == "at":
                texts.append(f"@{el.get('user_name', 'user')}")
            elif tag == "img" and (key := el.get("image_key")):
                images.append(key)
    return " ".join(texts).strip(), images
