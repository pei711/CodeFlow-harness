"""把 Stored Session 渲染为 Transcript，或导出为可自验证 Portable Envelope。

Pure ``render_transcript`` 与 I/O ``write_transcript`` 分离。Public Portable Export 写 Canonical JSON，
保留 Complete Session Payload、Human-readable Markdown 与 SHA-256 Digest；`verify_export` 只能证明
结构与 Payload 未被改动，不证明其中 Message、Tool Result 或外部事实真实，也不能恢复 Runtime
Side Effect。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from codeflow.session.manager import Session
from codeflow.utils.helpers import ensure_dir, safe_filename

_ROLE_HEADINGS = {
    "user": "## 🧑 User",
    "assistant": "## 🤖 Assistant",
    "system": "## ⚙️ System",
    "tool": "## 🛠 Tool result",
}
_EXPORT_SCHEMA = "codeflow.session.export.v1"


def render_transcript(session: Session) -> str:
    """把 ``session`` 渲染为 Full-fidelity Markdown Transcript。

    Header 包含 Key、Timestamps、Message Count 与可选 Title；随后按原顺序为 User/Assistant/System/
    Tool 使用不同 Heading，保留 Assistant Reasoning、Thinking Block、Tool Calls 与 Fenced Results。
    Multimodal Image 渲染为 ``[image]``，其他 Structured Content 转 JSON。函数 Pure，无 I/O，也不
    修改 Session。
    """
    parts: list[str] = [_render_header(session)]
    for msg in session.messages:
        parts.append(_render_message(msg))
    return "\n\n".join(p for p in parts if p) + "\n"


def default_export_path(workspace: Path, key: str) -> Path:
    """返回 Portable Session Export 的 Default Destination。

    Path 是 ``<workspace>/exports/<safe key>.codeflow-session.json``；Key 经 safe_filename，函数不创建
    Directory、不检查冲突，也不写 File。Caller 可传 Custom Dest 覆盖。
    """

    return Path(workspace) / "exports" / f"{safe_filename(key)}.codeflow-session.json"


def write_transcript(session: Session, dest: Path) -> Path:
    """渲染 ``session`` 并以 UTF-8 写入 ``dest``，返回 Absolute Path。

    Parent 缺失时创建，Existing File 直接覆盖，使 Re-export 反映 Current Session State。它写的是
    Markdown Transcript，不包含 Portable Schema/Digest；需要完整可验证 Payload 应用
    `write_portable_export`。
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    dest.write_text(render_transcript(session), encoding="utf-8")
    return dest.resolve()


def build_portable_export(session: Session) -> dict[str, Any]:
    """构建包含 Complete Payload 与 Digest 的 Self-verifying Session Export Envelope。

    Payload 保存 Key、Time、Metadata、last_consolidated、pending_clarification、全部 Messages、Count
    与 Rendered Transcript；外层写固定 Schema 与 Canonical Payload SHA-256。返回 Dict 尚未写盘，
    Digest 只覆盖 Payload，不覆盖外层格式空白。
    """
    payload = {
        "key": session.key,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "metadata": session.metadata,
        "last_consolidated": session.last_consolidated,
        "pending_clarification": session.pending_clarification,
        "messages": session.messages,
        "message_count": len(session.messages),
        "transcript_markdown": render_transcript(session),
    }
    return {
        "schema": _EXPORT_SCHEMA,
        "payload": payload,
        "sha256": _payload_digest(payload),
    }


def write_portable_export(session: Session, dest: Path) -> Path:
    """写入 Canonical Portable Export，并返回 Destination Absolute Path。

    Parent 自动创建，Envelope 使用 UTF-8、ensure_ascii=False、Sorted Keys、2-space Indent 与末尾换行。
    Existing File 覆盖。写入成功才返回 Path；函数不在写后自动 Verify，Caller 可调用 verify_export。
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    envelope = build_portable_export(session)
    dest.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dest.resolve()


def verify_export(path: Path) -> bool:
    """验证 ``path`` 是否是结构有效且 Payload Untampered 的 Portable Export。

    检查 UTF-8 JSON、Envelope Dict、Schema Version、Payload/Digest Type、Messages List 与 Message Count，
    并要求全部 Required Fields 存在；最后重新计算 Canonical SHA-256。任何 I/O/Decode/Structure/
    Digest Failure 返回 False。成功不验证 Timestamp Semantic 或 Message Trustworthiness。
    """
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(envelope, dict) or envelope.get("schema") != _EXPORT_SCHEMA:
        return False
    payload = envelope.get("payload")
    digest = envelope.get("sha256")
    if not isinstance(payload, dict) or not isinstance(digest, str):
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or payload.get("message_count") != len(messages):
        return False
    required = {
        "key",
        "created_at",
        "updated_at",
        "metadata",
        "last_consolidated",
        "pending_clarification",
        "messages",
        "message_count",
        "transcript_markdown",
    }
    return required <= payload.keys() and digest == _payload_digest(payload)


def _payload_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ── 内部实现 ─────────────────────────────────────────────────────────


def _render_header(session: Session) -> str:
    title = (session.metadata or {}).get("title")
    meta = (
        f"_{session.created_at.isoformat(timespec='seconds')}"
        f" → {session.updated_at.isoformat(timespec='seconds')}"
        f" · {len(session.messages)} messages_"
    )
    lines = [f"# Session `{session.key}`", meta]
    if title:
        lines.insert(1, f"**{title}**")
    return "\n".join(lines)


def _render_message(msg: dict[str, Any]) -> str:
    role = msg.get("role", "")
    heading = _ROLE_HEADINGS.get(role, f"## {role or 'message'}")
    if role == "tool":
        name = msg.get("name") or msg.get("tool_call_id") or ""
        suffix = f": `{name}`" if name else ""
        return f"{heading}{suffix}\n\n{_fenced(_as_text(msg.get('content')))}"

    blocks: list[str] = [heading]
    reasoning = _reasoning_text(msg)
    if reasoning:
        quoted = "\n".join(f"> {line}" for line in reasoning.splitlines() or [""])
        blocks.append(f"> 💭 _thinking_\n{quoted}")
    content = _as_text(msg.get("content"))
    if content:
        blocks.append(content)
    for call in msg.get("tool_calls") or []:
        blocks.append(_render_tool_call(call))
    return "\n\n".join(blocks)


def _render_tool_call(call: dict[str, Any]) -> str:
    fn = call.get("function") or {}
    name = fn.get("name") or call.get("name") or "tool"
    args = fn.get("arguments")
    if args is None:
        args = call.get("arguments")
    return f"⏺ **{name}**\n\n{_fenced(_as_text(args))}"


def _reasoning_text(msg: dict[str, Any]) -> str:
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return rc
    blocks = msg.get("thinking_blocks")
    if isinstance(blocks, list):
        texts = [b.get("thinking", "") for b in blocks if isinstance(b, dict) and b.get("thinking")]
        if texts:
            return "\n".join(texts)
    return ""


def _as_text(content: Any) -> str:
    """把 Message Content 的 String、Multimodal List 或 Dict Flatten 为 Transcript Text。

    None 变空 String，String 原样；List 中 Text Block 提取 text，Image URL 变 ``[image]``，其他 Dict
    JSON 编码，非 Dict 转 str；Top-level Structured Value 也 JSON 编码。函数只服务显示，不保证可
    Round-trip 恢复原 Multimodal Shape。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif part.get("type") == "image_url":
                    out.append("[image]")
                else:
                    out.append(json.dumps(part, ensure_ascii=False))
            else:
                out.append(str(part))
        return "\n".join(out)
    return json.dumps(content, ensure_ascii=False)


def _fenced(text: str) -> str:
    return f"```\n{text}\n```"


__all__ = [
    "build_portable_export",
    "default_export_path",
    "render_transcript",
    "verify_export",
    "write_portable_export",
    "write_transcript",
]
