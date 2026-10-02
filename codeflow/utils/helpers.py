"""CodeFlow 常用的 Text、Path、Token Estimation 与 Workspace Utility Functions。

这个模块汇集没有独立 Domain Ownership 的小型 Helper：识别图片 Magic Bytes、创建目录、规范文件名、
拆分 Channel Message、构造 Provider-safe Assistant Message、估算 Prompt Tokens，以及初始化 Workspace
模板。它们被多个运行路径复用，但返回值通常只是结构或估算，不应被当作 Provider 验证或任务完成
证据。
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken


def detect_image_mime(data: bytes) -> str | None:
    """根据 Magic Bytes 检测 Image MIME Type，忽略 File Extension。

    当前识别 PNG、JPEG、GIF 与 WEBP；未知或数据过短时返回 `None`。使用内容签名可避免仅凭用户提供的
    后缀误判格式，但这里只做轻量 Header Check，不验证整张图片是否完整、可解码或安全。
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def ensure_dir(path: Path) -> Path:
    """确保 Directory 存在，并返回同一个 `Path`。

    使用 `parents=True` 创建缺失父目录，`exist_ok=True` 允许目录已经存在。若目标存在但不是目录、或
    OS 拒绝创建，异常原样传播；函数不清空已有内容，也不改变权限。
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    """返回当前 Local Time 的 ISO Timestamp。

    结果来自 `datetime.now().isoformat()`，不附加强制 UTC 规范。它适合人类可读的本地时间戳；需要
    跨时区证据时应使用带 Timezone 的专门调用，不能假设这里的字符串是 UTC。
    """
    return datetime.now().isoformat()


_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    """把 Unsafe Path Characters 替换为 Underscores。

    ``<>:\"/\\|?*`` 会统一折叠成 ``_``，随后移除两端空白。转换不是可逆编码，不同原字符串可能
    得到同一 Filename；调用方若需要稳定唯一性，应另外加入 ID 或 Digest。
    """
    return _UNSAFE_CHARS.sub("_", name).strip()


def split_message(content: str, max_len: int = 2000) -> list[str]:
    """把 Content 拆成不超过 ``max_len`` 的 Chunks，并优先在 Line Breaks 处分段。

    Args:
        content: 待拆分 Text Content。空字符串返回空列表，短文本原样作为唯一 Chunk。
        max_len: 每个 Chunk 的 Maximum Length；默认 2000，以兼容 Discord 限制。

    Returns:
        Message Chunks 列表，每项长度都不超过 ``max_len``。截断优先级是换行、空格、最后才是强制
        位置；续段会 `lstrip`，因此分隔处的前导空白不会保留。这里按 Python Character Count，而不是
        UTF-8 Bytes 或平台 Token Count。
    """
    if not content:
        return []
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # 优先在换行符处截断，其次是空格，最后才强制截断。
        pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


def build_assistant_message(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[dict] | None = None,
) -> dict[str, Any]:
    """构造 Provider-safe Assistant Message，并按需加入 Reasoning Fields。

    基础结构始终包含 ``role="assistant"`` 与 `content`。非空 `tool_calls`、非 `None` 的
    `reasoning_content`、非空 `thinking_blocks` 才会进入结果，避免向不需要的 Provider 发送空协议
    字段。函数不验证 Tool Call Schema，也不隐藏 Reasoning；调用方仍负责遵守目标 Provider 的数据
    边界。
    """
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg


def _message_token_parts(message: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False))

    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    if message.get("tool_calls"):
        parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    if message.get("thinking_blocks"):
        parts.append(json.dumps(message["thinking_blocks"], ensure_ascii=False))
    return parts


def estimate_prompt_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """使用 Tiktoken 估算完整 Prompt Tokens。

    函数收集 Messages 中 Text、Tool Calls、Reasoning、Thinking Blocks 与 Tools Schema，序列化后用
    ``cl100k_base`` 编码。空 Payload 返回 0；Tiktoken 失败时按每 4 Characters 约一个 Token 回退，
    非空输入至少返回 1。结果适合预算门禁的近似值，不是任意模型的 Provider-billed Exact Usage。
    """
    parts: list[str] = []
    for msg in messages:
        parts.extend(_message_token_parts(msg))

    if tools:
        parts.append(json.dumps(tools, ensure_ascii=False))

    payload = "\n".join(parts)
    if not payload:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(payload)))
    except Exception:
        return max(1, len(payload) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """估算一条 Persisted Message 对 Prompt 贡献的 Tokens。

    使用与完整 Prompt 相同的字段提取与 ``cl100k_base`` 编码，异常时按字符数回退。即使 Message 没有
    可计数 Payload 也返回 1，给结构开销保留最低预算；该数值用于裁剪近似，不等于 Provider 对单条
    消息的独立账单。
    """
    payload = "\n".join(_message_token_parts(message))
    if not payload:
        return 1
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(payload)))
    except Exception:
        return max(1, len(payload) // 4)


def estimate_prompt_tokens_chain(
    provider: Any,
    model: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> tuple[int, str]:
    """先使用 Provider Counter 估算 Prompt Tokens，再回退到 Tiktoken。

    如果 Provider 暴露 Callable `estimate_prompt_tokens` 且返回正数，结果为 ``(tokens, source)``；
    Counter 异常或无效值不会阻断调用。随后使用本地 `estimate_prompt_tokens`，成功时 Source 为
    ``tiktoken``，仍无结果才返回 ``(0, "none")``。Source 让上层区分估算证据来源，而不是只看到一个
    无上下文数字。
    """
    provider_counter = getattr(provider, "estimate_prompt_tokens", None)
    if callable(provider_counter):
        try:
            tokens, source = provider_counter(messages, tools, model)
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens), str(source or "provider_counter")
        except Exception:
            pass

    estimated = estimate_prompt_tokens(messages, tools)
    if estimated > 0:
        return estimated, "tiktoken"
    return 0, "none"


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """把 Bundled Templates 同步到 Workspace，只创建 Missing Files。

    流程先把 Legacy Memory/Profile Files 一次性迁移到 L4 Layout，再为仍缺失的支柱文件写入包内模板
    或空文件，最后确保 ``skills`` 目录存在。已有 Destination 永不覆盖，因此用户直接修改的 L4 内容
    优先。返回本次 Added Relative Paths；`silent=False` 时同时在 Stderr 打印 Created Items。

    模板资源不可用时返回空列表。成功返回只说明文件布局准备完成，不验证其中的 Agent Persona、
    Memory 或 Skills 内容是否有效。
    """
    from importlib.resources import files as pkg_files

    try:
        tpl = pkg_files("codeflow") / "templates"
    except Exception:
        return []
    if not tpl.is_dir():
        return []

    added: list[str] = []

    def _write(src, dest: Path):
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8") if src else "", encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    def _migrate(src: Path, dest: Path):
        """把 Legacy Content One-shot Copy 到 L4 Path。

        Source Missing 或 Destination 已存在时 No-op，因此每次 Workspace Sync 重跑都安全，且绝不
        覆盖已经迁移或由用户创建的新文件。读取先采用 Binary，再用 UTF-8 ``errors="replace"`` 解码，
        使 Non-UTF-8 Windows Code Page 写出的 Legacy Files 也能迁移而不崩溃；不可解码字节会以替换
        字符保留证据，而不是静默丢弃整份内容。
        """
        if not src.is_file() or dest.exists():
            return
        try:
            text = src.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        added.append(f"{dest.relative_to(workspace)} (migrated from {src.relative_to(workspace)})")

    # 第一步：把旧版工作区文件迁移到 L4 布局。只有旧文件存在且 L4 目标
    # 尚不存在时规则才生效，因此用户直接对 L4 路径所做的修改优先。
    _migrate(workspace / "memory" / "MEMORY.md", workspace / "user_memory" / "profile" / "user.md")
    _migrate(workspace / "memory" / "HISTORY.md", workspace / "user_memory" / "episodic" / "episodes.md")
    _migrate(workspace / "SOUL.md", workspace / "agent_memory" / "profile" / "soul.md")
    _migrate(workspace / "AGENTS.md", workspace / "agent_memory" / "profile" / "agent.md")
    _migrate(workspace / "USER.md", workspace / "user_memory" / "profile" / "user.md")
    # 第二步：其余缺失文件回退到内置模板。先处理 L4 支柱文件；
    # TOOLS.md 仍保留在工作区根目录。
    _write(tpl / "SOUL.md", workspace / "agent_memory" / "profile" / "soul.md")
    _write(tpl / "AGENTS.md", workspace / "agent_memory" / "profile" / "agent.md")
    _write(tpl / "USER.md", workspace / "user_memory" / "profile" / "user.md")
    _write(None, workspace / "user_memory" / "episodic" / "episodes.md")
    # 程序性记忆文件在旧版布局中没有对应来源。
    _write(None, workspace / "agent_memory" / "procedural" / "skills.md")
    _write(None, workspace / "agent_memory" / "procedural" / "case.md")
    _write(tpl / "TOOLS.md", workspace / "TOOLS.md")
    (workspace / "skills").mkdir(exist_ok=True)

    if added and not silent:
        from rich.console import Console

        _c = Console(stderr=True)
        for name in added:
            _c.print(f"  [dim]Created {name}[/dim]")
    return added
