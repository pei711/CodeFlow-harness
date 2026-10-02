"""Channel Adapters 共用的 Media Persistence Boundary。

Adapter 先用 Own SDK Fetch Bytes，再交这里写盘。集中写入提供 Two Guarantees：Server-supplied Name 会 Strip
Directory Components，Crafted ``../../x`` 无法逃出 Media Dir；Saved Name 加 Content Hash Prefix，Two
Senders 的 ``report.pdf`` 不会 Silent Collision，而 Identical Bytes Re-send 保持 Idempotent。

成功保存只证明 Bytes 写入 Local Path，不验证 MIME、恶意内容或后续 LLM/Channel 使用。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from codeflow.config.paths import get_media_dir


def safe_name(name: str | None) -> str:
    """从 Server-supplied Filename Strip Directory Components。

    Empty/None 或 Basename Empty 时回退 ``file``。它防 Path Traversal，但不清理所有平台特殊字符。
    """
    return os.path.basename(name or "") or "file"


def save_media_bytes(channel: str, data: bytes, name: str | None) -> Path:
    """把 *data* 保存到 ``<media dir>/<content-hash>_<safe name>`` 并返回 Path。

    SHA-256 前 16 Hex 作为 Content Hash，Path Traversal/Collision Semantics 见 Module Docstring。IO Error
    向上传播；函数不限制大小或检测 MIME。
    """
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = get_media_dir(channel) / f"{digest}_{safe_name(name)}"
    path.write_bytes(data)
    return path
