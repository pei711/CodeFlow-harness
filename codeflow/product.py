"""集中定义 CodeFlow Product Identity 与 Persistent-state Roots。

模块统一保存产品名、CLI 名、Distribution Name、Logo 与 `.codeflow` 状态目录约定，并提供从
`CODEFLOW_HOME`、用户 Home 和 Workspace 派生实际路径的函数。Project State Directory 使用解析后的
Workspace 名称加 SHA-256 摘要，既保持可读性，也避免不同绝对路径的同名目录发生碰撞。

这里仅计算路径，不创建目录。调用方应区分 Global Product Home、Default Workspace、Workspace-local
State 与按项目隔离的 Global State，避免把一个项目的持久化状态写进另一个项目。
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

PRODUCT_NAME = "CodeFlow"
PRODUCT_LOGO = "✦"
CLI_NAME = "codeflow"
DISTRIBUTION_NAME = "codeflow-harness"
GLOBAL_STATE_DIRNAME = ".codeflow"
WORKSPACE_STATE_DIRNAME = ".codeflow"
DEFAULT_WORKSPACE_SPEC = "~/.codeflow/workspace"


def get_product_home() -> Path:
    override = os.environ.get("CODEFLOW_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / GLOBAL_STATE_DIRNAME


def get_default_workspace() -> Path:
    return get_product_home() / "workspace"


def get_workspace_state_dir(workspace: Path) -> Path:
    return workspace / WORKSPACE_STATE_DIRNAME


def get_project_state_dir(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved.name).strip(".-") or "workspace"
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return get_product_home() / "projects" / f"{slug}-{digest}"
