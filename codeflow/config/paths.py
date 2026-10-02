"""从 Active Config Context 派生 Runtime Paths 的 Helpers。

路径模型区分 Foreground Workspace/Project State 与 Long-running Service Workspace：默认 Foreground 命令在
Current Working Directory 运行，并把状态隔离到 Product Home 的 Project Hash；显式 Workspace 或 Service
则使用同一路径保存状态。其他 Helpers 从 Active Config File Parent 派生 Instance Data Subdirs。

多数 Getter 会 `ensure_dir` 产生目录；仅返回 History File Path 的接口不创建文件。路径解析成功不证明
权限可写或 Backend 可启动。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeflow.config.loader import get_config_path
from codeflow.product import DEFAULT_WORKSPACE_SPEC, get_default_workspace, get_product_home, get_project_state_dir
from codeflow.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from codeflow.config.schema import Config


@dataclass(frozen=True)
class RuntimePaths:
    workspace: Path
    state: Path


def resolve_foreground_paths(
    config: "Config",
    *,
    workspace: str | None = None,
    cwd: Path | None = None,
) -> RuntimePaths:
    if workspace:
        explicit = Path(workspace).expanduser()
        return RuntimePaths(workspace=explicit, state=explicit)
    configured = Path(config.workspace_path).expanduser()
    if config.agents.defaults.workspace != DEFAULT_WORKSPACE_SPEC:
        return RuntimePaths(workspace=configured, state=configured)
    resolved_workspace = (cwd or Path.cwd()).expanduser().resolve()
    return RuntimePaths(
        workspace=resolved_workspace,
        state=get_project_state_dir(resolved_workspace),
    )


def resolve_service_paths(config: "Config") -> RuntimePaths:
    workspace = Path(config.workspace_path).expanduser()
    return RuntimePaths(workspace=workspace, state=workspace)


def get_data_dir() -> Path:
    """返回并创建 Instance-level Runtime Data Directory。

    路径是 Active Config File 的 Parent，因此 ``--config`` 可隔离多个 CodeFlow Instance State。
    """
    return ensure_dir(get_config_path().parent)


def get_runtime_subdir(name: str) -> Path:
    """返回并创建 Instance Data Dir 下的 Named Runtime Subdirectory。

    `name` 由受信调用点提供；函数不清理 Path Separator，不应直接传入用户任意路径。
    """
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """返回并创建 Media Directory，可选按 Channel Namespace。

    无 Channel 返回 Shared ``media`` Root；有值时创建 Child。它不验证 Channel Name 是否安全，Caller 应
    传稳定内部标识符。
    """
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """返回并创建 Persistent Cron Storage Directory。

    Cron Service 在此保存 Job Store/Lock 等实例级状态；函数不启动 Scheduler。
    """
    return get_runtime_subdir("cron")


def get_cache_dir() -> Path:
    """返回并创建 Disposable、Refetchable On-disk Cache Directory。

    其中数据不应成为唯一业务事实，清空后应可从 Authoritative Source 重建。
    """
    return get_runtime_subdir("cache")


def get_sandbox_dir(backend: str) -> Path:
    """返回并创建给定 Backend 的 Sandbox Runtime Home Directory。

    例如 ``backend='boxlite'`` → ``<data_dir>/sandbox/boxlite``，作为 BoxLite ``home_dir``，使 DB、Images、
    Layers 位于 CodeFlow Data Dir 而非 ``~/.boxlite``。函数只分配路径，不创建 VM。
    """
    return ensure_dir(get_runtime_subdir("sandbox") / backend)


def get_logs_dir() -> Path:
    """返回并创建当前 Instance 的 Logs Directory。

    Gateway 等 Long-running Components 在此写日志；函数不配置 Rotation 或打开文件。
    """
    return get_runtime_subdir("logs")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve 并 Ensure Agent Workspace Path。

    `None` 或 ``DEFAULT_WORKSPACE_SPEC`` 使用 Product Default Workspace；其他值 Expanduser 后使用。方法会
    创建目录，不执行 Project State Hashing，Foreground Path Policy 由 `resolve_foreground_paths` 负责。
    """
    path = (
        get_default_workspace()
        if workspace is None or workspace == DEFAULT_WORKSPACE_SPEC
        else Path(workspace).expanduser()
    )
    return ensure_dir(path)


def get_cli_history_path() -> Path:
    """返回 Shared CLI History File Path ``<product_home>/.codeflow_history``。

    函数不创建文件；多个 CLI Session 共用该路径。
    """
    return get_product_home() / ".codeflow_history"
