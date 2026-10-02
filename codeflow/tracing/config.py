"""CodeFlow In-tree Tracing 的 Lightweight Configuration Accessors。

模块保持 Light 与 Side-effect-free：Process Startup 时由 CLI ``main()`` Callback 读取，决定是否安装
Instrumentation。Environment Variables 是 Explicit Overrides；否则 CodeFlow Config ``[tracing]`` Section
驱动行为，Default Enabled。

所有解析采用 Best-effort Fallback，Tracing Config Error 不能阻断 Host Startup。Accessor 返回的是当前
读取结果，不缓存配置，也不创建 Trace Directory 或 Viewer。
"""

from __future__ import annotations

import os
from pathlib import Path

from codeflow.product import get_product_home

_OFF = {"0", "false", "off", "no"}


def _config_section() -> dict:
    """Best-effort 读取 CodeFlow Config File 中的 ``[tracing]`` Block。

    使用 CodeFlow 自己的 Config-path Resolver，所以 ``--config`` Override 设置后会被 Honor。Missing File、
    Invalid JSON、Section 非 Dict 或任意 IO/Import Error 都返回空 Dict。Never Raises，因为 Tracing 不能
    Break Startup。
    """
    try:
        import json

        from codeflow.config.loader import get_config_path

        path = get_config_path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
        section = data.get("tracing")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def enabled() -> bool:
    """返回 Tracing 是否 Enabled，Default On。

    ``CODEFLOW_TRACING`` Env 优先；值为 ``0/false/off/no``（Case-insensitive）时关闭，其他显式值开启。Env
    未设置时读取 ``[tracing].enabled``，Section 缺失默认 `True`。
    """
    env = os.environ.get("CODEFLOW_TRACING")
    if env is not None:
        return env.strip().lower() not in _OFF
    return bool(_config_section().get("enabled", True))


def state_dir() -> Path:
    """返回 Trace State Dir，默认 ``~/.codeflow/traces``。

    Spans 写入 ``<dir>/logs/audit-spans.log``。``CODEFLOW_TRACING_DIR`` 可直接 Override，并执行 Expanduser；
    否则基于受 ``CODEFLOW_HOME`` 影响的 Product Home。函数只计算 Path，不创建目录。
    """
    override = os.environ.get("CODEFLOW_TRACING_DIR")
    if override:
        return Path(override).expanduser()
    return get_product_home() / "traces"


def port() -> int:
    """返回 Dashboard Viewer Port。

    ``TRACING_UI_PORT`` Env 优先，其次 ``[tracing].port``，Default 4318。Env/Config 无法转成 Integer 时
    回退 4318；函数不检查 Port 是否可绑定或已被占用。
    """
    env = os.environ.get("TRACING_UI_PORT")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            return 4318
    try:
        return int(_config_section().get("port", 4318))
    except (ValueError, TypeError):
        return 4318


def preview_len() -> int:
    """返回 Span Inline Preview 保留的 Maximum Characters。

    Full Payloads 应进入 Artifacts。``CODEFLOW_TRACING_PREVIEW`` Env 优先，其次 ``previewLen`` Config，默认
    500；结果最小为 0，Invalid Value 回退。Preview Limit 只控制可见摘要，不改变 Artifact 原始内容。
    """
    env = os.environ.get("CODEFLOW_TRACING_PREVIEW")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            return 500
    try:
        return max(0, int(_config_section().get("previewLen", 500)))
    except (ValueError, TypeError):
        return 500
