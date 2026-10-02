"""对 ``~/.codeflow/config.json`` 执行 Minimal In-place Updates。

``save_config`` 会 Re-serialize Entire Pydantic Model，把所有 Runtime Defaults Bake 回文件；这里改为读取
Raw JSON，只 Patch 少量 Fields，再用 Temp-file + Rename Atomic Rewrite。``codeflow cron config set`` 与
Onboarding Wizard 用它持久化修改，同时不触碰 Unrelated Fields。

所有 Public Update 必须通过 `read_raw_or_raise`，损坏 Config Fail Closed，避免 Data Loss。Atomic Write
成功表示新 JSON Published，不验证下次 Runtime External Dependency。
"""

from __future__ import annotations

import json
import os
import types
import typing
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from pydantic_core import PydanticUndefined

from codeflow.config.loader import get_config_path, read_raw_or_raise
from codeflow.config.schema import CronConfig


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _unwrap_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_model_class(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _annotation_str(annotation: Any) -> str:
    annotation = _unwrap_optional(annotation)
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return "Literal"
    if origin is list:
        args = typing.get_args(annotation)
        return f"list[{_annotation_str(args[0])}]" if args else "list"
    if origin is dict:
        args = typing.get_args(annotation)
        if args and len(args) == 2:
            return f"dict[{_annotation_str(args[0])}, {_annotation_str(args[1])}]"
        return "dict"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _coerce_value(value: Any, annotation: Any) -> Any:
    if not isinstance(value, str):
        return value

    base = _unwrap_optional(annotation)
    if base is bool:
        value_lower = value.strip().lower()
        if value_lower in ("true", "1", "yes", "on"):
            return True
        if value_lower in ("false", "0", "no", "off"):
            return False
        return value
    if base is int:
        try:
            return int(value)
        except ValueError:
            return value
    if base is float:
        try:
            return float(value)
        except ValueError:
            return value

    origin = typing.get_origin(base)
    if origin is list:
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in value.split(",") if item.strip()]
    if origin is dict:
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return value


def _field_default(field_info: Any) -> Any:
    if field_info.default_factory is not None:
        try:
            return field_info.default_factory()
        except Exception:
            return None
    if field_info.default is PydanticUndefined:
        return None
    return field_info.default


def _flatten_instance(instance: BaseModel, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name in type(instance).model_fields:
        value = getattr(instance, field_name)
        path = f"{prefix}{field_name}"
        if isinstance(value, BaseModel):
            out.update(_flatten_instance(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def _walk_nested_path(model_cls: type[BaseModel], dotted_key: str) -> tuple[type[BaseModel], str]:
    segments = dotted_key.split(".")
    cls = model_cls
    for segment in segments[:-1]:
        field_info = cls.model_fields.get(segment)
        if field_info is None:
            raise KeyError(f"Unknown nested field '{segment}' in {cls.__name__}")
        annotation = _unwrap_optional(field_info.annotation)
        if not _is_model_class(annotation):
            raise KeyError(f"Field '{segment}' in {cls.__name__} is not a nested model")
        cls = annotation
    leaf = segments[-1]
    if leaf not in cls.model_fields:
        raise KeyError(f"Unknown field '{leaf}' in {cls.__name__}")
    return cls, leaf


def _set_nested(dotted_key: str, value: Any, target: dict[str, Any]) -> Any:
    segments = dotted_key.split(".")
    cursor = target
    for segment in segments[:-1]:
        child = cursor.get(segment)
        if not isinstance(child, dict):
            child = {}
            cursor[segment] = child
        cursor = child
    previous = cursor.get(segments[-1])
    cursor[segments[-1]] = value
    return previous


def update_cron_config(
    key: str,
    value: Any,
    *,
    config_path: Path | None = None,
) -> Any:
    """Patch On-disk `CronConfig` 的 Single Field。

    返回 Previous Raw Value，Absent 时 `None`。``key`` 不属于 `CronConfig` 时抛 ``KeyError``；这是 Defensive
    Check，CLI ``_KEY_HANDLERS`` 已预验证。`value` Type Validation 由 Caller/CLI Parser 负责。Field 使用
    Camel Alias 写入，完成后 Atomic Replace。
    """
    if key not in CronConfig.model_fields:
        raise KeyError(f"Unknown cron config key: {key!r}. Supported: {sorted(CronConfig.model_fields)}")
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    cron_section = data.setdefault("cron", {})
    camel_key = to_camel(key)
    prev = cron_section.get(camel_key)
    cron_section[camel_key] = value
    _write_atomic(path, data)
    logger.info("config/update: cron.{} set to {!r} (was {!r})", key, value, prev)
    return prev


def reset_cron_config(*, config_path: Path | None = None) -> None:
    """从 On-disk Config 移除 Entire ``cron`` Section。

    下次 Load 使用 Schema Defaults ``forward_channels=["*"]`` / ``default_timezone="Asia/Shanghai"``，保持
    ``never bake defaults to disk`` Principle。Section Absent 也会重写相同数据，方法无返回值。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    removed = data.pop("cron", None)
    _write_atomic(path, data)
    logger.info("config/update: cron section reset (was {!r})", removed)


def set_language(
    language: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch On-disk Top-level ``language``，返回 Previous Value。

    Onboarding Language Screen 设置；CLI/Wizard Copy 通过 ``_t`` 读取，并注入 Agent System Prompt，指导
    Reply Language。函数不限制 Literal，Schema 在 Next Load 验证。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    prev = data.get("language")
    data["language"] = language
    _write_atomic(path, data)
    logger.info("config/update: language set to {!r} (was {!r})", language, prev)
    return prev


def set_default_model(
    model: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch On-disk ``agents.defaults.model``，返回 Previous Value。

    Onboarding 在 User 选 Provider 后，把 Default Model 换成 Matching Vendor；否则 ``codeflow run`` 仍使用 Fresh
    ``Config()`` Baked 的其他 Vendor Model。函数只写 Name，不验证 Provider Config/Credentials。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    defaults = data.setdefault("agents", {}).setdefault("defaults", {})
    prev = defaults.get("model")
    defaults["model"] = model
    _write_atomic(path, data)
    logger.info("config/update: default model set to {} (was {})", model, prev)
    return prev


def set_sandbox_backend(
    backend: str,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch On-disk ``tools.sandbox.backend``，返回 Previous Value。

    Onboarding Run-location Step 使用。``backend`` 必须是 `SandboxConfig` Literal ``none`` / ``auto`` /
    ``boxlite``，由 Next Load Validation；函数不尝试启动 VM。注意 Sandbox Nested under Tools，不是 Root。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    # sandbox 位于 tools 下（Config.tools.sandbox），而不是根节点。根 Config
    # 禁止额外字段，因此顶层 "sandbox" 键会在下次加载时导致 schema 校验失败。
    section = data.setdefault("tools", {}).setdefault("sandbox", {})
    prev = section.get("backend")
    section["backend"] = backend
    _write_atomic(path, data)
    logger.info("config/update: tools.sandbox.backend set to {!r} (was {!r})", backend, prev)
    return prev


def init_extension_block_defaults(*, config_path: Path | None = None) -> None:
    """把 Memory / Plugins / SkillForge 的 User-facing Defaults Seed 到 Fresh Config。

    Onboarding Bootstrap 调用一次，使新 ``~/.codeflow/config.json`` 展示可编辑 Knobs。每 Field 只在 Absent 时
    ``setdefault``，因此 Idempotent 且不 Clobber User/Earlier Wizard Value。Defaults 从 Pydantic Models 获取，
    避免 Schema Drift；``memory.backend`` Seed 为 Schema Default ``"myna"``。这里的 ``myna`` 仅是
    Backend Contribution Name，不包含外部仓库链接。Plugin Config 保持 Empty。

    `SkillForgeConfig` Optional Service Fields ``embedding_url`` / ``embedding_api_key`` / ``reranker_url`` /
    ``reranker_api_key`` / ``mass_library_db`` **NOT Written**，Hosted Deployment 手工显式添加。Key Casing
    遵循 Block Convention：``memory`` / ``skillForge`` 使用 CamelCase Alias，``plugins.config`` Verbatim
    ``snake_case`` Pass-through。
    写入 Defaults 不代表对应 Backend 已初始化。
    """
    from codeflow.config.codeflow import (
        MemoryConfig,
        PluginsConfig,
        SkillForgeRouterConfig,
    )

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)

    mem = MemoryConfig()
    memory = data.setdefault("memory", {})
    memory.setdefault("backend", mem.backend)
    memory.setdefault("userId", mem.user_id)
    memory.setdefault("memoryTopK", mem.memory_top_k)

    plugins = data.setdefault("plugins", {})
    plugins.setdefault("disabled", list(PluginsConfig().disabled))
    plugins.setdefault("config", {})

    router_defaults = SkillForgeRouterConfig()
    skill_forge = data.setdefault("skillForge", {})
    skill_forge.setdefault("enabled", True)
    router = skill_forge.setdefault("router", {})
    router.setdefault("enabled", router_defaults.enabled)

    _write_atomic(path, data)
    logger.info("config/update: seeded memory/plugins/skillForge extension defaults")


def set_memory_backend(
    backend: str | None,
    *,
    config_path: Path | None = None,
) -> str | None:
    """Patch On-disk ``memory.backend``，返回 Previous Value。

    ``"myna"`` 选择 Repository Memory；`None` 禁用 Implicit Memory，同时保留 Sessions 与 Local Skills。
    函数只保存 Selection，不 Resolve/Start Plugin。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    section = data.setdefault("memory", {})
    prev = section.get("backend")
    section["backend"] = backend
    _write_atomic(path, data)
    logger.info("config/update: memory.backend set to {!r} (was {!r})", backend, prev)
    return prev


__all__ = [
    "update_cron_config",
    "reset_cron_config",
    "set_default_model",
    "set_sandbox_backend",
    "set_memory_backend",
    "init_extension_block_defaults",
]
