"""Channel Config Sections 的 Atomic Read-modify-write Operations。

本模块是 Channel Configuration **ONLY Write Path**。CLI Commands、Future Wizard/WebUI/REPL Slash 等所有
Entry Points 都必须调用这里；禁止对 Channels Section 直接 `load_config` / `save_config`。共享路径统一完成
Schema Reflection、Secret Detection/Redaction、Coercion、Validate-before-write 与 Atomic Replace。

写入成功只表示 Config Published；Channel Restart/Connection/Delivery 仍由 Runtime 验证。
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ValidationError

from codeflow.config.loader import get_config_path, read_raw_or_raise
from codeflow.config.schema import ChannelsConfig
from codeflow.config.update import (
    _annotation_str,
    _coerce_value,
    _field_default,
    _flatten_instance,
    _is_model_class,
    _set_nested,
    _unwrap_optional,
    _walk_nested_path,
    _write_atomic,
)


def _channel_names() -> list[str]:
    """返回 `ChannelsConfig` 中代表 Channel 的 Field Names，仅包含 `BaseModel` Subfields。

    `send_progress` 等 Global Fields 不进入结果。
    """
    return [
        name
        for name, field in ChannelsConfig.model_fields.items()
        if _is_model_class(_unwrap_optional(field.annotation))
    ]


def _channel_schema_cls(name: str) -> type[BaseModel]:
    """按 Channel Name 查找其 Pydantic Schema Class。

    Unknown Name 或目标不是 Nested Model 时抛 ``KeyError``，并列出 Available Channels。
    """
    field = ChannelsConfig.model_fields.get(name)
    if field is None:
        raise KeyError(f"Unknown channel '{name}'. Available channels: {sorted(_channel_names())}")
    ann = _unwrap_optional(field.annotation)
    if not _is_model_class(ann):
        raise KeyError(f"'{name}' is not a channel section. Available channels: {sorted(_channel_names())}")
    return ann


_SECRET_EXACT = {"token", "secret", "password", "api_key"}
_SECRET_SUFFIXES = (
    "_token",
    "_secret",
    "_key",
    "_password",
)


def _is_secret_field(field_name: str, field_info: Any) -> bool:
    """按优先顺序 Detect Secret Fields。

    先看 Explicit ``field_info.json_schema_extra.get('secret') is True``，再匹配 Exact Names ``token`` /
    ``secret`` / ``password`` / ``api_key``，最后匹配 ``_token`` / ``_secret`` / ``_key`` / ``_password``
    Suffix。返回值用于 UI Redaction，不对实际 String 加密。
    """
    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict) and extra.get("secret") is True:
        return True
    if field_name in _SECRET_EXACT:
        return True
    return any(field_name.endswith(suf) for suf in _SECRET_SUFFIXES)


def _is_required_field(field_info: Any) -> bool:
    """显式 ``json_schema_extra={'required': True}`` 时 Field 才视为 Required。

    每个 Channel Field 都有 Pydantic Default，使 Partial/Disabled Config 可 Load，因此 Pydantic Own Required
    Flag 始终 False；这里的 Requiredness 是 UX Marker，Runtime 仍需在 Enable/Start 时检查非空值。
    """
    extra = getattr(field_info, "json_schema_extra", None)
    return isinstance(extra, dict) and extra.get("required") is True


def _flatten_fields(cls: type[BaseModel], prefix: str = "") -> dict[str, dict[str, Any]]:
    """递归 Nested ``BaseModel`` Fields，生成 Flat Dotted-path Specs Dict。

    每项包含 Type、Default、Secret、Required、``description``。``Literal[...]`` 若无 User Description，会把
    Choice List 渲染进 Description，供 CLI Surface。函数只反射 Schema，不读取用户值。
    """
    out: dict[str, dict[str, Any]] = {}
    for fname, finfo in cls.model_fields.items():
        ann = _unwrap_optional(finfo.annotation)
        path = f"{prefix}{fname}"
        if _is_model_class(ann):
            out.update(_flatten_fields(ann, prefix=f"{path}."))
            continue
        description = finfo.description or ""
        origin = typing.get_origin(ann)
        if origin is typing.Literal and not description:
            choices = ", ".join(str(a) for a in typing.get_args(ann))
            description = f"Choices: {choices}"
        out[path] = {
            "type": _annotation_str(ann),
            "default": _field_default(finfo),
            "is_secret": _is_secret_field(fname, finfo),
            "required": _is_required_field(finfo),
            "description": description,
        }
    return out


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def channel_field_specs(name: str) -> dict[str, dict[str, Any]]:
    """把 Channel Schema Reflect 成 Flat ``dotted-path -> spec`` Map。

    每项 Keys 为 ``type``、``default``、``is_secret``、``required``、``description``。CLI Parsers、
    ``channels help`` 与 `get_channel_config` 用它确定合法 Fields 与 Redaction；Unknown Channel 抛 KeyError。
    """
    cls = _channel_schema_cls(name)
    return _flatten_fields(cls)


def enable_channel(
    name: str,
    fields: dict[str, Any] | None = None,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """设置 ``channels.<name>.enabled = True``，并可同时 Patch Credential Fields。

    Atomic Contract：所有 Fields 在任何 Write 前完成 Validation。返回 Patched Fields 的 Previous Values，
    供 Caller Log。Unknown Channel/Field 抛 `KeyError`，违反 Pydantic Schema 抛 `ValidationError`。Enabled
    Published 不表示 Channel 已连接。
    """
    payload = dict(fields or {})
    payload["enabled"] = True
    return _patch_channel(name, payload, config_path)


def disable_channel(
    name: str,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """设置 ``channels.<name>.enabled = False``，保留 Credential Fields。

    返回 Previous Enabled Value Map；Runtime 是否立即 Stop Channel 取决于 Caller 后续 Reload/Restart。
    """
    return _patch_channel(name, {"enabled": False}, config_path)


def set_channel_fields(
    name: str,
    fields: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Patch Channel 的 Specific Fields。

    返回 ``{field_path: previous_value}`` 供 Logging；Empty Input 返回 Empty Dict 且不写。Atomic Validation
    Contract 与 :func:`enable_channel` 相同。
    """
    if not fields:
        return {}
    return _patch_channel(name, dict(fields), config_path)


def get_channel_config(
    name: str,
    *,
    redact_secrets: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """以 Flat ``dotted-path -> value`` Dict 返回 Current Channel Configuration。

    Secret Fields 默认 Redacted：Non-empty 渲染 ``'****set****'``，Empty/None 渲染 ``'(empty)'``。Raw Section
    Validation 失败时显示 Schema Defaults，而不修改 Disk；`redact_secrets=False` 仅供受信 Caller。
    """
    cls = _channel_schema_cls(name)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    raw_section = (data.get("channels") or {}).get(name) or {}

    try:
        instance = cls.model_validate(raw_section)
    except ValidationError:
        instance = cls()

    specs = channel_field_specs(name)
    flat = _flatten_instance(instance)
    out: dict[str, Any] = {}
    for path_key, spec in specs.items():
        val = flat.get(path_key)
        if redact_secrets and spec["is_secret"]:
            if val in (None, "", [], {}):
                out[path_key] = "(empty)"
            else:
                out[path_key] = "****set****"
        else:
            out[path_key] = val
    return out


def reset_channel(
    name: str,
    *,
    config_path: Path | None = None,
) -> None:
    """把 ``channels.<name>`` Reset 为 Schema Defaults。

    保留 Section Key，使 Downstream Discovery 仍看到 Channel，只 Revert Field Values。等价于 Fresh Pydantic
    Instance 的 ``model_dump(by_alias=True)`` 后 Atomic Write；Credentials 会被清空为 Defaults。
    """
    cls = _channel_schema_cls(name)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    data.setdefault("channels", {})
    instance = cls()
    data["channels"][name] = instance.model_dump(by_alias=True)
    _write_atomic(path, data)
    logger.info("update_channels: {} reset to defaults", name)


# ---------------------------------------------------------------------------
# 内部实现：共享写入路径
# ---------------------------------------------------------------------------


def _patch_channel(
    name: str,
    fields: dict[str, Any],
    config_path: Path | None,
) -> dict[str, Any]:
    """Enable / Disable / Set 共用的 Validate-then-write Core。

    先反射 Specs 并拒绝 Unknown Fields，读取 Raw Section，构造 Current Model，按 Annotation Coerce Dotted
    Values，再用完整 Schema Validate，最后 Alias Dump + Atomic Write。返回 Previous Raw Leaf Values。
    """
    cls = _channel_schema_cls(name)
    specs = channel_field_specs(name)

    unknown = [k for k in fields if k not in specs]
    if unknown:
        raise KeyError(f"Unknown field(s) {unknown} for channel '{name}'. Available fields: {sorted(specs.keys())}")

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    raw_section = (data.get("channels") or {}).get(name) or {}

    try:
        current = cls.model_validate(raw_section)
    except ValidationError:
        current = cls()

    working = current.model_dump()

    prev: dict[str, Any] = {}
    for path_key, raw_val in fields.items():
        leaf_cls, leaf_field = _walk_nested_path(cls, path_key)
        leaf_info = leaf_cls.model_fields[leaf_field]
        coerced = _coerce_value(raw_val, leaf_info.annotation)
        prev[path_key] = _set_nested(path_key, coerced, working)

    validated = cls.model_validate(working)

    data.setdefault("channels", {})
    data["channels"][name] = validated.model_dump(by_alias=True)
    _write_atomic(path, data)
    return prev


__all__ = [
    "channel_field_specs",
    "enable_channel",
    "disable_channel",
    "set_channel_fields",
    "get_channel_config",
    "reset_channel",
]
