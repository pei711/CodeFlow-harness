"""LLM Provider Config Sections 的 Atomic Operations 与 Credential Health Probe。

本模块是 Provider Configuration **ONLY Write Path**；CLI/Wizard/Future REPL 必须调用这里，禁止对 Providers
Section 直接 ``load_config`` / ``save_config``。共享逻辑完成 Schema Reflection、Secret Redaction、
Validate-before-write 与 Atomic Replace。

OAuth Providers ``openai_codex`` / ``github_copilot`` 通过
``provider_commands._LOGIN_HANDLERS`` + ``oauth_cli_kit`` 单独鉴权，Token 不写 ``config.json``。
`set_provider_fields` 拒绝写 OAuth ``api_key``，要求 ``provider login``；`reset_provider` 在
``is_oauth=True`` 时同时 Reset Config 与 Unlink Token File。Registry 的 ``is_oauth`` 决定分支。配置写入、
Credential Probe 与真实 LLM Call Success 是不同证据层。
"""

from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

from codeflow.config.loader import get_config_path, read_raw_or_raise
from codeflow.config.schema import ProvidersConfig
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
from codeflow.providers.registry import ProviderSpec, find_by_name


def _provider_names() -> list[str]:
    """返回 ``ProvidersConfig`` 声明的 Provider Field Names，仅包含 Nested Model Fields。"""
    return [
        name
        for name, field in ProvidersConfig.model_fields.items()
        if _is_model_class(_unwrap_optional(field.annotation))
    ]


def _provider_schema_cls(name: str) -> type[BaseModel]:
    """查找 Provider 的 Pydantic Class，例如 ``'gemini' -> GeminiProviderConfig``。

    Unknown/Non-section Name 抛 `KeyError` 并列出 Available Providers。
    """
    field = ProvidersConfig.model_fields.get(name)
    if field is None:
        raise KeyError(f"Unknown provider '{name}'. Available providers: {sorted(_provider_names())}")
    ann = _unwrap_optional(field.annotation)
    if not _is_model_class(ann):
        raise KeyError(f"'{name}' is not a provider section. Available providers: {sorted(_provider_names())}")
    return ann


def _provider_spec(name: str) -> ProviderSpec:
    """从 Registry 查找 ``ProviderSpec``；Absent 时抛 `KeyError` 并提示补 Registry Entry。"""
    spec = find_by_name(name)
    if spec is None:
        raise KeyError(f"No registry entry for provider '{name}'. Add a ProviderSpec to codeflow/providers/registry.py.")
    return spec


_SECRET_EXACT = {"token", "secret", "password", "api_key"}
_SECRET_SUFFIXES = ("_token", "_secret", "_key", "_password")

# 需要脱敏，但既不匹配 _SECRET_EXACT、也不以 secret 后缀结尾的名称。
# 目前仅涵盖 Gemini 的 ``api_key_list``（其后缀是
# ``_list`` 而非 ``_key``）。随着 schema.py 在底层字段上增加
# ``json_schema_extra={"secret": True}`` 标记，应删除此处对应条目。
_KNOWN_SECRET_FIELDS: set[str] = {"api_key_list"}


def _is_secret_field(field_name: str, field_info: Any) -> bool:
    """按 Priority 检测 Secret Fields。

    依次检查 Explicit ``field_info.json_schema_extra.get('secret') is True``、``_KNOWN_SECRET_FIELDS`` Patch
    （覆盖 Gemini ``api_key_list``）、Exact ``token`` / ``secret`` / ``password`` / ``api_key``，以及
    ``_token`` / ``_secret`` / ``_key`` / ``_password`` Suffix。返回值用于 Show/Log Redaction，不加密 Disk Value。
    """
    extra = getattr(field_info, "json_schema_extra", None)
    if isinstance(extra, dict) and extra.get("secret") is True:
        return True
    if field_name in _KNOWN_SECRET_FIELDS:
        return True
    if field_name in _SECRET_EXACT:
        return True
    return any(field_name.endswith(suf) for suf in _SECRET_SUFFIXES)


def _flatten_fields(cls: type[BaseModel], prefix: str = "") -> dict[str, dict[str, Any]]:
    """把 Provider Schema Flatten 为 ``path -> spec``。

    当前无 Nested Fields，但保持与 `update_channels` 一致，使同一 CLI Parser 可复用；Literal 无 Description
    时自动列 Choices。
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
            "description": description,
        }
    return out


def _redact(value: Any) -> Any:
    """Redact Single Value 或 Value List。

    Empty 返回 ``(empty)``，List 每项分别显示 ``****set****``，其他 Non-empty 返回单一 Marker；不泄露长度
    以外的 List 内容。
    """
    if value in (None, "", [], {}):
        return "(empty)"
    if isinstance(value, list):
        return ["****set****" for _ in value]
    return "****set****"


def _oauth_token_path(provider_name: str) -> Path:
    """解析 ``oauth_cli_kit`` 写入的 On-disk Token File Path。

    Honor Kit 自己的 ``OAUTH_CLI_KIT_TOKEN_PATH`` Override，使 Tests 可指向 ``tmp_path``。无 Override 时用
    `platformdirs`，Dependency Missing 再回退 ``~/.local/share/oauth-cli-kit/auth/<provider>.json``。函数不
    读取 Token。
    """
    override = os.environ.get("OAUTH_CLI_KIT_TOKEN_PATH")
    if override:
        return Path(override)
    try:
        from platformdirs import user_data_dir
    except ImportError:
        return Path.home() / ".local" / "share" / "oauth-cli-kit" / "auth" / f"{provider_name}.json"
    base_dir = Path(user_data_dir("oauth-cli-kit", appauthor=False))
    return base_dir / "auth" / f"{provider_name}.json"


# ---------------------------------------------------------------------------
# 公共 API：反射
# ---------------------------------------------------------------------------


def provider_field_specs(name: str) -> dict[str, dict[str, Any]]:
    """把 Provider Schema Reflect 成 Flat ``path -> spec`` Map。

    每项包含 ``type``、``default``、``is_secret``、``description``，供 CLI Parser、``provider show`` 与
    `get_provider_config` 确定合法 Fields/Redaction。Unknown Provider 抛 KeyError。
    """
    cls = _provider_schema_cls(name)
    return _flatten_fields(cls)


# ---------------------------------------------------------------------------
# 公共 API：读取
# ---------------------------------------------------------------------------


def list_providers(*, config_path: Path | None = None) -> list[dict[str, Any]]:
    """返回 ``ProvidersConfig`` 中每个 Provider 的 Schema + Current Configured Status。

    每项含 ``name``、``display_name``、``is_oauth`` / ``is_local`` / ``is_gateway``、``configured``、
    ``api_key_redacted`` 与 ``api_base``（Untouched 时 ``None``）。Redacted Value 为 ``****set****``、
    ``(empty)`` 或 ``(not needed for local)``。OAuth 以
    Token File Presence，Local 以 Base/Key，其他以 Key/List 判断 Configured。该状态只说明凭据材料存在，
    不说明有效；真实健康需 `test_provider`。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    raw_providers = data.get("providers") or {}

    out: list[dict[str, Any]] = []
    for fname in _provider_names():
        cls = _provider_schema_cls(fname)
        section = raw_providers.get(fname) or {}
        try:
            instance = cls.model_validate(section)
        except ValidationError:
            instance = cls()

        spec = find_by_name(fname)
        is_oauth = bool(spec and spec.is_oauth)
        is_local = bool(spec and spec.is_local)
        is_gateway = bool(spec and spec.is_gateway)
        display_name = spec.label if spec else fname.replace("_", " ").title()

        api_key = getattr(instance, "api_key", "") or ""
        api_base = getattr(instance, "api_base", None)
        api_key_list = list(getattr(instance, "api_key_list", []) or [])

        if is_oauth:
            configured = _oauth_token_path(fname).exists()
            api_key_redacted = "OAuth token" if configured else "(empty)"
        elif is_local:
            configured = bool(api_base) or bool(api_key)
            api_key_redacted = "(not needed for local)" if not api_key else "****set****"
        else:
            configured = bool(api_key) or bool(api_key_list)
            api_key_redacted = "****set****" if configured else "(empty)"

        out.append(
            {
                "name": fname,
                "display_name": display_name,
                "is_oauth": is_oauth,
                "is_local": is_local,
                "is_gateway": is_gateway,
                "configured": configured,
                "api_key_redacted": api_key_redacted,
                "api_base": api_base,
            }
        )
    return out


def get_provider_config(
    name: str,
    *,
    redact_secrets: bool = True,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """以 Flat ``path -> value`` Dict 返回一个 Provider Configuration。

    Secret 默认渲染 ``'****set****'`` / ``'(empty)'``；`redact_secrets=False` 返回 Plaintext，仅供
    `test_provider` 等受信路径调用 ``/v1/models``。Raw Section Validation 失败时显示 Schema Defaults，不
    修改 Disk。
    """
    cls = _provider_schema_cls(name)
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    raw_section = (data.get("providers") or {}).get(name) or {}

    try:
        instance = cls.model_validate(raw_section)
    except ValidationError:
        instance = cls()

    specs = provider_field_specs(name)
    flat = _flatten_instance(instance)
    out: dict[str, Any] = {}
    for path_key, spec in specs.items():
        val = flat.get(path_key)
        if redact_secrets and spec["is_secret"]:
            out[path_key] = _redact(val)
        else:
            out[path_key] = val
    return out


# ---------------------------------------------------------------------------
# 公共 API：写入
# ---------------------------------------------------------------------------


def set_provider_fields(
    name: str,
    fields: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Patch Provider Specific Fields，返回 ``{path: previous_value}``。

    Unknown Provider/Field 抛 `KeyError`；OAuth Provider 写 ``api_key`` / ``api_key_list`` 等 Secret 抛
    `RuntimeError`，Caller 必须用 ``provider login``；Pydantic Schema Violation 抛 `ValidationError`。所有
    Values Coerce/Validate 后才 Atomic Write。Empty Fields No-op。
    """
    if not fields:
        return {}

    cls = _provider_schema_cls(name)
    spec = _provider_spec(name)
    field_specs = provider_field_specs(name)

    unknown = [k for k in fields if k not in field_specs]
    if unknown:
        raise KeyError(
            f"Unknown field(s) {unknown} for provider '{name}'. Available fields: {sorted(field_specs.keys())}"
        )

    if spec.is_oauth:
        forbidden = [k for k in fields if field_specs[k]["is_secret"]]
        if forbidden:
            raise RuntimeError(
                f"Provider '{name}' uses OAuth — cannot set credential fields "
                f"{forbidden} directly. Run: codeflow provider login "
                f"{name.replace('_', '-')}"
            )

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    raw_section = (data.get("providers") or {}).get(name) or {}

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

    data.setdefault("providers", {})
    data["providers"][name] = validated.model_dump(by_alias=True)
    _write_atomic(path, data)
    return prev


def reset_provider(
    name: str,
    *,
    config_path: Path | None = None,
) -> None:
    """把 Provider 恢复为 Schema Defaults，保留 Section Key、Reset Values。

    Two Cleanup Paths 按 ``ProviderSpec.is_oauth`` 自动运行；``is_oauth=True`` 时执行 Token Cleanup。Config Fields 始终写 Fresh Pydantic Defaults，如
    ``api_key=""``、``api_base=None``、Gemini ``vertex=False`` / ``api_key_list=[]``；OAuth Provider 还
    Unlink Token File，使 User Logged Out，Path 遵循 ``oauth_cli_kit`` 与 ``OAUTH_CLI_KIT_TOKEN_PATH``。

    Token Unlink 使用 ``missing_ok``，可重复；其他 OSError Warning 但不撤销已写 Config。Caller 无需区分
    API-key/OAuth，但返回也不证明 Remote Session 已撤销。
    """
    cls = _provider_schema_cls(name)
    spec = _provider_spec(name)

    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    data.setdefault("providers", {})
    data["providers"][name] = cls().model_dump(by_alias=True)
    _write_atomic(path, data)

    if spec.is_oauth:
        token_path = _oauth_token_path(name)
        try:
            token_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "update_providers: failed to unlink OAuth token {}: {}",
                token_path,
                exc,
            )

    logger.info("update_providers: {} reset to defaults", name)


def _load_provider_models(name: str, data: dict[str, Any]) -> tuple[type, list[str]]:
    cls = _provider_schema_cls(name)
    section = (data.get("providers") or {}).get(name) or {}
    try:
        instance = cls.model_validate(section)
    except ValidationError:
        instance = cls()
    return cls, list(getattr(instance, "models", []) or [])


def add_provider_model(
    name: str,
    model: str,
    *,
    config_path: Path | None = None,
) -> list[str]:
    """Idempotently 把 ``model`` Append 到 Provider Curated ``models`` List。

    已存在时不写盘；否则 Validate 完整 Section 后 Atomic Write。返回 New Model List，Unknown Provider 抛
    KeyError。加入 List 不验证模型在 Provider 可用。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    cls, models = _load_provider_models(name, data)
    if model not in models:
        models.append(model)
        section = (data.get("providers") or {}).get(name) or {}
        section["models"] = models
        validated = cls.model_validate(section)
        data.setdefault("providers", {})
        data["providers"][name] = validated.model_dump(by_alias=True)
        _write_atomic(path, data)
    return models


def remove_provider_model(
    name: str,
    model: str,
    *,
    config_path: Path | None = None,
) -> list[str]:
    """从 Provider Curated ``models`` List 移除 ``model``，Absent 时 No-op。

    返回 New List；真正变化时 Validate + Atomic Write。Unknown Provider 抛 KeyError。
    """
    path = config_path or get_config_path()
    data = read_raw_or_raise(path)
    cls, models = _load_provider_models(name, data)
    if model in models:
        models = [m for m in models if m != model]
        section = (data.get("providers") or {}).get(name) or {}
        section["models"] = models
        validated = cls.model_validate(section)
        data.setdefault("providers", {})
        data["providers"][name] = validated.model_dump(by_alias=True)
        _write_atomic(path, data)
    return models


# ---------------------------------------------------------------------------
# 公共 API：凭据健康检查
# ---------------------------------------------------------------------------


# 将 HTTP 状态映射为 CLI 提示表使用的用户可见状态关键字。
_HTTP_STATUS_MAP: dict[int, str] = {
    200: "valid",
    401: "invalid_key",
    402: "no_credits",
    403: "invalid_key",
    429: "rate_limited",
}


def test_provider(
    name: str,
    *,
    timeout_s: int = 10,
    config_path: Path | None = None,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """通过 Free ``GET /v1/models`` Request 验证 Provider Credentials。

    不用 Chat Completion 的原因与 Hermes Agent ``doctor._probe_apikey_provider`` 相同：Metadata Endpoint
    Zero Token Cost、不消耗 Inference Quota、几乎所有 OpenAI-compatible Providers 支持，也无需维护 Which
    Test Model。

    流程先取得 Config ``api_key``，OAuth 则用 ``oauth_cli_kit.get_token()``，并回退
    ``ProviderSpec.default_api_base``；随后用 ``Authorization: Bearer {key}`` 执行
    ``GET {api_base}/v1/models``；Status 映射 `_HTTP_STATUS_MAP`，Unknown
    为 ``http_{code}``，Network Error 为 ``network_error``。200 时 Best-effort 解析 Model IDs。

    返回 Dict、Never Raises；`transport` 可注入 `httpx.MockTransport`。``ok=True`` 证明免费 Metadata Probe
    成功，不证明指定 Chat Model、配额或完整 LLM Call 可用。
    """
    import time

    try:
        spec = _provider_spec(name)
    except KeyError as exc:
        return {
            "ok": False,
            "status": "unknown_provider",
            "elapsed_ms": 0,
            "http_status": None,
            "models_count": None,
            "model_ids": None,
            "error": str(exc),
        }

    cfg = get_provider_config(name, redact_secrets=False, config_path=config_path)
    api_key = cfg.get("api_key") or ""
    api_base = cfg.get("api_base") or spec.default_api_base or ""

    if spec.is_oauth:
        try:
            from oauth_cli_kit import get_token
        except ImportError:
            return {
                "ok": False,
                "status": "oauth_token_missing",
                "elapsed_ms": 0,
                "http_status": None,
                "models_count": None,
                "model_ids": None,
                "error": "oauth_cli_kit not installed",
            }
        try:
            token = get_token()
        except Exception as exc:
            return {
                "ok": False,
                "status": "oauth_token_missing",
                "elapsed_ms": 0,
                "http_status": None,
                "models_count": None,
                "model_ids": None,
                "error": str(exc),
            }
        if not (token and getattr(token, "access", None)):
            return {
                "ok": False,
                "status": "oauth_token_missing",
                "elapsed_ms": 0,
                "http_status": None,
                "models_count": None,
                "model_ids": None,
                "error": "no OAuth token stored",
            }
        api_key = token.access

    if not api_key and not spec.is_local:
        return {
            "ok": False,
            "status": "not_configured",
            "elapsed_ms": 0,
            "http_status": None,
            "models_count": None,
            "model_ids": None,
            "error": "api_key is empty",
        }

    if not api_base:
        return {
            "ok": False,
            "status": "not_configured",
            "elapsed_ms": 0,
            "http_status": None,
            "models_count": None,
            "model_ids": None,
            "error": "api_base is empty and provider has no default",
        }

    base_url = api_base.rstrip("/")
    url = f"{base_url}/models" if "/v1" in api_base else f"{base_url}/v1/models"

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    start = time.monotonic()
    client_kwargs: dict[str, Any] = {"timeout": timeout_s}
    if transport is not None:
        client_kwargs["transport"] = transport

    try:
        with httpx.Client(**client_kwargs) as client:
            resp = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "status": "network_error",
            "elapsed_ms": int((time.monotonic() - start) * 1000),
            "http_status": None,
            "models_count": None,
            "model_ids": None,
            "error": str(exc),
        }

    elapsed_ms = int((time.monotonic() - start) * 1000)
    status_keyword = _HTTP_STATUS_MAP.get(resp.status_code, f"http_{resp.status_code}")

    models_count: int | None = None
    model_ids: list[str] | None = None
    if resp.status_code == 200:
        try:
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list):
                models_count = len(data)
                ids: list[str] = []
                for item in data:
                    if isinstance(item, dict):
                        mid = item.get("id") or item.get("name")
                        if isinstance(mid, str) and mid:
                            ids.append(mid)
                model_ids = ids
        except Exception:
            models_count = None
            model_ids = None

    return {
        "ok": resp.status_code == 200,
        "status": status_keyword,
        "elapsed_ms": elapsed_ms,
        "http_status": resp.status_code,
        "models_count": models_count,
        "model_ids": model_ids,
        "error": None if resp.status_code == 200 else f"HTTP {resp.status_code}",
    }


__all__ = [
    "provider_field_specs",
    "list_providers",
    "get_provider_config",
    "set_provider_fields",
    "reset_provider",
    "test_provider",
]
