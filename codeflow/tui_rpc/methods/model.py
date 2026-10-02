"""实现 TUI ``/model`` v1 picker 背后的 ``model.*`` RPC handlers。

五个 method 共同驱动选择器：

* ``model.options`` 返回 current model/provider 和每个 provider 的一行信息，不访问网络；
* ``model.save_key`` 保存 provider 的 ``api_key`` 与可选 ``api_base``；
* ``model.disconnect`` 清除 provider 已保存的 credentials；
* ``model.add_model`` / ``model.remove_model`` 编辑 provider 的 curated model list。

所有写 helper 都位于 ``codeflow.config.update_providers``，它是 provider config 的 single write
path；handler 使用 ``asyncio.to_thread`` 包装同步 disk IO，避免阻塞 RPC event loop。
OAuth provider 不允许从 picker 写 key，只能运行 ``codeflow provider login``，此限制暴露为
``-32012``。

picker 操作成功只说明本地配置已读取或写入；它不发起 provider 网络探测，也不证明 key
有效、model routable、下一次调用成功或 Agent 任务完成。
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from codeflow.config.update_providers import (
    add_provider_model,
    get_provider_config,
    list_providers,
    remove_provider_model,
    reset_provider,
    set_provider_fields,
)
from codeflow.providers.common_models import common_models_for
from codeflow.providers.registry import find_by_model, find_by_name
from codeflow.tui_rpc.errors import (
    ConfigValidationError,
    NotSupportedInV01Error,
)
from codeflow.tui_rpc.models import (
    ModelAddModelParams,
    ModelDisconnectParams,
    ModelOptionsParams,
    ModelRemoveModelParams,
    ModelSaveKeyParams,
)

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher


_NEEDS_API_BASE = {"custom", "azure_openai"}


def _parse(model_cls: type, params: dict) -> Any:
    try:
        return model_cls.model_validate(params)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"invalid params for {model_cls.__name__}",
            data={"errors": exc.errors(include_url=False)},
        ) from exc


def _provider_models(slug: str) -> list[str]:
    try:
        cfg = get_provider_config(slug, redact_secrets=False)
    except KeyError:
        cfg = {}
    configured = cfg.get("models", [])
    configured = list(configured) if isinstance(configured, list) else []
    # 优先放用户配置的模型（通过 ``model.add_model`` 手动录入），再放精选的
    # 常用模型短列表，并去重。这样无需网络请求，选择器开箱即用。
    seen = set(configured)
    return configured + [m for m in common_models_for(slug) if m not in seen]


def _build_provider_entry(
    slug: str,
    *,
    current_provider: str | None,
    provider_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = find_by_name(slug)
    info = provider_info
    if info is None:
        providers = {p["name"]: p for p in list_providers()}
        info = providers.get(slug, {})

    is_oauth = bool(spec and spec.is_oauth)
    configured = bool(info.get("configured"))
    warning = ""
    if is_oauth and not configured:
        warning = f"run `codeflow provider login {slug.replace('_', '-')}` to authenticate"

    models = _provider_models(slug)
    return {
        "slug": slug,
        "name": info.get("display_name") or (spec.label if spec else slug),
        "authenticated": configured,
        "is_current": slug == current_provider,
        "auth_type": "oauth" if is_oauth else "api_key",
        "key_env": (spec.env_key or None) if spec else None,
        "models": models,
        "total_models": len(models),
        "needs_api_base": slug in _NEEDS_API_BASE,
        "warning": warning,
    }


def _current_selection() -> tuple[str, str | None]:
    from codeflow.cli._helpers import load_runtime_config

    config = load_runtime_config(None, None)
    current_model = config.agents.defaults.model
    provider = config.agents.defaults.provider
    if not provider or provider == "auto":
        spec = find_by_model(current_model) if current_model else None
        provider = spec.name if spec else None
    return current_model, provider


# ---------------------------------------------------------------------------
# 处理器
# ---------------------------------------------------------------------------


async def model_options(params: dict) -> dict:
    """返回当前选择和无需网络即可构建的 provider option 列表。

    ``params`` 按 ``ModelOptionsParams`` 校验。当前 model/provider 来自 Runtime config；
    provider 为 ``auto`` 或为空时尝试从 model 推导。每个 entry 包含认证状态、auth type、
    env key、配置与 common model 合并后的列表、``needs_api_base`` 和可能的登录提示。

    返回 snapshot 不验证远端 credentials 或 model availability；``authenticated=True`` 只
    表示本地存在相应配置。
    """
    _parse(ModelOptionsParams, params)
    current_model, current_provider = _current_selection()
    providers = list_providers()
    entries = [
        _build_provider_entry(
            provider["name"],
            current_provider=current_provider,
            provider_info=provider,
        )
        for provider in providers
    ]
    return {
        "model": current_model,
        "provider": current_provider or "",
        "providers": entries,
    }


async def model_save_key(params: dict) -> dict:
    """为 API-key provider 保存凭证并返回更新后的 provider entry。

    参数按 ``ModelSaveKeyParams`` 校验。unknown provider 抛出
    ``ConfigValidationError``；OAuth provider 抛出 ``NotSupportedInV01Error`` 并提示使用
    ``codeflow provider login``；``custom`` 与 ``azure_openai`` 还必须提供 ``api_base``。
    实际写入通过 ``set_provider_fields`` 在线程中完成。

    返回的 ``provider`` 反映写入后的本地配置。方法不会向 provider 发请求，因此无法证明
    ``api_key`` 或 ``api_base`` 可用。
    """
    parsed = _parse(ModelSaveKeyParams, params)

    spec = find_by_name(parsed.slug)
    if spec is None:
        raise ConfigValidationError(
            f"unknown provider '{parsed.slug}'",
            data={"slug": parsed.slug},
        )
    if spec.is_oauth:
        raise NotSupportedInV01Error(
            f"{spec.label} uses OAuth; run `codeflow provider login {parsed.slug.replace('_', '-')}`",
            data={"slug": parsed.slug},
        )
    if parsed.slug in _NEEDS_API_BASE and not parsed.api_base:
        raise ConfigValidationError(
            f"{spec.label} requires an api_base",
            data={"slug": parsed.slug, "field": "api_base"},
        )

    fields: dict[str, Any] = {"api_key": parsed.api_key}
    if parsed.api_base:
        fields["api_base"] = parsed.api_base

    try:
        await asyncio.to_thread(set_provider_fields, parsed.slug, fields)
    except RuntimeError as exc:
        raise NotSupportedInV01Error(str(exc), data={"slug": parsed.slug}) from exc
    except KeyError as exc:
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc

    _, current_provider = _current_selection()
    return {
        "provider": _build_provider_entry(parsed.slug, current_provider=current_provider),
    }


async def model_disconnect(params: dict) -> dict:
    """清除指定 provider 的本地配置凭证。

    ``params`` 按 ``ModelDisconnectParams`` 校验，``reset_provider`` 在线程中执行；unknown
    slug 转成 ``ConfigValidationError``。成功返回 ``{"disconnected": True}``，只表示本地
    reset 完成，不会撤销 provider 端 token，也不会终止已经在运行的远端请求。
    """
    parsed = _parse(ModelDisconnectParams, params)
    try:
        await asyncio.to_thread(reset_provider, parsed.slug)
    except KeyError as exc:
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc
    return {"disconnected": True}


async def model_add_model(params: dict) -> dict:
    """把 model 名称加入 provider 的本地 curated list。

    参数按 ``ModelAddModelParams`` 校验，写入委托给 ``add_provider_model``；unknown provider
    抛出 ``ConfigValidationError``。返回合并 common models 后的 provider entry。添加名称
    不会下载模型、验证路由或切换当前 AgentLoop。
    """
    parsed = _parse(ModelAddModelParams, params)
    try:
        await asyncio.to_thread(add_provider_model, parsed.slug, parsed.model)
    except KeyError as exc:
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc
    _, current_provider = _current_selection()
    return {
        "provider": _build_provider_entry(parsed.slug, current_provider=current_provider),
    }


async def model_remove_model(params: dict) -> dict:
    """从 provider 的本地 curated list 删除 model 名称。

    参数按 ``ModelRemoveModelParams`` 校验，写入委托给 ``remove_provider_model``；unknown
    provider 抛出 ``ConfigValidationError``。返回更新后的 provider entry。common model
    仍可能由 ``common_models_for()`` 补回，因此返回列表才是 picker 的最终可见 snapshot。
    """
    parsed = _parse(ModelRemoveModelParams, params)
    try:
        await asyncio.to_thread(remove_provider_model, parsed.slug, parsed.model)
    except KeyError as exc:
        raise ConfigValidationError(str(exc), data={"slug": parsed.slug}) from exc
    _, current_provider = _current_selection()
    return {
        "provider": _build_provider_entry(parsed.slug, current_provider=current_provider),
    }


def register_model_methods(dispatcher: "Dispatcher") -> None:
    """在 Dispatcher 上注册五个 ``model.*`` handler。

    注册项为 ``model.options``、``model.save_key``、``model.disconnect``、
    ``model.add_model`` 和 ``model.remove_model``。函数不读取或修改 provider config；重复
    注册由 Dispatcher 抛出 ``ValueError``。
    """
    dispatcher.register("model.options", model_options)
    dispatcher.register("model.save_key", model_save_key)
    dispatcher.register("model.disconnect", model_disconnect)
    dispatcher.register("model.add_model", model_add_model)
    dispatcher.register("model.remove_model", model_remove_model)


__all__ = [
    "model_options",
    "model_save_key",
    "model_disconnect",
    "model_add_model",
    "model_remove_model",
    "register_model_methods",
    "_build_provider_entry",
]
