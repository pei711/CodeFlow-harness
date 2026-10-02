"""实现探测 provider 配置状态的 ``setup.status`` RPC handler。

契约来自 ``docs/openspec/changes/tui-ipc-bridge/specs/tui-ipc.md §3.9`` 与
``design.md §3a.1``。它存在的原因是 hermes fork-import 的
``useSessionLifecycle.ts:127,206`` 和 ``setupHandoff.ts:43`` 会在 app boot 时硬调用
``setup.status``；若返回 ``{provider_configured: false}``，UI 会把用户停在 *Setup
required* panel，并拒绝创建 Session，因此 v0.1 也必须遵守该 contract。

Q9 的 partial answer 是把 ``agents.defaults.provider`` 视为 canonical provider field。
具体名称（``"anthropic"``、``"openai"`` 等）算 configured；sentinel ``"auto"`` 算
not-yet-configured，因为用户尚未选择，CodeFlow 也未执行 auto-detection。实际 onboarding gate
还要求 model 与 provider signal 同时存在。

若 config read 因 file missing、unparseable JSON 或 unexpected shape 等原因失败，v0.1 按
design §3a.1 fail-open 返回 ``{"provider_configured": true}``，避免 hermes UI 因 transient
I/O hiccup 永久阻塞。proper provider auto-detection 完成后，v0.2 可以收紧此 signal。
该布尔值只是本地 setup gate，不验证 credentials，也不证明一次模型调用能成功。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from codeflow.config.loader import get_config_path

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher


_AUTO_SENTINEL = "auto"


def _config_path() -> Path:
    return get_config_path()


def _detect_provider_configured(payload: dict) -> bool:
    """判断 loaded config payload 是否具备 onboarding 所需 provider signal。

    gate 的 ``"required config complete"`` 标准是：``agents.defaults.model`` 已设置，AND
    至少一个 provider 有 ``apiKey``。任一条件单独存在都不能驱动 Turn，UI 仍应停在 setup
    panel。显式、非 ``auto`` 的 ``agents.defaults.provider`` 也算 provider signal，用于兼容
    早于 per-provider section 的 legacy config。

    ``payload`` 不是 dict、model 缺失或没有任何 provider signal 时返回 ``False``。返回
    ``True`` 不会读取 secret 内容之外的信息，也不发起远端认证。
    """
    if not isinstance(payload, dict):
        return False

    agents = payload.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    defaults = defaults if isinstance(defaults, dict) else {}

    model = defaults.get("model")
    if not (isinstance(model, str) and model):
        return False

    provider = defaults.get("provider")
    if isinstance(provider, str) and provider and provider != _AUTO_SENTINEL:
        return True

    providers = payload.get("providers")
    return isinstance(providers, dict) and any(
        isinstance(value, dict) and value.get("apiKey") for value in providers.values()
    )


async def setup_status(params: dict) -> dict:
    """执行 ``setup.status``，返回 provider 是否已满足本地配置 gate。

    方法读取 ``get_config_path()`` 指向的 UTF-8 JSON，并调用
    ``_detect_provider_configured()``。file missing、其他 ``OSError`` 或 JSON decode failure
    都按 v0.1 fallback 返回 ``{"provider_configured": true}``，使 hermes UI 不因读取问题
    停在 *Setup required* panel。``params`` 当前不参与判断。

    正常 ``False`` 表示需要 setup；fallback ``True`` 只为 UI 可用性降级，不能作为 provider
    已配置或凭证有效的证据。
    """
    path = _config_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("setup.status: {} missing → v0.1 fallback true", path)
        return {"provider_configured": True}
    except OSError as exc:
        logger.warning("setup.status: read failed for {}: {} → fallback true", path, exc)
        return {"provider_configured": True}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("setup.status: invalid JSON in {}: {} → fallback true", path, exc)
        return {"provider_configured": True}

    return {"provider_configured": _detect_provider_configured(payload)}


def register_setup_methods(dispatcher: "Dispatcher") -> None:
    """在 Dispatcher 上注册 ``setup.status``。

    注册不读取 config；重复注册由 Dispatcher 抛出 ``ValueError``。
    """
    dispatcher.register("setup.status", setup_status)


__all__ = ["setup_status", "register_setup_methods"]
