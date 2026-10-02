"""实现 ``config.get`` / ``config.set`` RPC handlers（specs §3.6）。

契约来源是 ``docs/openspec/changes/tui-ipc-bridge/specs/tui-ipc.md §3.6``。
v0.1 surface 只开放四个 hot-changeable key；其他 write target 抛出
:class:`ConfigFieldReadonlyError`（``-32010``）。值以 dotted-path nesting 写入
``~/.codeflow/config.json``，例如 ``tui.theme`` 对应
``{"tui": {"theme": "..."}}``，使 legacy ``codeflow.config.loader`` 无需额外 schema
转换就能读取同一文件。

每个 key 的验证边界如下：

* ``agent.thinking_budget`` 必须是 non-negative integer，且 bool 不算 integer；
* ``agent.temperature`` 必须是 int/float，范围闭区间 ``[0.0, 2.0]``；
* ``tui.theme`` 必须是匹配 ``[A-Za-z0-9_-]+`` 的 non-empty string；
* ``tui.show_token_usage`` 必须是 boolean。

其他不合规输入抛出 :class:`ConfigValidationError`（``-32011``）。读取成功只表示拿到
磁盘值或默认值；写入成功表示 JSON 已保存并不等于所有已运行组件都热重载。特殊 key
``model`` 会先构造 prospective provider，再持久化并替换 live loop，避免构造失败留下
半写入状态。本模块不负责完整配置 schema、secret 管理或跨进程配置同步。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from codeflow.cli._helpers import load_runtime_config, make_provider
from codeflow.config.loader import get_config_path
from codeflow.providers.registry import find_by_model
from codeflow.tui_rpc.errors import (
    ConfigFieldReadonlyError,
    ConfigValidationError,
    ModelNotAvailableError,
    ModelSwitchInTurnError,
)
from codeflow.tui_rpc.methods.turn import has_active_turns

if TYPE_CHECKING:
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.methods.session import AgentLoopFactory


# 磁盘配置缺少键时，config.get 返回的默认值。
_DEFAULTS: dict[str, Any] = {
    "agent.thinking_budget": 0,
    "agent.temperature": 1.0,
    "tui.theme": "default",
    "tui.show_token_usage": True,
}


# ---------------------------------------------------------------------------
# 各配置键的校验器
# ---------------------------------------------------------------------------


_THEME_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_thinking_budget(value: Any) -> int:
    # 布尔值是 int 的子类，因此需显式拒绝，避免 True 被静默转成 1。
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(
            "agent.thinking_budget must be a non-negative integer",
            data={"field": "agent.thinking_budget", "got": repr(value)},
        )
    if value < 0:
        raise ConfigValidationError(
            "agent.thinking_budget must be non-negative",
            data={"field": "agent.thinking_budget", "value": value},
        )
    return value


def _validate_temperature(value: Any) -> float:
    if isinstance(value, bool):  # bool 是 int 的子类，需提前拒绝
        raise ConfigValidationError(
            "agent.temperature must be a number in [0, 2]",
            data={"field": "agent.temperature", "got": repr(value)},
        )
    if not isinstance(value, (int, float)):
        raise ConfigValidationError(
            "agent.temperature must be a number in [0, 2]",
            data={"field": "agent.temperature", "got": repr(value)},
        )
    if not (0.0 <= float(value) <= 2.0):
        raise ConfigValidationError(
            "agent.temperature out of range [0, 2]",
            data={"field": "agent.temperature", "value": value},
        )
    return float(value)


def _validate_theme(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigValidationError(
            "tui.theme must be a non-empty string",
            data={"field": "tui.theme", "got": repr(value)},
        )
    if not _THEME_NAME_RE.match(value):
        raise ConfigValidationError(
            "tui.theme must match [A-Za-z0-9_-]+",
            data={"field": "tui.theme", "value": value},
        )
    return value


def _validate_show_token_usage(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(
            "tui.show_token_usage must be a boolean",
            data={"field": "tui.show_token_usage", "got": repr(value)},
        )
    return value


_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "agent.thinking_budget": _validate_thinking_budget,
    "agent.temperature": _validate_temperature,
    "tui.theme": _validate_theme,
    "tui.show_token_usage": _validate_show_token_usage,
}

# 公开的标准可写键集合；调用方可通过迭代它枚举默认值，
# 而无需直接修改 ``_DEFAULTS``。
CONFIG_WRITABLE_KEYS: tuple[str, ...] = tuple(_VALIDATORS.keys())


# ---------------------------------------------------------------------------
# 持久化辅助函数
# ---------------------------------------------------------------------------


def _config_path() -> Path:
    return get_config_path()


def _load_config() -> dict[str, Any]:
    """为 get/set/_set_model 的 read-modify-write 读取 ``config.json``。

    文件 absent 或 empty 时返回 ``{}``，允许安全创建新配置。文件存在但无法解析时抛出
    ``ConfigValidationError``，不能沿用旧版 empty-dict fallback：若此时返回 ``{}``，
    后续 ``_save_config`` 会用单个改动 key 覆盖用户完整配置，造成 data loss。

    on-disk file 是 source of truth，下游 loader 会独立读取同一文件；本函数返回的是本次
    read-modify-write 使用的 dict snapshot，不提供锁或跨进程事务保证。
    """
    from codeflow.config.loader import ConfigReadError, read_raw_or_raise

    try:
        return read_raw_or_raise(_config_path())
    except ConfigReadError as exc:
        raise ConfigValidationError(str(exc)) from exc


def _save_config(payload: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any | None:
    """读取 ``payload`` 中 ``dotted_key`` 指向的值，缺失时返回 ``None``。

    路径用 ``.`` 分段；任一中间节点不是 dict，或 key 不存在，都视为 absent。调用方需
    注意：该返回约定无法区分“路径缺失”和“磁盘显式保存 JSON null”，两者都会得到
    ``None``，当前 config.get 会在这种情况下使用默认值。
    """
    parts = dotted_key.split(".")
    cur: Any = payload
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: dict[str, Any] = payload
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
# 处理器
# ---------------------------------------------------------------------------


async def config_get(params: dict) -> dict:
    """返回请求的 writable whitelist 配置值。

    ``params["keys"]`` 缺失时读取全部 ``CONFIG_WRITABLE_KEYS``；提供时必须是
    ``list[str]``，否则抛出 ``ConfigValidationError``。每个已知 key 优先返回磁盘值，
    absent 时返回 ``_DEFAULTS``。

    按 Spec §3.6，unknown key 必须 silently omitted，而不是 error。返回形状是
    ``{"config": {...}}``。调用成功只表示读取 snapshot 成功，不保证其他进程看到相同值。
    """
    requested_raw = params.get("keys") if isinstance(params, dict) else None
    if requested_raw is None:
        requested: list[str] = list(CONFIG_WRITABLE_KEYS)
    else:
        if not isinstance(requested_raw, list) or not all(isinstance(k, str) for k in requested_raw):
            raise ConfigValidationError(
                "config.get params.keys must be a list[str] if provided",
                data={"field": "keys", "got": repr(requested_raw)},
            )
        requested = requested_raw

    payload = _load_config()
    out: dict[str, Any] = {}
    for key in requested:
        if key not in _VALIDATORS:
            # 按协议静默忽略未知或不在白名单中的键。
            continue
        value = _get_nested(payload, key)
        out[key] = value if value is not None else _DEFAULTS[key]
    return {"config": out}


async def config_set(
    params: dict,
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> dict:
    """验证并写入单个 whitelist key，返回 ``{applied, previous}``。

    ``params`` 必须包含 non-empty string ``key`` 和 ``value``。普通 key 先由专属 validator
    规范化，再 read-modify-write 到 ``config.json``。特殊 key ``"model"`` 会切换 live
    agent loop 的 provider/model，并返回 ``{applied, previous, value}``；细节见
    :func:`_set_model`。``agent_loop_factory`` 只在 model 切换时使用。

    参数 shape 或 value 无效时抛出 ``ConfigValidationError``（``-32011``）；key 不在
    writable whitelist 时抛出 ``ConfigFieldReadonlyError``（``-32010``）。普通配置的
    ``applied=True`` 表示文件已写入，不表示所有运行组件立即采用新值。
    """
    if not isinstance(params, dict):
        raise ConfigValidationError(
            "config.set params must be an object",
            data={"got": type(params).__name__},
        )

    key = params.get("key")
    if not isinstance(key, str) or not key:
        raise ConfigValidationError(
            "config.set params.key is required and must be a non-empty string",
            data={"field": "key", "got": repr(key)},
        )
    if "value" not in params:
        raise ConfigValidationError(
            "config.set params.value is required",
            data={"field": "value"},
        )
    raw_value = params["value"]

    if key == "model":
        return _set_model(params, raw_value, agent_loop_factory)

    if key not in _VALIDATORS:
        raise ConfigFieldReadonlyError(
            f"key '{key}' is not in the v0.1 hot-changeable whitelist",
            data={"field": key, "writable": list(CONFIG_WRITABLE_KEYS)},
        )

    validated = _VALIDATORS[key](raw_value)

    payload = _load_config()
    previous = _get_nested(payload, key)
    _set_nested(payload, key, validated)
    _save_config(payload)

    return {"applied": True, "previous": previous}


def _set_model(
    params: dict,
    raw_value: Any,
    agent_loop_factory: "AgentLoopFactory | None",
) -> dict:
    """切换 global model（及 provider），并更新 live AgentLoop。

    ``raw_value`` 必须是 non-empty model string；可选 ``params["provider"]`` 必须是
    string。未显式提供 provider 时，函数尝试用 ``find_by_model()`` 推导；gateway 或 local
    model 无匹配时保留原强制 provider。存在 active Turn 时抛出
    ``ModelSwitchInTurnError``，避免执行中的 loop 被中途换模。

    关键顺序是 BEFORE 持久化，先基于 prospective config 构造 provider：构造失败会抛出
    ``ModelNotAvailableError``，on-disk model 保持 untouched。保存成功后，如果 live loop
    提供 ``replace_provider`` 就调用它，否则直接更新 ``provider`` 与 ``model`` 属性。
    返回 ``{applied, previous, value}``；它表示切换步骤完成，不证明后续模型调用一定成功。
    """
    if not isinstance(raw_value, str) or not raw_value:
        raise ConfigValidationError(
            "config.set model value must be a non-empty string",
            data={"field": "value", "got": repr(raw_value)},
        )
    new_provider = params.get("provider")
    if new_provider is not None and not isinstance(new_provider, str):
        raise ConfigValidationError(
            "config.set model provider must be a string",
            data={"field": "provider", "got": repr(new_provider)},
        )
    # 简写 `/model <name>` 不携带提供商，需从模型推导，避免之前强制的提供商
    # 将新模型静默路由到错误位置。网关或本地模型（无关键词匹配）保留原强制提供商。
    if new_provider is None:
        spec = find_by_model(raw_value)
        if spec is not None:
            new_provider = spec.name

    session_id = params.get("session_id")
    if has_active_turns():
        raise ModelSwitchInTurnError(
            "cannot switch the global model while a turn is active",
            data={"session_id": session_id},
        )

    payload = _load_config()
    previous = _get_nested(payload, "agents.defaults.model")

    loop = agent_loop_factory() if agent_loop_factory is not None else None
    built_provider = None
    if loop is not None:
        runtime = load_runtime_config(None, None)
        runtime.agents.defaults.model = raw_value
        if new_provider is not None:
            runtime.agents.defaults.provider = new_provider
        try:
            built_provider = make_provider(runtime)
        except (SystemExit, RuntimeError, ValueError) as exc:
            raise ModelNotAvailableError(
                f"cannot build provider for model {raw_value!r}",
                data={"model": raw_value, "error": str(exc)},
            ) from exc

    _set_nested(payload, "agents.defaults.model", raw_value)
    if new_provider is not None:
        _set_nested(payload, "agents.defaults.provider", new_provider)
    _save_config(payload)

    if loop is not None:
        replace_provider = getattr(loop, "replace_provider", None)
        if callable(replace_provider):
            replace_provider(built_provider, model=raw_value)
        else:
            loop.provider = built_provider
            loop.model = raw_value

    return {"applied": True, "previous": previous, "value": raw_value}


def register_config_methods(
    dispatcher: "Dispatcher",
    *,
    agent_loop_factory: "AgentLoopFactory | None" = None,
) -> None:
    """在 ``dispatcher`` 上注册 ``config.get`` 与 ``config.set``。

    ``agent_loop_factory`` 被闭包捕获，只供特殊 ``model`` 写入更新 live loop。重复注册由
    Dispatcher 抛出 ``ValueError``。函数不读取配置，也不触发模型切换。
    """

    async def _set(params: dict) -> dict:
        return await config_set(params, agent_loop_factory=agent_loop_factory)

    dispatcher.register("config.get", config_get)
    dispatcher.register("config.set", _set)


__all__ = [
    "config_get",
    "config_set",
    "register_config_methods",
    "CONFIG_WRITABLE_KEYS",
]
