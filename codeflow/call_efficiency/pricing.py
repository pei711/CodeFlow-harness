"""LLM Call Cost Estimation 的 Single Source of Truth。

`CallEfficiency`、Tracing 与 Historical TokenWise Compatibility Code 共同使用这里的结果。所有
路径从同一处计算 Cost，可防止 Recorded Estimate 与 Reported Estimate 随时间产生 Drift。

Pricing Sources 按以下顺序查找：

1. ``litellm.cost_per_token``：覆盖多数 Public Models，先尝试 ``openrouter/<model>`` Alias，再尝试
   Bare Model ID；
2. 之前 Refresh 的 OpenRouter Model Catalog：普通 Call Path 只读 Local Memory 或 Disk，Remote
   Refresh 必须由 Operator 显式执行；
3. ``_FALLBACK_PRICING``：为前两者都缺失的模型维护 Manual Rate Table；
4. `None`：所有来源都不认识该模型，Caller 应 Gracefully Degrade，不能把未知成本当零。

Anthropic Ephemeral Cache Pricing 叠加在 Base Rate 上：Cache Read 是 Prompt Rate 的 10%，Cache
Write 是 125%，对应 Ephemeral 5-min TTL。DeepSeek V4 改用其公开的 Automatic Disk-cache Hit、Miss
与 Output Rates。其他 Provider 传入 ``cache_read_tokens=0``、``cache_write_tokens=0`` 时，公式自然
退化为标准输入输出计费。
"""

from __future__ import annotations

import time

import httpx
from loguru import logger

from codeflow.call_efficiency import model_catalog_cache

# 费率对：(prompt_cost_per_token, completion_cost_per_token)，单位为美元。
# 保持此表精简：它只为 LiteLLM 尚未收录的新模型提供回退。
# 添加条目前应先检查 LiteLLM。
_FALLBACK_PRICING: dict[str, tuple[float, float]] = {
    # OpenRouter 模型页面快照（2026-03）。
    "z-ai/glm-4.5-air": (0.13e-6, 0.85e-6),  # 每百万令牌 $0.13/$0.85
}

# DeepSeek 2026-08-06“模型与定价”页面的快照。这些 Provider 直连费率
# 优先于 LiteLLM，因为普通的 prompt/completion 费率对无法表达缓存命中价格。
_DEEPSEEK_V4_PRICING: dict[str, tuple[float, float, float]] = {
    # 依次为未命中缓存输入、命中缓存输入、输出，单位为美元/令牌。
    "deepseek-v4-flash": (0.14e-6, 0.0028e-6, 0.28e-6),
    "deepseek-v4-pro": (0.435e-6, 0.003625e-6, 0.87e-6),
}

# 记录已警告过的未知模型，确保每个模型只输出一次日志。
_WARNED_UNKNOWN: set[str] = set()

# 实时 OpenRouter 价格表，惰性获取并在进程内缓存 1 小时。
# 同时映射完整 id（``deepseek/deepseek-v4-pro``）和不带命名空间的别名。
# （``deepseek-v4-pro``）映射到 OpenRouter 的逐令牌 ``pricing`` 字典。
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_CACHE_TTL = 3600
_OPENROUTER_CACHE: dict[str, dict] = {}
_OPENROUTER_CACHE_TIME: float = 0.0
_ALLOW_NETWORK_CATALOG = False


def _try_litellm_rates(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    allow_import: bool,
) -> tuple[float, float] | None:
    """向 LiteLLM 查询 Per-token Rates，返回 ``(prompt_rate, completion_rate)`` 或 `None`。

    `allow_import=False` 且当前 Process 尚未加载 LiteLLM 时直接跳过，避免 Cost Path 引入沉重 Import。
    查询会依次尝试 OpenRouter Alias 与原 Model，并用至少一个合成 Token 反推单 Token 费率。LiteLLM
    异常、缺值或对未知模型返回的 ``(0, 0)`` 都视为 Miss，交给后续 Pricing Source。
    """
    import sys

    if not allow_import and "litellm" not in sys.modules:
        return None
    try:
        from codeflow.providers.litellm_setup import import_litellm

        litellm = import_litellm()
    except Exception:
        return None

    candidates = [model]
    if not model.startswith("openrouter/"):
        candidates.insert(0, f"openrouter/{model}")

    # litellm.cost_per_token 至少需要 1 个非零 token 才能计算。
    # 这里传入合成 token 来反推出单 token 费率。
    probe_in = input_tokens if input_tokens else 1
    probe_out = output_tokens if output_tokens else 1

    for candidate in candidates:
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=candidate, prompt_tokens=probe_in, completion_tokens=probe_out
            )
        except Exception:
            continue
        if prompt_cost is None or completion_cost is None:
            continue
        if prompt_cost == 0 and completion_cost == 0:
            # LiteLLM 对未知模型返回 (0, 0)，应按未命中处理。
            continue
        return prompt_cost / probe_in, completion_cost / probe_out

    return None


def _fetch_openrouter_models() -> dict[str, dict]:
    """返回 OpenRouter Model Table，Live Fetch 后在 Process 内缓存 1h。

    每项结构为 ``{"pricing": ..., "context_length": ...}``，同时用 Full ID 与 Bare Alias 建立 Double
    Keys。函数先检查新鲜内存缓存，再检查磁盘层，只有两者都不可用或过期时才访问网络。任何 Network
    Failure 都返回 Stale Cache 或 Empty Dict；Pricing 绝不能把目录刷新异常传播进 Cost Path。

    成功联网后会同时更新 In-process 与 On-disk Catalog。返回表只是一次价格目录 Snapshot，不代表
    模型当前可调用。
    """
    global _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME

    now = time.time()
    if _OPENROUTER_CACHE and (now - _OPENROUTER_CACHE_TIME) < _OPENROUTER_CACHE_TTL:
        return _OPENROUTER_CACHE

    # 磁盘层：从仍然新鲜的磁盘缓存热启动（也可复用同级进程刚获取的数据），
    # 无需访问网络。
    disk = model_catalog_cache.load()
    if disk is not None and (now - disk[1]) < _OPENROUTER_CACHE_TTL:
        _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME = disk
        return _OPENROUTER_CACHE

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(_OPENROUTER_MODELS_URL)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("pricing: OpenRouter models fetch failed ({}), degrading", exc)
        if _OPENROUTER_CACHE:
            return _OPENROUTER_CACHE
        if disk is not None:
            _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME = disk
            return _OPENROUTER_CACHE
        return {}

    cache: dict[str, dict] = {}
    for model in data.get("data", []):
        model_id = model.get("id", "")
        if not model_id:
            continue
        entry = {
            "pricing": model.get("pricing") or {},
            "context_length": model.get("context_length"),
        }
        cache[model_id] = entry
        if "/" in model_id:
            cache.setdefault(model_id.split("/", 1)[1], entry)

    _OPENROUTER_CACHE = cache
    _OPENROUTER_CACHE_TIME = time.time()
    model_catalog_cache.save(cache)
    return cache


def _cached_openrouter_models() -> dict[str, dict]:
    global _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME

    if _OPENROUTER_CACHE:
        return _OPENROUTER_CACHE
    disk = model_catalog_cache.load()
    if disk is not None:
        _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME = disk
    return _OPENROUTER_CACHE


def _lookup_openrouter_entry(model: str, *, allow_network: bool) -> dict | None:
    """把 Model 解析为对应的 OpenRouter Catalog Entry。

    先移除 Leading ``openrouter/``，再尝试 Remaining ID 与其 Bare Alias。它是 LiteLLM 无法映射时的
    Cross-provider Fallback，因此 Catalog 也能覆盖例如 Direct ``deepseek/...`` Route。是否允许因
    Miss 触发联网由 `allow_network` 明确控制；无匹配返回 `None`。
    """
    key = model.removeprefix("openrouter/")
    table = _fetch_openrouter_models() if allow_network else _cached_openrouter_models()
    entry = table.get(key)
    if entry is None and "/" in key:
        entry = table.get(key.split("/", 1)[1])
    return entry


def _try_openrouter_rates(model: str, *, allow_network: bool) -> tuple[float, float] | None:
    """查询 Optional OpenRouter Catalog Rates，返回费率对或 `None`。

    找到 Entry 后把 ``pricing.prompt`` 与 ``pricing.completion`` 转成 Float。字段缺失、类型错误或数值
    非法均视为 Miss，不在这里记录 Unknown Warning；上层仍可继续尝试 Manual Fallback。
    """
    entry = _lookup_openrouter_entry(model, allow_network=allow_network)
    if not entry:
        return None
    pricing = entry.get("pricing") or {}
    try:
        return float(pricing["prompt"]), float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None


def _try_litellm_context_window(model: str, *, allow_import: bool) -> int | None:
    """从 LiteLLM Static Model Metadata 读取 Context Window，Offline 覆盖多数 Mapped Providers。

    与费率查询一样，`allow_import=False` 可阻止首次 Import。函数尝试 OpenRouter Alias 与原 Model，
    优先读取 ``max_input_tokens``，再读 ``max_tokens``；无法取得有效整数时返回 `None`，由 Catalog
    Fallback 接手。
    """
    import sys

    if not allow_import and "litellm" not in sys.modules:
        return None
    try:
        from codeflow.providers.litellm_setup import import_litellm

        litellm = import_litellm()
    except Exception:
        return None

    candidates = [model]
    if not model.startswith("openrouter/"):
        candidates.insert(0, f"openrouter/{model}")

    for candidate in candidates:
        try:
            info = litellm.get_model_info(candidate)
        except Exception:
            continue
        if not info:
            continue
        window = info.get("max_input_tokens") or info.get("max_tokens")
        if window:
            try:
                return int(window)
            except (TypeError, ValueError):
                continue
    return None


def resolve_context_window(
    model: str,
    *,
    allow_network: bool | None = None,
    allow_litellm_import: bool = True,
) -> int | None:
    """返回 Model 的真实 Context Window Token 数；未知时返回 `None`。

    Sources 依次为 LiteLLM Static Model Metadata 与 Locally Cached OpenRouter Catalog。Default Call
    Path 永不为此自动联网 Refresh Catalog；只有显式 `allow_network=True` 才可访问远端。未知或字段
    非法时返回 `None`，让 Caller 保留 Configured Default，而不是用猜测值覆盖用户配置。
    """
    window = _try_litellm_context_window(model, allow_import=allow_litellm_import)
    if window:
        return window

    network = _ALLOW_NETWORK_CATALOG if allow_network is None else allow_network
    entry = _lookup_openrouter_entry(model, allow_network=network)
    if entry:
        try:
            length = int(entry.get("context_length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length:
            return length
    return None


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    *,
    allow_network: bool | None = None,
    allow_litellm_import: bool = True,
) -> float | None:
    """估算一次 LLM Call 的 USD Cost；Unknown Models 返回 `None`。

    ``input_tokens`` 必须是 Fresh、即 Non-cache Prompt Tokens。Anthropic 的 ``usage.input_tokens``
    已排除 Cache Tokens，应原样传入；DeepSeek 会把 Cache Hits 与 Total Prompt Tokens 分别报告，Caller
    必须先把 Total 规范为 Fresh Tokens 再调用本函数。

    DeepSeek V4 Direct Route 优先使用专用 Hit/Miss/Output 费率；其他模型按 LiteLLM、OpenRouter
    Catalog、Manual Fallback 的顺序寻找费率，并应用 Anthropic-style Cache 倍率。返回数字是基于当前
    Price Snapshot 的 Estimate，不是 Provider Invoice；未知模型只 Warning 一次并返回 `None`。
    """
    deepseek_model = model.removeprefix("deepseek/")
    if not model.startswith("openrouter/") and deepseek_model in _DEEPSEEK_V4_PRICING:
        miss_rate, hit_rate, output_rate = _DEEPSEEK_V4_PRICING[deepseek_model]
        return (
            input_tokens * miss_rate
            + output_tokens * output_rate
            + cache_read_tokens * hit_rate
            + cache_write_tokens * miss_rate
        )

    network = _ALLOW_NETWORK_CATALOG if allow_network is None else allow_network
    rates = _try_litellm_rates(
        model,
        input_tokens,
        output_tokens,
        allow_import=allow_litellm_import,
    )
    if rates is None:
        rates = _try_openrouter_rates(model, allow_network=network)
    if rates is None:
        key = model.removeprefix("openrouter/")
        if key in _FALLBACK_PRICING:
            rates = _FALLBACK_PRICING[key]
        else:
            if model not in _WARNED_UNKNOWN:
                logger.warning("pricing: unknown model '{}', cost estimate = None", model)
                _WARNED_UNKNOWN.add(model)
            return None

    prompt_rate, completion_rate = rates
    cost = (
        input_tokens * prompt_rate
        + output_tokens * completion_rate
        + cache_read_tokens * prompt_rate * 0.1
        + cache_write_tokens * prompt_rate * 1.25
    )
    return cost


def estimate_cost_from_rates(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    input_usd_per_token: float,
    output_usd_per_token: float,
    cache_read_usd_per_token: float,
    cache_write_usd_per_token: float | None = None,
) -> float:
    """依据 Frozen Evidence Price Snapshot 估算一次调用。

    函数直接用调用记录冻结的 Input、Output、Cache Read 与可选 Cache Write Per-token Rates 乘以对应
    Token Counts，不再查询当前 Catalog。未提供 Write Rate 时沿用 Input Rate。这适合重放历史证据，
    避免今天的价格变化改写过去记录的成本含义。
    """
    write_rate = input_usd_per_token if cache_write_usd_per_token is None else cache_write_usd_per_token
    return (
        input_tokens * input_usd_per_token
        + output_tokens * output_usd_per_token
        + cache_read_tokens * cache_read_usd_per_token
        + cache_write_tokens * write_rate
    )


def reset_warning_cache() -> None:
    """清空已经记录过 ``unknown`` Warning 的 Model Set。

    该接缝 Only Useful for Tests，用于让多个测试用例分别观察首次 Warning。Production Code 应保持
    每个未知模型只落一次日志，避免高频 Cost Estimation 刷屏。
    """
    _WARNED_UNKNOWN.clear()


def reset_openrouter_cache() -> None:
    """清空 In-process OpenRouter Catalog Cache。

    该函数 Only Useful for Tests。应与 ``model_catalog_cache._CACHE_PATH`` Test Seam 配合，才能在不
    触碰真实 ``~/.codeflow/cache/`` 的情况下覆盖 Disk Tiers。它不会删除磁盘文件，也不会自动重新联网。
    """
    global _OPENROUTER_CACHE, _OPENROUTER_CACHE_TIME
    _OPENROUTER_CACHE = {}
    _OPENROUTER_CACHE_TIME = 0.0


def refresh_model_catalog() -> dict[str, dict]:
    """通过网络 Explicitly Refresh Optional OpenRouter Model Catalog。

    这是 Operator Action，而不是普通 Call Path 的隐式副作用。返回刷新后或降级得到的 Catalog；
    Network Failure 仍可能返回 Stale/Empty Data，因此调用成功不等于远端一定更新。
    """
    return _fetch_openrouter_models()
