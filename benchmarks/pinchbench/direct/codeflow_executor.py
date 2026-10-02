"""
CodeFlow Agent Executor - replaces OpenClaw CLI subprocess calls.

Drives CodeFlow's AgentLoop.run_turn() programmatically to execute
benchmark tasks, capturing the full session transcript for grading.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from task_loader import Task

logger = logging.getLogger(__name__)

# 默认 OpenAI 兼容基准配置。
DEFAULT_API_KEY = (
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
)
DEFAULT_API_BASE = (
    os.environ.get("OPENROUTER_API_BASE")
    or os.environ.get("DEEPSEEK_API_BASE")
    or os.environ.get("OPENAI_BASE_URL")
    or ""
)
DEFAULT_PROVIDER = os.environ.get("CODEFLOW_BENCH_PROVIDER", "custom")
DEFAULT_MODEL = os.environ.get("CODEFLOW_BENCH_MODEL", "deepseek-v4-flash")


async def _fetch_openrouter_model_ids(api_key: str) -> set[str]:
    """Fetch the set of valid model IDs from OpenRouter's /models endpoint."""
    import httpx

    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            ids = {m["id"] for m in data.get("data", []) if "id" in m}
            logger.info("OpenRouter: %d available models fetched", len(ids))
            return ids
    except Exception as e:
        logger.warning("Could not fetch OpenRouter model list: %s", e)
        return set()


# ---------------------------------------------------------------------------
# 仅标准模型的 ModelRouter（仅供基准使用，过滤非标准模型标识）。
# ---------------------------------------------------------------------------

# 合法 OpenRouter 供应商使用小写名称，如 "anthropic"、"z-ai"、"minimax"。
# 用户命名空间提交的供应商名称含大小写混合或下划线，如 "Jobeous_II"、
# "JoePro"；拒绝不符合规范的项。
_STANDARD_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9\-\.]*$")


def _is_standard_model_id(model_id: str) -> bool:
    """Return True only for 'provider/model' IDs with a lowercase provider name."""
    parts = model_id.split("/")
    if len(parts) != 2:
        return False
    return bool(_STANDARD_PROVIDER_RE.match(parts[0]))


class _StandardModelRouter:
    """ModelRouter wrapper that restricts selection to models available on OpenRouter.

    PinchBench includes user-tagged submissions (e.g. "Jobeous_II/…") that are
    not publicly accessible. This wrapper fetches the live OpenRouter model
    list and removes any benchmark entries that don't appear in it.

    Even when the live fetch succeeds, provider-name validation is applied as
    a second gate to catch user-namespace models that happen to be listed on
    OpenRouter but are effectively private/inaccessible.
    """

    def __init__(self, api_key: str, profile: str, fallback_model: str):
        from codeflow.routing.router import ModelRouter

        self._inner = ModelRouter(api_key=api_key, profile=profile, fallback_model=fallback_model)
        self._api_key = api_key
        self._fallback_model = fallback_model
        self._valid_ids: set[str] = set()

    async def initialize(self) -> None:
        # 并行获取有效 OpenRouter 模型标识并初始化基准数据。
        self._valid_ids, _ = await asyncio.gather(
            _fetch_openrouter_model_ids(self._api_key),
            self._inner.initialize(),
        )
        # 将基准数据过滤为 OpenRouter 上实际可用的模型。只要有基准数据就始终过滤；
        # 若获取失败导致 _valid_ids 为空，则记录警告，但仍要求不带用户前缀的
        # "provider/model" 格式，以移除非标准模型。
        if self._inner._data:
            before = len(self._inner._data)
            if self._valid_ids:
                # 仅保留同时位于 OpenRouter 实时列表且供应商名为标准小写的模型，
                # 对用户命名空间实施双重门禁。
                self._inner._data = {
                    k: v for k, v in self._inner._data.items() if k in self._valid_ids and _is_standard_model_id(k)
                }
            else:
                # 实时获取失败时回退为要求小写供应商名。即使 "Jobeous_II/…"、
                # "JoePro/…" 等恰有一个 "/" 分段，也会被拒绝。
                logger.warning(
                    "OpenRouter model list unavailable; filtering benchmark data "
                    "to standard lowercase-provider 'provider/model' entries only"
                )
                self._inner._data = {k: v for k, v in self._inner._data.items() if _is_standard_model_id(k)}
            removed = before - len(self._inner._data)
            logger.info(
                "Filtered %d unavailable model(s) from benchmark data (%d OpenRouter-accessible models remain)",
                removed,
                len(self._inner._data),
            )

    async def select_model_id(self, prompt: str) -> str | None:
        return await self._inner.select_model_id(prompt)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# 用量跟踪供应商包装器（仅供基准使用，不修改 loop.py）。
# ---------------------------------------------------------------------------


class _UsageTrackingProvider:
    """Wraps any LLMProvider to accumulate token usage and track model calls.

    Intercepts chat_with_retry() so every LLM call in the agent loop is
    recorded per-model, enabling accurate cost estimation across mixed-model
    routing sessions.  Delegates everything else transparently to the real provider.
    """

    def __init__(self, inner):
        self._inner = inner
        self.accumulated: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.model_calls: List[str] = []  # 按调用顺序排列的模型列表。
        # 各模型 token 明细：model -> {prompt_tokens, completion_tokens}。
        self.per_model_usage: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        self.accumulated = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model_calls = []
        self.per_model_usage = {}

    # --- 拦截唯一关键方法 ---

    async def chat_with_retry(self, messages, tools=None, model=None, **kwargs):
        response = await self._inner.chat_with_retry(messages, tools=tools, model=model, **kwargs)
        # 累计总用量。
        for k in self.accumulated:
            self.accumulated[k] += response.usage.get(k, 0)
        # 记录实际调用的模型。
        effective = model or self._inner.get_default_model()
        self.model_calls.append(effective)
        # 累计各模型用量。
        if effective not in self.per_model_usage:
            self.per_model_usage[effective] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.per_model_usage[effective]["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
        self.per_model_usage[effective]["completion_tokens"] += response.usage.get("completion_tokens", 0)
        return response

    # --- 透明委托 ---

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------


def prepare_workspace(task: Task, workspace: Path, assets_dir: Path) -> Path:
    """Prepare an isolated workspace for a task, copying fixture files."""
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    for file_spec in task.workspace_files:
        if "content" in file_spec:
            dest = workspace / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"])
            continue

        source_key = file_spec.get("source", "")
        dest_key = file_spec.get("dest", source_key)
        source = assets_dir / source_key
        dest = workspace / dest_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            logger.error("Asset not found: %s", source)
            continue
        dest.write_bytes(source.read_bytes())

    return workspace


def _session_to_openclaw_transcript(
    session_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert CodeFlow session messages (OpenAI format) to PinchBench/OpenClaw
    transcript format so that existing grading functions work unchanged.

    CodeFlow format:
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [...]}
        {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

    OpenClaw/PinchBench format:
        {"type": "message", "message": {"role": "user", "content": [...]}}
        {"type": "message", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "..."},
            {"type": "toolCall", "name": "...", "arguments": {...}}
        ]}}
        {"type": "message", "message": {"role": "toolResult", "content": [...]}}
    """
    transcript: List[Dict[str, Any]] = []

    for msg in session_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            text = content if isinstance(content, str) else str(content)
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "user",
                        "content": [text],
                    },
                }
            )

        elif role == "assistant":
            items: List[Dict[str, Any]] = []
            if content:
                items.append({"type": "text", "text": content})

            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    import json

                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args}

                items.append(
                    {
                        "type": "toolCall",
                        "name": func.get("name", ""),
                        "arguments": args,
                    }
                )

            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": items,
                    },
                }
            )

        elif role == "tool":
            result_text = content if isinstance(content, str) else str(content)
            transcript.append(
                {
                    "type": "message",
                    "message": {
                        "role": "toolResult",
                        "content": [result_text],
                    },
                }
            )

    return transcript


def _make_benchmark_provider(model: str, api_key: str, api_base: str, provider_name: str):
    """Create the benchmark LLM provider."""
    from codeflow.providers.base import GenerationSettings
    from codeflow.providers.custom_provider import CustomProvider
    from codeflow.providers.litellm_provider import LiteLLMProvider

    if provider_name == "custom":
        provider = CustomProvider(
            api_key=api_key,
            api_base=api_base,
            default_model=model,
        )
    else:
        provider = LiteLLMProvider(
            api_key=api_key,
            api_base=api_base or ("https://openrouter.ai/api/v1" if provider_name == "openrouter" else None),
            default_model=model,
            provider_name=provider_name,
        )
    provider.generation = GenerationSettings(
        temperature=0.7,
        max_tokens=8192,
    )
    return provider


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate USD cost using LiteLLM's pricing database with manual fallbacks.

    Falls back to _fallback_pricing for models not yet in LiteLLM's DB.
    Returns None if the model is unknown to both.
    """
    # LiteLLM 数据库缺失模型的手工回退定价（美元/token）。来源为截至 2026-03 的
    # OpenRouter 模型页面。
    _fallback_pricing: Dict[str, tuple[float, float]] = {
        "z-ai/glm-4.5-air": (0.13e-6, 0.85e-6),  # 每百万 token 为 $0.13/$0.85。
    }

    try:
        import litellm

        # LiteLLM 要求 OpenRouter 模型采用 "openrouter/<provider>/<model>" 格式。
        or_model = f"openrouter/{model}" if not model.startswith("openrouter/") else model
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=or_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return prompt_cost + completion_cost
    except Exception:
        pass

    # LiteLLM 数据库中不存在模型时的回退路径。
    base_model = model.removeprefix("openrouter/")
    if base_model in _fallback_pricing:
        p_per_tok, c_per_tok = _fallback_pricing[base_model]
        return p_per_tok * prompt_tokens + c_per_tok * completion_tokens

    return None


async def _run_turn_text(agent, message: str, *, session_key: str, chat_id: str) -> str:
    """Run one USER turn through the spine ``run_turn`` and return the reply text
    (non-streaming → the reply arrives as Text events, which we accumulate)."""
    from codeflow.spine import ChatType, Origin, Source, Text, TurnRequest

    parts: list[str] = []

    async def _collect(ev: object) -> None:
        if isinstance(ev, Text):
            parts.append(ev.content)

    await agent.run_turn(
        TurnRequest(
            origin=Origin.USER,
            source=Source(channel="benchmark", chat_id=chat_id, sender_id="user", chat_type=ChatType.DM),
            text=message,
            conversation=session_key,
        ),
        _collect,
        lambda: [],
        stream=False,
    )
    return "".join(parts)


async def execute_task(
    task: Task,
    workspace: Path,
    assets_dir: Path,
    model: str = DEFAULT_MODEL,
    api_key: str = DEFAULT_API_KEY,
    api_base: str = DEFAULT_API_BASE,
    provider_name: str = DEFAULT_PROVIDER,
    timeout_multiplier: float = 1.0,
    verbose: bool = False,
    routing_profile: str | None = None,
) -> Dict[str, Any]:
    """
    Execute a single benchmark task using CodeFlow's AgentLoop.

    When ``routing_profile`` is given (e.g. "eco"), a ModelRouter is
    attached to the AgentLoop so each turn selects the best model for
    that specific prompt.  Token usage and estimated cost are recorded
    via a lightweight provider wrapper — no changes to loop.py required.

    Returns a result dict compatible with PinchBench grading:
        task_id, status, transcript, workspace, execution_time, timed_out,
        usage, cost_usd, models_used
    """
    from codeflow.agent.loop import AgentLoop
    from codeflow.config.schema import ExecToolConfig
    from codeflow.session.manager import SessionManager

    # 准备工作区。
    task_workspace = prepare_workspace(task, workspace, assets_dir)

    # 创建带用量跟踪包装的供应商。
    raw_provider = _make_benchmark_provider(model, api_key, api_base, provider_name)
    tracked_provider = _UsageTrackingProvider(raw_provider)

    # 可选附加 EcoClaw 风格路由器，仅支持标准 OpenRouter 模型。
    router = None
    if routing_profile:
        if provider_name != "openrouter":
            raise ValueError("--routing-profile requires --provider openrouter and an OpenRouter API key")
        router = _StandardModelRouter(api_key=api_key, profile=routing_profile, fallback_model=model)
        await router.initialize()
        logger.info("ModelRouter enabled with profile=%s", routing_profile)

    session_mgr = SessionManager(task_workspace)
    session_key = f"bench:{task.task_id}"

    # 加载 skill_forge 配置，使基准运行遵守 injection_mode、inject_max、
    # mass_library_db 等设置。否则 AgentLoop 收到 ``skill_forge_config=None``，
    # SkillService 会忽略用户配置并回退到数据类默认值，如 injection_mode="summary"。
    from codeflow.config.codeflow import load_codeflow_config

    _ec_cfg = load_codeflow_config()
    skill_forge_cfg = getattr(_ec_cfg, "skill_forge", None)

    agent = AgentLoop(
        provider=tracked_provider,  # 包装后的供应商，对 AgentLoop 透明。
        workspace=task_workspace,
        model=model,
        max_iterations=40,
        context_window_tokens=65_536,
        exec_config=ExecToolConfig(),
        restrict_to_workspace=True,  # 为基准安全限制在工作区沙箱内。
        session_manager=session_mgr,
        router=router,
        skill_forge_config=skill_forge_cfg,
        runtime_config=getattr(_ec_cfg, "runtime", None),
        # 基准是非交互式批量运行，因此禁用 Bug2 的逐轮 shadow-git 检查点；既没有
        # 可注入恢复信息的渠道，也不希望任务工作区出现 ``.codeflow/shadow.git``。
        interactive=False,
    )

    timeout_seconds = task.timeout_seconds * timeout_multiplier
    start_time = time.time()
    status = "success"
    timed_out = False
    response = ""

    logger.info(
        "Executing task %s (%s) — timeout %.0fs%s",
        task.task_id,
        task.name,
        timeout_seconds,
        f" [routing={routing_profile}]" if routing_profile else "",
    )

    try:
        response = await asyncio.wait_for(
            _run_turn_text(
                agent,
                task.prompt,
                session_key=session_key,
                chat_id=task.task_id,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        timed_out = True
        status = "timeout"
        logger.warning("Task %s timed out after %.0fs", task.task_id, timeout_seconds)
    except Exception as exc:
        status = "error"
        logger.error("Task %s failed: %s", task.task_id, exc, exc_info=True)
    finally:
        try:
            await agent.close_mcp()
        except Exception:
            pass

    execution_time = time.time() - start_time

    # 从跟踪包装器收集用量与成本。
    usage = dict(tracked_provider.accumulated)
    models_used = list(tracked_provider.model_calls)
    # 按各模型实际 token 数量汇总全部模型成本。
    total_cost: float | None = None
    for m, m_usage in tracked_provider.per_model_usage.items():
        partial = _estimate_cost_usd(
            m,
            m_usage.get("prompt_tokens", 0),
            m_usage.get("completion_tokens", 0),
        )
        if partial is not None:
            total_cost = (total_cost or 0.0) + partial
    cost_usd = total_cost

    # 从会话中提取记录。
    session = session_mgr.get_or_create(session_key)
    raw_messages = list(session.messages)
    transcript = _session_to_openclaw_transcript(raw_messages)

    if verbose:
        logger.info("  Response: %s", (response[:500] + "...") if len(response) > 500 else response)
        logger.info("  Transcript entries: %d", len(transcript))
        logger.info("  Execution time: %.2fs", execution_time)
        logger.info(
            "  Tokens: prompt=%d  completion=%d  total=%d",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )
        if cost_usd is not None:
            logger.info("  Estimated cost: $%.6f", cost_usd)
        logger.info("  Models called: %s", models_used)
        if task_workspace.exists():
            logger.info("  Workspace files:")
            for f in sorted(task_workspace.rglob("*")):
                if f.is_file():
                    logger.info("    %s (%d bytes)", f.relative_to(task_workspace), f.stat().st_size)

    return {
        "task_id": task.task_id,
        "status": status,
        "transcript": transcript,
        "workspace": str(task_workspace),
        "execution_time": execution_time,
        "timed_out": timed_out,
        "response": response,
        "raw_messages": raw_messages,
        # 成本字段。
        "usage": usage,
        "cost_usd": cost_usd,
        "models_used": models_used,
    }
