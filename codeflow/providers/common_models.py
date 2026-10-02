"""按 Provider Slug 维护人工 Curated ``common models`` Shortlist。

Provider ``/v1/models`` 只返回 Full Catalog，OpenRouter 单独就有 300+ Models，却没有
``popular``/``common`` Flag，所以可识别 Default Set 必须 Hand-maintained，不能可靠推导。

TUI ``/model`` Picker 先展示 User 在 ``config.providers.<slug>.models`` 配置的项，再补本 Shortlist；
User 也始终可用 ``model.add_model`` 手输任意 ID。因此这里覆盖 Common Case，不声称完整或实时
可用。Model ID 会随 Release Drift，需要按当前 Provider 更新；未列 Provider 回退其 Config List。
"""

from __future__ import annotations

COMMON_MODELS: dict[str, list[str]] = {
    "openrouter": [
        "anthropic/claude-opus-4.8",
        "anthropic/claude-opus-4.7",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-fable-5",
        "openai/gpt-5.5",
        "openai/gpt-5.4-mini",
        "google/gemini-3.5-flash",
        "google/gemini-3-flash-preview",
        "x-ai/grok-4.3",
        "meta-llama/llama-4-maverick",
        "mistralai/mistral-medium-3-5",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
        "xiaomi/mimo-v2.5",
        "minimax/minimax-m3",
        "z-ai/glm-5.2",
        "tencent/hy3",
        "moonshotai/kimi-k2.6",
        "qwen/qwen3.7-max",
    ],
    "openai": [
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
        "openai/gpt-5.3-codex",
        "openai/gpt-4.1",
        "openai/gpt-4o-mini",
    ],
    "anthropic": [
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-fable-5",
    ],
    "gemini": [
        "gemini/gemini-3.5-flash",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-3.1-pro-preview",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3-flash-preview",
    ],
    "groq": [
        "groq/openai/gpt-oss-120b",
        "groq/openai/gpt-oss-20b",
        "groq/llama-3.3-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/qwen/qwen3.6-27b",
    ],
    "deepseek": [
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    ],
    "zhipu": [
        "zai/glm-5.2",
        "zai/glm-5.1",
        "zai/glm-5",
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5-air",
        "zai/glm-4.5",
        "zai/glm-4.7-flash",
        "zai/glm-4.5-flash",
    ],
    "dashscope": [
        "dashscope/qwen-plus",
        "dashscope/qwen-max",
        "dashscope/qwen-flash",
        "dashscope/qwen-turbo",
        "dashscope/qwen3.5-plus",
        "dashscope/qwen3.6-plus",
        "dashscope/qwen3.7-max",
        "dashscope/qwq-plus",
        "dashscope/qwen3-coder-plus",
        "dashscope/qwen3-coder-flash",
        "dashscope/qwen3-vl-plus",
    ],
}


def common_models_for(slug: str) -> list[str]:
    """返回 ``slug`` 对应 Curated Common-model Shortlist 的新副本。

    未知 Slug 返回空 List；Caller 修改结果不会改变 Module-level COMMON_MODELS。函数不请求
    ``/v1/models``、不验证 Credential，也不合并 User Config，该顺序由 TUI Caller 所有。
    """
    return list(COMMON_MODELS.get(slug, []))
