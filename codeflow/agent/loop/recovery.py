"""为 Agent Loop 提供 Empty-response Recovery 的纯 Decision Logic。

本模块没有 I/O；Loop 拥有 Append Message、Counter Increment 与 ``continue`` 等 Side Effect，所以
分支可以独立 Unit Test。若 Turn 在无 Visible Text 时直接结束，弱模型会交付 canned "no response
to give" dud；这里在发送前用三种 Bounded Mode 恢复。

``PREFILL`` 处理 thinking-only：模型只有 Structured Reasoning 或 Inline <think>，回填自身推理
继续正文；``NUDGE`` 处理 post-tool empty：注入短 User Prompt 要求消费 Tool Result；``RETRY``
处理 plain empty：原样再请求。每种都有 Per-turn Budget，耗尽后 COMPLETE，绝不形成 Infinite
Loop。Synthetic Recovery Message 在持久化前由 AgentLoop 移除。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from codeflow.providers.base import LLMResponse

# 内容中的思考标记。部分模型网关将推理以 <think>…</think> 形式放入 ``content``，
# 而不是结构化 reasoning_content 字段，因此必须扫描内容；只检查结构化字段会漏掉它们。
_THINK_TAG_RE = re.compile(r"<think>|<thinking>|<reasoning>", re.IGNORECASE)

POST_TOOL_NUDGE = (
    "You executed tool calls but returned an empty response. Use the tool "
    "results above to continue the task, or give your final answer now."
)


class RecoveryAction(Enum):
    """枚举 Loop 面对 Empty Assistant Response 时下一步唯一动作。

    COMPLETE 表示已有 Visible Text、Recovery Disabled 或所有 Budget 耗尽，应结束 Turn；PREFILL
    回填 Reasoning，NUDGE 在 Tool Result 后插入推动消息，RETRY 不改 Message 再请求。枚举只表达
    Decision，具体 Side Effect 由 AgentLoop 执行。
    """

    COMPLETE = auto()  # 存在可见文本或预算耗尽 → 结束 Turn
    PREFILL = auto()  # 只有思考内容 → 回填推理后重试
    NUDGE = auto()  # 工具后空响应 → 注入空响应和用户提示后重试
    RETRY = auto()  # 普通空响应 → 原样重试


@dataclass(frozen=True)
class RecoveryLimits:
    """保存 Empty Recovery 的 Per-turn Enable Flag 与独立 Retry Budgets。

    默认允许 1 次 Post-tool Nudge、2 次 Thinking Prefill、3 次 Plain Empty Retry。每次新 Turn 在
    AgentLoop 内重置 Counter，不能放在 Long-lived Loop instance 上跨 Session 泄漏。Frozen Object
    防止分类过程中修改 Configuration。
    """

    enabled: bool = True
    post_tool_empty_max_nudges: int = 1
    thinking_prefill_max_retries: int = 2
    empty_content_max_retries: int = 3


def limits_from_defaults(defaults: object) -> RecoveryLimits:
    """从 Duck-typed ``agents.defaults`` Config 构建 `RecoveryLimits`。

    函数集中 Config→RecoveryLimits Mapping，避免多个 AgentLoop Construction Site 重复 Field
    Plumbing 或使用不同 Default。缺失属性回退 enabled=True、Nudge=1、Prefill=2、Retry=3；它不
    校验外部 Config Range，Schema Validation 由配置层负责。
    """
    return RecoveryLimits(
        enabled=getattr(defaults, "empty_recovery_enabled", True),
        post_tool_empty_max_nudges=getattr(defaults, "post_tool_empty_max_nudges", 1),
        thinking_prefill_max_retries=getattr(defaults, "thinking_prefill_max_retries", 2),
        empty_content_max_retries=getattr(defaults, "empty_content_max_retries", 3),
    )


def has_inline_thinking(content: str | None) -> bool:
    """判断 Raw Content 是否包含 ``<think>``/``<thinking>``/``<reasoning>`` Marker。

    匹配忽略大小写，只检测 Opening Marker；空 Content 返回 ``False``。它用于识别把 Reasoning
    塞进 Content 而非 Structured Field 的 Gateway，不负责删除 Tag 或验证 Closing Pair。
    """
    return bool(content) and bool(_THINK_TAG_RE.search(content))


def has_thinking(response: LLMResponse) -> bool:
    """判断 LLMResponse 是否以任一 Supported Form 产生了 Reasoning。

    Structured ``reasoning_content``、``thinking_blocks`` 或 Content Inline Marker 任一存在即返回
    ``True``。该结果只帮助区分 Thinking-only Empty，不表示推理有效，也不会把 Reasoning 暴露为
    User-visible Text。
    """
    return bool(response.reasoning_content or response.thinking_blocks or has_inline_thinking(response.content))


def classify_empty_response(
    response: LLMResponse,
    visible: str | None,
    *,
    prev_had_tool_calls: bool,
    nudges_done: int,
    prefill_retries: int,
    empty_retries: int,
    limits: RecoveryLimits,
) -> RecoveryAction:
    """决定如何处理 No-tool-call Assistant Response，并返回 `RecoveryAction`。

    ``visible`` 是从 ``response.content`` Strip <think> Blocks 后的 User-facing Text；非空或 Recovery
    Disabled 立即 COMPLETE。随后优先 PREFILL，使 Thinking-only Response 延续自身 Reasoning，
    不浪费 Post-tool Nudge；NUDGE 还带 ``not thinking`` Guard，二者互斥。

    Prefill Budget 耗尽后，仍含 Thinking 的模型可进入 Plain RETRY；这个 ``prefill_exhausted``
    条件避免始终填 Reasoning Field 的 Provider 被永久阻止重试。各 Counter 与 Limit 比较后都
    不可用则 COMPLETE。函数不改变 Response、Message 或 Counter。
    """
    if visible or not limits.enabled:
        return RecoveryAction.COMPLETE

    thinking = has_thinking(response)

    # 只有思考的预填充：模型已推理，但没有生成正文。
    if thinking and prefill_retries < limits.thinking_prefill_max_retries:
        return RecoveryAction.PREFILL

    # 工具后空响应提示：排除只有思考的情况，因为上方已处理。
    if prev_had_tool_calls and not thinking and nudges_done < limits.post_tool_empty_max_nudges:
        return RecoveryAction.NUDGE

    # 兜底的普通重试。``prefill_exhausted`` 条件至关重要：部分模型始终填充推理字段，
    # 如果只在 ``not thinking`` 时允许重试，预填充用尽后所有推理模型都会被永久阻止重试。
    prefill_exhausted = thinking and prefill_retries >= limits.thinking_prefill_max_retries
    if empty_retries < limits.empty_content_max_retries and (not thinking or prefill_exhausted):
        return RecoveryAction.RETRY

    return RecoveryAction.COMPLETE
