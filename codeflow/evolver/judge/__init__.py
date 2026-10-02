"""Evolver 使用的 LLM-judge 子系统。

Judge 读取 compressed trajectory，按 spec §3 把 failure 分为 L1/L2/L3，并为 L2/L3 生成
structured ``(WHERE, WHY)`` patch proposal（spec §12.4-§12.5）。``IssueType``、
``PatchWhere``、``PatchWhy``、``ActionKind`` 定义 taxonomy；``JudgeAction``/
``JudgeResult`` 是 validated dataclass；``build_judge_messages`` 组装一次 LLM call 的 system/
user message；``parse_judge_output`` 把 raw text 转为 typed result，malformed output 抛出
``JudgeParseError``。LLM client 位于 ``codeflow.evolver.judge.llm_client``（B3）。

Judge output 是 candidate design 输入，不是 ground truth。parse 成功只证明 schema 合规；L1
仍需 human/infrastructure review，L2/L3 patch 仍必须经过 manifest、activation、benchmark gate
与 sealed evidence，才能产生正向结论。
"""

from .llm_client import (
    JudgeLLM,
    JudgeLLMBackend,
    JudgeLLMConfig,
    LitellmBackend,
    MockBackend,
    Mode,
    OpenRouterBackend,
    TrajectoryFormat,
    build_backend,
    build_judge_llm,
)
from .parser import JudgeParseError, parse_judge_output
from .prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    WHERE_DESCRIPTIONS,
    WHY_DESCRIPTIONS,
    build_judge_messages,
)
from .schema import (
    ActionKind,
    IssueType,
    JudgeAction,
    JudgeResult,
    PatchWhere,
    PatchWhy,
    ProposedComponent,
)

__all__ = [
    # 枚举
    "ActionKind",
    "IssueType",
    "PatchWhere",
    "PatchWhy",
    # 数据类
    "JudgeAction",
    "JudgeResult",
    "ProposedComponent",
    # 提示构建
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_USER_TEMPLATE",
    "WHERE_DESCRIPTIONS",
    "WHY_DESCRIPTIONS",
    "build_judge_messages",
    # 解析
    "JudgeParseError",
    "parse_judge_output",
    # LLM 客户端（B3）
    "JudgeLLM",
    "JudgeLLMBackend",
    "JudgeLLMConfig",
    "LitellmBackend",
    "MockBackend",
    "Mode",
    "OpenRouterBackend",
    "TrajectoryFormat",
    "build_backend",
    "build_judge_llm",
]
