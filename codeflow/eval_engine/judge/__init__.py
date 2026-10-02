"""Eval Engine 的 LLM Judge Invocation 入口。

公开 `EvalJudge` 与离散的 `JudgeVerdict`。Judge 根据 User Goal 与 Final Response 生成评估意见；它是
辅助认知判断，不拥有 Runtime Outcome，也不能验证外部系统中的真实副作用。
"""

from codeflow.eval_engine.judge.judge import EvalJudge, JudgeVerdict

__all__ = ["EvalJudge", "JudgeVerdict"]
