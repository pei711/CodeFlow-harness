"""CodeFlow 的 EcoClaw-style 模型路由入口。

模型路由解决的是“这一轮请求应该交给哪个模型”这一问题：调用方只需导入这里公开的
`ModelRouter`，不必了解候选筛选、任务分类、价格约束或缓存分别由哪个子模块完成。这里沿用
EcoClaw-style 这一设计来源名称，表示路由器会结合任务特征与模型能力作出选择；它不是模型
Provider，也不直接执行 LLM 请求。
"""

from codeflow.routing.router import ModelRouter

__all__ = ["ModelRouter"]
