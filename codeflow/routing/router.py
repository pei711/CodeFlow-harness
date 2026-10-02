"""CodeFlow 的 EcoClaw-style Model Router，对外串联分类、评测数据与模型选择。

典型 Usage 是先创建实例并预热，再按用户消息取得模型：

    router = ModelRouter(api_key="sk-or-...", profile="balanced")
    await router.initialize()
    model_id = await router.select_model_id(user_message)

路由器只给出本轮应该优先尝试的 Model ID 与 Fallback Chain，不直接发送 LLM 请求。任何分类、数据
加载或选择失败都会返回 `None`，让 Agent 保留已配置的默认模型，因此 Routing 是可降级增强而不是
Turn 能否运行的单点依赖。
"""

from __future__ import annotations

from loguru import logger

from codeflow.routing.cache import BenchmarkCache
from codeflow.routing.classifier import PromptClassifier
from codeflow.routing.fetcher import BenchmarkData
from codeflow.routing.selector import select_model
from codeflow.routing.types import RoutingProfileName, SelectionResult


class ModelRouter:
    """依据 PinchBench Benchmark Data，把 User Messages 路由到 Best-value Model。

    每条消息的 Flow 分三步：

    1. `classify(prompt)` 用 Embedding Cosine Similarity 得到 `TaskCategory`；
    2. `select_model(data, category, profile)` 用 Composite Score 得到 `ModelScore` 排名；
    3. 返回 Primary Model ID，格式采用 OpenRouter 的 ``provider/model``。

    实例持有 Benchmark Cache、Prompt Classifier 与当前 Routing Profile。`initialize` 可以在进程启动
    时预加载数据，未预热也可在第一次 `route` 时惰性加载。它不判断模型调用结果是否完成用户任务，
    只负责在请求发出前提供候选顺序。
    """

    def __init__(
        self,
        api_key: str,
        profile: RoutingProfileName = "balanced",
        fallback_model: str | None = None,
    ):
        self._api_key = api_key
        self._profile = profile
        self._fallback_model = fallback_model
        self._cache = BenchmarkCache()
        self._classifier = PromptClassifier(api_key=api_key)
        self._data: BenchmarkData | None = None

    async def initialize(self) -> None:
        """预加载 Benchmark Data，通常在 Startup 时调用一次。

        方法通过 `BenchmarkCache.load` 建立当前数据 Snapshot 并记录模型数量与 Profile。缓存内部会
        处理在线刷新和降级，因此成功返回的数据仍可能来自磁盘旧缓存或随包快照；重复调用是安全的，
        但一般没有必要用它强制刷新。
        """
        self._data = await self._cache.load()
        logger.info(
            "ModelRouter initialized: {} models, profile={}",
            len(self._data),
            self._profile,
        )

    async def route(self, prompt: str) -> SelectionResult | None:
        """分类 Prompt 并选择 Models；任一阶段失败时返回 `None`。

        若尚未初始化，先惰性加载 Benchmark Data；随后取得任务类别，再按当前 Profile 对模型排序并
        返回完整 `SelectionResult`。数据加载、Classification 或 Model Selection 的异常都被视为可
        降级问题，记录日志后交由上层使用 Default Model，而不会让 Agent Turn 因路由失败而终止。
        """
        if self._data is None:
            try:
                self._data = await self._cache.load()
            except Exception:
                return None

        try:
            classification = await self._classifier.classify(prompt)
        except Exception as e:
            logger.warning("Classification failed: {}", e)
            return None

        try:
            result = select_model(self._data, classification.category, self._profile)
            logger.info(
                "Routed to {} (category={}, score={:.3f})",
                result.primary.model,
                result.primary.task_score,
                result.primary.composite_score,
            )
            return result
        except Exception as e:
            logger.warning("Model selection failed: {}", e)
            return None

    async def select_model_id(self, prompt: str) -> str | None:
        """返回 Prompt 对应的 Best Model ID，或用 `None` 指示使用 Default。

        这是只关心 Primary Model 的便捷接口：内部仍执行完整 `route`，但丢弃 Runner-up 信息。
        `None` 的准确含义是使用已配置的 Default Model、不要应用 Routing Override；它不区分数据
        加载失败、分类失败或没有可选结果，诊断原因需要查看日志或调用 `route`。
        """
        result = await self.route(prompt)
        if result is None:
            return None
        return result.primary.model

    async def select_model_chain(self, prompt: str) -> tuple[str | None, list[str]]:
        """返回 Prompt 的 ``(primary_id, [fallback_ids])`` Model Chain。

        Routing 没有结果时，``primary`` 为 `None` 且 Chain 为空，调用方应使用其 Configured Default
        Model。成功时，Fallbacks 先采用 Selector 排出的 Runner-up Models；若配置了
        ``fallback_model``，且它既不是 Primary 也尚未出现，则追加到末尾作为 Last Resort。这个顺序
        表达尝试优先级，不会在本方法内实际调用或探活任何模型。
        """
        result = await self.route(prompt)
        if result is None:
            return None, []
        fallbacks = [f.model for f in result.fallbacks]
        if (
            self._fallback_model
            and self._fallback_model != result.primary.model
            and self._fallback_model not in fallbacks
        ):
            fallbacks.append(self._fallback_model)
        return result.primary.model, fallbacks

    @property
    def profile(self) -> RoutingProfileName:
        return self._profile

    @profile.setter
    def profile(self, value: RoutingProfileName) -> None:
        self._profile = value
        logger.info("ModelRouter profile changed to '{}'", value)
