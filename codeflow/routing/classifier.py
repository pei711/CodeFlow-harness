"""基于 Embedding 的 Prompt Classifier，对应 EcoClaw 的 `classifier.ts`。

分类器先把用户 Prompt 转成向量，再与预计算的任务向量做 Nearest-neighbour Search。距离最近的任务
决定本轮属于哪个能力类别，路由器随后可使用该类别对应的 Benchmark 分数挑选模型。这里定义的 23
Categories 与 PinchBench Task IDs 1:1 映射；这个映射是评测类别与运行时分类之间的契约，不能只改
其中一侧。

Classification 只描述 Prompt 与已有任务样本的语义相似度。它不验证用户请求最终能否完成，也不
代表模型已经调用；缺少 Embedding Data 或在线 API 失败时会保守回退到 `sanity`。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
from loguru import logger

from codeflow.routing.types import ClassificationResult, TaskCategory

# ── 23 个类别映射到任务 ID ─────────────────────────────────────────────

TASK_CATEGORIES: dict[TaskCategory, str] = {
    "sanity": "task_00_sanity",
    "calendar": "task_01_calendar",
    "stock": "task_02_stock",
    "blog": "task_03_blog",
    "tool_use": "task_04_weather",
    "summary": "task_05_summary",
    "events": "task_06_events",
    "email": "task_07_email",
    "memory": "task_08_memory",
    "files": "task_09_files",
    "workflow": "task_10_workflow",
    "clawdhub": "task_11_clawdhub",
    "skill_search": "task_12_skill_search",
    "image_gen": "task_13_image_gen",
    "humanizer": "task_14_humanizer",
    "daily_summary": "task_15_daily_summary",
    "email_triage": "task_16_email_triage",
    "email_search": "task_17_email_search",
    "market_research": "task_18_market_research",
    "spreadsheet": "task_19_spreadsheet_summary",
    "eli5_pdf": "task_20_eli5_pdf_summary",
    "comprehension": "task_21_openclaw_comprehension",
    "second_brain": "task_22_second_brain",
}

_TASK_TO_CATEGORY: dict[str, TaskCategory] = {v: k for k, v in TASK_CATEGORIES.items()}

_EMBEDDING_DATA_PATH = Path(__file__).parent / "embedding_data.json"

# ── 嵌入 API ───────────────────────────────────────────────────────────

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "text-embedding-3-small"


async def fetch_embedding(
    text: str,
    api_key: str,
    model: str = EMBEDDING_MODEL,
) -> list[float]:
    """调用 OpenRouter `/embeddings`，返回输入文本的浮点向量。

    `text` 是待分类的用户 Prompt，`api_key` 用于 Bearer 鉴权，`model` 默认采用
    `text-embedding-3-small`。HTTP 状态异常由 `raise_for_status` 原样抛给调用方，响应结构缺失也会触发
    解析异常；本函数不做回退，因为 `PromptClassifier.classify` 统一负责把这些失败降级为 `sanity`。
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OPENROUTER_API_BASE}/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"input": text, "model": model},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的 Cosine Similarity，并通过 Truncating 处理维度不一致。

    只比较 `a` 与 `b` 的共同前缀长度，避免 `zip` 因维度差异产生不可见的偏差；任一向量在该范围内
    范数为零时返回 `0.0`，而不是除零。返回值越大表示方向越接近，但这里只是数学相似度，不自动判断
    分类置信度是否足够，也不会拒绝低相似度结果。
    """
    length = min(len(a), len(b))
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a[:length]))
    norm_b = math.sqrt(sum(x * x for x in b[:length]))
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0


# ── 预计算嵌入 ─────────────────────────────────────────────────────────


def _load_embedding_data() -> list[dict] | None:
    """从 `embedding_data.json` 加载 Pre-computed Task Embeddings。

    文件不存在、JSON 无法解析或读取失败时返回 `None`，让上层进入明确的 Fallback；合法文件则返回
    `tasks` 数组，缺少该字段时返回空列表。这里不校验每个任务的向量内容，单条空向量会在分类循环中
    跳过，因而“文件加载成功”不等于“每个类别都有可用样本”。
    """
    if not _EMBEDDING_DATA_PATH.exists():
        return None
    try:
        raw = json.loads(_EMBEDDING_DATA_PATH.read_text(encoding="utf-8"))
        return raw.get("tasks", [])
    except Exception as e:
        logger.warning("Failed to load embedding_data.json: {}", e)
        return None


# ── 分类器 ─────────────────────────────────────────────────────────────


class PromptClassifier:
    """把一个 Prompt 归入 23 个 PinchBench Task Categories 之一。

    实例保存 OpenRouter API Key，并在第一次分类时惰性加载本地预计算向量。生命周期中会复用同一份
    任务数据，避免每轮重复读盘；若首次加载得到 `None`，后续调用仍会再次尝试加载。它只负责语义
    类别判断，与模型价格、可用 Provider 和最终 Model Selection 无关，这些职责属于后续路由模块。
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._tasks: list[dict] | None = None  # 惰性加载

    def _get_tasks(self) -> list[dict] | None:
        if self._tasks is None:
            self._tasks = _load_embedding_data()
        return self._tasks

    async def classify(self, prompt: str) -> ClassificationResult:
        """返回 Prompt 的 Best-matching Task Category，任何失败都回退到 `sanity`。

        方法先取得预计算 Tasks，再在线获取 `prompt` 的 Embedding，对每个非空任务向量计算 Cosine
        Similarity，并保留最大值对应的 Task ID。返回的 `ClassificationResult` 同时携带 Category 与原始
        Similarity，便于上层记录路由依据。

        缺少本地数据或 Embedding API 调用失败时返回 `category="sanity"`、`similarity=0.0`；未知 Task ID
        同样映射到 `sanity`。这个 Fallback 保证路由链仍可继续，但不是对 Prompt 确实属于基础任务的判断。
        """
        tasks = self._get_tasks()
        if not tasks:
            logger.warning(
                "Embedding data missing — run `python -m codeflow.routing.generate_embeddings`. Falling back to 'sanity'."
            )
            return ClassificationResult(category="sanity", similarity=0.0)

        try:
            query_vec = await fetch_embedding(prompt, self._api_key)
        except Exception as e:
            logger.warning("Embedding API failed ({}), falling back to 'sanity'", e)
            return ClassificationResult(category="sanity", similarity=0.0)

        best_category: TaskCategory = "sanity"
        best_sim = float("-inf")

        for task in tasks:
            task_vec: list[float] = task.get("embedding", [])
            if not task_vec:
                continue
            sim = cosine_similarity(query_vec, task_vec)
            if sim > best_sim:
                best_sim = sim
                task_id: str = task.get("task_id", "")
                best_category = _TASK_TO_CATEGORY.get(task_id, "sanity")

        logger.debug("Classified prompt → {} (similarity={:.3f})", best_category, best_sim)
        return ClassificationResult(category=best_category, similarity=best_sim)
