"""面向单个 Task 的 KNN Model Router。

每个新任务到达时，路由器先生成 Embedding，再从预构建 Memory 中检索 K 个最近的训练任务。它对
这些 Neighbours 上每个模型的期望价值取平均，价值公式为
``reward - lambda_cost * cost``，最后选择得分最高的模型。模块公开与 EcoClaw `ModelRouter` 相同的
`select_model_chain` Interface，因此 Agent Loop 可以互换两种路由器而不改变调用协议。

Memory Schema 是 JSON List，每个训练任务一条记录：

    {"task_name": str,
     "text": str,               # 任务描述；缺少 "embedding" 时在加载阶段生成，
                                 # 这样既缩小文件，也让训练任务与在线查询使用一致向量模型
     "embedding": [float, ...], # 可选预计算向量；存在时原样使用
     "rewards": {model_name: float, ...},
     "costs":   {model_name: float, ...}}

Routing Candidates 是配置模型与 Memory 中出现模型的 Intersection。缺少 Memory、Embedding Error、
No Candidates 或证据门禁不足都会返回 ``(None, [])``，让 Caller 回退到 Default Model。这个返回值
表示“不建议覆盖默认选择”，不是一次失败的模型调用。
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import httpx
import numpy as np
from loguru import logger

from codeflow.product import get_product_home


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, 1e-8)


class KNNModelRouter:
    """通过 Per-model Rewards 上的 KNN，把每个任务路由给 Best-value Model。

    初始化时读取路由参数与训练 Memory，归一化任务向量，并只保留配置和 Memory 共同支持的模型。
    运行期 `select_model_chain` 为 Prompt 生成向量、检索邻居并应用相似度、样本数与 Margin 门禁。实例
    不执行最终 LLM Call，也不修改 Agent 的默认模型；证据不足时始终把决定权交回调用方。

    Memory Embedding 可以预先提供，也可以通过配置的 Endpoint 在加载时补齐并缓存。缓存只用于减少
    重复计算，写入失败不会改变路由正确性，只会使下次加载重新生成向量。
    """

    def __init__(self, routing_cfg, default_model: str | None = None):
        self._k = max(1, int(routing_cfg.k))
        self._lambda = float(routing_cfg.lambda_cost)
        self._embed_url = routing_cfg.embedding_endpoint
        self._config_models = [m.model for m in routing_cfg.models if m.model]
        # Agent 配置的默认模型是安全基线，路由器仅在证据充分时才离开它。
        # None 会使“已经是默认模型”和 margin 门禁失效；返回 None 时调用方
        # 仍会回退到自己的模型。
        self._default_model = default_model
        self._min_similarity = float(getattr(routing_cfg, "min_similarity", 0.6))
        self._min_similar = max(1, int(getattr(routing_cfg, "min_similar_neighbors", 4)))
        self._min_memory_size = max(1, int(getattr(routing_cfg, "min_memory_size", 10)))
        self._min_margin = float(getattr(routing_cfg, "min_margin", 0.0))

        self._embeddings = np.empty((0, 0))
        self._rewards: list[dict[str, float]] = []
        self._costs: list[dict[str, float]] = []
        self._candidates: list[str] = []
        self._load_memory(routing_cfg.memory_path)

    def _load_memory(self, path: str) -> None:
        if not path:
            logger.warning("KNNModelRouter: no memory_path configured; routing disabled")
            return
        try:
            entries = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("KNNModelRouter: failed to load memory {}: {}", path, e)
            return
        if not entries:
            logger.warning("KNNModelRouter: empty memory at {}", path)
            return

        self._rewards = [e.get("rewards", {}) for e in entries]
        self._costs = [e.get("costs", {}) for e in entries]

        mat = self._resolve_embeddings(entries, path)
        if mat is None:
            return
        self._embeddings = _normalize(mat)

        mem_models = {m for r in self._rewards for m in r}
        self._candidates = [m for m in self._config_models if m in mem_models]
        missing_reward = [m for m in self._config_models if m not in mem_models]
        if missing_reward:
            logger.warning("KNNModelRouter: configured models absent from memory (skipped): {}", missing_reward)
        logger.info(
            "KNNModelRouter: loaded {} tasks, candidates={}, k={}, lambda={}",
            len(entries),
            self._candidates,
            self._k,
            self._lambda,
        )

    def _resolve_embeddings(self, entries: list[dict], path: str) -> "np.ndarray | None":
        """优先使用 Precomputed ``embedding``，否则逐条生成每个 Entry 的 ``text`` 向量。

        所有 Entry 都携带向量时直接构造 `np.ndarray`。只要存在缺失，就从 ``text`` 取任务描述，缺失文本
        时退回 ``task_name``，并通过配置的 Endpoint 以 batch=1 请求；逐条处理是为了与 Live Queries 的
        Embedding 方式一致。生成结果缓存在由 Memory Path 与 Endpoint 共同派生的用户目录文件中，而
        不是写到可能只读的 Memory File 旁边。

        未配置 Endpoint 或任一网络/响应解析过程失败时返回 `None`，调用方据此禁用本次路由。返回数组
        仅说明每条任务取得了数值向量，归一化由 `_load_memory` 随后完成。
        """
        if all("embedding" in e for e in entries):
            return np.array([e["embedding"] for e in entries], dtype=np.float32)

        texts = [e.get("text") or e.get("task_name", "") for e in entries]
        if not self._embed_url:
            logger.warning("KNNModelRouter: memory has no embeddings and no embedding_endpoint; routing disabled")
            return None

        cache = self._read_emb_cache(path)
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        try:
            for t in missing:
                payload = json.dumps({"texts": [t]}).encode()
                req = urllib.request.Request(
                    self._embed_url, data=payload, headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    cache[t] = json.loads(resp.read())["embeddings"][0]
        except Exception as e:
            logger.warning("KNNModelRouter: failed to embed memory texts: {}", e)
            return None
        if missing:
            self._write_emb_cache(path, cache)
            logger.info("KNNModelRouter: embedded {} memory texts at load", len(missing))
        return np.array([cache[t] for t in texts], dtype=np.float32)

    def _emb_cache_path(self, path: str) -> Path:
        # 缓存在用户目录中，而不是可能只读的仓库记忆文件旁。
        # 以 endpoint + memory path 为键，让不同嵌入器或记忆文件使用独立缓存。
        key = hashlib.sha1(f"{self._embed_url}|{Path(path).resolve()}".encode()).hexdigest()[:16]
        d = get_product_home() / "knn_embcache"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d / f"{key}.json"

    def _read_emb_cache(self, path: str) -> dict:
        try:
            return json.loads(self._emb_cache_path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_emb_cache(self, path: str, cache: dict) -> None:
        try:
            self._emb_cache_path(path).write_text(json.dumps(cache), encoding="utf-8")
        except Exception:
            pass  # 只读位置：跳过缓存，下次加载时重新嵌入

    async def _embed(self, prompt: str) -> np.ndarray | None:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(self._embed_url, json={"texts": [prompt]})
                resp.raise_for_status()
                vec = np.array(resp.json()["embeddings"][0], dtype=np.float32)
            return vec / max(float(np.linalg.norm(vec)), 1e-8)
        except Exception as e:
            logger.warning("KNNModelRouter: embedding failed: {}", e)
            return None

    async def select_model_chain(self, prompt: str) -> tuple[str | None, list[str]]:
        """返回 ``(primary_model, [fallback_models])``；``(None, [])`` 表示使用 Default。

        方法先执行冷启动结构门禁：候选模型少于两个或 Memory Size 不足时不参与决策。随后生成 Prompt
        Embedding，取 K 个最近邻，只保留 Cosine Similarity 达到阈值的样本，并要求 Similar Neighbours
        数量足够。每个候选模型只在确实含有其 Reward 的同一批邻居上计算平均 Reward 与 Cost，避免缺失值
        被错误地当作零分。

        排名第一的模型若就是 `_default_model`，或相对默认模型的得分优势未达到 `min_margin`，仍返回
        ``(None, [])``；真正切换时才返回 Primary 与其余 Ranked Fallbacks。Embedding、矩阵维度或数据
        计算发生任何异常也会降级而不让 Turn 崩溃，因此成功选出模型只代表历史证据通过门禁，不保证本轮
        任务最终完成。
        """
        # 冷启动/结构门禁：候选或记忆过少时无法可靠决策，保留调用方默认模型。
        if len(self._candidates) < 2 or self._embeddings.shape[0] < self._min_memory_size:
            return None, []

        q = await self._embed(prompt)
        if q is None:
            return None, []

        # 路由只是可选增强，KNN 或数据错误不得让 Turn 崩溃；
        # 任何失败都降级到调用方默认模型。
        try:
            sims = self._embeddings @ q
            top = np.argsort(-sims)[: self._k]

            # 相似样本门禁：只有足够多的检索邻居确实相似
            # （cosine >= min_similarity）时才信任所选模型。分布外查询
            # （如闲聊）的相似邻居很少，因此保留默认模型。评分只使用这些
            # 相似邻居，避免无关任务稀释奖励。
            similar = [int(i) for i in top if float(sims[i]) >= self._min_similarity]
            if len(similar) < self._min_similar:
                return None, []

            scores: dict[str, float] = {}
            for m in self._candidates:
                # 在同一批邻居（包含该模型奖励的邻居）上计算平均奖励和成本，
                # 避免缺少该模型数据的邻居用零值稀释成本。
                pairs = [(self._rewards[i][m], self._costs[i].get(m, 0.0)) for i in similar if m in self._rewards[i]]
                if not pairs:
                    continue
                reward = float(np.mean([r for r, _ in pairs]))
                cost = float(np.mean([c for _, c in pairs]))
                scores[m] = reward - self._lambda * cost

            if not scores:
                return None, []

            ranked = sorted(scores, key=lambda m: scores[m], reverse=True)
            primary = ranked[0]

            # 已经选择默认模型，无需切换。
            if primary == self._default_model:
                return None, []

            # margin 门禁：只有所选模型明确胜出时才离开默认模型。
            baseline = scores.get(self._default_model)
            if baseline is not None and scores[primary] - baseline < self._min_margin:
                return None, []

            fallbacks = ranked[1:]
            logger.info("KNNModelRouter: routed to {} (scores={})", primary, scores)
            return primary, fallbacks
        except Exception as e:
            logger.warning("KNNModelRouter: routing failed, using default: {}", e)
            return None, []
