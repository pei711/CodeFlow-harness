"""`LocalPool`：对 File-based Skills 执行 BM25 Keyword Retrieval。

``local`` Pool 覆盖 Disk 上全部 ``SKILL.md``：User-authored ``workspace/skills/``、Packaged Builtin（原始设计
为 9 个 Shipped Skills）、Configured External 与 Mirror Directories。池规模通常几十到几百且频繁编辑，
因此 In-memory BM25 合适：不加载 Embedding Model，Milliseconds 启动；Specific Tool Intent 如 ``pdf`` /
``weather`` 的 Keyword Match 往往优于 Dense Semantic；该规模下 Query Tokenize 成本低。

Index 在 ``__init__`` Eager Build，并由 Public ``rebuild_index()`` Refresh。Catalog 的
``invalidate_skill_cache`` 把 File-watcher/Evolver Invalidation 传入 BM25。历史 ``SkillService.select``
调用该检索层；Steady-state ``search`` 只需
Query Tokenize + 对 Precomputed ``doc_freqs`` 的 BM25 Scoring；Per-doc Tokenize/IDF 仅在文件变化时运行。

BM25/Tokenization 来自 :mod:`codeflow.utils.bm25`，无 ``rank_bm25`` / ``jieba`` / ``nltk`` 依赖且 CJK-aware，
与 Agent Tool Catalog 共享。Ranking 是关键词证据，不代表 Skill Workflow 可执行。
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from codeflow.memory_engine.skill_local.types import ScoredSkill, SkillMeta
from codeflow.utils.bm25 import BM25Okapi as _BM25Okapi
from codeflow.utils.bm25 import tokenize as _tokenize

if TYPE_CHECKING:
    from codeflow.memory_engine.skill_local.registry import SkillRegistry


def _format_skill_text(meta: SkillMeta, body_max: int = 4000) -> str:
    """构造送入 BM25 Index 的 One-line Representation。

    Name 与 Description 重复加权，Body 最多取 `body_max` Characters；这样 ``"weather"`` 即使 Body 主要
    谈 HTTP/Caching，也能命中 Weather Skill。函数只拼接检索文本，不修改 `SkillMeta`。
    """
    body = (meta.content or "")[:body_max]
    # 重复名称和描述，使它们在 BM25 词频中的权重高于较长正文。
    return f"{meta.name} {meta.name} {meta.description or ''} {body}"


class LocalPool:
    """围绕 File-based ``SkillRegistry`` 的 BM25 Retrieval Wrapper。

    实例持有 Current Registry Contents 的 Prebuilt ``_BM25Okapi``。历史 :class:`SkillService` 从
    ``invalidate_skill_cache`` 调用 :meth:`rebuild_index`，当前由 Catalog 承接，让 File-watcher 与 Evolver Writes 直接 Refresh，
    `search` 只做 Query-side Tokenize + BM25 Dot Product。

    Thread-safety：内部 :class:`threading.Lock` 保护 ``(metas, _BM25Okapi)`` Pair。Rebuild 在 Lock 外完成
    Expensive Tokenize/Construction，只在 Atomic Swap 时持锁；Search 仅捕获两个 References 后在锁外
    Score/Sort。In-flight Search 因而使用一致 Snapshot。
    """

    def __init__(self, registry: "SkillRegistry") -> None:
        self._registry = registry
        self._metas: list[SkillMeta] = []
        self._bm25: _BM25Okapi | None = None
        # 使用普通 Lock 而非 RLock，因为方法之间不会重入。
        self._lock = threading.Lock()
        # 急切执行首次构建，与服务其余部分一致：提前支付磁盘遍历成本，
        # 不让第一次用户查询承担。
        self.rebuild_index()

    def rebuild_index(self) -> None:
        """重新读取 Registry，并 In-place 替换 BM25 Index Snapshot。

        初始由 ``__init__`` 调用；历史入口是 :meth:`SkillService.invalidate_skill_cache`，当前每次 Watcher
        Event / Evolver Write 通过 Catalog Invalidation 调用。
        方法 Idempotent 且可 Concurrent Call：Last Writer Index Wins；In-flight Searches 保留此前捕获
        References，在 Consistent Snapshot 上完成。Registry Empty 时原子清空 Meta 与 Index。
        """
        metas = self._registry.list_all()
        if not metas:
            with self._lock:
                self._metas = []
                self._bm25 = None
            return
        tokenized_corpus = [_tokenize(_format_skill_text(m)) for m in metas]
        bm25 = _BM25Okapi(tokenized_corpus)
        # 防御性复制：注册表的 ``list_all`` 按引用返回缓存列表，后续重建会替换而非就地修改它。
        # 复制后即可解耦，保证向读取方提供的快照不会与配对的 BM25 索引分离。
        metas_snapshot = list(metas)
        with self._lock:
            self._metas = metas_snapshot
            self._bm25 = bm25

    def search(self, query: str, top_k: int = 50) -> list[ScoredSkill]:
        """在 Prebuilt Index 上返回 BM25 Top-K Matches。

        Empty Query/Index 返回空。只保留 Positive Score，并排除 ``always`` Skills，因为它们已通过
        ``# Active Skills`` 注入，避免同一 Body 重复。结果按 Score Descending，返回 Legacy
        `ScoredSkill(name, score, source)`，Body 由后续 `LocalSkillSource` 从 Registry O(1) 补齐。
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        with self._lock:
            bm25 = self._bm25
            metas = self._metas
        if bm25 is None or not metas:
            return []
        scores = bm25.get_scores(query_tokens)
        # 丢弃零分文档，按分数降序排列后取 top_k。排除 ``m.always`` Skill，因为它们已由
        # ``ContextBuilder`` 通过 ``get_always_skills()`` 注入 ``# Active Skills`` 块。如果仍在此处出现，
        # 同一正文会在系统提示词中重复：一次作为 Active，一次作为 top-K。Mass pool 项
        # 不会重复注入，因为 ``get_always_skills`` 只读取 ``_registry``，从不读取 ``_mass_registry``，
        # 所以该过滤仅对本地池有意义。
        ranked = sorted(
            ((s, m) for s, m in zip(scores, metas) if s > 0.0 and not m.always),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]
        return [ScoredSkill(name=m.name, score=float(s), source=m.source) for s, m in ranked]
