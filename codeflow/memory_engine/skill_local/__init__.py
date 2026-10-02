"""SkillForge 的 Local-pool Storage、Retrieval 与 Shared Types Primitives。

Phase B 已删除 ``Retrieval``（Local PyTorch Matmul）、``Reranker``、``SqliteStore``、
``SqliteSkillRegistry``、Sync 与 ``skill_library/`` Offline Tooling。Mass-library Retrieval 曾设计为
:mod:`codeflow.memory_engine.skill_forge` 下的 :class:`MassSkillSource` HTTP Client；``SkillService`` Aggregate 及其 ``select`` /
LLM-gate / Query-rewriter Path 在 :class:`SkillForgeRouter` 成为 Live Path 后退役进
:class:`LocalSkillCatalog`。No-op ``SkillEvolver`` Seam 也在 Feedback 移到 :class:`MemoryBackend` Plugin
的 ``backend.feedback`` / ``backend.store`` 后删除。

现在只保留 LOCAL Primitive Layer：:class:`SkillRegistry` 扫描 Workspace + Builtin ``SKILL.md``，
:class:`LocalPool` 在 Registry 上做 BM25，:class:`SkillMeta` / :class:`ScoredSkill` 传递本地元数据与分数。
"""

from codeflow.memory_engine.skill_local.local_pool import LocalPool
from codeflow.memory_engine.skill_local.registry import SkillRegistry
from codeflow.memory_engine.skill_local.types import ScoredSkill, SkillMeta

__all__ = [
    # 数据层
    "SkillRegistry",
    "LocalPool",
    # 共享类型
    "SkillMeta",
    "ScoredSkill",
]
