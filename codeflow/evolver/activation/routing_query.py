"""离线探测真实 Skill discovery 与 selection 路径的 ``dry_query``。

Round-4 forensics 发现，写在 ``skill_library/tb2_gap_fill/`` 的 custom Skill 从未注入：
benchmark Agent 构造 ``SkillForgeConfig(enabled=False)`` 且 ``local_dirs`` 为空，目录未挂载
为 discovery layer，``select()`` 在 retrieval 前就返回 ``[]``。``dry_query`` 不依赖 LLM 或
SR server，通过真实 :class:`LocalSkillCatalog` 与 :class:`SkillForgeRouter`，把
``library_root`` 放入 ``local_dirs``，运行实际 BM25 retrieval + resolve path，回答“这个 task
会注入哪些 Skill name”。

LLM gate 与 query rewriter 被禁用，所以 selection 退化为 filesystem discovery + lexical
BM25 scoring，即 benchmark 必须接线的 deterministic core。query 命中只证明离线路由可发现
Skill，不证明 benchmark live Runtime 已挂载同一目录、Skill 已注入或任务效果改善。
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

__all__ = ["dry_query"]


def dry_query(task_text: str, *, library_root: Path | None = None) -> list[str]:
    """返回 routing 会为 ``task_text`` 注入的 Skill name 列表。

    ``task_text`` 是交给 Skill selection 的任务描述。``library_root`` 非空时作为额外
    discovery layer；recursive ``SKILL.md`` scan 会遍历子树，因此 ``.../skill_library`` 能
    暴露 ``tb2_gap_fill/<skill>/SKILL.md``。``None`` 只测试 workspace + builtin 默认层。

    返回按 injection order 排列的 flat ``SkillMeta.name`` list。顺序镜像 benchmark 在
    ``ContextBuilder.build_system_prompt`` 的两类注入面：先是 ``always: true``、渲染在
    ``Active Skills`` 下的 Skill，再是 retrieval ``select()`` hit，最后按 name 去重。

    函数创建 temporary workspace 且关闭 watcher，不持久化 catalog；发现结果是当前文件与
    BM25 规则的 snapshot。
    """
    from codeflow.config.codeflow import LocalDirConfig, SkillForgeConfig

    # CodeFlow 将旧统一 SkillService 拆成发现目录（常驻技能加注册表/池）和跨来源检索路由器。
    from codeflow.memory_engine.skill_forge.catalog import LocalSkillCatalog
    from codeflow.memory_engine.skill_forge.local_source import LocalSkillSource
    from codeflow.memory_engine.skill_forge.router import SkillForgeRouter

    local_dirs: list[LocalDirConfig] = []
    if library_root is not None:
        local_dirs.append(LocalDirConfig(path=str(Path(library_root)), name="tb2_gap_fill"))

    config = SkillForgeConfig(
        enabled=True,
        local_dirs=local_dirs,
        llm_gate_enabled=False,
        rewrite_enabled=False,
        reranker_enabled=False,
        disable_always=False,
    )

    with tempfile.TemporaryDirectory() as ws:
        catalog = LocalSkillCatalog(
            Path(ws),
            config=config,
            llm_provider=None,
            start_watcher=False,
        )
        always = catalog.get_always_skills()
        router = SkillForgeRouter([LocalSkillSource(catalog.pool, catalog.registry)])
        selected = asyncio.run(router.select(task_text, []))

    names: list[str] = []
    seen: set[str] = set()
    for meta in [*always, *selected]:
        if meta.name in seen:
            continue
        seen.add(meta.name)
        names.append(meta.name)
    return names
