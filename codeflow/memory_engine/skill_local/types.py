"""SkillForge Local Layer 使用的 Shared Dataclasses。

`SkillMeta` 保存 Registry 从 ``SKILL.md`` 与 Frontmatter 解析的完整本地元数据；`ScoredSkill` 是 BM25
Hot Path 的轻量 Hit，只携带 Name、Score、Source。Router Layer 随后把两者合成 Self-contained
`RouterHit`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class SkillMeta:
    """一条 Skill 的完整 Metadata 与已加载 Body。

    历史接口可通过 ``SkillRegistry.get_body()`` 另读 Full File；当前 `content` 已保存移除 Frontmatter 的
    Body，供 Local Source O(1) Hydrate。对象还记录 Physical Source、Always/Requires 与可选 Provenance。
    """

    id: str
    """Unique Skill Identifier，通常包含 Source 与 Stable Directory Key。"""

    name: str
    """Skill Display Name，优先来自 Frontmatter，缺失时使用 Directory Name。"""

    description: str
    """展示给 LLM/Gate 的 One-line Description；缺失时可回退到 Name。"""

    path: Path
    """``SKILL.md`` 的 Absolute Path，也是 Relative Ref Resolution 的根依据。"""

    content: str
    """``SKILL.md`` Body Content，Frontmatter 已移除；是否注入由 Router/Resolver 决定。"""

    source: str
    """Physical Origin，例如 ``workspace``、``builtin`` 或 ``mirror/*``。"""

    always: bool = False
    """是否要求每 Turn Force-inject 到 System Prompt；仍受 Availability 与 Catalog Cap 限制。"""

    requires: dict = field(default_factory=dict)
    """Dependency Declarations：``{"bins": [...], "env": [...]}``；仅做 Binary/Env Presence Check。"""

    # ---- 后续字段（由摄取过程填充，当前为 None 或空）----

    scope: str | None = None
    """Owning Pool：Personal / Team / Official / Community / Mass；当前文件扫描通常未填充。"""

    license: str | None = None
    """SPDX License，例如 MIT / Apache-2.0；`None` 表示未声明而非默认许可。"""

    imported_at: datetime | None = None
    """Skill 从 External Source Pulled In 的时间；Local Authoring 可为 `None`。"""

    raw_frontmatter: dict = field(default_factory=dict)
    """Full Original Frontmatter，保留给 Downstream Consumers；不代表所有键已验证。"""


@dataclass
class ScoredSkill:
    """一条轻量 Retrieval Hit，包含 Name + Score + Source。

    LocalPool 返回该结构以避免在 BM25 Scoring 阶段复制 Body；LocalSkillSource 随后从 Registry 补齐内容。
    """

    name: str
    """Skill Name，用于回查 Registry。"""

    score: float
    """Source-local Relevance Score；Higher 更相关，但不可跨 Source 直接比较。"""

    source: str = ""
    """与 `SkillMeta.source` 对齐的 Physical Origin；Empty String 为 Backward Compatibility 保留。"""
