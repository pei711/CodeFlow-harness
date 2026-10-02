"""定义 LLM judge output 的 typed schema 与 cross-field invariant。

Judge 从 trajectory 生成两个 orthogonal axis：spec §3 的 Issue type L1/L2/L3 决定路由，L1
进入 human review，L2/L3 进入 Evolver patch；spec §12.4-§12.5 的 ``(WHERE, WHY)`` 只在
L2/L3 填充，说明改哪里、针对哪类 pathology。

所有 Enum 都继承 ``str``，无需 custom encoder 即可 JSON serialize，也能与 LLM output 解析的
string 直接比较，例如 ``IssueType.L1 == "L1"``。``PatchWhy.other`` 接受
``"other:<new_category>"`` free-form sub-name，支持 spec §12.5 的 evolving WHY taxonomy；
新 pathology 必须先 human review，不能自动成为 first-class Enum。

schema validation 只保证 Judge output 内部一致，不证明分类真实、patch 可行或 task 已完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class IssueType(str, Enum):
    """spec §3 的三态 failure classification，决定 downstream routing。

    L1 是 infrastructure defect、阻塞 Evolver；L2 是 evaluation framework config 问题；L3 是
    Harness capability gap。Enum 不表示严重度排序。
    """

    L1 = "L1"  # 基础设施缺陷——阻塞 evolver 并交由人工处理
    L2 = "L2"  # 评测框架配置错误——evolver 修补文档或配置
    L3 = "L3"  # 评测框架能力缺口——evolver 修补技能、记忆或钩子


class PatchWhere(str, Enum):
    """proposal 的 structural patch location（spec §12.4，14 categories）。

    每个 value 对应 mutable surface（spec §2.2）中的 file class，Judge 按 failure signature
    选择。``control`` 于 2026-06-10 为 evolution-restart round 1 加入，表示没有真实 patch
    surface 的 control-arm node。
    """

    system_prompt_template = "system_prompt_template"  # 路径：templates/*.md
    task_wrapper_prompt = "task_wrapper_prompt"  # benchmarks/<bench>/ 的任务包装器
    judge_prompt = "judge_prompt"  # 路径：codeflow/eval_engine/prompts/*.py
    tool_description = "tool_description"  # codeflow/agent/tools/*.py 中的描述
    hook_new = "hook_new"  # 新建 codeflow/agent/hook/<name>.py
    hook_modify = "hook_modify"  # 修改 codeflow/eval_engine/hooks/*.py
    skill = "skill"  # 路径：codeflow/memory_engine/skills/*
    memory = "memory"
    tool_new = "tool_new"  # 新建 codeflow/agent/tools/<name>.py
    loop_override = "loop_override"  # 有限作用域的循环覆盖
    context_override = "context_override"  # 有限作用域的上下文引擎覆盖
    tool_override = "tool_override"  # 有限作用域的工具覆盖（代码类）
    config = "config"  # yaml/json 默认值
    control = "control"  # 对照组，不涉及补丁表面


class PatchWhy(str, Enum):
    """patch 要处理的 pathology category（spec §12.5，11 named + other）。

    taxonomy 来自 244-paired SWE-bench failure analysis。Judge 可用 ``other`` + free-form
    sub-name 提议新类，human review 后才 promote 为 first-class Enum。

    ``reasoning_visibility`` 于 2026-06-01 从 B2 dry-run 的 ``patch_why_extra`` accumulation
    promote，是 dominant uncategorized class：10 个 other 中 5 个来自
    reasoning_visibility_improvement、communication_traceability、communication verbosity
    nudge、explanatory_text_nudge、trajectory_logging_quality。``empty_response_recovery``、
    ``method_lock_in_remedy``、``infra_neutrality_control`` 于 2026-06-10 为 restart round 1
    加入，分别表示 empty-response streak recovery、early method lock-in remedy、control-arm
    bookkeeping。
    """

    repetition_breaker = "repetition_breaker"  # 轨迹尾部重复率达 72%
    test_starvation_remedy = "test_starvation_remedy"  # 通过样本测试占 25%，失败样本占 12%
    budget_awareness = "budget_awareness"  # 失败样本 100% 达到 maxIter
    tool_clarity = "tool_clarity"  # 工具文档缺失或有误导性
    env_contract_clarify = "env_contract_clarify"  # 环境规则相互矛盾（如禁止提示）
    skill_gap_fill = "skill_gap_fill"  # 反复出现的任务类型缺少技能
    memory_recall_fix = "memory_recall_fix"  # 重复读取或验证已知事实
    reasoning_visibility = "reasoning_visibility"  # 长时间只有工具调用而无叙述解释
    empty_response_recovery = "empty_response_recovery"  # 连续空响应恢复
    method_lock_in_remedy = "method_lock_in_remedy"  # 修复过早锁定方法
    infra_neutrality_control = "infra_neutrality_control"  # 对照组记录，并非真实病理
    other = "other"  # 由评判器提出；子名称存入 patch_why_extra


class ActionKind(str, Enum):
    """downstream 对 Judge output 采取的动作类型。

    ``human_review_needed`` 对应 L1 且禁止 auto patch；``patch_proposal`` 对应 L2/L3。
    """

    human_review_needed = "human_review_needed"  # L1：evolver 暂停，由工程师修复
    patch_proposal = "patch_proposal"  # L2/L3：evolver 应用补丁


@dataclass
class ProposedComponent:
    """multi-file patch 中 Judge 提议修改的一个 file component。

    tree layer counterpart 是带 actual diff 的
    :class:`codeflow.evolver.tree.node.PatchComponent`。Judge 阶段只有 natural-language
    ``summary``；mutation operator（GEPA library）随后把它变成 unified diff。

    spec §12.2/§18.5.1.x 允许一个 semantic improvement 由多个 file edit 组成，例如新 hook +
    config registration；``depends_on`` graph 使 regression 时 component-level bisect 能安全
    删除 subset。简单 one-file fix 的 ``JudgeAction.components`` length 为 1。
    """

    component_id: str  # "comp_1" / "comp_2"，在单个 JudgeAction 内唯一
    target_file: str  # 待编辑文件的仓库相对路径
    summary: str  # 对预期编辑的自然语言描述
    depends_on: list[str] = field(default_factory=list)  # 同级组件 ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "target_file": self.target_file,
            "summary": self.summary,
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProposedComponent":
        missing = [k for k in ("component_id", "target_file", "summary") if k not in d]
        if missing:
            raise ValueError(f"ProposedComponent missing required fields: {missing}")
        return cls(
            component_id=d["component_id"],
            target_file=d["target_file"],
            summary=d["summary"],
            depends_on=list(d.get("depends_on") or []),
        )


@dataclass
class JudgeAction:
    """Judge 推荐的下一步动作及其结构化 proposal。

    ``kind=human_review_needed``（L1）只填写 ``reasoning``，Evolver MUST NOT apply patch，只
    向 human operator 暴露问题。``kind=patch_proposal``（L2/L3）必须包含 WHERE、WHY 与非空
    components；mutation operator 后续生成 unified diff。

    ``patch_why_extra`` 仅在 ``patch_why == other`` 时非空，携带 Judge 新 pathology sub-name，
    例如 ``other:plan_action_disconnect``；review/promote sub-name 是 spec §12.5 演化 WHY
    taxonomy 的机制。构造时还验证 component_id unique 与 depends_on sibling existence。
    """

    kind: ActionKind
    reasoning: str

    # 仅 patch_proposal 会填充：
    patch_where: Optional[PatchWhere] = None
    patch_why: Optional[PatchWhy] = None
    patch_why_extra: Optional[str] = None
    components: list[ProposedComponent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind == ActionKind.patch_proposal:
            if not self.components:
                raise ValueError("patch_proposal JudgeAction requires at least one component")
            ids = [c.component_id for c in self.components]
            if len(ids) != len(set(ids)):
                raise ValueError(f"JudgeAction.components have duplicate component_id: {ids}")
            valid_ids = set(ids)
            for c in self.components:
                for dep in c.depends_on:
                    if dep not in valid_ids:
                        raise ValueError(
                            f"ProposedComponent {c.component_id!r} depends_on {dep!r} which is not a sibling component"
                        )

    def is_patch(self) -> bool:
        return self.kind == ActionKind.patch_proposal

    def is_human_review(self) -> bool:
        return self.kind == ActionKind.human_review_needed

    @property
    def target_file(self) -> Optional[str]:
        """返回首个 component target_file，作为 backwards-compat primary file。

        无 component 时返回 ``None``。multi-component caller 应直接遍历 components。
        """
        return self.components[0].target_file if self.components else None

    @property
    def patch_summary(self) -> Optional[str]:
        """返回兼容旧 caller 的 patch summary string。

        single-component 返回该 summary；无 component 返回 ``None``。multi-component 没有单一
        summary，规范用法是 ``components[i].summary`` 或整体 ``reasoning``；compat path 用
        `` | `` 拼接。
        """
        if not self.components:
            return None
        if len(self.components) == 1:
            return self.components[0].summary
        # 多组件：为仍请求单字符串的旧调用方拼接，但规范访问方式是逐组件读取。
        return " | ".join(c.summary for c in self.components)


@dataclass
class JudgeResult:
    """一段 trajectory 的 complete Judge analysis。

    字段遵循 spec §3.2。``evidence_turn_range`` 是 Judge 引用的 inclusive Turn index interval，
    用于把 patch anchor 到具体 failure point。``confidence`` 必须在 ``[0.0, 1.0]``；downstream
    可拒绝 low-confidence judgment，例如跳过 confidence < 0.5 proposal。

    cross-field invariant 要求 L1 -> human_review_needed，L2/L3 -> patch_proposal + WHERE/WHY，
    且 ``PatchWhy.other`` 必须有 extra sub-name。``raw_response`` 只用于 audit。
    """

    trajectory_id: str
    issue_type: IssueType
    confidence: float
    signal_description: str
    proposed_action: JudgeAction
    evidence_turn_range: Optional[tuple[int, int]] = None
    raw_response: Optional[str] = None  # 原始 LLM 文本，用于审计

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence!r}")
        # 跨字段不变量：L1 必须使用 human_review_needed。
        if self.issue_type == IssueType.L1 and not self.proposed_action.is_human_review():
            raise ValueError(
                f"L1 issues must have proposed_action.kind=human_review_needed; got {self.proposed_action.kind!r}"
            )
        # L2/L3 必须使用 patch_proposal，并填充 where/why。
        if self.issue_type in (IssueType.L2, IssueType.L3):
            if not self.proposed_action.is_patch():
                raise ValueError(f"{self.issue_type.value} issues must have proposed_action.kind=patch_proposal")
            if self.proposed_action.patch_where is None:
                raise ValueError(f"{self.issue_type.value} patch must have patch_where set")
            if self.proposed_action.patch_why is None:
                raise ValueError(f"{self.issue_type.value} patch must have patch_why set")
            if self.proposed_action.patch_why == PatchWhy.other and not self.proposed_action.patch_why_extra:
                raise ValueError("patch_why=other requires patch_why_extra to carry the judge-proposed sub-name")


@dataclass(frozen=True)
class PassFailResult:
    """无 benchmark verifier 时，一段 trajectory 的 pass/fail verdict。

    它不同于只诊断 failure mode、不承载 pass/fail 的 :class:`JudgeResult`。LLM scorer 在没有
    verifier 时返回该对象，orchestrator 把 ``passed`` 映射到 ``TaskEval.passes``。这仍是 LLM
    judgment，不等价于 deterministic verifier evidence。
    """

    trajectory_id: str
    passed: bool
    reasoning: str = ""
    raw_response: Optional[str] = None


__all__ = [
    "IssueType",
    "PatchWhere",
    "PatchWhy",
    "ActionKind",
    "ProposedComponent",
    "JudgeAction",
    "JudgeResult",
    "PassFailResult",
]
