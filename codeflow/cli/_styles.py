"""Shared :mod:`questionary` ``Style`` for all interactive CLI prompts.

Centralizing the palette here lets future commands (sessions picker,
provider login, etc.) stay visually consistent without re-declaring colors
inline. Import is deferred to module load — callers that only need other
helpers should still import lazily so a missing :mod:`questionary` install
doesn't break the rest of the CLI.
"""

from __future__ import annotations

from questionary import Style

CODEFLOW_STYLE = Style(
    [
        # 开头的 "?" 符号和问题文本。
        ("qmark", "fg:#fbe23f bold"),
        ("question", "bold"),
        # 提示结束后回显的已提交答案。
        ("answer", "fg:#fbe23f bold"),
        # "❯" 指针及其所在行（悬停状态）。
        ("pointer", "fg:#fbe23f bold"),
        # 活动行：文本颜色与其他行相同，仅加粗；黄色 "❯" 指针是唯一选择提示，使所有选项
        # 颜色一致。noreverse 用于阻止 prompt_toolkit 基础样式反色高亮活动行，否则会画出实心色块。
        ("highlighted", "fg:#FFF5EA bold noreverse"),
        # 先前选中的值（如复选框）；同理使用 noreverse，以金色而非背景色块表示选中。
        ("selected", "fg:#c8a900 noreverse"),
        # 选项组之间的淡色分隔线。
        ("separator", "fg:#444444"),
        # 问题后的 "(Use arrow keys)" 样式提示。
        ("instruction", "fg:#6c6c6c italic"),
        # 不可选择的行。
        ("disabled", "fg:#585858 italic"),
        # 行内校验错误工具栏。
        ("validation-toolbar", "fg:#ff5f5f bold"),
        # 用户正在输入的自由文本。
        ("text", "fg:#FFF5EA"),
    ]
)

__all__ = ["CODEFLOW_STYLE"]
