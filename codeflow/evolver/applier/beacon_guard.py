"""拒绝没有 activation beacon 的 code-class patch。

design section 3 规定，每条 evolved code path 必须调用 ``activation_beacon(node_id)``，让
Gate 1 能证明 path 实际执行。diff 中没有 beacon 就无法监测，因此必须在 eval 前拒绝。
guard 只检查 token presence，不证明 beacon 位于正确控制流、Runtime 已触发或 ledger 写入成功。
"""

from __future__ import annotations

# WHERE 按行为目标而非文件位置分类：直接编辑基准智能体代码（如终止策略）时，根据实际变更
# 分类为 loop_override、tool_override 或 context_override，因此没有单独的
# "benchmark_agent_code" 值。
CODE_CLASS_WHERES = {
    "tool_new",
    "loop_override",
    "context_override",
    "tool_override",
}
BEACON_TOKEN = "activation_beacon("


class MissingBeaconError(ValueError):
    """code-class patch 缺少 activation beacon 时抛出的拒绝异常。"""

    pass


def assert_beacon_present(
    node_id: str,
    *,
    patch_where: str,
    diff_text: str,
) -> None:
    """验证 code-class patch 是否包含 activation beacon。

    ``node_id`` 用于 error message，``patch_where`` 是 ``loop_override``、``skill`` 等行为位置，
    ``diff_text`` 是 unified diff 或 code text。非 ``CODE_CLASS_WHERES`` 直接通过；code class
    中找不到 ``activation_beacon(`` token 时抛出 ``MissingBeaconError``。

    这是 lexical gate，不解析 AST，也不验证传入 node_id 是否正确。
    """
    if patch_where not in CODE_CLASS_WHERES:
        return
    if BEACON_TOKEN not in diff_text:
        raise MissingBeaconError(
            f"{node_id}: {patch_where} patch contains no {BEACON_TOKEN}...) call - unmonitorable, rejected"
        )
