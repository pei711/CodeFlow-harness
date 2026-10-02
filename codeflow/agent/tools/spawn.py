"""实现创建后台 Subagent 并把完成结果路由回 Host Session 的 Spawn Tool。

Tool 本身不运行子 Agent Loop，而是把 task、label 和本 Turn Origin 交给 `SubagentManager`。
共享实例用 ContextVar 隔离 Channel、Chat 与 session_key，避免并发 Lane 的完成通知串到错误
会话。子 Agent 最多 15 次内部迭代且无墙上时间，因此 Registry 兜底 timeout 放宽到 900 秒。
"""

from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from codeflow.agent.tools.base import Tool

if TYPE_CHECKING:
    from codeflow.agent.subagent import SubagentManager


@dataclass(frozen=True)
class _SpawnOrigin:
    """保存 Subagent announcement 的 per-turn Origin，并按 asyncio Task 隔离。

    SpawnTool 是共享实例，每个 Turn 在独立 Lane Task 运行；Channel、Chat 与 session_key 放在
    ContextVar，防止并发 Turn 覆盖。对象采用 Frozen + copy-on-write，Child Task 即使继承 Parent
    值也只会重新绑定自己的 Context，不能通过共享引用写回 Parent。
    """

    channel: str
    chat_id: str
    session_key: str


class SpawnTool(Tool):
    """请求 SubagentManager 为可独立任务创建后台子 Agent。

    ``task`` 是完整工作说明，``label`` 只用于展示；execute 将它们连同当前 Origin 路由信息交给
    Manager，并 await Manager 的 spawn 结果。并发数量、小时限额、Agent 生命周期与结果重注入
    都由 Manager 拥有，Tool 不绕过门禁，也不把“已创建”冒充“任务已完成”。
    """

    # 子 Agent 运行自己的循环，最多 15 次迭代，内部没有墙上时间上限，
    # 因此使用宽裕的兜底超时，而非默认值。
    timeout_seconds = 900.0

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._default = _SpawnOrigin(channel="cli", chat_id="direct", session_key="cli:direct")
        self._origin: ContextVar[_SpawnOrigin] = ContextVar("spawn_origin")

    def _cur(self) -> _SpawnOrigin:
        return self._origin.get(None) or self._default

    def set_context(self, channel: str, chat_id: str, session_key: str) -> None:
        """设置 Subagent 完成通知的 turn-local Channel、Chat 与 Session Origin。

        方法以 `replace` 创建新 `_SpawnOrigin` 并绑定当前 ContextVar；应在实际运行 Turn 的 Task
        内调用。此信息决定子 Agent 结果回注到哪里，不改变子 Agent 自身任务内容或资源额度。
        """
        self._origin.set(replace(self._cur(), channel=channel, chat_id=chat_id, session_key=session_key))

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        """把给定 ``task`` 交给 Manager 创建 Subagent，并返回 Manager 的执行结果文本。

        调用会读取当前 `_SpawnOrigin`，把 origin_channel、origin_chat_id、session_key 与可选 label
        一并传入。缺省 Context 使用 CLI direct 基线。方法不自行创建 asyncio Task，也不吞掉
        Manager 的异常；900 秒 Tool ceiling 仍由 Registry 外层控制。
        """
        org = self._cur()
        return await self._manager.spawn(
            task=task,
            label=label,
            origin_channel=org.channel,
            origin_chat_id=org.chat_id,
            session_key=org.session_key,
        )
