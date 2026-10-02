"""定义一次请求的意图、归属与排队策略，也就是 `submit()` 接受的唯一输入。

Spine 不从消息文本猜测“这是谁的任务”或“忙时怎么办”，而是要求 Host 在创建
`TurnRequest` 时显式给出 `Origin`、`Source` 和 `BusyPolicy`。这些字段随后决定并发池、
conversation lane、回复地址、控制资格和排队行为；正文与媒体只是本轮要处理的载荷。
阅读调度链路时应从 `TurnRequest` 开始，再看 Scheduler 如何按 conversation 串行化请求。
"""

from dataclasses import dataclass
from enum import StrEnum

from codeflow.spine.message import Media, Source


class Origin(StrEnum):
    """标明请求归属方，并驱动并发池、控制资格与账目归属。

    ``USER`` 是真实用户发起的 Turn；``CRON`` 是 Runtime 定时任务；``SUBAGENT`` 是子 Agent
    结果重新注入 Host 的 Turn。它们可能携带相似文本，但资源与信任边界不同，所以调度器
    必须读取这个枚举而不能从内容推断。新增 Origin 时还必须在 `OriginPools` 中明确选择
    pool，否则调度会拒绝它。
    """

    USER = "user"
    CRON = "cron"
    SUBAGENT = "subagent"


class BusyPolicy(StrEnum):
    """规定同一 conversation lane 已忙时，新请求如何进入执行域。

    ``APPEND`` 把请求排到队尾；``INJECT`` 尝试在当前 Turn 的模型—Tool 迭代间隙合并，若
    未被 drain 则回退成普通 APPEND；``INTERRUPT`` 取消当前 payload task，并把新请求插到
    既有积压前。策略只在同一 Lane 内生效，不改变其他 conversation 的并发，也不允许提交
    方直接完成正在运行 Turn 的 terminal future。
    """

    APPEND = "append"
    INJECT = "inject"
    INTERRUPT = "interrupt"


@dataclass(frozen=True)
class TurnRequest:
    """封装交给 Spine 处理的一次完整请求。

    ``origin`` 决定资源与控制边界，``source`` 保存 channel、chat、sender 等回复地址，
    ``text`` 与不可变 ``media`` 元组组成用户可见载荷。``conversation`` 可显式指定 Lane key；
    为空时 Host 通常从 Source 构造会话标识。``busy`` 默认 ``BusyPolicy.APPEND``，避免未声明
    的请求抢占正在执行的工作。

    ``message_id`` 是入站消息自身的 id，也是出站回复默认串回原消息的锚点；在输出侧它会
    作为 Text 的 ``reply_to`` 字段携带。数据类使用 ``frozen=True``，提交后不会被调度器原地
    改写，媒体也用 tuple 防止排队期间载荷漂移。
    """

    origin: Origin
    source: Source
    text: str
    media: tuple[Media, ...] = ()
    message_id: str | None = None
    conversation: str | None = None
    busy: BusyPolicy = BusyPolicy.APPEND
