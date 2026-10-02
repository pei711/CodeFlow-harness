"""定义消息来自哪里、回复送往哪里以及消息携带什么的空间词汇。

`Source` 保存 Channel、chat、sender 与会话形态，`Media` 保存已经验证的附件位置和可选字节
快照，`ChatType` 只保留真实分支需要的私聊/群聊区别。这些对象都是 frozen data，没有发送、
下载或权限判断行为；Spine 与 Channel rewrite 共享它们，避免各层用不兼容字典重复描述同一
地址事实。
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ChatType(StrEnum):
    """把会话形态收敛为当前行为真正分支的两种形式。

    ``DM`` 表示一对一私聊，``GROUP`` 表示群聊。Channel adapter 负责把平台特有类型映射到
    这两个协议值；下游据此选择提及、权限或回复行为，而不是携带各平台枚举。没有消费者
    使用的更细类别不在这里提前建模。
    """

    DM = "dm"
    GROUP = "group"


@dataclass(frozen=True)
class Source:
    """记录一条入站消息的来源，也是出站回复的默认投递地址。

    ``channel`` 选择 DeliveryHub 的 Outlet，``chat_id`` 指向平台会话，``sender_id`` 标识发件人，
    ``chat_type`` 区分 DM 与 GROUP。``extras`` 保存 Channel 特有但核心协议不解释的只读扩展
    字段。对象冻结后随 Turn 传播，Runner 和 Hub 只能补入事件而不能改写来源事实。
    """

    channel: str
    chat_id: str
    sender_id: str
    chat_type: ChatType
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Media:
    """描述一个已验证附件，并可携带入站时取得的不可变字节快照。

    ``path`` 是运行时可访问位置，``mime`` 与 ``kind`` 供 Context 和 Outlet 选择处理方式；
    ``content`` 可在文件随后变化或消失前冻结原始 bytes，并通过 ``repr=False`` 避免日志展开
    大载荷。该对象不负责读取、解码或上传附件，验证与生命周期仍由创建它的 Channel/Host
    承担。
    """

    path: str
    mime: str
    kind: str
    content: bytes | None = field(default=None, repr=False)
