"""集中注册当前保留的 conversational TUI-RPC method surface。

``RpcServer`` 用这里的组合函数把 system、setup、config、session、terminal、image、
model、turn、confirm 与 question handler 装入同一个 ``Dispatcher``。可选依赖决定某些
method group 是否可用：例如没有 ``SubscriptionEmitter`` 就不注册 ``turn.*``，没有
对应 Broker 就不注册 confirm 或 question 往返。

本模块只完成装配，不执行 handler，也不验证 Scheduler、AgentLoopFactory 或 Session
backend 已经可用。method 出现在 Dispatcher 中只证明注册成功，不证明 Runtime 能完成
相应操作。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflow.tui_rpc.methods.config import register_config_methods
from codeflow.tui_rpc.methods.confirm import register_confirm_methods
from codeflow.tui_rpc.methods.image import register_image_methods
from codeflow.tui_rpc.methods.model import register_model_methods
from codeflow.tui_rpc.methods.question import register_question_methods
from codeflow.tui_rpc.methods.session import register_session_methods
from codeflow.tui_rpc.methods.setup import register_setup_methods
from codeflow.tui_rpc.methods.system import register_system_methods
from codeflow.tui_rpc.methods.terminal import register_terminal_methods
from codeflow.tui_rpc.methods.turn import register_turn_methods

if TYPE_CHECKING:
    from codeflow.spine.scheduler import Scheduler
    from codeflow.tui_rpc.confirm_broker import ConfirmBroker
    from codeflow.tui_rpc.dispatcher import Dispatcher
    from codeflow.tui_rpc.errors import RpcError
    from codeflow.tui_rpc.methods.session import AgentLoopFactory
    from codeflow.tui_rpc.methods.turn import SchedulerFactory
    from codeflow.tui_rpc.question_broker import QuestionBroker
    from codeflow.tui_rpc.subscriptions import SubscriptionEmitter


def register_aligned_methods(
    dispatcher: "Dispatcher",
    *,
    emitter: "SubscriptionEmitter | None" = None,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    confirm_broker: "ConfirmBroker | None" = None,
    question_broker: "QuestionBroker | None" = None,
    scheduler: "Scheduler | None" = None,
    scheduler_factory: "SchedulerFactory | None" = None,
    turn_ids: "dict[int, str] | None" = None,
    submission_ids: "dict[int, str] | None" = None,
    build_error: "RpcError | None" = None,
) -> None:
    """在 ``dispatcher`` 上注册全部 aligned RPC handler。

    方法先注册 ``system.*``，再调用 ``register_aligned_methods_except_system()`` 完成其余
    retained surface。``emitter`` 与 build_tui bundle（``scheduler``、``turn_ids``、
    ``submission_ids``、``build_error`` 和 ``scheduler_factory``）转交给
    :func:`register_turn_methods`；``emitter`` 为 ``None`` 时跳过整个 ``turn.*`` group。
    ``agent_loop_factory`` 转交 config/session methods，``confirm_broker`` 通过
    ``register_confirm_methods`` 转交确认交互，``question_broker`` 转交提问交互。

    重复对同一 Dispatcher 注册同名 method 会由 ``Dispatcher.register()`` 抛出
    ``ValueError``。函数无返回值；完成只表示 handler 映射已经建立。
    """
    register_system_methods(dispatcher)
    register_aligned_methods_except_system(
        dispatcher,
        emitter=emitter,
        agent_loop_factory=agent_loop_factory,
        confirm_broker=confirm_broker,
        question_broker=question_broker,
        scheduler=scheduler,
        scheduler_factory=scheduler_factory,
        turn_ids=turn_ids,
        submission_ids=submission_ids,
        build_error=build_error,
    )


def register_aligned_methods_except_system(
    dispatcher: "Dispatcher",
    *,
    emitter: "SubscriptionEmitter | None" = None,
    agent_loop_factory: "AgentLoopFactory | None" = None,
    confirm_broker: "ConfirmBroker | None" = None,
    question_broker: "QuestionBroker | None" = None,
    scheduler: "Scheduler | None" = None,
    scheduler_factory: "SchedulerFactory | None" = None,
    turn_ids: "dict[int, str] | None" = None,
    submission_ids: "dict[int, str] | None" = None,
    build_error: "RpcError | None" = None,
) -> None:
    """注册除 ``system.*`` 之外的 retained RPC handler。

    该入口用于 server 需要单独控制 handshake method 注册顺序的场景。基础的 setup、
    config、session、terminal、image 和 model groups 总会注册；turn、confirm、question
    则仅在相应 ``emitter`` 或 Broker 存在时注册。各参数与
    ``register_aligned_methods()`` 含义一致，且同样不会启动 Runtime 或执行任何 Turn。
    """
    register_setup_methods(dispatcher)
    register_config_methods(dispatcher, agent_loop_factory=agent_loop_factory)
    register_session_methods(
        dispatcher,
        agent_loop_factory=agent_loop_factory,
        confirm_broker=confirm_broker,
    )
    register_terminal_methods(dispatcher)
    register_image_methods(dispatcher)
    register_model_methods(dispatcher)
    if emitter is not None:
        register_turn_methods(
            dispatcher,
            emitter=emitter,
            scheduler=scheduler,
            scheduler_factory=scheduler_factory,
            turn_ids=turn_ids,
            submission_ids=submission_ids,
            build_error=build_error,
        )
    if confirm_broker is not None:
        register_confirm_methods(dispatcher, confirm_broker=confirm_broker)
    if question_broker is not None:
        register_question_methods(dispatcher, question_broker=question_broker)


__all__ = [
    "register_aligned_methods",
    "register_aligned_methods_except_system",
    "register_system_methods",
    "register_setup_methods",
    "register_config_methods",
    "register_session_methods",
    "register_terminal_methods",
    "register_model_methods",
    "register_turn_methods",
    "register_confirm_methods",
    "register_image_methods",
    "register_question_methods",
]
