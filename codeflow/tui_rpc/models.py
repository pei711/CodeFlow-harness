"""定义 tui-ipc-bridge JSON-RPC contract 的 Pydantic v2 models。

这些类型是 ``ui-tui/rpc-schema/openrpc.json`` 的 Python-side mirror。
``specs/tui-ipc.md`` §3.12 中每个 public type 都有对应
:class:`pydantic.BaseModel`，每个 RPC method 都有 ``<Method>Params`` 与
``<Method>Result`` model。``TurnEvent`` 使用 ``type`` discriminator 把 streaming event
建模为可辨别 union；``METHOD_MODELS`` 则把 method name 连接到参数和结果 schema。

``tests/test_rpc_schema_match.py`` 在 CI 中检测本模块与 OpenRPC schema 的 drift。任何一侧
变更都 MUST 在同一 commit 同步另一侧。model validation 成功只证明 wire shape 与类型约束
成立，不证明 method 已执行、副作用已持久化或 Agent 任务完成。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 可复用的模型配置。``extra="forbid"`` 会让 Pydantic 在生成的 JSON Schema 中输出
# ``additionalProperties: false``，与 OpenRPC 模式对每个对象的显式定义一致。
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    """所有 RPC model 的严格基类，默认禁止 extra field。

    ``extra="forbid"`` 让未知字段在 validation 时失败，并使生成的 JSON Schema 包含
    ``additionalProperties: false``，与 OpenRPC 对每个 object 的显式定义一致。该类只提供
    schema policy，不承载具体协议字段。
    """

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 公开类型（specs/tui-ipc.md §3.12）
# ---------------------------------------------------------------------------


# JSON Schema 中的 ``JsonValue`` 是全部基本类型和容器类型的联合。此处使用 ``Any``，
# 因为模式将 ``JsonValue`` 定义为宽松的基本类型任选项，而模式匹配测试将 JsonValue
# 锁定到 OpenRPC 规范而非 Pydantic 模式（见 ``components/schemas/JsonValue``）。
JsonValue = Any


class SessionInfo(_Strict):
    """retained TUI Session panel 渲染所需的 init bundle。

    ``model``、``skills``、``tools`` 是前端必需字段；provider、memory、context window、
    lazy 状态、usage、version、cwd 与 MCP server 信息用于展示和诊断。它是 Session 打开
    时的 snapshot，不保证后续 Runtime 状态持续不变。
    """

    model: str
    skills: dict[str, list[str]]
    tools: dict[str, list[str]]
    provider: str | None = None
    memory: str | None = None
    context_window: int | None = None
    lazy: bool | None = None
    usage: dict[str, JsonValue] | None = None
    version: str | None = None
    cwd: str | None = None
    mcp_servers: list[dict[str, JsonValue]] | None = None


class SessionMessage(_Strict):
    """``session.resume`` 返回的一条 transcript message。

    ``role`` 只能是 ``user``、``assistant``、``system`` 或 ``tool``；``text``、``context``
    与 ``name`` 可选。该 wire model 面向 TUI 渲染，不保留原始 multimodal non-text block。
    """

    role: Literal["user", "assistant", "system", "tool"]
    text: str | None = None
    context: str | None = None
    name: str | None = None


class UsageSnapshot(_Strict):
    """Turn 结束时报告的 token 与 cost usage snapshot。

    三类 token counter 必填；``cost_usd`` 和 context 使用量/上限/百分比可选。snapshot 反映
    已报告 usage，不单独证明 provider 已结算、任务完成质量或正向结论可用。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None = None
    context_used: int | None = None
    context_max: int | None = None
    context_percent: int | None = None


# ---------------------------------------------------------------------------
# TurnEvent：线上事件变体的可辨别联合。
# ---------------------------------------------------------------------------


class MessageStartPayload(_Strict):
    submission_id: str
    turn_id: str


class MessageStartEvent(_Strict):
    type: Literal["message.start"]
    payload: MessageStartPayload


class TokenDeltaPayload(_Strict):
    text: str


class TokenDeltaEvent(_Strict):
    type: Literal["token.delta"]
    payload: TokenDeltaPayload


class ThinkingDeltaPayload(_Strict):
    text: str


class ThinkingDeltaEvent(_Strict):
    type: Literal["thinking.delta"]
    payload: ThinkingDeltaPayload


class ToolStartPayload(_Strict):
    tool_call_id: str
    name: str
    arguments: dict[str, JsonValue]


class ToolStartEvent(_Strict):
    type: Literal["tool.start"]
    payload: ToolStartPayload


class ToolProgressPayload(_Strict):
    tool_call_id: str
    preview: str


class ToolProgressEvent(_Strict):
    type: Literal["tool.progress"]
    payload: ToolProgressPayload


class ToolCompletePayload(_Strict):
    tool_call_id: str
    result_preview: str
    truncated: bool
    failed: bool = False


class ToolCompleteEvent(_Strict):
    type: Literal["tool.complete"]
    payload: ToolCompletePayload


class MessageCompletePayload(_Strict):
    submission_id: str
    turn_id: str
    usage: UsageSnapshot


class MessageCompleteEvent(_Strict):
    type: Literal["message.complete"]
    payload: MessageCompletePayload


class ErrorEventPayload(_Strict):
    attachments_discarded: bool = False
    code: int
    message: str
    reason: Literal["cancelled_by_client", "internal"] | None = None
    submission_id: str | None = None
    turn_id: str | None = None


class ErrorEvent(_Strict):
    type: Literal["error"]
    payload: ErrorEventPayload


class CronDeliveredPayload(_Strict):
    job_id: str
    name: str
    text: str
    fired_at: str


class CronDeliveredEvent(_Strict):
    type: Literal["cron.delivered"]
    payload: CronDeliveredPayload


class SubagentDeliveredPayload(_Strict):
    text: str


class SubagentDeliveredEvent(_Strict):
    type: Literal["subagent.delivered"]
    payload: SubagentDeliveredPayload


TurnEvent = Annotated[
    Union[
        MessageStartEvent,
        TokenDeltaEvent,
        ThinkingDeltaEvent,
        ToolStartEvent,
        ToolProgressEvent,
        ToolCompleteEvent,
        MessageCompleteEvent,
        ErrorEvent,
        CronDeliveredEvent,
        SubagentDeliveredEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# session.* 方法
# ---------------------------------------------------------------------------


class SessionListItem(_Strict):
    """Session picker 中的一行，对应 ``gatewayTypes.ts:130 SessionListItem``。

    ``id`` 是完整 ``<channel>:<chat_id>``，``started_at`` 是由 ``created_at`` 转换的 Unix
    timestamp；``message_count``、``preview``、``source`` 与 ``title`` 供列表展示。它只是
    metadata row，不包含 transcript。
    """

    id: str = Field(..., description="Full session_key: <channel>:<chat_id>.")
    message_count: int
    preview: str
    source: str | None = None
    started_at: float = Field(..., description="Unix timestamp from created_at.")
    title: str


class SessionListParams(_Strict):
    limit: int | None = Field(default=None, description="Max sessions to return.")


class SessionListResult(_Strict):
    sessions: list[SessionListItem]


class SessionCreateParams(_Strict):
    cols: int | None = None


class SessionCreateResult(_Strict):
    session_id: str
    info: SessionInfo


class SessionResumeParams(_Strict):
    session_id: str


class SessionResumeResult(_Strict):
    session_id: str
    info: SessionInfo
    messages: list[SessionMessage]


class SessionCloseParams(_Strict):
    session_id: str | None = None


class SessionCloseResult(_Strict):
    ok: bool


class SessionDeleteParams(_Strict):
    session_id: str = Field(..., description="Full session_key as sent by the UI.")


class SessionDeleteResult(_Strict):
    deleted: str | None = Field(
        default=None,
        description=(
            "The session_id whose current logical generation was deleted; null when no matching generation existed."
        ),
    )
    cancelled: bool | None = Field(
        default=None,
        description="True when the user declined the TUI confirmation.",
    )


class SessionMostRecentParams(_Strict):
    pass


class SessionMostRecentResult(_Strict):
    """``gatewayTypes.ts:147 SessionMostRecentResponse`` 对应的 response shape。

    ``session_id`` 是完整 ``tui:<chat_id>``，无 Session 时为 null；source、started_at 与
    title 为兼容 frontend 的可选字段。返回 ID 不表示 Session file 已加载。
    """

    session_id: str | None = Field(
        default=None,
        description="Full tui:<chat_id> key, or null when no sessions exist.",
    )
    source: str | None = None
    started_at: float | None = None
    title: str | None = None


class SessionTitleParams(_Strict):
    """``slash/commands/core.ts:201,218`` 使用的 title 参数。

    ``session_id`` 是完整 Session key；``title`` 提供时为 set path，省略时为 get path。
    """

    session_id: str = Field(..., description="Full session_key.")
    title: str | None = None


class SessionTitleResult(_Strict):
    """``gatewayTypes.ts:154 SessionTitleResponse`` 对应的 response。

    ``pending=True`` 表示 title 仍保存在 lazy、never-saved Session 的 memory 中，会在该
    Session 第一次 save 时落盘；``False`` 表示本次调用无需等待后续首次 save。
    """

    title: str | None = None
    session_key: str
    pending: bool


class SessionClearParams(_Strict):
    """``session.clear`` 参数：原地清空 messages，同时保留 sid。

    ``session_id`` 必须是要改写的完整 Session key；model 只验证 shape，不检查 active Turn。
    """

    session_id: str = Field(..., description="Full session_key to clear.")


class SessionClearResult(_Strict):
    session_id: str = Field(..., description="The same session_key (no new id minted).")
    cleared: bool = Field(..., description="True when the in-place wipe ran.")


class SessionUndoParams(_Strict):
    """``session.undo`` 参数：删除最后 ``n`` 个 Turn，默认 1。

    Turn boundary 由 handler 使用 ``role==user`` 规则解释；本 model 只承载 session key 与 n。
    """

    session_id: str = Field(..., description="Full session_key to undo.")
    n: int = Field(1, description="Trailing turns to drop (role==user boundary).")


class SessionUndoResult(_Strict):
    removed: int = Field(..., description="Messages dropped (0 = nothing to undo).")


class SessionExportParams(_Strict):
    """``session.export`` 参数：写出 portable Session artifact。

    ``session_id`` 可为完整 key、prefix，或省略以表示 current Session；歧义与 not-found 由
    handler 的 shared resolver 区分。
    """

    session_id: str | None = Field(
        default=None,
        description="Session id / prefix / full key to export; current session when omitted.",
    )


class SessionExportResult(_Strict):
    exported: bool = Field(..., description="True when a verified portable artifact was written.")
    path: str | None = Field(..., description="Absolute path of the written file, or null on failure.")
    reason: str | None = Field(
        default=None,
        description="Failure reason: not_found | ambiguous | write_failed | verification_failed.",
    )
    candidates: list[str] | None = Field(
        default=None,
        description="Candidate full keys when reason is ambiguous.",
    )


class SessionBranchParams(_Strict):
    session_id: str
    name: str | None = None


class SessionBranchResult(_Strict):
    session_id: str | None = None
    title: str | None = None
    message_count: int | None = None


# ---------------------------------------------------------------------------
# turn.* 方法
# ---------------------------------------------------------------------------


class TurnSendParams(_Strict):
    session_key: str
    content: str
    submission_id: str | None = None
    channel: str | None = None
    chat_id: str | None = None
    sender_id: str | None = None


class TurnSendResult(_Strict):
    turn_id: str
    accepted: bool


class TurnSubscribeParams(_Strict):
    session_key: str


class TurnSubscribeResult(_Strict):
    subscription_id: str


class TurnUnsubscribeParams(_Strict):
    subscription_id: str


class TurnUnsubscribeResult(_Strict):
    unsubscribed: bool


class TurnCancelParams(_Strict):
    session_key: str


class TurnCancelResult(_Strict):
    cancelled: bool


class ImageAttachParams(_Strict):
    session_id: str
    path: str


class ImageAttachResult(_Strict):
    name: str
    remainder: str


# ---------------------------------------------------------------------------
# model.* 方法
# ---------------------------------------------------------------------------


class ModelOptionProvider(_Strict):
    """``/model`` picker 中的一条 provider row。

    字段描述本地 provider identity、认证配置、是否 current、auth type、env key、可选模型、
    ``api_base`` 要求与 warning。``authenticated`` 表示本地配置状态，不是远端连通性证明。
    """

    slug: str
    name: str
    authenticated: bool
    is_current: bool
    auth_type: str
    key_env: str | None = None
    models: list[str]
    total_models: int
    needs_api_base: bool
    warning: str


class ModelOptionsParams(_Strict):
    session_id: str | None = None


class ModelOptionsResult(_Strict):
    model: str
    provider: str
    providers: list[ModelOptionProvider]


class ModelSaveKeyParams(_Strict):
    slug: str
    api_key: str
    api_base: str | None = None
    session_id: str | None = None


class ModelSaveKeyResult(_Strict):
    provider: ModelOptionProvider


class ModelDisconnectParams(_Strict):
    slug: str
    session_id: str | None = None


class ModelDisconnectResult(_Strict):
    disconnected: bool


class ModelAddModelParams(_Strict):
    slug: str
    model: str
    session_id: str | None = None


class ModelAddModelResult(_Strict):
    provider: ModelOptionProvider


class ModelRemoveModelParams(_Strict):
    slug: str
    model: str
    session_id: str | None = None


class ModelRemoveModelResult(_Strict):
    provider: ModelOptionProvider


# ---------------------------------------------------------------------------
# config.* 方法
# ---------------------------------------------------------------------------


class ConfigGetParams(_Strict):
    keys: list[str] | None = Field(
        default=None,
        description=("If omitted, return all whitelisted fields. Unknown keys are silently dropped."),
    )


class ConfigGetResult(_Strict):
    config: dict[str, JsonValue]


class ConfigSetParams(_Strict):
    key: str
    value: JsonValue
    provider: str | None = None
    session_id: str | None = None


class ConfigSetResult(_Strict):
    applied: bool
    # ``previous`` 是必填字段，但值可以合法地为 ``null``。将它标注为 ``JsonValue``
    # （``Any``），因为 ``JsonValue`` 已包含 ``null``；模式中冗余的
    # ``oneOf: [JsonValue, null]`` 最终也会收敛为同一标准的“任意值”形式。
    previous: JsonValue = Field(...)
    value: str | None = None


# ---------------------------------------------------------------------------
# system.* 方法
# ---------------------------------------------------------------------------


class SystemHelloParams(_Strict):
    client_version: str
    client_capabilities: list[str] | None = None


class SystemHelloSession(_Strict):
    default_channel: Literal["tui"]
    default_session_key: str


class SystemHelloResult(_Strict):
    server_version: str
    server_capabilities: list[str]
    session: SystemHelloSession


class SystemPingParams(_Strict):
    pass


class SystemPingResult(_Strict):
    pong: Literal[True]
    server_time_ms: float


class SystemVersionParams(_Strict):
    pass


class SystemVersionResult(_Strict):
    server_version: str
    schema_version: str = Field(..., description="OpenRPC info.version mirrored back to client.")
    codeflow_version: str


class SetupStatusParams(_Strict):
    pass


class SetupStatusResult(_Strict):
    provider_configured: bool


class TerminalResizeParams(_Strict):
    cols: int | None = None
    rows: int | None = None
    session_id: str | None = None


class TerminalResizeResult(_Strict):
    ok: bool


class ConfirmRespondParams(_Strict):
    request_id: str
    answer: bool


class ConfirmRespondResult(_Strict):
    ok: bool


class ClarifyRespondParams(_Strict):
    request_id: str | None = None
    conversation_id: str | None = None
    answer: str


class ClarifyRespondResult(_Strict):
    ok: bool


# ---------------------------------------------------------------------------
# 方法注册表：tests/test_rpc_schema_match.py 用它遍历每个方法，并将其 Pydantic
# Params/Result 模型与 OpenRPC 模式比较。键必须与 openrpc.json 中的
# ``method.name`` 字符串一致。
# ---------------------------------------------------------------------------

METHOD_MODELS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    # session.* 方法
    "session.list": (SessionListParams, SessionListResult),
    "session.create": (SessionCreateParams, SessionCreateResult),
    "session.close": (SessionCloseParams, SessionCloseResult),
    "session.resume": (SessionResumeParams, SessionResumeResult),
    "session.delete": (SessionDeleteParams, SessionDeleteResult),
    "session.most_recent": (SessionMostRecentParams, SessionMostRecentResult),
    "session.title": (SessionTitleParams, SessionTitleResult),
    "session.clear": (SessionClearParams, SessionClearResult),
    "session.undo": (SessionUndoParams, SessionUndoResult),
    "session.branch": (SessionBranchParams, SessionBranchResult),
    "session.export": (SessionExportParams, SessionExportResult),
    # turn.* 方法
    "turn.send": (TurnSendParams, TurnSendResult),
    "turn.subscribe": (TurnSubscribeParams, TurnSubscribeResult),
    "turn.unsubscribe": (TurnUnsubscribeParams, TurnUnsubscribeResult),
    "turn.cancel": (TurnCancelParams, TurnCancelResult),
    "image.attach": (ImageAttachParams, ImageAttachResult),
    # model.* 方法
    "model.options": (ModelOptionsParams, ModelOptionsResult),
    "model.save_key": (ModelSaveKeyParams, ModelSaveKeyResult),
    "model.disconnect": (ModelDisconnectParams, ModelDisconnectResult),
    "model.add_model": (ModelAddModelParams, ModelAddModelResult),
    "model.remove_model": (ModelRemoveModelParams, ModelRemoveModelResult),
    # config.* 方法
    "config.get": (ConfigGetParams, ConfigGetResult),
    "config.set": (ConfigSetParams, ConfigSetResult),
    # system.* 方法
    "system.hello": (SystemHelloParams, SystemHelloResult),
    "system.ping": (SystemPingParams, SystemPingResult),
    "system.version": (SystemVersionParams, SystemVersionResult),
    "setup.status": (SetupStatusParams, SetupStatusResult),
    "terminal.resize": (TerminalResizeParams, TerminalResizeResult),
    "confirm.respond": (ConfirmRespondParams, ConfirmRespondResult),
    "clarify.respond": (ClarifyRespondParams, ClarifyRespondResult),
}

__all__ = [
    # 公开类型
    "SessionInfo",
    "SessionListItem",
    "SessionMessage",
    "ModelOptionProvider",
    "UsageSnapshot",
    "TurnEvent",
    "ImageAttachParams",
    "ImageAttachResult",
    "SessionMostRecentParams",
    "SessionMostRecentResult",
    "SessionTitleParams",
    "SessionTitleResult",
    "SessionClearParams",
    "SessionClearResult",
    "SessionUndoParams",
    "SessionUndoResult",
    "SessionExportParams",
    "SessionExportResult",
    "MessageStartEvent",
    "TokenDeltaEvent",
    "ThinkingDeltaEvent",
    "ToolStartEvent",
    "ToolProgressEvent",
    "ToolCompleteEvent",
    "MessageCompleteEvent",
    "ErrorEvent",
    "CronDeliveredEvent",
    "CronDeliveredPayload",
    "SubagentDeliveredEvent",
    "SubagentDeliveredPayload",
    # 注册表
    "METHOD_MODELS",
]
