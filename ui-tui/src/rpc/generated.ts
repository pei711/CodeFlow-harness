// AUTO-GENERATED — DO NOT EDIT — run `npm run gen:rpc`
//
// Source of truth: ui-tui/rpc-schema/openrpc.json (OpenRPC 1.2.6).
// Regenerate via: cd ui-tui && npm run gen:rpc
// Lint (drift check) via: cd ui-tui && npm run lint:rpc
//
// CodeFlow method-scoped types + components/schemas + JSON-RPC 2.0 envelopes.

/* eslint-disable */
/* tslint:disable */

/**
 * Any valid JSON value.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "JsonValue".
 */
export type JsonValue = string | number | boolean | {} | unknown[] | null;
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnEvent".
 */
export type TurnEvent =
  | MessageStartEvent
  | TokenDeltaEvent
  | ThinkingDeltaEvent
  | ToolStartEvent
  | ToolProgressEvent
  | ToolCompleteEvent
  | MessageCompleteEvent
  | ErrorEvent
  | CronDeliveredEvent
  | SubagentDeliveredEvent;

/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ClarifyRespondResult".
 */
export interface ClarifyRespondResult {
  ok: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigGetResult".
 */
export interface ConfigGetResult {
  config: {
    [k: string]: JsonValue;
  };
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigSetResult".
 */
export interface ConfigSetResult {
  applied: boolean;
  previous: JsonValue;
  value?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ConfirmRespondResult".
 */
export interface ConfirmRespondResult {
  ok: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "CronDeliveredEvent".
 */
export interface CronDeliveredEvent {
  type: 'cron.delivered';
  payload: CronDeliveredPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "CronDeliveredPayload".
 */
export interface CronDeliveredPayload {
  job_id: string;
  name: string;
  text: string;
  fired_at: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ErrorEvent".
 */
export interface ErrorEvent {
  type: 'error';
  payload: ErrorEventPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ErrorEventPayload".
 */
export interface ErrorEventPayload {
  attachments_discarded?: boolean;
  code: number;
  message: string;
  reason?: 'cancelled_by_client' | 'internal';
  submission_id?: string;
  turn_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ImageAttachResult".
 */
export interface ImageAttachResult {
  name: string;
  remainder: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "MessageCompleteEvent".
 */
export interface MessageCompleteEvent {
  type: 'message.complete';
  payload: MessageCompletePayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "MessageCompletePayload".
 */
export interface MessageCompletePayload {
  submission_id: string;
  turn_id: string;
  usage: UsageSnapshot;
}
/**
 * Token / cost usage reported at the end of a turn.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "UsageSnapshot".
 */
export interface UsageSnapshot {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost_usd?: number;
  context_used?: number;
  context_max?: number;
  context_percent?: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "MessageStartEvent".
 */
export interface MessageStartEvent {
  type: 'message.start';
  payload: MessageStartPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "MessageStartPayload".
 */
export interface MessageStartPayload {
  submission_id: string;
  turn_id: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelAddModelResult".
 */
export interface ModelAddModelResult {
  provider: ModelOptionProvider;
}
/**
 * One provider row in the ``/model`` picker.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOptionProvider".
 */
export interface ModelOptionProvider {
  slug: string;
  name: string;
  authenticated: boolean;
  is_current: boolean;
  auth_type: string;
  key_env?: string;
  models: string[];
  total_models: number;
  needs_api_base: boolean;
  warning: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelDisconnectResult".
 */
export interface ModelDisconnectResult {
  disconnected: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOptionsResult".
 */
export interface ModelOptionsResult {
  model: string;
  provider: string;
  providers: ModelOptionProvider[];
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelRemoveModelResult".
 */
export interface ModelRemoveModelResult {
  provider: ModelOptionProvider;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelSaveKeyResult".
 */
export interface ModelSaveKeyResult {
  provider: ModelOptionProvider;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionBranchResult".
 */
export interface SessionBranchResult {
  session_id?: string;
  title?: string;
  message_count?: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionClearResult".
 */
export interface SessionClearResult {
  /**
   * The same session_key (no new id minted).
   */
  session_id: string;
  /**
   * True when the in-place wipe ran.
   */
  cleared: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCloseResult".
 */
export interface SessionCloseResult {
  ok: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCreateResult".
 */
export interface SessionCreateResult {
  session_id: string;
  info: SessionInfo;
}
/**
 * The init bundle rendered by the CodeFlow TUI Session panel.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionInfo".
 */
export interface SessionInfo {
  model: string;
  skills: {
    [k: string]: string[];
  };
  tools: {
    [k: string]: string[];
  };
  provider?: string;
  memory?: string;
  context_window?: number;
  lazy?: boolean;
  usage?: {
    [k: string]: JsonValue;
  };
  version?: string;
  cwd?: string;
  mcp_servers?: {
    [k: string]: JsonValue;
  }[];
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionDeleteResult".
 */
export interface SessionDeleteResult {
  /**
   * True when the user declined the TUI confirmation.
   */
  cancelled?: boolean;
  deleted?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionExportResult".
 */
export interface SessionExportResult {
  /**
   * True when a verified portable artifact was written.
   */
  exported: boolean;
  path?: string;
  /**
   * Failure reason: not_found | ambiguous | write_failed | verification_failed.
   */
  reason?: string;
  candidates?: string[];
}
/**
 * One row in the session picker (gatewayTypes.ts:130 SessionListItem).
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionListItem".
 */
export interface SessionListItem {
  /**
   * Full session_key: <channel>:<chat_id>.
   */
  id: string;
  message_count: number;
  preview: string;
  source?: string;
  /**
   * Unix timestamp from created_at.
   */
  started_at: number;
  title: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionListResult".
 */
export interface SessionListResult {
  sessions: SessionListItem[];
}
/**
 * A transcript message returned by ``session.resume``.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionMessage".
 */
export interface SessionMessage {
  role: 'user' | 'assistant' | 'system' | 'tool';
  text?: string;
  context?: string;
  name?: string;
}
/**
 * Response shape per gatewayTypes.ts:147 SessionMostRecentResponse.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionMostRecentResult".
 */
export interface SessionMostRecentResult {
  session_id?: string;
  source?: string;
  started_at?: number;
  title?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionResumeResult".
 */
export interface SessionResumeResult {
  session_id: string;
  info: SessionInfo;
  messages: SessionMessage[];
}
/**
 * Response per gatewayTypes.ts:154 SessionTitleResponse.
 *
 * pending=True means the title is held in memory for a lazy (never-saved)
 * session and lands with the session's first save.
 *
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionTitleResult".
 */
export interface SessionTitleResult {
  title?: string;
  session_key: string;
  pending: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionUndoResult".
 */
export interface SessionUndoResult {
  /**
   * Messages dropped (0 = nothing to undo).
   */
  removed: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SetupStatusResult".
 */
export interface SetupStatusResult {
  provider_configured: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SubagentDeliveredEvent".
 */
export interface SubagentDeliveredEvent {
  type: 'subagent.delivered';
  payload: SubagentDeliveredPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SubagentDeliveredPayload".
 */
export interface SubagentDeliveredPayload {
  text: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemHelloResult".
 */
export interface SystemHelloResult {
  server_version: string;
  server_capabilities: string[];
  session: SystemHelloSession;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemHelloSession".
 */
export interface SystemHelloSession {
  default_channel: 'tui';
  default_session_key: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemPingResult".
 */
export interface SystemPingResult {
  pong: true;
  server_time_ms: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemVersionResult".
 */
export interface SystemVersionResult {
  server_version: string;
  /**
   * OpenRPC info.version mirrored back to client.
   */
  schema_version: string;
  codeflow_version: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TerminalResizeResult".
 */
export interface TerminalResizeResult {
  ok: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ThinkingDeltaEvent".
 */
export interface ThinkingDeltaEvent {
  type: 'thinking.delta';
  payload: ThinkingDeltaPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ThinkingDeltaPayload".
 */
export interface ThinkingDeltaPayload {
  text: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TokenDeltaEvent".
 */
export interface TokenDeltaEvent {
  type: 'token.delta';
  payload: TokenDeltaPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TokenDeltaPayload".
 */
export interface TokenDeltaPayload {
  text: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ToolCompleteEvent".
 */
export interface ToolCompleteEvent {
  type: 'tool.complete';
  payload: ToolCompletePayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ToolCompletePayload".
 */
export interface ToolCompletePayload {
  tool_call_id: string;
  result_preview: string;
  failed?: boolean;
  truncated: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ToolProgressEvent".
 */
export interface ToolProgressEvent {
  type: 'tool.progress';
  payload: ToolProgressPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ToolProgressPayload".
 */
export interface ToolProgressPayload {
  tool_call_id: string;
  preview: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ToolStartEvent".
 */
export interface ToolStartEvent {
  type: 'tool.start';
  payload: ToolStartPayload;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ToolStartPayload".
 */
export interface ToolStartPayload {
  tool_call_id: string;
  name: string;
  arguments: {
    [k: string]: JsonValue;
  };
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnCancelResult".
 */
export interface TurnCancelResult {
  cancelled: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSendResult".
 */
export interface TurnSendResult {
  turn_id: string;
  accepted: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSubscribeResult".
 */
export interface TurnSubscribeResult {
  subscription_id: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnUnsubscribeResult".
 */
export interface TurnUnsubscribeResult {
  unsubscribed: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionListParams".
 */
export interface SessionListParams {
  limit?: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCreateParams".
 */
export interface SessionCreateParams {
  cols?: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCloseParams".
 */
export interface SessionCloseParams {
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionResumeParams".
 */
export interface SessionResumeParams {
  session_id: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionDeleteParams".
 */
export interface SessionDeleteParams {
  /**
   * Full session_key as sent by the UI.
   */
  session_id: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionMostRecentParams".
 */
export interface SessionMostRecentParams {}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionTitleParams".
 */
export interface SessionTitleParams {
  /**
   * Full session_key.
   */
  session_id: string;
  title?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionClearParams".
 */
export interface SessionClearParams {
  /**
   * Full session_key to clear.
   */
  session_id: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionUndoParams".
 */
export interface SessionUndoParams {
  /**
   * Full session_key to undo.
   */
  session_id: string;
  /**
   * Trailing turns to drop (role==user boundary).
   */
  n?: number;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionBranchParams".
 */
export interface SessionBranchParams {
  session_id: string;
  name?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SessionExportParams".
 */
export interface SessionExportParams {
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSendParams".
 */
export interface TurnSendParams {
  session_key: string;
  content: string;
  submission_id?: string;
  channel?: string;
  chat_id?: string;
  sender_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSubscribeParams".
 */
export interface TurnSubscribeParams {
  session_key: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnUnsubscribeParams".
 */
export interface TurnUnsubscribeParams {
  subscription_id: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TurnCancelParams".
 */
export interface TurnCancelParams {
  session_key: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ImageAttachParams".
 */
export interface ImageAttachParams {
  session_id: string;
  path: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOptionsParams".
 */
export interface ModelOptionsParams {
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelSaveKeyParams".
 */
export interface ModelSaveKeyParams {
  slug: string;
  api_key: string;
  api_base?: string;
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelDisconnectParams".
 */
export interface ModelDisconnectParams {
  slug: string;
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelAddModelParams".
 */
export interface ModelAddModelParams {
  slug: string;
  model: string;
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ModelRemoveModelParams".
 */
export interface ModelRemoveModelParams {
  slug: string;
  model: string;
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigGetParams".
 */
export interface ConfigGetParams {
  keys?: string[];
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigSetParams".
 */
export interface ConfigSetParams {
  key: string;
  value: JsonValue;
  provider?: string;
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemHelloParams".
 */
export interface SystemHelloParams {
  client_version: string;
  client_capabilities?: string[];
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemPingParams".
 */
export interface SystemPingParams {}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SystemVersionParams".
 */
export interface SystemVersionParams {}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "SetupStatusParams".
 */
export interface SetupStatusParams {}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "TerminalResizeParams".
 */
export interface TerminalResizeParams {
  cols?: number;
  rows?: number;
  session_id?: string;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ConfirmRespondParams".
 */
export interface ConfirmRespondParams {
  request_id: string;
  answer: boolean;
}
/**
 * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
 * via the `definition` "ClarifyRespondParams".
 */
export interface ClarifyRespondParams {
  request_id?: string;
  conversation_id?: string;
  answer: string;
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 envelope (specs/tui-ipc.md §2.1/2.2/2.3/2.4)
// ---------------------------------------------------------------------------

export interface JsonRpcRequest<P = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  method: string;
  params: P;
}

export interface JsonRpcSuccess<R = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  result: R;
}

export interface JsonRpcErrorObject {
  code: number;
  message: string;
  data?: unknown;
}

export interface JsonRpcErrorResponse {
  jsonrpc: '2.0';
  id: string | number;
  error: JsonRpcErrorObject;
}

export type JsonRpcResponse<R = unknown> = JsonRpcSuccess<R> | JsonRpcErrorResponse;

export interface JsonRpcNotification<P = unknown> {
  jsonrpc: '2.0';
  method: string;
  params: P;
}

export interface EventNotificationParams<E = unknown> {
  subscription_id: string;
  event: E;
}

export function isJsonRpcError<R>(
  resp: JsonRpcResponse<R>,
): resp is JsonRpcErrorResponse {
  return (resp as JsonRpcErrorResponse).error !== undefined;
}
