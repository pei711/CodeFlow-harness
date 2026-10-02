// CodeFlow TUI RPC 面向消费者的统一公开入口。
//
// ui-tui 消费者应统一从 `./rpc` 导入：
//   import { RpcClient, SessionNotFoundError, type TurnEvent } from './rpc';

export { RpcClient } from './client.js'
export type { RpcClientOptions } from './client.js'
export { SubscriptionRegistry } from './subscriptions.js'
export {
  RpcError,
  SessionNotFoundError,
  SessionLockedError,
  TurnInProgressError,
  McpServerNotConnectedError,
  McpToolCallFailedError,
  SkillNotFoundError,
  SkillPinConflictError,
  ModelNotAvailableError,
  ModelSwitchInTurnError,
  ConfigFieldReadonlyError,
  ConfigValidationError,
  NotSupportedInV01Error,
  CliCommandFailedError,
  CliCommandTimeoutError,
  NotDispatchCompatibleError,
  rpcErrorFromFrame
} from './errors.js'
export * from './generated.js'
