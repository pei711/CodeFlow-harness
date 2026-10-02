// CodeFlow TUI RPC 服务端推送订阅注册表。
//
// 服务端发送 `method: "event"` 的 JSON-RPC 通知帧（规范 §2.4），参数为
// `{ subscription_id, event }`。注册表将 subscription_id 映射到处理器，使
// `RpcClient` 能把每一帧路由到正确的消费者。
//
// 未知 subscription_id 不会使客户端崩溃，只会向 stderr 记录警告并丢弃该帧。

import type { EventNotificationParams } from './generated.js'

type AnyHandler = (event: unknown) => void

export class SubscriptionRegistry {
  private readonly handlers = new Map<string, AnyHandler>()

  /** 为服务端返回的 subscription_id 注册处理器。 */
  register<E>(subscriptionId: string, handler: (event: E) => void): void {
    this.handlers.set(subscriptionId, handler as AnyHandler)
  }

  /** 删除处理器；此后该标识的通知会被丢弃。 */
  unregister(subscriptionId: string): boolean {
    return this.handlers.delete(subscriptionId)
  }

  /** 返回当前是否已为该标识注册处理器。 */
  has(subscriptionId: string): boolean {
    return this.handlers.has(subscriptionId)
  }

  /** 活动订阅数量，主要供测试使用。 */
  size(): number {
    return this.handlers.size
  }

  /**
   * 将入站事件通知分派给已注册的处理器。调用了处理器时返回 true，否则返回
   * false。遇到未知 subscription_id 时向 stderr 发出警告，但不会抛出异常。
   */
  dispatch(params: EventNotificationParams<unknown>): boolean {
    const handler = this.handlers.get(params.subscription_id)
    if (!handler) {
      process.stderr.write(`[rpc-subscriptions] event for unknown subscription_id=${params.subscription_id} dropped\n`)
      return false
    }
    try {
      handler(params.event)
    } catch (err) {
      // 处理器崩溃不能破坏读取循环。
      process.stderr.write(`[rpc-subscriptions] handler for ${params.subscription_id} threw: ${String(err)}\n`)
    }
    return true
  }

  /** 清除全部注册项，供 `RpcClient.close()` 使用。 */
  clear(): void {
    this.handlers.clear()
  }
}
