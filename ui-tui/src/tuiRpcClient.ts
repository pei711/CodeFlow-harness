// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// 基于 Python 所有的 JSON-RPC 套接字实现生产环境 TUI 客户端。

import { EventEmitter } from 'node:events'

import type { GatewayEvent } from './gatewayTypes.js'

import { RpcClient } from './rpc/index.js'

const CLIENT_VERSION = '0.0.2'
// 与 `system.hello` 期间发送的协商能力列表保持一致。服务端尚未强制校验，
// 但发送规范集合可为后续能力协商提供稳定基线。
const CLIENT_CAPABILITIES: string[] = ['chat', 'config', 'confirm', 'attachments', 'sessions']

export interface TuiRpcClientOptions {
  /** Unix 套接字路径，默认读取环境变量 `CODEFLOW_RPC_SOCKET`。 */
  socketPath?: string
  /** 注入的 RpcClient（测试接缝）；传入后由其自行管理生命周期。 */
  rpcClient?: RpcClient
}

export class TuiRpcClient extends EventEmitter {
  // 聊天流通过同一客户端订阅，服务端通知则流经 EventEmitter 适配器。共享客户端
  // 可将套接字数量维持为一个，并避免握手竞争。
  public readonly rpcClient: RpcClient
  private bufferedEvents: GatewayEvent[] = []
  private subscribed = false
  private startPromise: Promise<void> | null = null
  private killed = false

  constructor(opts: TuiRpcClientOptions = {}) {
    super()
    this.setMaxListeners(0)
    this.rpcClient =
      opts.rpcClient ??
      new RpcClient({
        // 将一等顶层通知（confirm.request 和 clarify.request）桥接到
        // `createGatewayEventHandler` 消费的 `'event'` 总线；订阅流中的 `event`
        // 通知仍通过带类型的注册表流转。
        onNotification: (method, params) => this.publishServerNotification(method, params),
        socketPath: opts.socketPath
      })
  }

  /**
   * 将顶层服务端通知（`{method, params}`）映射为 `GatewayEvent`
   * （`{type, payload}`）并发布。`createGatewayEventHandler` 按 `type` 分派且
   * 忽略未知类型，因此保持向前兼容。
   */
  private publishServerNotification(method: string, params: unknown): void {
    this.publish({ payload: params, type: method } as GatewayEvent)
  }

  /**
   * 启动顺序：
   *   1. 等待 `system.hello`（RpcServer 服务端超时为 5 秒）。
   *   2. 合成传输就绪事件。
   *   3. 缓冲事件；若已调用 `drain()`，则直接发出。
   *
   * 重复调用返回同一个 Promise。
   */
  start(): Promise<void> {
    if (this.startPromise) {
      return this.startPromise
    }

    this.startPromise = (async () => {
      await this.rpcClient.rpc('system.hello', {
        client_version: CLIENT_VERSION,
        client_capabilities: CLIENT_CAPABILITIES
      })
      setTimeout(() => {
        if (this.killed) {
          return
        }

        const ready: GatewayEvent = { payload: { skin: {} }, type: 'gateway.ready' }
        this.publish(ready)
      }, 0)
    })()

    return this.startPromise
  }

  /**
   * 切换为直接投递事件，并重放启动期间缓冲的事件。
   */
  drain(): void {
    this.subscribed = true
    const queued = this.bufferedEvents
    this.bufferedEvents = []

    for (const ev of queued) {
      this.emit('event', ev)
    }
  }

  /** 调用 RPC 方法，直接委托给 `RpcClient.rpc()`。 */
  async request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    return this.rpcClient.rpc<T>(method, params)
  }

  /**
   * 拆除客户端：关闭 RPC 套接字并发出 `exit`，让 `useMainApp.ts` 清理界面
   * 状态。可安全重复调用。
   */
  kill(): void {
    if (this.killed) {
      return
    }

    this.killed = true
    this.bufferedEvents = []
    this.subscribed = false
    try {
      this.rpcClient.close()
    } finally {
      this.emit('exit', 0)
    }
  }

  /**
   * RPC 传输没有可追踪的子进程 stdout 缓冲区；服务端日志由 Python 的
   * `loguru` 写入其自身文件。
   */
  getLogTail(_limit = 20): string {
    return '(RPC server logs are available through the Python loguru sink)'
  }

  private publish(ev: GatewayEvent): void {
    if (this.subscribed) {
      this.emit('event', ev)
    } else {
      this.bufferedEvents.push(ev)
    }
  }
}
