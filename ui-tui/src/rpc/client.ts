// CodeFlow TUI RPC——生产用 JSON-RPC 2.0 客户端。
//
// 传输使用跨平台 TCP 回环，因为 Windows 没有可用 AF_UNIX。Python 父进程监听
// 127.0.0.1:<临时端口>；Node 子进程通过 `net.createConnection({host, port})` 连接，首行发送
// CODEFLOW_RPC_TOKEN（父进程在分发前校验），随后传输 JSON 帧。旧配置仍接受 Unix 套接字路径。
// 不采用裸文件描述符继承 pass_fds=(3,4)，因为 Node 无法可靠包装继承的管道文件描述符。
//
// 帧格式为换行分隔的 UTF-8 JSON（规范第 2.5 节）。每帧为 `JSON.stringify(obj) + '\n'`，
// 单帧上限 1 MiB。
//
// 写入通过 `writeQueue` 串行化（单写者模型），使并发 `rpc()` / `subscribe()` 调用绝不交错字节。
//
// 错误映射：传入 JSON-RPC 错误帧通过 `errors.ts::rpcErrorFromFrame` 转为带类型的
// `RpcError` 子类。

import type { Socket } from 'node:net'

import { createConnection } from 'node:net'

import type { EventNotificationParams, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse } from './generated.js'

import { rpcErrorFromFrame } from './errors.js'
import { isJsonRpcError } from './generated.js'
import { SubscriptionRegistry } from './subscriptions.js'

const MAX_FRAME_BYTES = 1024 * 1024 // 1 MiB（规范第 2.5 节）

type Pending = {
  resolve: (value: unknown) => void
  reject: (err: Error) => void
}

export interface RpcClientOptions {
  /** RPC 目标："host:port"（TCP 回环）或 Unix 套接字路径，默认读取 `CODEFLOW_RPC_SOCKET`。 */
  socketPath?: string
  /** 用于非致命协议异常的可选日志器，默认写入 stderr。 */
  warn?: (msg: string) => void
  /**
   * 接收服务端主动发起且方法不是 `event`（订阅流信封）的通知。确认往返
   * `confirm.request` 会通过此路径到达，它是一等顶层方法，而不是每个订阅的
   * 流事件。省略时，此类通知会按未知通知记录并丢弃。
   */
  onNotification?: (method: string, params: unknown) => void
}

export class RpcClient {
  private readonly socket: Socket
  private readonly registry = new SubscriptionRegistry()
  private readonly pending = new Map<number, Pending>()
  private readonly warn: (msg: string) => void
  private readonly onNotification?: (method: string, params: unknown) => void

  private nextId = 1
  private readBuffer = ''
  private writeQueue: Promise<void> = Promise.resolve()
  private closed = false
  private connected = false
  private readonly connectPromise: Promise<void>

  constructor(opts: RpcClientOptions = {}) {
    const target = opts.socketPath ?? process.env.CODEFLOW_RPC_SOCKET
    if (!target) {
      throw new Error('RpcClient: no RPC target supplied; pass `socketPath` or set ' + 'CODEFLOW_RPC_SOCKET env var.')
    }
    this.warn = opts.warn ?? (m => process.stderr.write(`[rpc-client] ${m}\n`))
    this.onNotification = opts.onNotification

    // 跨平台传输：父进程导出 TCP 回环 "host:port"（当前 Python 父进程，Windows 也支持），
    // 旧配置则可导出 Unix 套接字路径。尾部 ":<数字>" 用于识别 TCP。
    const tcp = /^(.+):(\d+)$/.exec(target)
    if (tcp) {
      this.socket = createConnection({ host: tcp[1], port: Number(tcp[2]) })
    } else {
      this.socket = createConnection(target)
    }
    this.socket.setEncoding('utf-8')

    // 父进程在所有帧前校验首行共享密钥；任何本地进程都能访问回环端口，因此需要该门控。
    const authToken = process.env.CODEFLOW_RPC_TOKEN

    this.connectPromise = new Promise<void>((resolve, reject) => {
      const onConnect = () => {
        this.connected = true
        this.socket.off('error', onError)
        // 必须是线路上的首批字节，先于所有 RPC 帧。
        if (authToken) {
          this.socket.write(authToken + '\n')
        }
        resolve()
      }
      const onError = (err: Error) => {
        this.socket.off('connect', onConnect)
        reject(err)
      }
      this.socket.once('connect', onConnect)
      this.socket.once('error', onError)
    })

    this.socket.on('data', (chunk: string | Buffer) => {
      const text = typeof chunk === 'string' ? chunk : chunk.toString('utf-8')
      this.readBuffer += text
      if (this.readBuffer.length > MAX_FRAME_BYTES * 2) {
        // 防御性处理：对端持续发送无换行数据时中止，避免内存溢出。
        this.warn(
          `incoming read buffer exceeded ${MAX_FRAME_BYTES * 2} bytes without ` + 'newline — closing connection'
        )
        this.failAll(new Error('rpc-client: frame size limit exceeded'))
        this.socket.destroy()
        return
      }
      this.drainBuffer()
    })
    this.socket.on('end', () => this.failAll(new Error('socket closed by peer')))
    this.socket.on('error', err => this.failAll(err))
  }

  /** 套接字连接建立后完成的可等待句柄。 */
  ready(): Promise<void> {
    return this.connectPromise
  }

  private drainBuffer(): void {
    let nl = this.readBuffer.indexOf('\n')
    while (nl !== -1) {
      const line = this.readBuffer.slice(0, nl).trim()
      this.readBuffer = this.readBuffer.slice(nl + 1)
      if (line.length > 0) {
        this.handleFrame(line)
      }
      nl = this.readBuffer.indexOf('\n')
    }
  }

  private handleFrame(line: string): void {
    if (line.length > MAX_FRAME_BYTES) {
      this.warn(`oversized frame (${line.length} bytes) dropped`)
      return
    }
    let frame: unknown
    try {
      frame = JSON.parse(line)
    } catch {
      this.warn(`malformed frame ignored: ${line.slice(0, 120)}`)
      return
    }
    if (!frame || typeof frame !== 'object') {
      this.warn('non-object frame ignored')
      return
    }
    const obj = frame as Record<string, unknown>

    // 通知帧：没有 `id`，包含 `method`。
    if (obj.id === undefined && typeof obj.method === 'string') {
      if (obj.method === 'event') {
        const params = obj.params as EventNotificationParams<unknown> | undefined
        if (params && typeof params.subscription_id === 'string') {
          this.registry.dispatch(params)
        } else {
          this.warn('event notification missing subscription_id/event')
        }
      } else if (this.onNotification) {
        // confirm.request 等一等顶层通知不是订阅流事件，应交给使用方出口。
        this.onNotification(obj.method, obj.params)
      } else {
        this.warn(`unknown notification method: ${obj.method}`)
      }
      return
    }

    // 响应帧：包含 `id`。
    const resp = frame as JsonRpcResponse<unknown>
    const id = resp.id
    if (typeof id !== 'number' && typeof id !== 'string') {
      this.warn('response frame has no valid id')
      return
    }
    const idKey = typeof id === 'number' ? id : Number(id)
    const pending = this.pending.get(idKey)
    if (!pending) {
      this.warn(`response for unknown id ${String(id)}`)
      return
    }
    this.pending.delete(idKey)
    if (isJsonRpcError(resp)) {
      pending.reject(rpcErrorFromFrame((resp as JsonRpcErrorResponse).error))
    } else {
      pending.resolve(resp.result)
    }
  }

  private failAll(err: Error): void {
    if (this.closed) {
      return
    }
    this.closed = true
    for (const [, p] of this.pending) {
      p.reject(err)
    }
    this.pending.clear()
    this.registry.clear()
  }

  private async writeFrame(frame: string): Promise<void> {
    if (frame.length > MAX_FRAME_BYTES) {
      throw new Error(`rpc-client: outgoing frame ${frame.length} bytes exceeds ${MAX_FRAME_BYTES} limit`)
    }
    // 串行化全部写入。即使套接字本身支持并发写入，也不能让两帧在线路上交错。
    const prev = this.writeQueue
    this.writeQueue = (async () => {
      await prev
      if (!this.connected) {
        await this.connectPromise
      }
      await new Promise<void>((resolve, reject) => {
        this.socket.write(frame, err => (err ? reject(err) : resolve()))
      })
    })()
    return this.writeQueue
  }

  /** 调用 JSON-RPC 方法并等待带类型的结果。 */
  async rpc<R = unknown, P = unknown>(method: string, params: P): Promise<R> {
    if (this.closed) {
      throw new Error('rpc-client: closed')
    }
    const id = this.nextId++
    const req: JsonRpcRequest<P> = { jsonrpc: '2.0', id, method, params }
    const frame = JSON.stringify(req) + '\n'
    const result = new Promise<R>((resolve, reject) => {
      this.pending.set(id, {
        resolve: v => resolve(v as R),
        reject
      })
    })
    try {
      await this.writeFrame(frame)
    } catch (err) {
      this.pending.delete(id)
      throw err
    }
    return result
  }

  /**
   * 订阅服务端推送流，例如 `turn.subscribe`。
   *
   * 服务端返回 `{subscription_id}`；本方法为该标识注册处理器，并返回
   * `unsubscribe()` 延迟函数。若提供 `unsubscribeMethod`，该函数会调用配对的
   * 服务端方法，同时在本地解除处理器。
   */
  async subscribe<E = unknown, P = unknown, R extends { subscription_id: string } = { subscription_id: string }>(
    method: string,
    params: P,
    handler: (event: E) => void,
    opts: { unsubscribeMethod?: string } = {}
  ): Promise<{ subscription_id: string; unsubscribe: () => Promise<void> }> {
    const result = await this.rpc<R, P>(method, params)
    const subscriptionId = result.subscription_id
    this.registry.register<E>(subscriptionId, handler)
    const unsubscribeMethod = opts.unsubscribeMethod
    const unsubscribe = async (): Promise<void> => {
      this.registry.unregister(subscriptionId)
      if (unsubscribeMethod && !this.closed) {
        await this.rpc<unknown, { subscription_id: string }>(unsubscribeMethod, {
          subscription_id: subscriptionId
        })
      }
    }
    return { subscription_id: subscriptionId, unsubscribe }
  }

  /** 待处理请求数量，主要供测试使用。 */
  pendingCount(): number {
    return this.pending.size
  }

  /** 活动订阅数量，主要供测试使用。 */
  subscriptionCount(): number {
    return this.registry.size()
  }

  /** 拆除套接字并拒绝所有待处理的 Promise。 */
  close(): void {
    this.failAll(new Error('rpc-client: closed by caller'))
    try {
      this.socket.end()
    } catch {
      /* noop */
    }
    try {
      this.socket.destroy()
    } catch {
      /* noop */
    }
  }
}
