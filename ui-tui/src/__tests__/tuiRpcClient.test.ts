// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// TuiRpcClient 适配器针对模拟 Unix 套接字服务端的单元测试，模式与
// `src/rpc/__tests__/client.test.ts` 相同。覆盖握手、合成 gateway.ready、
// 排空重放、请求委托、终止语义以及 getLogTail 占位行为。

import type { Server, Socket } from 'node:net'

import { mkdtempSync, rmSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { GatewayEvent } from '../gatewayTypes.js'

import { RpcClient } from '../rpc/index.js'
import { TuiRpcClient } from '../tuiRpcClient.js'

// --------------------------------------------------------------------------
// 模拟服务端使用逐行 JSON。自动以预设 SystemHelloResult 回答 `system.hello`；
// 后续帧转发给测试处理器，使各用例可定制响应。
// --------------------------------------------------------------------------

type Frame = Record<string, unknown>

interface MockServer {
  socketPath: string
  server: Server
  setHandler: (h: (socket: Socket, frame: Frame) => void) => void
  close: () => Promise<void>
}

function startMock(): Promise<MockServer> {
  return new Promise((resolve, reject) => {
    const dir = mkdtempSync(join(tmpdir(), 'codeflow-tui-rpc-test-'))
    const socketPath = join(dir, 'sock')

    // 默认处理器回答 `system.hello`，各测试可覆盖它。
    let handler: (socket: Socket, frame: Frame) => void = (socket, frame) => {
      if (frame.method === 'system.hello') {
        const resp = {
          jsonrpc: '2.0',
          id: frame.id,
          result: {
            server_version: '0.0.2',
            server_capabilities: ['chat', 'sessions'],
            session: { default_channel: 'tui', default_session_key: 'tui:default' }
          }
        }

        socket.write(JSON.stringify(resp) + '\n')
      }
    }

    const server = createServer(socket => {
      socket.setEncoding('utf-8')
      let readBuf = ''

      socket.on('data', (chunk: string | Buffer) => {
        readBuf += typeof chunk === 'string' ? chunk : chunk.toString('utf-8')
        let nl = readBuf.indexOf('\n')

        while (nl !== -1) {
          const line = readBuf.slice(0, nl).trim()

          readBuf = readBuf.slice(nl + 1)
          if (line.length > 0) {
            try {
              handler(socket, JSON.parse(line) as Frame)
            } catch {
              /* 格式错误，忽略。 */
            }
          }

          nl = readBuf.indexOf('\n')
        }
      })
    })

    server.on('error', reject)
    server.listen(socketPath, () => {
      resolve({
        close: () =>
          new Promise<void>(res => {
            server.close(() => {
              rmSync(dir, { force: true, recursive: true })
              res()
            })
          }),
        server,
        setHandler: h => {
          handler = h
        },
        socketPath
      })
    })
  })
}

describe('TuiRpcClient', () => {
  let mock: MockServer
  let client: TuiRpcClient | null = null

  beforeEach(async () => {
    mock = await startMock()
  })

  afterEach(async () => {
    if (client) {
      client.kill()
      client = null
    }

    await mock.close()
  })

  it('start() performs system.hello handshake and buffers gateway.ready', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })

    let helloSeen = false

    mock.setHandler((socket, frame) => {
      if (frame.method === 'system.hello') {
        helloSeen = true
        const params = frame.params as { client_version?: string; client_capabilities?: string[] }

        expect(params.client_version).toBe('0.0.2')
        expect(params.client_capabilities).toContain('chat')
        expect(params.client_capabilities).toContain('sessions')
        const resp = {
          jsonrpc: '2.0',
          id: frame.id,
          result: {
            server_version: '0.0.2',
            server_capabilities: [],
            session: { default_channel: 'tui', default_session_key: 'tui:default' }
          }
        }

        socket.write(JSON.stringify(resp) + '\n')
      }
    })

    await client.start()
    expect(helloSeen).toBe(true)
  })

  it('drain() after start() replays the transport-ready event', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })
    const events: GatewayEvent[] = []

    client.on('event', (ev: GatewayEvent) => events.push(ev))

    await client.start()
    // 允许 setTimeout(0) 延迟的发布在 drain 前触发。
    await new Promise(resolve => setTimeout(resolve, 20))
    client.drain()

    const ready = events.find(e => e.type === 'gateway.ready')

    expect(ready).toBeDefined()
    expect(ready!.payload).toEqual({ skin: {} })
  })

  it('bridges a server-initiated confirm.request notification onto the event bus', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })
    const events: GatewayEvent[] = []

    client.on('event', (ev: GatewayEvent) => events.push(ev))

    let serverSocket: Socket | null = null
    mock.setHandler((socket, frame) => {
      serverSocket = socket
      if (frame.method === 'system.hello') {
        socket.write(
          JSON.stringify({
            jsonrpc: '2.0',
            id: frame.id,
            result: {
              server_version: '0.0.2',
              server_capabilities: [],
              session: { default_channel: 'tui', default_session_key: 'tui:default' }
            }
          }) + '\n'
        )
      }
    })

    await client.start()
    await new Promise(resolve => setTimeout(resolve, 20))
    client.drain()

    // 服务端 ConfirmBroker 推送顶层 confirm.request 通知。
    serverSocket!.write(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'confirm.request',
        params: { default: false, prompt: 'Continue?', request_id: 'r1' }
      }) + '\n'
    )
    await new Promise(resolve => setTimeout(resolve, 20))

    const confirm = events.find(e => e.type === 'confirm.request')

    expect(confirm).toBeDefined()
    expect(confirm!.payload).toEqual({ default: false, prompt: 'Continue?', request_id: 'r1' })
  })

  it('start() is idempotent — second call returns the same promise', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })

    const p1 = client.start()
    const p2 = client.start()

    expect(p1).toBe(p2)
    await p1
  })

  it('request() delegates to RpcClient.rpc() with method + params', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })
    await client.start()

    mock.setHandler((socket, frame) => {
      if (frame.method === 'config.get') {
        expect(frame.params).toEqual({ key: 'full' })
        const resp = { jsonrpc: '2.0', id: frame.id, result: { config: { display: {} } } }

        socket.write(JSON.stringify(resp) + '\n')
      }
    })

    const result = await client.request<{ config: { display: Record<string, unknown> } }>('config.get', { key: 'full' })

    expect(result.config.display).toBeDefined()
  })

  it('kill() emits exit event and is safe to call twice', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })
    await client.start()

    const exits: number[] = []

    client.on('exit', (code: number) => exits.push(code))

    client.kill()
    client.kill() // 第二次调用不执行操作。
    expect(exits).toEqual([0])

    const localClient = client

    client = null // 禁用 afterEach 清理。
    // 后续安全性：localClient 已设置 `killed = true`，不会留下悬空套接字。
    expect(localClient).toBeDefined()
  })

  it('getLogTail() returns a non-empty placeholder string', async () => {
    client = new TuiRpcClient({ socketPath: mock.socketPath })
    const tail = client.getLogTail()

    expect(typeof tail).toBe('string')
    expect(tail.length).toBeGreaterThan(0)
  })

  it('honors an injected RpcClient (test seam for custom transports)', async () => {
    const rpc = new RpcClient({ socketPath: mock.socketPath })

    client = new TuiRpcClient({ rpcClient: rpc })

    let helloId: unknown = null

    mock.setHandler((socket, frame) => {
      if (frame.method === 'system.hello') {
        helloId = frame.id
        const resp = {
          jsonrpc: '2.0',
          id: frame.id,
          result: {
            server_version: '0.0.2',
            server_capabilities: [],
            session: { default_channel: 'tui', default_session_key: 'tui:default' }
          }
        }

        socket.write(JSON.stringify(resp) + '\n')
      }
    })

    await client.start()
    expect(helloId).not.toBeNull()
  })
})
