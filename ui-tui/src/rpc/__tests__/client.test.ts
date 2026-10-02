// CodeFlow TUI RPC client.ts 针对模拟 Unix 套接字服务端的集成测试。覆盖正常路径、
// 并行请求、带类型错误映射、突然断开连接、帧大小限制以及代码生成类型基线。

import type { Server, Socket } from 'node:net'

import { mkdtempSync, rmSync } from 'node:fs'
import { createServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect, beforeEach, afterEach } from 'vitest'

import type { SystemHelloResult, TerminalResizeParams, TerminalResizeResult, TurnSendParams } from '../generated.js'

import { RpcClient } from '../client.js'
import { SessionNotFoundError, ConfigValidationError, RpcError } from '../errors.js'

// --------------------------------------------------------------------------
// 模拟服务端使用逐行 JSON 并由处理器驱动。每个连接的帧流入 `onFrame`，后者可
// 写回零个或多个响应帧。
// --------------------------------------------------------------------------

type Frame = Record<string, unknown>

interface MockServer {
  socketPath: string
  server: Server
  /** 向当前已连接客户端推送响应或通知帧。 */
  send: (frame: Frame) => void
  /** 替换逐帧处理器。 */
  setHandler: (h: (frame: Frame) => void) => void
  /** 强制关闭活动连接，以模拟服务端崩溃。 */
  killConnection: () => void
  close: () => Promise<void>
}

function startMock(): Promise<MockServer> {
  return new Promise((resolve, reject) => {
    const dir = mkdtempSync(join(tmpdir(), 'eve-rpc-test-'))
    const socketPath = join(dir, 'sock')
    let activeSocket: Socket | null = null
    let handler: (frame: Frame) => void = () => {}
    let readBuf = ''

    const server = createServer(socket => {
      activeSocket = socket
      socket.setEncoding('utf-8')
      socket.on('data', (chunk: string | Buffer) => {
        readBuf += typeof chunk === 'string' ? chunk : chunk.toString('utf-8')
        let nl = readBuf.indexOf('\n')
        while (nl !== -1) {
          const line = readBuf.slice(0, nl).trim()
          readBuf = readBuf.slice(nl + 1)
          if (line.length > 0) {
            try {
              handler(JSON.parse(line))
            } catch (e) {
              // 吞掉异常，由测试断言结果。
              void e
            }
          }
          nl = readBuf.indexOf('\n')
        }
      })
      socket.on('error', () => {
        activeSocket = null
      })
      socket.on('close', () => {
        activeSocket = null
      })
    })

    server.listen(socketPath, () => {
      resolve({
        socketPath,
        server,
        send: frame => {
          if (!activeSocket) {
            throw new Error('no active connection')
          }
          activeSocket.write(JSON.stringify(frame) + '\n')
        },
        setHandler: h => {
          handler = h
        },
        killConnection: () => {
          if (activeSocket) {
            activeSocket.destroy()
            activeSocket = null
          }
        },
        close: () =>
          new Promise<void>(res => {
            if (activeSocket) {
              activeSocket.destroy()
            }
            server.close(() => {
              rmSync(dir, { recursive: true, force: true })
              res()
            })
          })
      })
    })
    server.on('error', reject)
  })
}

// --------------------------------------------------------------------------
// 测试。
// --------------------------------------------------------------------------

describe('RpcClient', () => {
  let mock: MockServer
  let client: RpcClient

  beforeEach(async () => {
    mock = await startMock()
  })
  afterEach(async () => {
    if (client) {
      client.close()
    }
    await mock.close()
  })

  it('completes a happy-path request (system.hello echo)', async () => {
    mock.setHandler(frame => {
      expect(frame.method).toBe('system.hello')
      expect(frame.jsonrpc).toBe('2.0')
      mock.send({
        jsonrpc: '2.0',
        id: frame.id as number,
        result: {
          server_version: '0.1.0',
          server_capabilities: ['sessions'],
          session: { default_channel: 'tui', default_session_key: 'tui:default' }
        }
      })
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    const result = await client.rpc<SystemHelloResult>('system.hello', {
      client_version: '0.0.1'
    })
    expect(result.server_version).toBe('0.1.0')
    expect(result.session.default_channel).toBe('tui')
  })

  it('routes parallel responses to the correct caller by id', async () => {
    // 缓冲入站请求并乱序响应，以验证标识匹配。
    const seen: Frame[] = []
    mock.setHandler(frame => {
      seen.push(frame)
      if (seen.length === 3) {
        // 以相反顺序响应。
        for (let i = 2; i >= 0; i--) {
          mock.send({
            jsonrpc: '2.0',
            id: seen[i].id as number,
            result: { pong: true, server_time_ms: (i + 1) * 100 }
          })
        }
      }
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    const [a, b, c] = await Promise.all([
      client.rpc<{ server_time_ms: number }>('system.ping', {}),
      client.rpc<{ server_time_ms: number }>('system.ping', {}),
      client.rpc<{ server_time_ms: number }>('system.ping', {})
    ])
    expect(a.server_time_ms).toBe(100)
    expect(b.server_time_ms).toBe(200)
    expect(c.server_time_ms).toBe(300)
  })

  it('maps -32001 errors to SessionNotFoundError', async () => {
    mock.setHandler(frame => {
      mock.send({
        jsonrpc: '2.0',
        id: frame.id as number,
        error: {
          code: -32001,
          message: 'session_not_found',
          data: { session_key: 'cli:nope' }
        }
      })
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    await expect(client.rpc('session.resume', { session_id: 'tui:nope' })).rejects.toBeInstanceOf(SessionNotFoundError)
  })

  it('maps -32011 to ConfigValidationError and preserves code+data', async () => {
    mock.setHandler(frame => {
      mock.send({
        jsonrpc: '2.0',
        id: frame.id as number,
        error: { code: -32011, message: 'config_validation_error', data: { field: 'channel' } }
      })
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    try {
      await client.rpc('config.set', { key: '', value: 'x' })
      throw new Error('should have rejected')
    } catch (err) {
      expect(err).toBeInstanceOf(ConfigValidationError)
      expect(err).toBeInstanceOf(RpcError)
      const re = err as RpcError
      expect(re.code).toBe(-32011)
      expect((re.data as { field: string }).field).toBe('channel')
    }
  })

  it('rejects all pending promises when peer disconnects mid-flight', async () => {
    // 捕获请求但永不响应，随后断开连接。
    mock.setHandler(() => {
      setTimeout(() => mock.killConnection(), 10)
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    const p1 = client.rpc('system.ping', {})
    const p2 = client.rpc('system.ping', {})
    await expect(p1).rejects.toThrow(/socket closed|rpc-client/)
    await expect(p2).rejects.toThrow(/socket closed|rpc-client/)
  })

  it('refuses to write frames over 1 MiB', async () => {
    mock.setHandler(() => {
      // 不应执行到这里。
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    await client.ready()
    const huge = 'x'.repeat(1024 * 1024 + 100) // JSON 序列化后大于 1 MiB。
    await expect(client.rpc('turn.send', { content: huge, session_key: 'tui:test' } as TurnSendParams)).rejects.toThrow(
      /frame.*exceeds/
    )
  })

  it('typecheck: retained method types are reachable from generated.ts', async () => {
    // 此测试主要用于编译期 `tsc` 检查；若代码生成损坏，导入或字段访问会编译失败。
    mock.setHandler(frame => {
      const params = frame.params as TerminalResizeParams
      expect(typeof params.cols).toBe('number')
      const result: TerminalResizeResult = { ok: true }
      mock.send({ jsonrpc: '2.0', id: frame.id as number, result })
    })
    client = new RpcClient({ socketPath: mock.socketPath })
    const out = await client.rpc<TerminalResizeResult, TerminalResizeParams>('terminal.resize', {
      cols: 80,
      rows: 24
    })
    expect(out.ok).toBe(true)
  })
})
