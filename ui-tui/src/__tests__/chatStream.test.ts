// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// createChatStream 的纵深防御测试。Python 端已修复服务端根因，即按轮次取消时
// 错误拆除会话订阅；这些测试保护客户端，确保没有产生终止事件的轮次不再卡死
// 界面：
//   1. watchdog：窗口内没有服务端事件时清除 busy/turnId 并显示错误，由时间驱动
//      恢复；
//   2. forceReset：Ctrl+C 逃生路径使用的本地强制重置，由按键驱动恢复且无需
//      服务端往返。

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { Msg } from '../types.js'

import { createChatStream, type ChatStreamRpcClient } from '../app/chatStream.js'
import { turnController } from '../app/turnController.js'
import { resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import {
  ModelNotAvailableError,
  TurnInProgressError,
  type TurnEvent,
  type TurnSendParams,
  type TurnSendResult
} from '../rpc/index.js'

interface FakeRpc extends ChatStreamRpcClient {
  __pushEvent: (event: TurnEvent) => void
  __pushRawEvent: (event: TurnEvent) => void
  __submissionIds: string[]
}

const deferred = <T>() => {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res, rej) => {
    reject = rej
    resolve = res
  })
  return { promise, reject, resolve }
}

const makeFakeRpc = (sendResult?: TurnSendResult): FakeRpc => {
  let handler: ((event: TurnEvent) => void) | null = null
  const submissionIds: string[] = []
  const fake: FakeRpc = {
    __pushEvent: (event: TurnEvent) => {
      if (handler) {
        const submissionId = submissionIds.at(-1)
        if (
          submissionId &&
          (event.type === 'message.start' || event.type === 'message.complete' || event.type === 'error') &&
          !event.payload.submission_id
        ) {
          handler({
            ...event,
            payload: { ...event.payload, submission_id: submissionId }
          } as TurnEvent)
          return
        }
        handler(event)
      }
    },
    __pushRawEvent: (event: TurnEvent) => {
      handler?.(event)
    },
    __submissionIds: submissionIds,
    async rpc<R, P>(method: string, params: P): Promise<R> {
      if (method === 'turn.send') {
        submissionIds.push((params as TurnSendParams).submission_id!)
        return (sendResult ?? { turn_id: `turn-${submissionIds.length}`, accepted: true }) as unknown as R
      }
      if (method === 'turn.cancel') {
        return { cancelled: true } as unknown as R
      }
      return {} as R
    },
    async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
      handler = h as unknown as (event: TurnEvent) => void
      return {
        subscription_id: 'sub-1',
        unsubscribe: async () => {
          handler = null
        }
      }
    }
  }
  return fake
}

describe('createChatStream — wedge defenses', () => {
  beforeEach(() => {
    resetTurnState()
    resetUiState()
    turnController.fullReset()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('treats an accepted turn.send response as the server ack', async () => {
    vi.useFakeTimers()
    const fake = makeFakeRpc()
    const sysCalls: string[] = []
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m),
      watchdogMs: 5000
    })
    await stream.attach()

    // 模拟提交处理器将界面标记为忙碌。即使执行仍在排队，RPC 接受结果也证明
    // 服务端已接管该轮次。
    patchUiState({ busy: true })
    await stream.send('hello')
    expect(stream.isTurnActive()).toBe(true)

    // 没有事件到达时，确认看门狗也不能回收服务端已接管的轮次。
    vi.advanceTimersByTime(5000)

    expect(stream.isTurnActive()).toBe(true)
    expect(getUiState().busy).toBe(true)
    expect(sysCalls.some(m => /no response/i.test(m))).toBe(false)
  })

  it('does not fire the watchdog when the turn completes normally', async () => {
    vi.useFakeTimers()
    const fake = makeFakeRpc()
    const sysCalls: string[] = []
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m),
      watchdogMs: 5000
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('hi')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'a' } })
    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        turn_id: 'turn-1',
        usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      }
    })

    // 即使远超窗口，已完成轮次也不能触发看门狗。
    vi.advanceTimersByTime(60000)
    expect(sysCalls.some(m => /no response/i.test(m))).toBe(false)
    expect(getUiState().busy).toBe(false)
  })

  it('appends a subagent delivery without binding it to the active user Turn', async () => {
    const appended: Msg[] = []
    const fake = makeFakeRpc()
    const stream = createChatStream({
      appendMessage: message => appended.push(message),
      rpcClient: fake,
      sessionKey: 'tui:default',
      submissionIdFactory: () => 'submission-1'
    })
    await stream.attach()
    await stream.send('user task')

    fake.__pushEvent({
      type: 'subagent.delivered',
      payload: { text: 'delegated result' }
    })

    expect(appended).toEqual([{ role: 'assistant', text: 'delegated result' }])
    expect(stream.isTurnActive()).toBe(true)
    expect(turnController.bufRef).toBe('')

    fake.__pushEvent({
      type: 'message.start',
      payload: { submission_id: 'submission-1', turn_id: 'turn-1' }
    })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'user result' } })
    expect(turnController.bufRef).toBe('user result')
    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        submission_id: 'submission-1',
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })
    expect(stream.isTurnActive()).toBe(false)
  })

  it('forceReset clears visible state while retaining unresolved backend ownership', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('long task')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    patchUiState({ busy: true })
    expect(stream.isTurnActive()).toBe(true)

    stream.forceReset()

    expect(stream.isTurnActive()).toBe(true)
    expect(getUiState().busy).toBe(false)
  })

  it('forceReset clears the armed Ctrl+C escape-hatch flag', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('long task')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    // 模拟首次 Ctrl+C 启用由 useInputHandlers 设置的逃生路径。
    patchUiState({ busy: true, escapeArmed: true })

    stream.forceReset()

    // 提示必须恢复；轮次重置后不能继续保持第二次 Ctrl+C 的启用状态。
    expect(getUiState().escapeArmed).toBe(false)
    expect(getUiState().busy).toBe(false)
  })

  it('does not false-positive when message.start arrives in the same packet as the turn.send accept (arming race)', async () => {
    vi.useFakeTimers()
    // 复现误报：首次提交渲染卡顿时，服务端接受响应与大模型前的 message.start
    // 落在同一网络包内。订阅回调会在 turn.send 内同步触发，早于接受响应完成，
    // 因此此时看门狗必须已启用，之后也不能再次启用。随后模型沉默超过窗口
    // （首 token 较慢），但健康轮次不能被判定为失效。
    let handler: ((event: TurnEvent) => void) | null = null
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, params: P): Promise<R> {
        if (method === 'turn.send') {
          if (handler) {
            handler({
              type: 'message.start',
              payload: {
                submission_id: (params as TurnSendParams).submission_id!,
                turn_id: 'turn-1'
              }
            })
          }
          return { turn_id: 'turn-1', accepted: true } as unknown as R
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const sysCalls: string[] = []
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m),
      watchdogMs: 5000
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('hello')

    vi.advanceTimersByTime(60000)

    expect(sysCalls.some(m => /no response/i.test(m))).toBe(false)
    expect(stream.isTurnActive()).toBe(true)
  })

  it('recovers the input when turn.send hangs and never returns (hung RPC)', async () => {
    vi.useFakeTimers()
    // turn.send 永不完成时，确认看门狗已在 await 前启用，因此挂起的 RPC 仍会
    // 恢复输入，而不是冻结界面。
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          return new Promise<R>(() => {})
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, _h: (event: E) => void) {
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const sysCalls: string[] = []
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m),
      watchdogMs: 5000
    })
    await stream.attach()
    patchUiState({ busy: true })
    void stream.send('hello')

    await vi.advanceTimersByTimeAsync(5000)

    expect(getUiState().busy).toBe(false)
    expect(sysCalls.some(m => /no response/i.test(m))).toBe(true)
  })

  it('detach during a hung turn.send clears the in-flight guard so the next send is accepted', async () => {
    let resumeSend: (() => void) | null = null
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          if (resumeSend === null) {
            // 首次发送会一直挂起到 detach，自身永不完成。
            return new Promise<R>(() => {})
          }
          return { turn_id: 'turn-2', accepted: true } as unknown as R
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, _h: (event: E) => void) {
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    void stream.send('first (hangs)')

    await stream.detach()
    resumeSend = () => {}
    await stream.attach()

    // 挂起发送使 sendInFlight 保持 true；detach 必须清除它，否则这里会抛出
    // 'turn already in progress'。
    await expect(stream.send('second')).resolves.toMatchObject({ accepted: true })
  })

  it('keeps a newer send guarded when an older detached send resolves', async () => {
    vi.useFakeTimers()
    const first = deferred<TurnSendResult>()
    const second = deferred<TurnSendResult>()
    let sendCount = 0
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          sendCount += 1
          return (sendCount === 1 ? first.promise : second.promise) as Promise<R>
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, _handler: (event: E) => void) {
        return { subscription_id: `sub-${sendCount}`, unsubscribe: async () => {} }
      }
    }
    const sys = vi.fn()
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys,
      watchdogMs: 5000
    })
    await stream.attach()
    const firstSend = stream.send('first')

    await stream.detach()
    await stream.attach()
    const secondSend = stream.send('second')

    first.resolve({ accepted: true, turn_id: 'turn-1' })
    await firstSend

    await expect(stream.send('third')).rejects.toThrow(/turn already in progress/i)
    await vi.advanceTimersByTimeAsync(5000)
    expect(stream.isTurnActive()).toBe(true)
    expect(sys).toHaveBeenCalledWith(expect.stringMatching(/no response/i))

    second.resolve({ accepted: true, turn_id: 'turn-2' })
    await secondSend
  })

  it('ignores a stale correlated error after detach without clearing the newer Turn', async () => {
    const first = deferred<TurnSendResult>()
    const submissions: string[] = []
    let handler: ((event: TurnEvent) => void) | null = null
    let sendCount = 0
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, params: P): Promise<R> {
        if (method === 'turn.send') {
          sendCount += 1
          submissions.push((params as TurnSendParams).submission_id!)
          if (sendCount === 1) {
            return first.promise as Promise<R>
          }
          return { accepted: true, turn_id: 'turn-2' } as unknown as R
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: `sub-${sendCount}`, unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    const firstSend = stream.send('first')

    await stream.detach()
    await stream.attach()
    await stream.send('second')
    handler?.({
      type: 'message.start',
      payload: { submission_id: submissions[1]!, turn_id: 'turn-2' }
    })
    first.resolve({ accepted: true, turn_id: 'turn-1' })
    await firstSend

    handler?.({
      type: 'error',
      payload: {
        code: -32099,
        message: 'late failure',
        reason: 'internal',
        submission_id: submissions[0]!,
        turn_id: 'turn-1'
      }
    })

    expect(stream.isTurnActive()).toBe(true)
  })

  it('ignores stale uncorrelated payloads until the current Turn starts', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('first')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'old partial' } })

    stream.forceReset()
    await stream.send('second')
    fake.__pushRawEvent({ type: 'token.delta', payload: { text: 'STALE-OLD-TOKEN' } })

    expect(turnController.bufRef).toBe('')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-2' } })
    fake.__pushRawEvent({ type: 'token.delta', payload: { text: 'fresh token' } })
    expect(turnController.bufRef).toBe('fresh token')
  })

  it('does not reactivate a watchdog-reset send when its accept and start arrive late', async () => {
    vi.useFakeTimers()
    const pending = deferred<TurnSendResult>()
    let handler: ((event: TurnEvent) => void) | null = null
    let submissionId = ''
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, params: P): Promise<R> {
        if (method === 'turn.send') {
          submissionId = (params as TurnSendParams).submission_id!
          return pending.promise as Promise<R>
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      watchdogMs: 5000
    })
    await stream.attach()
    const send = stream.send('slow accept')

    await vi.advanceTimersByTimeAsync(5000)
    expect(stream.isTurnActive()).toBe(true)

    pending.resolve({ accepted: true, turn_id: 'turn-1' })
    await send
    handler?.({
      type: 'message.start',
      payload: { submission_id: submissionId, turn_id: 'turn-1' }
    })

    expect(stream.isTurnActive()).toBe(true)
  })

  it('lets the next send start before the previous RPC response reconciles', async () => {
    const first = deferred<TurnSendResult>()
    const second = deferred<TurnSendResult>()
    const submissions: string[] = []
    let handler: ((event: TurnEvent) => void) | null = null
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, params: P): Promise<R> {
        if (method === 'turn.send') {
          submissions.push((params as TurnSendParams).submission_id!)
          return (submissions.length === 1 ? first.promise : second.promise) as Promise<R>
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    const firstSend = stream.send('first')
    const firstSubmission = submissions[0]!

    handler?.({
      type: 'message.start',
      payload: { submission_id: firstSubmission, turn_id: 'turn-1' }
    })
    handler?.({
      type: 'message.complete',
      payload: {
        submission_id: firstSubmission,
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })

    const secondSend = stream.send('second')
    first.resolve({ accepted: true, turn_id: 'turn-1' })
    await firstSend

    expect(stream.isTurnActive()).toBe(true)
    await expect(stream.send('third')).rejects.toThrow(/turn already in progress/i)

    second.resolve({ accepted: true, turn_id: 'turn-2' })
    await secondSend
  })

  it('recovers the prompt when a transport-scoped error cannot be correlated', async () => {
    const fake = makeFakeRpc()
    const sys = vi.fn()
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys
    })
    await stream.attach()
    await stream.send('hello')

    fake.__pushRawEvent({
      type: 'error',
      payload: { code: -32016, message: 'subscription_capacity_exceeded', reason: 'internal' }
    })

    expect(stream.isTurnActive()).toBe(true)
    expect(getUiState().busy).toBe(false)
    expect(sys).toHaveBeenCalledWith(expect.stringMatching(/subscription_capacity_exceeded/i))
  })

  it('reattaches before the next send after a subscription overflow', async () => {
    let handler: ((event: TurnEvent) => void) | null = null
    let subscribeCount = 0
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          return { accepted: true, turn_id: 'turn-retry' } as unknown as R
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        subscribeCount += 1
        handler = h as unknown as (event: TurnEvent) => void
        return {
          subscription_id: `sub-${subscribeCount}`,
          unsubscribe: async () => {
            handler = null
          }
        }
      }
    }
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()

    handler?.({
      type: 'error',
      payload: { code: -32016, message: 'subscription_capacity_exceeded', reason: 'internal' }
    })
    await stream.send('retry')

    expect(subscribeCount).toBe(2)
  })

  it('measures server-ack liveness only: model silence past the window after message.start is not a false positive', async () => {
    vi.useFakeTimers()
    const fake = makeFakeRpc()
    const sysCalls: string[] = []
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m),
      watchdogMs: 5000
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('slow first token')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    // 已收到大模型前确认，随后模型远超窗口仍未产生增量。看门狗不能重新启用并
    // 变成大模型首 token 延迟判定器。
    vi.advanceTimersByTime(60000)

    expect(sysCalls.some(m => /no response/i.test(m))).toBe(false)
    expect(stream.isTurnActive()).toBe(true)
  })
})

describe('createChatStream — attachment lifecycle', () => {
  beforeEach(() => {
    resetTurnState()
    resetUiState()
    turnController.fullReset()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('releases Session images when an accepted Turn completes', async () => {
    const fake = makeFakeRpc()
    const releaseSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    await stream.send('hello', 'claim-1')

    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })

    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
  })

  it('keeps unsent images when an idle subscription reports an error', async () => {
    const fake = makeFakeRpc()
    const releaseSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()

    fake.__pushEvent({
      type: 'error',
      payload: { code: -32016, message: 'subscription_capacity_exceeded', reason: 'internal' }
    })

    expect(releaseSessionImages).not.toHaveBeenCalled()
  })

  it('releases images when the backend explicitly discards them before submission', async () => {
    const fake = makeFakeRpc()
    const releaseSessionImages = vi.fn()
    const sys = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys
    })
    await stream.attach()

    fake.__pushEvent({
      type: 'error',
      payload: {
        attachments_discarded: true,
        code: -32008,
        message: 'model_not_available',
        reason: 'internal'
      }
    })

    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default')
    expect(sys).toHaveBeenCalledWith('pending image attachment discarded; attach it again before retrying')
  })

  it('quarantines a force-reset image claim until its late terminal event releases it', async () => {
    const fake = makeFakeRpc()
    const releaseSessionImages = vi.fn()
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    await stream.send('hello', 'claim-1')

    stream.forceReset()
    expect(rollbackSessionImages).not.toHaveBeenCalled()
    expect(releaseSessionImages).not.toHaveBeenCalled()

    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })

    expect(rollbackSessionImages).not.toHaveBeenCalled()
    expect(releaseSessionImages).toHaveBeenCalledOnce()
    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
  })

  it('does not time-release an accepted image claim after local invalidation', async () => {
    vi.useFakeTimers()
    const fake = makeFakeRpc()
    const releaseSessionImages = vi.fn()
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    await stream.send('hello', 'claim-1')

    stream.forceReset()
    await vi.advanceTimersByTimeAsync(10 * 60 * 1000)
    expect(releaseSessionImages).not.toHaveBeenCalled()
    expect(rollbackSessionImages).not.toHaveBeenCalled()
    expect(stream.isTurnActive()).toBe(true)
  })

  it('rolls back the exact claim on a definite TurnInProgress rejection', async () => {
    const fake = makeFakeRpc()
    const rejection = new TurnInProgressError({
      code: -32003,
      message: 'turn already in progress'
    })
    vi.spyOn(fake, 'rpc').mockRejectedValueOnce(rejection)
    const releaseSessionImages = vi.fn()
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()

    await expect(stream.send('hello', 'claim-1')).rejects.toBe(rejection)

    expect(rollbackSessionImages).toHaveBeenCalledOnce()
    expect(rollbackSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(releaseSessionImages).not.toHaveBeenCalled()
    expect(stream.isTurnActive()).toBe(false)
  })

  it('rolls back the exact claim on any authoritative RPC rejection', async () => {
    const fake = makeFakeRpc()
    const rejection = new ModelNotAvailableError({
      code: -32008,
      message: 'model not available'
    })
    vi.spyOn(fake, 'rpc').mockRejectedValueOnce(rejection)
    const releaseSessionImages = vi.fn()
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()

    await expect(stream.send('hello', 'claim-1')).rejects.toBe(rejection)

    expect(rollbackSessionImages).toHaveBeenCalledOnce()
    expect(rollbackSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(releaseSessionImages).not.toHaveBeenCalled()
    expect(stream.isTurnActive()).toBe(false)
  })

  it('rolls back the exact claim when subscription attach fails before dispatch', async () => {
    const attachError = new Error('subscription unavailable')
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(_method: string, _params: P): Promise<R> {
        throw new Error('turn.send must not run')
      },
      async subscribe<E, P>(_method: string, _params: P, _handler: (event: E) => void) {
        throw attachError
      }
    }
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })

    await expect(stream.send('hello', 'claim-1')).rejects.toBe(attachError)

    expect(rollbackSessionImages).toHaveBeenCalledOnce()
    expect(rollbackSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(stream.isTurnActive()).toBe(false)
  })

  it('rolls back a new claim rejected by the local in-progress guard', async () => {
    const fake = makeFakeRpc()
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    await stream.send('first')

    await expect(stream.send('second', 'claim-2')).rejects.toThrow(/turn already in progress/i)

    expect(rollbackSessionImages).toHaveBeenCalledOnce()
    expect(rollbackSessionImages).toHaveBeenCalledWith('tui:default', 'claim-2')
  })

  it('quarantines an ambiguous transport claim until its correlated terminal event', async () => {
    vi.useFakeTimers()
    const fake = makeFakeRpc()
    vi.spyOn(fake, 'rpc').mockRejectedValueOnce(new Error('socket closed'))
    const subscribe = vi.spyOn(fake, 'subscribe')
    const releaseSessionImages = vi.fn()
    const rollbackSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default',
      submissionIdFactory: () => 'submission-1'
    })
    await stream.attach()

    await expect(stream.send('hello', 'claim-1')).resolves.toMatchObject({ accepted: true })
    expect(stream.isTurnActive()).toBe(true)
    expect(rollbackSessionImages).not.toHaveBeenCalled()
    expect(releaseSessionImages).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(10 * 60 * 1000)
    expect(releaseSessionImages).not.toHaveBeenCalled()
    expect(rollbackSessionImages).not.toHaveBeenCalled()

    await stream.attach()
    expect(subscribe).toHaveBeenCalledTimes(2)
    fake.__pushRawEvent({
      type: 'message.complete',
      payload: {
        submission_id: 'submission-1',
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })

    expect(releaseSessionImages).toHaveBeenCalledOnce()
    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(rollbackSessionImages).not.toHaveBeenCalled()
    expect(stream.isTurnActive()).toBe(false)
  })

  it('does not roll back or surface an RPC rejection after its terminal event', async () => {
    const pending = deferred<TurnSendResult>()
    let handler: ((event: TurnEvent) => void) | null = null
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          return pending.promise as Promise<R>
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const releaseSessionImages = vi.fn()
    const rollbackSessionImages = vi.fn()
    const sys = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default',
      submissionIdFactory: () => 'submission-1',
      sys
    })
    await stream.attach()
    const send = stream.send('hello', 'claim-1')

    handler?.({
      type: 'message.start',
      payload: { submission_id: 'submission-1', turn_id: 'turn-1' }
    })
    handler?.({
      type: 'message.complete',
      payload: {
        submission_id: 'submission-1',
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })
    pending.reject(new Error('late socket close'))

    await expect(send).resolves.toMatchObject({ accepted: true, turn_id: 'turn-1' })
    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(rollbackSessionImages).not.toHaveBeenCalled()
    expect(sys).not.toHaveBeenCalledWith(expect.stringMatching(/delivery status is unknown/i))
    expect(stream.isTurnActive()).toBe(false)
  })

  it('does not let a stale RPC rejection clear or surface over a newer Turn', async () => {
    const first = deferred<TurnSendResult>()
    const submissions: string[] = []
    let handler: ((event: TurnEvent) => void) | null = null
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, params: P): Promise<R> {
        if (method === 'turn.send') {
          submissions.push((params as TurnSendParams).submission_id!)
          if (submissions.length === 1) {
            return first.promise as Promise<R>
          }
          return { accepted: true, turn_id: 'turn-2' } as unknown as R
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const sys = vi.fn()
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys
    })
    await stream.attach()
    const firstSend = stream.send('first')

    await stream.detach()
    await stream.attach()
    await stream.send('second')
    handler?.({
      type: 'message.start',
      payload: { submission_id: submissions[1]!, turn_id: 'turn-2' }
    })

    first.reject(new Error('late socket close'))
    await expect(firstSend).resolves.toMatchObject({ accepted: true })

    await expect(stream.send('third')).rejects.toThrow(/turn already in progress/i)
    expect(sys).not.toHaveBeenCalledWith(expect.stringMatching(/delivery status is unknown/i))

    handler?.({
      type: 'message.complete',
      payload: {
        submission_id: submissions[0]!,
        turn_id: 'turn-1',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })
    expect(stream.isTurnActive()).toBe(true)
    handler?.({
      type: 'message.complete',
      payload: {
        submission_id: submissions[1]!,
        turn_id: 'turn-2',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })
    expect(stream.isTurnActive()).toBe(false)
  })

  it('suppresses a stale accepted-false result after a newer Turn starts', async () => {
    const first = deferred<TurnSendResult>()
    let sendCount = 0
    const rollbackSessionImages = vi.fn()
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          sendCount += 1

          return (
            sendCount === 1 ? first.promise : Promise.resolve({ accepted: true, turn_id: 'turn-2' })
          ) as Promise<R>
        }

        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, _handler: (event: E) => void) {
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    const firstSend = stream.send('first', 'claim-1')

    stream.forceReset()
    await stream.send('second')
    first.resolve({ accepted: false, turn_id: 'turn-1' })

    await expect(firstSend).resolves.toMatchObject({ accepted: true, turn_id: 'turn-1' })
    expect(rollbackSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(stream.isTurnActive()).toBe(true)
  })

  it('suppresses a stale authoritative rejection after a newer Turn starts', async () => {
    const first = deferred<TurnSendResult>()
    let sendCount = 0
    const rollbackSessionImages = vi.fn()
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, _params: P): Promise<R> {
        if (method === 'turn.send') {
          sendCount += 1

          return (
            sendCount === 1 ? first.promise : Promise.resolve({ accepted: true, turn_id: 'turn-2' })
          ) as Promise<R>
        }

        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, _handler: (event: E) => void) {
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({
      rollbackSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    const firstSend = stream.send('first', 'claim-1')

    stream.forceReset()
    await stream.send('second')
    first.reject(
      new TurnInProgressError({
        code: -32003,
        message: 'late authoritative rejection'
      })
    )

    await expect(firstSend).resolves.toMatchObject({ accepted: true })
    expect(rollbackSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')
    expect(stream.isTurnActive()).toBe(true)
  })

  it('settles an older Turn without clearing or leaking the newer Turn', async () => {
    const fake = makeFakeRpc()
    const releaseSessionImages = vi.fn()
    const stream = createChatStream({
      releaseSessionImages,
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    await stream.send('first', 'claim-1')
    const firstSubmission = fake.__submissionIds[0]!
    fake.__pushEvent({
      type: 'message.start',
      payload: { submission_id: firstSubmission, turn_id: 'turn-1' }
    })

    stream.forceReset()
    await stream.send('second', 'claim-2')
    const secondSubmission = fake.__submissionIds[1]!
    fake.__pushEvent({
      type: 'message.start',
      payload: { submission_id: secondSubmission, turn_id: 'turn-2' }
    })
    fake.__pushEvent({
      type: 'error',
      payload: {
        code: -32099,
        message: 'late failure',
        reason: 'internal',
        submission_id: firstSubmission,
        turn_id: 'turn-1'
      }
    })

    expect(stream.isTurnActive()).toBe(true)
    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default', 'claim-1')

    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        submission_id: secondSubmission,
        turn_id: 'turn-2',
        usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 }
      }
    })

    expect(stream.isTurnActive()).toBe(false)
    expect(releaseSessionImages).toHaveBeenCalledWith('tui:default', 'claim-2')
  })

  it('does not restore a Turn that ended before turn.send returned', async () => {
    let handler: ((event: TurnEvent) => void) | null = null
    const fake: ChatStreamRpcClient = {
      async rpc<R, P>(method: string, params: P): Promise<R> {
        if (method === 'turn.send' && handler) {
          const submissionId = (params as TurnSendParams).submission_id!
          handler({
            type: 'message.start',
            payload: { submission_id: submissionId, turn_id: 'turn-1' }
          })
          handler({
            type: 'error',
            payload: {
              code: -32008,
              message: 'model_not_available',
              reason: 'internal',
              submission_id: submissionId,
              turn_id: 'turn-1'
            }
          })
          return { accepted: true, turn_id: 'turn-1' } as unknown as R
        }
        return {} as R
      },
      async subscribe<E, P>(_method: string, _params: P, h: (event: E) => void) {
        handler = h as unknown as (event: TurnEvent) => void
        return { subscription_id: 'sub-1', unsubscribe: async () => {} }
      }
    }
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()

    await stream.send('hello')

    expect(stream.isTurnActive()).toBe(false)
  })
})

describe('createChatStream — cancel preserves streamed content', () => {
  beforeEach(() => {
    resetTurnState()
    resetUiState()
    turnController.fullReset()
  })

  it('keeps the streamed partial in the transcript on a server cancelled_by_client', async () => {
    const appended: Msg[] = []
    const fake = makeFakeRpc()
    const stream = createChatStream({
      appendMessage: m => appended.push(m),
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('question')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'Partial answer so far' } })
    fake.__pushEvent({
      type: 'error',
      payload: { code: -32000, message: 'cancelled', reason: 'cancelled_by_client' }
    })

    const assistant = appended.find(m => m.role === 'assistant')
    expect(assistant).toBeDefined()
    expect(assistant!.text).toContain('Partial answer so far')
    expect(assistant!.text).toContain('[interrupted]')
  })

  it('keeps the streamed partial in the transcript on a local forceReset', async () => {
    const appended: Msg[] = []
    const fake = makeFakeRpc()
    const stream = createChatStream({
      appendMessage: m => appended.push(m),
      rpcClient: fake,
      sessionKey: 'tui:default'
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('question')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'Half a reply' } })

    stream.forceReset()

    const assistant = appended.find(m => m.role === 'assistant')
    expect(assistant).toBeDefined()
    expect(assistant!.text).toContain('Half a reply')
    expect(assistant!.text).toContain('[interrupted]')
  })

  it('does not emit a redundant interrupted note when a second cancel re-enters after content was already preserved', async () => {
    const appended: Msg[] = []
    const sysCalls: string[] = []
    const fake = makeFakeRpc()
    const stream = createChatStream({
      appendMessage: m => appended.push(m),
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m)
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('question')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'A streamed reply' } })

    // 首次 Ctrl+C 逃生路径保留部分输出。
    stream.forceReset()
    // 随后服务端取消错误到达；取消时未清除 turnId，因此会在当前空状态上再次
    // 进入 finalize。
    fake.__pushEvent({
      type: 'error',
      payload: { code: -32000, message: 'cancelled', reason: 'cancelled_by_client' }
    })

    // 内容只保留一次，不能额外生成第二条虚假的中断系统提示。
    expect(appended.filter(m => m.role === 'assistant')).toHaveLength(1)
    expect(sysCalls.filter(m => m === 'interrupted')).toHaveLength(0)
  })

  it('does not append an empty assistant message when nothing was streamed', async () => {
    const appended: Msg[] = []
    const sysCalls: string[] = []
    const fake = makeFakeRpc()
    const stream = createChatStream({
      appendMessage: m => appended.push(m),
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: m => sysCalls.push(m)
    })
    await stream.attach()
    patchUiState({ busy: true })
    await stream.send('question')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    // 没有 token.delta，轮次在任何内容流出前就被取消。
    stream.forceReset()

    expect(appended.some(m => m.role === 'assistant')).toBe(false)
    expect(sysCalls).toContain('interrupted')
  })
})
