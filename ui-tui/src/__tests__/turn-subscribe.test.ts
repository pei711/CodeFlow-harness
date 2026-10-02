// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// 带类型聊天订阅路径的测试。
//
// 这些测试覆盖 `createChatStream`：它是轻量工厂，将
// `RpcClient.subscribe('turn.subscribe', ...)` 事件桥接到现有
// `turnController`，使聊天界面无需经过旧版 `TuiRpcClient.gw.on('event', ...)`
// 适配器即可更新。测试使用只包含聊天路径所需两个方法的虚假 RpcClient 接口：
// `rpc(method, params)` 与 `subscribe(method, params, handler)`，因而无需为每个
// 用例启动 Unix 套接字。

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { TurnEvent, TurnSendParams, TurnSendResult, TurnSubscribeParams } from '../rpc/index.js'

import { createChatStream, type ChatStreamRpcClient } from '../app/chatStream.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

type FakeUnsubscribe = () => Promise<void>

interface FakeRpc extends ChatStreamRpcClient {
  __pushEvent: (event: TurnEvent) => void
  __sendCalls: Array<{ method: string; params: unknown }>
  __cancelCalls: number
  __subParams: TurnSubscribeParams | null
  __unsubscribeCalls: number
}

const makeFakeRpc = (opts: { sendResult?: TurnSendResult; subId?: string } = {}): FakeRpc => {
  const sendCalls: Array<{ method: string; params: unknown }> = []
  let unsubscribeCalls = 0
  let cancelCalls = 0
  let subParams: TurnSubscribeParams | null = null
  let handler: ((event: TurnEvent) => void) | null = null

  const fake: FakeRpc = {
    __sendCalls: sendCalls,
    __cancelCalls: 0,
    __unsubscribeCalls: 0,
    __subParams: null,
    __pushEvent: (event: TurnEvent) => {
      if (handler) {
        const turnSend = [...sendCalls].reverse().find(call => call.method === 'turn.send')
        const submissionId = (turnSend?.params as TurnSendParams | undefined)?.submission_id
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
    async rpc<R, P>(method: string, params: P): Promise<R> {
      sendCalls.push({ method, params })
      if (method === 'turn.send') {
        const result = opts.sendResult ?? { turn_id: 'turn-1', accepted: true }
        return result as unknown as R
      }
      if (method === 'turn.cancel') {
        cancelCalls += 1
        fake.__cancelCalls = cancelCalls
        return { cancelled: true } as unknown as R
      }
      return {} as R
    },
    async subscribe<E, P, R extends { subscription_id: string } = { subscription_id: string }>(
      method: string,
      params: P,
      h: (event: E) => void
    ): Promise<{ subscription_id: string; unsubscribe: FakeUnsubscribe }> {
      void method
      subParams = params as unknown as TurnSubscribeParams
      fake.__subParams = subParams
      handler = h as unknown as (event: TurnEvent) => void
      const subscription_id = opts.subId ?? 'sub-1'
      const unsubscribe: FakeUnsubscribe = async () => {
        unsubscribeCalls += 1
        fake.__unsubscribeCalls = unsubscribeCalls
        handler = null
      }
      return { subscription_id, unsubscribe } as unknown as {
        subscription_id: string
        unsubscribe: FakeUnsubscribe
      } & R
    }
  }
  return fake
}

describe('createChatStream', () => {
  beforeEach(() => {
    resetTurnState()
    resetUiState()
    turnController.fullReset()
  })

  it('streams token.delta sequence and closes the turn on message.complete', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()

    // 模拟用户提交。
    const sendResult = await stream.send('hello')
    expect(sendResult).toEqual({ turn_id: 'turn-1', accepted: true })
    expect(fake.__sendCalls[0]).toMatchObject({
      method: 'turn.send',
      params: { session_key: 'tui:default', content: 'hello' }
    })
    expect((fake.__sendCalls[0]!.params as TurnSendParams).submission_id).toEqual(expect.any(String))
    expect(fake.__subParams).toEqual({ session_key: 'tui:default' } satisfies TurnSubscribeParams)

    // 驱动事件流。
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    expect(getUiState().busy).toBe(true)

    for (const text of ['Hel', 'lo, ', 'wor', 'ld', '!']) {
      fake.__pushEvent({ type: 'token.delta', payload: { text } })
    }
    // 五个增量后，累计缓冲区应为拼接文本。控制器将原始增量存入 `bufRef`；可见
    // 流补丁由定时器调度，因此这里通过内部累加器断言。
    expect(turnController.bufRef).toBe('Hello, world!')

    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        turn_id: 'turn-1',
        usage: {
          prompt_tokens: 1,
          completion_tokens: 5,
          total_tokens: 6,
          cost_usd: 0.0012,
          context_used: 8378,
          context_max: 1048576,
          context_percent: 1
        }
      }
    })

    // 轮次关闭后应释放 busy 标记并清空 bufRef。
    expect(getUiState().busy).toBe(false)
    expect(turnController.bufRef).toBe('')

    // message.complete 中的用量会合并到界面状态，使状态栏上下文仪表与成本反映
    // 当前轮次，而不是停留在启动基线。
    const usage = getUiState().usage
    expect(usage.context_used).toBe(8378)
    expect(usage.context_max).toBe(1048576)
    expect(usage.context_percent).toBe(1)
    expect(usage.cost_usd).toBe(0.0012)

    await stream.detach()
    expect(fake.__unsubscribeCalls).toBe(1)
  })

  it('clears a prior cost when the completed turn has unknown pricing', async () => {
    patchUiState(state => ({ ...state, usage: { ...state.usage, cost_usd: 0.25 } }))
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('hello')

    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({
      type: 'message.complete',
      payload: {
        turn_id: 'turn-1',
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 }
      }
    })

    expect(getUiState().usage.cost_usd).toBeUndefined()
  })

  it('restores the input prompt when the server reports cancelled_by_client', async () => {
    const fake = makeFakeRpc({ subId: 'sub-cancel' })
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()

    await stream.send('long-task')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({ type: 'token.delta', payload: { text: 'partial...' } })
    expect(getUiState().busy).toBe(true)

    fake.__pushEvent({
      type: 'error',
      payload: { code: -32800, message: 'cancelled by client', reason: 'cancelled_by_client' }
    })

    // 输入提示恢复后不再忙碌，状态重置为 'ready'，或处于 'interrupted' 冷却；
    // 只要没有卡在 'running'，两者均可接受。
    const ui = getUiState()
    expect(ui.busy).toBe(false)
    expect(ui.status === 'ready' || ui.status === 'interrupted').toBe(true)
  })

  it('cancel() routes through turn.cancel when a turn is in flight', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('a long task that the user will interrupt')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })

    expect(stream.isTurnActive()).toBe(true)

    await stream.cancel()

    expect(fake.__cancelCalls).toBe(1)
    expect(fake.__sendCalls.some(c => c.method === 'turn.cancel')).toBe(true)
    const cancelCall = fake.__sendCalls.find(c => c.method === 'turn.cancel')
    expect(cancelCall?.params).toEqual({ session_key: 'tui:default' })
  })

  it('cancel() is a no-op when there is no active turn', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()

    expect(stream.isTurnActive()).toBe(false)
    await stream.cancel()
    expect(fake.__cancelCalls).toBe(0)
  })

  it('records tool.start / tool.complete onto the active turn', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('use a tool')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({
      type: 'tool.start',
      payload: { tool_call_id: 'tc-1', name: 'shell.exec', arguments: { command: 'ls' } }
    })
    // tool.start 后，活动工具列表包含已启动工具。
    expect(getTurnState().tools.some(t => t.name === 'shell.exec')).toBe(true)

    fake.__pushEvent({
      type: 'tool.complete',
      payload: { tool_call_id: 'tc-1', result_preview: 'a b c', truncated: false }
    })
    // tool.complete 后，活动工具被移除。
    expect(getTurnState().tools.some(t => t.id === 'tc-1')).toBe(false)
  })

  it('marks failed tool.complete events as errors', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('use a tool')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })
    fake.__pushEvent({
      type: 'tool.start',
      payload: { tool_call_id: 'tc-1', name: 'read_file', arguments: { path: 'missing.txt' } }
    })

    fake.__pushEvent({
      type: 'tool.complete',
      payload: {
        tool_call_id: 'tc-1',
        result_preview: 'Error: file not found',
        truncated: false,
        failed: true
      }
    })

    expect(getTurnState().streamPendingTools).toEqual([expect.stringMatching(/Error: file not found.*✗$/)])
  })

  it('surfaces non-cancellation errors and restores input prompt', async () => {
    const fake = makeFakeRpc()
    const sysCalls: string[] = []
    const stream = createChatStream({
      rpcClient: fake,
      sessionKey: 'tui:default',
      sys: msg => sysCalls.push(msg)
    })
    await stream.attach()
    await stream.send('bad request')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })

    fake.__pushEvent({
      type: 'error',
      payload: { code: -32008, message: 'model not available' }
    })

    expect(sysCalls.some(m => m.includes('model not available'))).toBe(true)
    const ui = getUiState()
    expect(ui.busy).toBe(false)
    // 状态应反映错误，或在控制器稳定后为 'ready'；约定的核心是不能卡在忙碌状态。
    expect(ui.status).not.toBe('running…')
  })

  it('blocks send while a turn is already active', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    await stream.send('first')
    fake.__pushEvent({ type: 'message.start', payload: { turn_id: 'turn-1' } })

    await expect(stream.send('second')).rejects.toThrow(/turn.*in.*progress|active/i)
    // 只有首次 turn.send 应写入线路。
    const sendCount = fake.__sendCalls.filter(c => c.method === 'turn.send').length
    expect(sendCount).toBe(1)
  })

  it('does not double-subscribe when attach() is called twice', async () => {
    const fake = makeFakeRpc()
    const stream = createChatStream({ rpcClient: fake, sessionKey: 'tui:default' })
    await stream.attach()
    const subscribeSpy = vi.spyOn(fake, 'subscribe')
    await stream.attach()
    expect(subscribeSpy).not.toHaveBeenCalled()
  })
})
