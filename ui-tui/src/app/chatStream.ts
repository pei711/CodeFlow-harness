// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// 将 `RpcClient.subscribe<TurnEvent>('turn.subscribe', ...)` 通知桥接到现有的
// `turnController`。工厂返回只包含 `attach / detach / send / cancel /
// isTurnActive` 的轻量句柄，既能通过虚假 RpcClient 做无套接字单元测试，
// 也能在 `useMainApp.ts` 中作为每个会话的生命周期对象接入。

import { randomUUID } from 'node:crypto'

import type { Msg } from '../types.js'

import { mergeUsage } from '../domain/usage.js'
import {
  RpcError,
  type ErrorEvent,
  type MessageCompleteEvent,
  type MessageStartEvent,
  type TokenDeltaEvent,
  type ToolCompleteEvent,
  type ToolStartEvent,
  type TurnEvent,
  type TurnSendParams,
  type TurnSendResult,
  type TurnSubscribeParams
} from '../rpc/index.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

/**
 * 聊天路径所需的最小 RpcClient 接口。局部定义后，测试可以注入替身而不接触
 * 真实的套接字 RpcClient。结构与 `src/rpc/client.ts::RpcClient` 的公开方法
 * 一致，包括 `subscribe` 的返回结构 `{subscription_id, unsubscribe}`。
 */
export interface ChatStreamRpcClient {
  rpc<R = unknown, P = unknown>(method: string, params: P): Promise<R>
  subscribe<E = unknown, P = unknown>(
    method: string,
    params: P,
    handler: (event: E) => void,
    opts?: { unsubscribeMethod?: string }
  ): Promise<{ subscription_id: string; unsubscribe: () => Promise<void> }>
}

export interface ChatStreamOptions {
  rpcClient: ChatStreamRpcClient
  sessionKey: string
  releaseSessionImages?: (sessionId: string, claimId?: string) => void
  rollbackSessionImages?: (sessionId: string, claimId?: string) => void
  submissionIdFactory?: () => string
  /** 用于展示非取消类错误的可选系统消息钩子。 */
  sys?: (msg: string) => void
  /**
   * 将完成的消息追加到 React 历史列表。`message.complete` 需要借此在界面中
   * 持久化助手轮次的最终文本和工具轨迹；否则流式 token 会在完成时消失。
   * 沿用旧版 `createGatewayEventHandler.ts:675` 的模式。
   */
  appendMessage?: (msg: Msg) => void
  /**
   * 服务端确认看门狗窗口（毫秒）。`send` 开始时启动；若窗口内未收到任何
   * 服务端事件，则认为轮次卡死（事件丢失、订阅未投递或 turn.send 挂起），
   * 恢复输入而不是让界面冻结。它只衡量服务端确认活性，首个入站事件即解除，
   * 不衡量大模型首 token 延迟。默认值为 {@link DEFAULT_WATCHDOG_MS}。
   */
  watchdogMs?: number
}

/** 默认服务端确认看门狗窗口，参见 {@link ChatStreamOptions.watchdogMs}。 */
export const DEFAULT_WATCHDOG_MS = 10_000

export interface ChatStreamHandle {
  attach: () => Promise<void>
  detach: () => Promise<void>
  send: (content: string, imageClaimId?: string) => Promise<TurnSendResult>
  cancel: () => Promise<void>
  isTurnActive: () => boolean
  sessionKey: string
  /**
   * 本地强制重置：不经过服务端往返，直接丢弃活动轮次并恢复提示符。它为
   * Ctrl+C 逃生路径和看门狗提供支撑，确保没有终止事件的轮次不会卡死界面。
   */
  forceReset: () => void
}

type AttemptInvalidation = 'detach' | 'force_reset' | 'transport_error' | 'watchdog'

interface SendAttempt {
  imageClaimId?: string
  invalidated: AttemptInvalidation | null
  imagesSettled: boolean
  rpcSettled: boolean
  started: boolean
  submissionId: string
  terminal: boolean
  turnId: string | null
  watchdog: null | ReturnType<typeof setTimeout>
}

interface InternalState {
  attached: boolean
  attempts: Map<string, SendAttempt>
  current: SendAttempt | null
  subscriptionEpoch: number
  unsubscribe: (() => Promise<void>) | null
  turnId: string | null
}

const dispatch = (
  state: InternalState,
  event: TurnEvent,
  sys?: (msg: string) => void,
  appendMessage?: (msg: Msg) => void,
  releaseSessionImages?: (claimId?: string) => void,
  invalidateTransport?: () => void
): void => {
  switch (event.type) {
    case 'message.start':
      onMessageStart(state, event)
      return
    case 'token.delta':
      if (acceptsTurnPayload(state)) {
        onTokenDelta(event)
      }
      return
    case 'thinking.delta':
      if (acceptsTurnPayload(state)) {
        turnController.recordReasoningDelta(event.payload.text)
      }
      return
    case 'tool.start':
      if (acceptsTurnPayload(state)) {
        onToolStart(event)
      }
      return
    case 'tool.progress':
      return
    case 'tool.complete':
      if (acceptsTurnPayload(state)) {
        onToolComplete(event)
      }
      return
    case 'message.complete':
      onMessageComplete(state, event, appendMessage, releaseSessionImages)
      return
    case 'error':
      onError(state, event, sys, appendMessage, releaseSessionImages, invalidateTransport)
      return
    case 'subagent.delivered':
      appendMessage?.({ role: 'assistant', text: event.payload.text })
      return
    case 'cron.delivered': {
      if (sys) {
        const { name, text, fired_at } = event.payload
        const tag = fired_at ? `${name} @ ${fired_at}` : name
        sys(`─── ⏰ ${tag} ───\n${text}\n${'─'.repeat(40)}`)
      }
      return
    }
    default: {
      // 用穷尽检查确保新增 TurnEvent 变体时类型检查器在此报错，迫使同步更新本文件。
      const exhaustive: never = event
      void exhaustive
    }
  }
}

const acceptsTurnPayload = (state: InternalState): boolean => {
  const attempt = state.current

  return Boolean(attempt?.started && !attempt.invalidated && !attempt.terminal)
}

const onMessageStart = (state: InternalState, ev: MessageStartEvent): void => {
  const attempt = state.attempts.get(ev.payload.submission_id)

  if (!attempt) {
    return
  }

  attempt.turnId = ev.payload.turn_id
  attempt.started = true
  if (attempt.watchdog !== null) {
    clearTimeout(attempt.watchdog)
    attempt.watchdog = null
  }
  if (state.current !== attempt || attempt.invalidated || attempt.terminal) {
    return
  }

  state.turnId = attempt.turnId
  turnController.startMessage()
  patchUiState({ status: 'running…' })
}

const onTokenDelta = (ev: TokenDeltaEvent): void => {
  turnController.recordMessageDelta({ text: ev.payload.text })
}

const onToolStart = (ev: ToolStartEvent): void => {
  const { tool_call_id, name, arguments: args } = ev.payload
  // 从首个标量参数值生成简短上下文，使活动工具列表显示有用预览，同时避免
  // 将完整参数块泄漏到界面中。
  const previewKey = Object.keys(args)[0]
  const previewVal = previewKey !== undefined ? args[previewKey] : undefined
  const context =
    typeof previewVal === 'string' ? previewVal : previewVal !== undefined ? JSON.stringify(previewVal) : ''
  turnController.recordToolStart(tool_call_id, name, context)
}

const onToolComplete = (ev: ToolCompleteEvent): void => {
  const { tool_call_id, result_preview, failed, truncated } = ev.payload
  const summary = truncated ? `${result_preview} (truncated)` : result_preview
  turnController.recordToolComplete(tool_call_id, undefined, failed ? summary : undefined, failed ? undefined : summary)
}

const releaseAttemptImages = (attempt: SendAttempt, releaseSessionImages?: (claimId?: string) => void): void => {
  if (attempt.imagesSettled) {
    return
  }

  if (attempt.imageClaimId) {
    releaseSessionImages?.(attempt.imageClaimId)
  }
  attempt.imagesSettled = true
}

const onMessageComplete = (
  state: InternalState,
  ev: MessageCompleteEvent,
  appendMessage?: (msg: Msg) => void,
  releaseSessionImages?: (claimId?: string) => void
): void => {
  const attempt = state.attempts.get(ev.payload.submission_id)

  if (!attempt || attempt.terminal) {
    return
  }

  attempt.terminal = true
  attempt.turnId = ev.payload.turn_id
  if (attempt.watchdog !== null) {
    clearTimeout(attempt.watchdog)
    attempt.watchdog = null
  }
  releaseAttemptImages(attempt, releaseSessionImages)
  state.attempts.delete(attempt.submissionId)
  if (state.current !== attempt || attempt.invalidated) {
    return
  }

  state.current = null
  state.turnId = null
  // 按 CAP-CHAT-1 线协议结构（B1 修复），带类型的 message.complete 携带
  // `{turn_id, usage}`；助手内容由 token.delta 累积到 `bufRef` 后重建。
  // payload.text 缺失时，recordMessageComplete 会读取 bufRef，并返回调用方
  // 必须写入历史记录的最终消息列表；否则流式 token 会在轮次中出现，却在
  // 轮次结束时消失。
  if (ev.payload.usage) {
    patchUiState(s => ({ ...s, usage: mergeUsage(s.usage, ev.payload.usage) }))
  }
  const { finalMessages, finalText, wasInterrupted } = turnController.recordMessageComplete({})
  if (!wasInterrupted && appendMessage) {
    const msgs: Msg[] = finalMessages.length > 0 ? finalMessages : [{ role: 'assistant', text: finalText }]
    msgs.forEach(appendMessage)
  }
  patchUiState({ status: 'ready' })
}

const onError = (
  state: InternalState,
  ev: ErrorEvent,
  sys?: (msg: string) => void,
  appendMessage?: (msg: Msg) => void,
  releaseSessionImages?: (claimId?: string) => void,
  invalidateTransport?: () => void
): void => {
  const { attachments_discarded: attachmentsDiscarded, submission_id: submissionId, reason, message, code } = ev.payload
  const attempt = submissionId ? state.attempts.get(submissionId) : undefined

  if (submissionId && !attempt) {
    return
  }

  if (!attempt) {
    const hadActiveAttempt = state.current !== null
    if (attachmentsDiscarded) {
      releaseSessionImages?.()
      sys?.('pending image attachment discarded; attach it again before retrying')
    }
    sys?.(`error: ${message} (code=${code})`)
    invalidateTransport?.()
    if (hadActiveAttempt) {
      turnController.finalizeInterruptedTurn({ appendMessage, sys })
      turnController.clearStatusTimer()
    } else {
      turnController.recordError()
    }
    patchUiState({ busy: false, status: `error: ${message.slice(0, 80)}` })
    patchTurnState({ activity: [], outcome: '' })
    return
  }
  if (attempt.terminal) {
    return
  }

  attempt.terminal = true
  attempt.turnId = ev.payload.turn_id ?? attempt.turnId
  if (attempt.watchdog !== null) {
    clearTimeout(attempt.watchdog)
    attempt.watchdog = null
  }
  releaseAttemptImages(attempt, releaseSessionImages)
  state.attempts.delete(attempt.submissionId)
  if (attachmentsDiscarded && sys) {
    sys('pending image attachment discarded; attach it again before retrying')
  }
  if (state.current !== attempt || attempt.invalidated) {
    return
  }

  state.current = null
  state.turnId = null
  if (reason === 'cancelled_by_client') {
    restoreInputPrompt(appendMessage, sys)
    return
  }
  // 非取消类错误需要显示系统提示、将轮次置为空闲并重置实时锚点，允许再次提交。
  if (sys) {
    sys(`error: ${message} (code=${code})`)
  }
  turnController.recordError()
  patchUiState({ busy: false, status: `error: ${message.slice(0, 80)}` })
  patchTurnState({ activity: [], outcome: '' })
}

const restoreInputPrompt = (appendMessage?: (msg: Msg) => void, sys?: (msg: string) => void): void => {
  // 轮次取消后保留已流式输出的内容，清除流式状态、释放 `busy` 并稳定可见状态。
  turnController.finalizeInterruptedTurn({ appendMessage, sys })
  turnController.clearStatusTimer()
  patchUiState({ busy: false, status: 'interrupted' })
  turnController.statusTimer = setTimeout(() => {
    turnController.statusTimer = null
    if (!getUiState().busy) {
      patchUiState({ status: 'ready' })
    }
  }, 800)
}

const isAuthoritativeRpcRejection = (error: unknown): error is RpcError => error instanceof RpcError

export const createChatStream = (opts: ChatStreamOptions): ChatStreamHandle => {
  const state: InternalState = {
    attached: false,
    attempts: new Map(),
    current: null,
    subscriptionEpoch: 0,
    unsubscribe: null,
    turnId: null
  }

  const watchdogMs = opts.watchdogMs ?? DEFAULT_WATCHDOG_MS

  const clearAttemptWatchdog = (attempt: SendAttempt): void => {
    if (attempt.watchdog !== null) {
      clearTimeout(attempt.watchdog)
      attempt.watchdog = null
    }
  }

  const rollbackImageClaim = (imageClaimId?: string): void => {
    if (!imageClaimId) {
      return
    }

    if (opts.rollbackSessionImages) {
      opts.rollbackSessionImages(opts.sessionKey, imageClaimId)
    } else {
      opts.releaseSessionImages?.(opts.sessionKey, imageClaimId)
    }
  }

  const rollbackAttemptImages = (attempt: SendAttempt): void => {
    if (attempt.imagesSettled) {
      return
    }

    rollbackImageClaim(attempt.imageClaimId)
    attempt.imagesSettled = true
  }

  const invalidateAttempt = (attempt: SendAttempt, reason: AttemptInvalidation): void => {
    attempt.invalidated ??= reason
    clearAttemptWatchdog(attempt)
    if (state.current === attempt) {
      state.current = null
    }
  }

  const invalidateCurrent = (reason: AttemptInvalidation): void => {
    const attempt = state.current

    if (attempt) {
      invalidateAttempt(attempt, reason)
    }
    state.turnId = null
  }

  const invalidateSubscription = (): void => {
    invalidateCurrent('transport_error')
    state.subscriptionEpoch += 1
    state.attached = false
    const unsubscribe = state.unsubscribe
    state.unsubscribe = null
    if (unsubscribe) {
      void unsubscribe().catch(() => undefined)
    }
  }

  const forceReset = (): void => {
    invalidateCurrent('force_reset')
    restoreInputPrompt(opts.appendMessage, opts.sys)
  }

  const armAckWatchdog = (attempt: SendAttempt): void => {
    clearAttemptWatchdog(attempt)
    attempt.watchdog = setTimeout(() => {
      if (state.current !== attempt || attempt.invalidated || attempt.terminal || attempt.started) {
        return
      }

      opts.sys?.('turn produced no response — input restored (press Enter to retry)')
      invalidateCurrent('watchdog')
      restoreInputPrompt(opts.appendMessage, opts.sys)
    }, watchdogMs)
  }

  async function attach(): Promise<void> {
    if (state.attached) {
      return
    }

    const subscriptionEpoch = ++state.subscriptionEpoch
    const params: TurnSubscribeParams = { session_key: opts.sessionKey }
    const result = await opts.rpcClient.subscribe<TurnEvent, TurnSubscribeParams>(
      'turn.subscribe',
      params,
      event => {
        if (subscriptionEpoch !== state.subscriptionEpoch) {
          return
        }
        dispatch(
          state,
          event,
          opts.sys,
          opts.appendMessage,
          opts.releaseSessionImages
            ? claimId =>
                claimId
                  ? opts.releaseSessionImages?.(opts.sessionKey, claimId)
                  : opts.releaseSessionImages?.(opts.sessionKey)
            : undefined,
          invalidateSubscription
        )
      },
      { unsubscribeMethod: 'turn.unsubscribe' }
    )

    if (subscriptionEpoch !== state.subscriptionEpoch) {
      await result.unsubscribe()
      return
    }

    state.unsubscribe = result.unsubscribe
    state.attached = true
  }

  const detach = async (): Promise<void> => {
    state.subscriptionEpoch += 1
    for (const attempt of [...state.attempts.values()]) {
      invalidateAttempt(attempt, 'detach')
    }
    state.current = null
    state.turnId = null
    const u = state.unsubscribe
    state.unsubscribe = null
    state.attached = false
    if (u) {
      await u()
    }
  }

  const send = async (content: string, imageClaimId?: string): Promise<TurnSendResult> => {
    if (!state.attached) {
      try {
        await attach()
      } catch (error) {
        rollbackImageClaim(imageClaimId)
        throw error
      }
    }

    if (!state.attached) {
      rollbackImageClaim(imageClaimId)
      throw new Error('turn subscription is not attached')
    }

    if (state.current) {
      rollbackImageClaim(imageClaimId)
      throw new Error('turn already in progress — wait for message.complete or cancel first')
    }

    const submissionId = opts.submissionIdFactory?.() ?? randomUUID()
    const attempt: SendAttempt = {
      imageClaimId,
      imagesSettled: false,
      invalidated: null,
      rpcSettled: false,
      started: false,
      submissionId,
      terminal: false,
      turnId: null,
      watchdog: null
    }
    const params: TurnSendParams = {
      session_key: opts.sessionKey,
      content,
      submission_id: submissionId
    }

    state.attempts.set(submissionId, attempt)
    state.current = attempt
    armAckWatchdog(attempt)

    let result: TurnSendResult
    try {
      result = await opts.rpcClient.rpc<TurnSendResult, TurnSendParams>('turn.send', params)
    } catch (err) {
      attempt.rpcSettled = true
      clearAttemptWatchdog(attempt)
      if (attempt.terminal) {
        return { accepted: true, turn_id: attempt.turnId ?? '' }
      }
      if (isAuthoritativeRpcRejection(err)) {
        state.attempts.delete(submissionId)
        rollbackAttemptImages(attempt)
        if (state.current === attempt) {
          state.current = null
          state.turnId = null
        }
        if (attempt.invalidated) {
          return { accepted: true, turn_id: attempt.turnId ?? '' }
        }
        throw err
      }
      if (attempt.invalidated) {
        return { accepted: true, turn_id: attempt.turnId ?? '' }
      }

      invalidateSubscription()
      opts.sys?.('turn delivery status is unknown; input restored while awaiting server settlement')
      restoreInputPrompt(opts.appendMessage, opts.sys)

      return { accepted: true, turn_id: attempt.turnId ?? '' }
    }

    attempt.rpcSettled = true
    attempt.turnId ??= result.turn_id
    if (attempt.terminal) {
      return { accepted: true, turn_id: attempt.turnId ?? result.turn_id }
    }
    if (!result.accepted) {
      clearAttemptWatchdog(attempt)
      state.attempts.delete(submissionId)
      rollbackAttemptImages(attempt)
      if (state.current === attempt) {
        state.current = null
        state.turnId = null
      }
      if (attempt.invalidated) {
        return { accepted: true, turn_id: attempt.turnId ?? result.turn_id }
      }
      return result
    }
    clearAttemptWatchdog(attempt)
    if (state.current === attempt && !attempt.invalidated && !attempt.terminal) {
      state.turnId = attempt.turnId
    }

    return result
  }

  const cancel = async (): Promise<void> => {
    if (!state.turnId) {
      return
    }
    await opts.rpcClient.rpc<{ cancelled: boolean }, { session_key: string }>('turn.cancel', {
      session_key: opts.sessionKey
    })
    // 此处不能清除 state.turnId：服务端应发送
    // `error(reason=cancelled_by_client)` 事件，再由 dispatch() 驱动实际的
    // 界面状态重置。本地提前清除会与事件投递竞争，导致轮次活动保护状态不一致。
  }

  const isTurnActive = (): boolean => state.attempts.size > 0

  return { attach, detach, send, cancel, isTurnActive, forceReset, sessionKey: opts.sessionKey }
}
