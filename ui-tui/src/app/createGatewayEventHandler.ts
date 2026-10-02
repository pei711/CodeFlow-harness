// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { GatewayEvent, GatewaySkin } from '../gatewayTypes.js'
import type { Msg, SubagentProgress } from '../types.js'
import type { GatewayEventHandlerContext } from './interfaces.js'

import { STARTUP_IMAGE, STARTUP_QUERY } from '../config/env.js'
import { STREAM_BATCH_MS } from '../config/timing.js'
import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import { mergeUsage } from '../domain/usage.js'
import { rpcErrorMessage } from '../lib/rpc.js'
import { formatToolCall, stripAnsi } from '../lib/text.js'
import { patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { applySkinTheme, getUiState, patchUiState } from './uiStore.js'

const NO_PROVIDER_RE = /\bNo (?:LLM|inference) provider configured\b/i

const statusFromBusy = () => (getUiState().busy ? 'running…' : 'ready')

const dropBgTask = (taskId: string) =>
  patchUiState(state => {
    const next = new Set(state.bgTasks)
    next.delete(taskId)

    return { ...state, bgTasks: next }
  })

const pushUnique =
  (max: number) =>
  <T>(xs: T[], x: T): T[] =>
    xs.at(-1) === x ? xs : [...xs, x].slice(-max)

const pushThinking = pushUnique(6)
const pushNote = pushUnique(6)
const pushTool = pushUnique(8)

export function createGatewayEventHandler(ctx: GatewayEventHandlerContext): (ev: GatewayEvent) => void {
  const releaseSessionImages = ctx.clipboard?.releaseSessionImages
  const { rpc } = ctx.gateway
  const { STARTUP_RESUME_ID, newSession, resumeById } = ctx.session
  const { bellOnComplete, stdout, sys } = ctx.system
  const { appendMessage, panel, setHistoryItems } = ctx.transcript
  const { submitRef } = ctx.submission

  let pendingThinkingStatus = ''
  let thinkingStatusTimer: null | ReturnType<typeof setTimeout> = null
  let startupPromptSubmitted = false

  const setStatus = (status: string) => {
    pendingThinkingStatus = ''

    if (thinkingStatusTimer) {
      clearTimeout(thinkingStatusTimer)
      thinkingStatusTimer = null
    }

    patchUiState({ status })
  }

  const scheduleThinkingStatus = (status: string) => {
    pendingThinkingStatus = status

    if (thinkingStatusTimer) {
      return
    }

    thinkingStatusTimer = setTimeout(() => {
      thinkingStatusTimer = null
      patchUiState({ status: pendingThinkingStatus || statusFromBusy() })
    }, STREAM_BATCH_MS)
  }

  const restoreStatusAfter = (ms: number) => {
    turnController.clearStatusTimer()
    turnController.statusTimer = setTimeout(() => {
      turnController.statusTimer = null
      patchUiState({ status: statusFromBusy() })
    }, ms)
  }

  const scheduleStartupPrompt = () => {
    if (startupPromptSubmitted || (!STARTUP_QUERY && !STARTUP_IMAGE)) {
      return
    }

    startupPromptSubmitted = true
    setTimeout(async () => {
      let sid = getUiState().sid

      for (let i = 0; !sid && i < 40; i += 1) {
        await new Promise(resolve => setTimeout(resolve, 100))
        sid = getUiState().sid
      }

      if (!sid) {
        return sys('startup query skipped: no active session')
      }

      if (STARTUP_IMAGE) {
        try {
          await rpc('image.attach', { path: STARTUP_IMAGE, session_id: sid })
        } catch (e) {
          sys(`startup image attach failed: ${rpcErrorMessage(e)}`)
        }
      }

      submitRef.current(STARTUP_QUERY || 'What do you see in this image?')
    }, 0)
  }

  // 终止状态不能被迟到的实时事件覆盖，否则陈旧的 `subagent.start` 或
  // `spawn_requested` 会覆盖 `failed` 或 `interrupted` 终止状态。
  const isTerminalStatus = (s: SubagentProgress['status']) => s === 'completed' || s === 'failed' || s === 'interrupted'

  const keepTerminalElseRunning = (s: SubagentProgress['status']) => (isTerminalStatus(s) ? s : 'running')

  const handleReady = (skin?: GatewaySkin) => {
    if (skin) {
      applySkinTheme(skin)
    }

    if (STARTUP_RESUME_ID) {
      patchUiState({ status: 'resuming…' })
      resumeById(STARTUP_RESUME_ID)
      scheduleStartupPrompt()

      return
    }

    patchUiState({ status: 'forging session…' })
    newSession()
    scheduleStartupPrompt()
  }

  return (ev: GatewayEvent) => {
    const sid = getUiState().sid

    if (ev.session_id && sid && ev.session_id !== sid && !ev.type.startsWith('gateway.')) {
      return
    }

    switch (ev.type) {
      case 'gateway.ready':
        handleReady(ev.payload?.skin)

        return

      case 'skin.changed':
        if (ev.payload) {
          applySkinTheme(ev.payload)
        }

        return
      case 'session.info': {
        const info = ev.payload

        patchUiState(state => ({
          ...state,
          info,
          status: state.status === 'starting agent…' ? 'ready' : state.status,
          usage: info.usage ? { ...state.usage, ...info.usage } : state.usage
        }))

        setHistoryItems(prev => prev.map(m => (m.kind === 'intro' ? { ...m, info } : m)))

        return
      }

      case 'thinking.delta': {
        const text = ev.payload?.text

        if (text !== undefined) {
          const value = String(text)
          scheduleThinkingStatus(value || statusFromBusy())

          if (value) {
            turnController.recordReasoningDelta(value)
          }
        }

        return
      }

      case 'message.start':
        turnController.startMessage()

        return
      case 'status.update': {
        const p = ev.payload

        if (!p?.text) {
          return
        }

        setStatus(p.text)

        if (p.kind === 'compressing') {
          sys(p.text)

          return
        }

        if (p.kind === 'goal') {
          sys(p.text)

          return
        }

        if (!p.kind || p.kind === 'status') {
          return
        }

        if (turnController.lastStatusNote !== p.text) {
          turnController.lastStatusNote = p.text
          turnController.pushActivity(p.text, p.kind === 'error' ? 'error' : p.kind === 'warn' ? 'warn' : 'info')
        }

        restoreStatusAfter(4000)

        return
      }

      case 'gateway.stderr': {
        const line = String(ev.payload.line).slice(0, 120)

        turnController.pushActivity(line, 'info')

        return
      }

      case 'browser.progress': {
        const message = String(ev.payload?.message ?? '').trim()

        if (message) {
          sys(message)
        }

        return
      }

      case 'gateway.start_timeout': {
        const { cwd, python, stderr_tail: stderrTail } = ev.payload ?? {}
        const trace = python || cwd ? ` · ${String(python || '')} ${String(cwd || '')}`.trim() : ''

        setStatus('gateway startup timeout')
        turnController.pushActivity(`gateway startup timed out${trace} · /logs to inspect`, 'error')

        // 在界面内显示最有用的 stderr 行，让用户无需离开 TUI 即可区分 Python
        // 版本错误、依赖缺失和配置解析失败。取最后 N 行前先过滤空行，避免缓冲区
        // 尾部空行挤掉实际内容；截断长度与 `gateway.stderr` 活动条目的 120 字符
        // 限制保持一致。
        const STDERR_LINE_CAP = 120
        const STDERR_LINES_MAX = 8

        const tailLines = (stderrTail ?? '')
          .split('\n')
          .map(l => l.trim())
          .filter(Boolean)
          .slice(-STDERR_LINES_MAX)

        for (const line of tailLines) {
          turnController.pushActivity(line.slice(0, STDERR_LINE_CAP), 'error')
        }

        return
      }

      case 'gateway.protocol_error':
        setStatus('protocol warning')
        restoreStatusAfter(4000)

        if (!turnController.protocolWarned) {
          turnController.protocolWarned = true
          turnController.pushActivity('protocol noise detected · /logs to inspect', 'info')
        }

        if (ev.payload?.preview) {
          turnController.pushActivity(`protocol noise: ${String(ev.payload.preview).slice(0, 120)}`, 'info')
        }

        return

      case 'reasoning.delta':
        if (ev.payload?.text) {
          turnController.recordReasoningDelta(ev.payload.text)
        }

        return

      case 'reasoning.available':
        turnController.recordReasoningAvailable(String(ev.payload?.text ?? ''))

        return

      case 'tool.progress':
        if (ev.payload?.preview && ev.payload.name) {
          turnController.recordToolProgress(ev.payload.name, ev.payload.preview)
        }

        return

      case 'tool.generating':
        if (ev.payload?.name) {
          turnController.pushTrail(`drafting ${ev.payload.name}…`)
        }

        return

      case 'tool.start':
        turnController.recordTodos(ev.payload.todos)
        turnController.recordToolStart(ev.payload.tool_id, ev.payload.name ?? 'tool', ev.payload.context ?? '')

        return
      case 'tool.complete': {
        const inlineDiffText =
          ev.payload.inline_diff && getUiState().inlineDiffs ? stripAnsi(String(ev.payload.inline_diff)).trim() : ''

        if (inlineDiffText) {
          turnController.recordInlineDiffToolComplete(
            inlineDiffText,
            ev.payload.tool_id,
            ev.payload.name,
            ev.payload.error,
            ev.payload.duration_s
          )
        } else {
          turnController.recordToolComplete(
            ev.payload.tool_id,
            ev.payload.name,
            ev.payload.error,
            ev.payload.summary,
            ev.payload.duration_s,
            ev.payload.todos
          )
        }

        return
      }

      case 'clarify.request':
        patchOverlayState({
          clarify: { choices: ev.payload.choices, question: ev.payload.question, requestId: ev.payload.request_id }
        })
        setStatus('waiting for input…')

        return
      case 'confirm.request':
        patchOverlayState({
          confirm: {
            defaultAnswer: ev.payload.default,
            prompt: ev.payload.prompt,
            requestId: ev.payload.request_id
          }
        })
        setStatus('confirm needed')

        return

      case 'background.complete':
        dropBgTask(ev.payload.task_id)
        sys(`[bg ${ev.payload.task_id}] ${ev.payload.text}`)

        return
      case 'review.summary': {
        // 自我改进后台评审会输出已保存到记忆或技能中的持久摘要。将其作为系统消息
        // 写入对话记录，避免被短暂状态提示淹没；Python 端已完成展示格式化。
        const text = String(ev.payload?.text ?? '').trim()

        if (text) {
          sys(text)
        }

        return
      }

      case 'subagent.spawn_requested':
        // 子智能体已创建但尚未运行，正在等待 ThreadPoolExecutor 槽位；若后续事件
        // 抢先到达，则保留已完成状态。
        turnController.upsertSubagent(ev.payload, c => (isTerminalStatus(c.status) ? {} : { status: 'queued' }))

        return

      case 'subagent.start':
        turnController.upsertSubagent(ev.payload, c => (isTerminalStatus(c.status) ? {} : { status: 'running' }))

        return
      case 'subagent.thinking': {
        const text = String(ev.payload.text ?? '').trim()

        if (!text) {
          return
        }

        // 这里只更新现有项，不能复活漏收 spawn_requested/start 或已通过
        // message.complete 刷新的子智能体。
        turnController.upsertSubagent(
          ev.payload,
          c => ({
            status: keepTerminalElseRunning(c.status),
            thinking: pushThinking(c.thinking, text)
          }),
          { createIfMissing: false }
        )

        return
      }

      case 'subagent.tool': {
        const line = formatToolCall(
          ev.payload.tool_name ?? 'delegate_task',
          ev.payload.tool_preview ?? ev.payload.text ?? ''
        )

        turnController.upsertSubagent(
          ev.payload,
          c => ({
            status: keepTerminalElseRunning(c.status),
            tools: pushTool(c.tools, line)
          }),
          { createIfMissing: false }
        )

        return
      }

      case 'subagent.progress': {
        const text = String(ev.payload.text ?? '').trim()

        if (!text) {
          return
        }

        turnController.upsertSubagent(
          ev.payload,
          c => ({
            notes: pushNote(c.notes, text),
            status: keepTerminalElseRunning(c.status)
          }),
          { createIfMissing: false }
        )

        return
      }

      case 'subagent.complete':
        turnController.upsertSubagent(
          ev.payload,
          c => ({
            durationSeconds: ev.payload.duration_seconds ?? c.durationSeconds,
            status: ev.payload.status ?? 'completed',
            summary: ev.payload.summary || ev.payload.text || c.summary
          }),
          { createIfMissing: false }
        )

        return

      case 'message.delta':
        turnController.recordMessageDelta(ev.payload ?? {})

        return
      case 'message.complete': {
        const { finalMessages, finalText, wasInterrupted } = turnController.recordMessageComplete(ev.payload ?? {})

        if (!wasInterrupted) {
          const msgs: Msg[] = finalMessages.length ? finalMessages : [{ role: 'assistant', text: finalText }]
          msgs.forEach(appendMessage)

          if (bellOnComplete && stdout?.isTTY) {
            stdout.write('\x07')
          }
        }

        setStatus('ready')

        const usage = ev.payload?.usage
        if (usage) {
          patchUiState(state => ({ ...state, usage: mergeUsage(state.usage, usage) }))
        }

        if (sid) {
          releaseSessionImages?.(sid)
        }

        return
      }

      case 'error': {
        const turnWasActive = getUiState().busy

        turnController.recordError()

        if (sid && (turnWasActive || ev.payload?.attachments_discarded)) {
          releaseSessionImages?.(sid)
        }

        const message = String(ev.payload?.message || 'unknown error')

        turnController.pushActivity(message, 'error')

        if (ev.payload?.attachments_discarded) {
          sys('pending image attachment discarded; attach it again before retrying')
        }

        if (NO_PROVIDER_RE.test(message)) {
          panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
          setStatus('setup required')

          return
        }

        sys(`error: ${message}`)
        setStatus('ready')

        return
      }
    }
  }
}
