// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { SubagentEventPayload } from '../gatewayTypes.js'
import type { ActiveTool, ActivityItem, Msg, SubagentProgress, TodoItem } from '../types.js'

import {
  REASONING_PULSE_MS,
  STREAM_BATCH_MS,
  STREAM_IDLE_BATCH_MS,
  STREAM_SCROLL_BATCH_MS,
  STREAM_TYPING_BATCH_MS
} from '../config/timing.js'
import { appendToolShelfMessage, isToolShelfMessage } from '../lib/liveProgress.js'
import { hasReasoningTag, splitReasoning } from '../lib/reasoning.js'
import {
  boundedLiveRenderText,
  buildToolTrailLine,
  estimateTokensRough,
  isTransientTrailLine,
  sameToolTrailGroup,
  toolTrailLabel
} from '../lib/text.js'
import { resetFlowOverlays } from './overlayStore.js'
import { pushSnapshot } from './spawnHistoryStore.js'
import { archiveDoneTodos, getTurnState, patchTurnState, resetTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const ACTIVITY_LIMIT = 8
const TRAIL_LIMIT = 8

// 从 pushInlineDiffSegment 生成的纯差异片段中提取原始补丁。message.complete 使用它与叙述
// 同一补丁的最终助手文本去重。其他内容返回 null，确保真实助手叙述不受影响。
const diffSegmentBody = (msg: Msg): null | string => {
  if (msg.kind !== 'diff') {
    return null
  }

  const m = msg.text.match(/^```diff\n([\s\S]*?)\n```$/)

  return m ? m[1]! : null
}

const hasDetails = (msg: Msg): boolean => Boolean(msg.thinking || msg.tools?.length || msg.toolTokens)

const isTodoStatus = (status: unknown): status is TodoItem['status'] =>
  status === 'pending' || status === 'in_progress' || status === 'completed' || status === 'cancelled'

const parseTodos = (value: unknown): null | TodoItem[] => {
  if (!Array.isArray(value)) {
    return null
  }

  return value
    .map(item => {
      if (!item || typeof item !== 'object') {
        return null
      }

      const row = item as Record<string, unknown>
      const status = row.status

      if (!isTodoStatus(status)) {
        return null
      }

      return {
        content: String(row.content ?? '').trim(),
        id: String(row.id ?? '').trim(),
        status
      }
    })
    .filter((item): item is TodoItem => Boolean(item?.id && item.content))
}

const textSegments = (segments: Msg[]) =>
  segments.filter(msg => msg.role === 'assistant' && msg.kind !== 'diff').map(msg => msg.text)

const finalTail = (finalText: string, segments: Msg[]) => {
  let tail = finalText

  for (const text of textSegments(segments)) {
    const trimmed = text.trim()

    if (trimmed && tail.startsWith(trimmed)) {
      tail = tail.slice(trimmed.length).trimStart()
    }
  }

  return tail
}

export interface FinalizeInterruptDeps {
  appendMessage?: (msg: Msg) => void
  sys?: (text: string) => void
}

type Timer = null | ReturnType<typeof setTimeout>

const clear = (t: Timer): null => {
  if (t) {
    clearTimeout(t)
  }

  return null
}

class TurnController {
  bufRef = ''
  interrupted = false
  lastStatusNote = ''
  persistedToolLabels = new Set<string>()
  persistSpawnTree?: (subagents: SubagentProgress[], sessionId: null | string) => Promise<void>
  protocolWarned = false
  reasoningText = ''
  segmentMessages: Msg[] = []
  pendingSegmentTools: string[] = []
  statusTimer: Timer = null
  toolTokenAcc = 0
  turnTools: string[] = []

  private activeTools: ActiveTool[] = []
  private activeReasoningText = ''
  private reasoningSegmentIndex: null | number = null
  private activityId = 0
  private reasoningStreamingTimer: Timer = null
  private reasoningTimer: Timer = null
  private streamTimer: Timer = null
  private streamDelay = STREAM_IDLE_BATCH_MS
  private toolProgressTimer: Timer = null

  boostStreamingForTyping() {
    this.streamDelay = STREAM_TYPING_BATCH_MS
  }

  boostStreamingForScroll() {
    this.streamDelay = Math.max(this.streamDelay, STREAM_SCROLL_BATCH_MS)
  }

  relaxStreaming() {
    this.streamDelay = STREAM_IDLE_BATCH_MS
  }

  clearReasoning() {
    this.reasoningTimer = clear(this.reasoningTimer)
    this.activeReasoningText = ''
    this.reasoningSegmentIndex = null
    this.reasoningText = ''
    this.toolTokenAcc = 0
    patchTurnState({ reasoning: '', reasoningTokens: 0, toolTokens: 0 })
  }

  clearStatusTimer() {
    this.statusTimer = clear(this.statusTimer)
  }

  endReasoningPhase() {
    this.reasoningStreamingTimer = clear(this.reasoningStreamingTimer)
    patchTurnState({ reasoningActive: false, reasoningStreaming: false })
  }

  idle() {
    this.endReasoningPhase()
    this.activeTools = []
    this.streamTimer = clear(this.streamTimer)
    this.bufRef = ''
    this.pendingSegmentTools = []
    this.segmentMessages = []

    patchTurnState({
      streamPendingTools: [],
      streamSegments: [],
      streaming: '',
      subagents: [],
      tools: [],
      turnTrail: []
    })
    patchUiState({ busy: false, escapeArmed: false })
    resetFlowOverlays()
  }

  // 在转录中保留中断轮次已流出的内容，并丢弃实时轮次状态。旧 `interruptTurn` 与带类型的 spine
  // 取消路径 chatStream.restoreInputPrompt 共用该逻辑，避免两者对取消后保留内容产生分歧。
  // `appendMessage`/`sys` 可选；没有转录出口的旧调用方或空操作场景只执行空闲清理，不追加内容。
  finalizeInterruptedTurn({ appendMessage, sys }: FinalizeInterruptDeps) {
    // 重入调用（如第二次 Ctrl+C 强制重置与服务器取消错误竞速）会发现轮次状态已排空；不得重复发出
    // 第一次调用已显示的裸中断提示。
    const reentrant = this.interrupted
    this.interrupted = true
    this.closeReasoningSegment()

    const segments = this.segmentMessages
    const partial = this.bufRef.trimStart()
    const tools = this.pendingSegmentTools

    // 将保留快照写入转录前，先从 nanostore 排空流式/片段状态；否则每个已刷新片段会在一帧内
    // 同时出现在 `turn.streamSegments` 和转录中。
    this.idle()
    this.clearReasoning()
    this.turnTools = []
    patchTurnState({ activity: [], outcome: '' })

    if (!appendMessage) {
      return
    }

    for (const msg of segments) {
      appendMessage(msg)
    }

    // 始终显示中断提示：若有正在生成的 `partial` 或待处理工具，将其折叠成单条助手消息；否则
    // 发出系统说明，使转录始终记录轮次已取消，即使只保留了此前 `segments`。
    if (partial || tools.length) {
      appendMessage({
        role: 'assistant',
        text: partial ? `${partial}\n\n*[interrupted]*` : '*[interrupted]*',
        ...(tools.length && { tools })
      })
    } else if (!reentrant) {
      sys?.('interrupted')
    }
  }

  pruneTransient() {
    this.turnTools = this.turnTools.filter(line => !isTransientTrailLine(line))
    patchTurnState(state => {
      const next = state.turnTrail.filter(line => !isTransientTrailLine(line))

      return next.length === state.turnTrail.length ? state : { ...state, turnTrail: next }
    })
  }

  private syncReasoningSegment() {
    const thinking = this.activeReasoningText.trim()

    if (!thinking) {
      return
    }

    const msg: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      thinking,
      thinkingTokens: estimateTokensRough(thinking),
      toolTokens: this.toolTokenAcc || undefined
    }

    if (this.reasoningSegmentIndex === null) {
      this.reasoningSegmentIndex = this.segmentMessages.length
      this.segmentMessages = [...this.segmentMessages, msg]
    } else {
      this.segmentMessages = this.segmentMessages.map((item, i) => (i === this.reasoningSegmentIndex ? msg : item))
    }

    patchTurnState({ streamSegments: this.segmentMessages })
  }

  private closeReasoningSegment() {
    this.syncReasoningSegment()
    this.activeReasoningText = ''
    this.reasoningSegmentIndex = null
  }

  private pushSegment(msg: Msg) {
    this.segmentMessages = appendToolShelfMessage(this.segmentMessages, msg)
  }

  flushStreamingSegment() {
    const raw = this.bufRef.trimStart()

    const split = raw
      ? hasReasoningTag(raw)
        ? splitReasoning(raw)
        : { reasoning: '', text: raw }
      : { reasoning: '', text: '' }

    if (split.reasoning && !this.reasoningText.trim()) {
      this.reasoningText = split.reasoning
      this.activeReasoningText = split.reasoning
      patchTurnState({ reasoning: this.reasoningText, reasoningTokens: estimateTokensRough(this.reasoningText) })
      this.syncReasoningSegment()
    }

    const msg: Msg = {
      role: split.text ? 'assistant' : 'system',
      text: split.text,
      ...(!split.text && { kind: 'trail' as const }),
      ...(this.pendingSegmentTools.length && { tools: this.pendingSegmentTools })
    }

    this.streamTimer = clear(this.streamTimer)

    if (split.text || hasDetails(msg)) {
      this.pushSegment(msg)
    }

    this.pendingSegmentTools = []
    this.bufRef = ''
    patchTurnState({ streamPendingTools: [], streamSegments: this.segmentMessages, streaming: '' })
  }

  pulseReasoningStreaming() {
    this.reasoningStreamingTimer = clear(this.reasoningStreamingTimer)
    patchTurnState({ reasoningActive: true, reasoningStreaming: true })

    this.reasoningStreamingTimer = setTimeout(() => {
      this.reasoningStreamingTimer = null
      patchTurnState({ reasoningStreaming: false })
    }, REASONING_PULSE_MS)
  }

  recordTodos(value: unknown) {
    if (this.interrupted) {
      return
    }

    const todos = parseTodos(value)

    if (todos !== null) {
      patchTurnState({ todos })
    }
  }

  private flushPendingToolsIntoLastSegment() {
    if (!this.pendingSegmentTools.length) {
      return false
    }

    const next = appendToolShelfMessage(this.segmentMessages, {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: this.pendingSegmentTools
    })

    if (next.length === this.segmentMessages.length + 1) {
      return false
    }

    this.segmentMessages = next
    this.pendingSegmentTools = []
    patchTurnState({ streamPendingTools: [], streamSegments: this.segmentMessages })

    return true
  }

  pushInlineDiffSegment(diffText: string, tools: string[] = []) {
    // 移除网关在统一差异前发出的 CLI 装饰，例如 `_emit_inline_diff` 为终端打印器写入的前导
    // "┊ review diff" 标题。该标题只适合作为标准输出装饰，不应出现在 Markdown ```diff 块内。
    const stripped = diffText.replace(/^\s*┊[^\n]*\n?/, '').trim()

    if (!stripped) {
      return
    }

    // 先把正在生成的流式文本刷新为独立片段，使差异落在编辑前的助手叙述和之后的流式内容之间，
    // 而不是粘到最终消息上。这正是片段锚定差异的目的：差异在编辑实际发生的位置渲染。
    this.flushStreamingSegment()

    const block = `\`\`\`diff\n${stripped}\n\`\`\``

    // 跳过连续重复项，例如同一工具触发两次 tool.complete，或两次编辑产生相同补丁。此处保持低成本；
    // 与最终助手文本的深度去重在 message.complete 时执行。
    if (this.segmentMessages.at(-1)?.text === block) {
      return
    }

    this.segmentMessages = [
      ...this.segmentMessages,
      { kind: 'diff', role: 'assistant', text: block, ...(tools.length && { tools }) }
    ]
    patchTurnState({ streamSegments: this.segmentMessages })
  }

  pushActivity(text: string, tone: ActivityItem['tone'] = 'info', replaceLabel?: string) {
    patchTurnState(state => {
      const base = replaceLabel
        ? state.activity.filter(item => !sameToolTrailGroup(replaceLabel, item.text))
        : state.activity

      const tail = base.at(-1)

      if (tail?.text === text && tail.tone === tone) {
        return state
      }

      return { ...state, activity: [...base, { id: ++this.activityId, text, tone }].slice(-ACTIVITY_LIMIT) }
    })
  }

  pushTrail(line: string) {
    if (this.interrupted) {
      return
    }

    patchTurnState(state => {
      if (state.turnTrail.at(-1) === line) {
        return state
      }

      const next = [...state.turnTrail.filter(item => !isTransientTrailLine(item)), line].slice(-TRAIL_LIMIT)

      this.turnTools = next

      return { ...state, turnTrail: next }
    })
  }

  recordError() {
    this.idle()
    this.clearReasoning()
    this.clearStatusTimer()
    this.pendingSegmentTools = []
    this.segmentMessages = []
    this.turnTools = []
    this.persistedToolLabels.clear()
  }

  recordMessageComplete(payload: { rendered?: string; reasoning?: string; text?: string }) {
    this.closeReasoningSegment()

    // Ink 通过 <Md> 渲染 Markdown；网关的 Rich ANSI 结果（`payload.rendered`）供无法渲染的终端
    // 使用。用户启用 `display.final_response_markdown: render` 时，若此处优先使用 `rendered`，
    // 原始 ANSI 转义会进入 React 树并造成乱码。应优先原始文本，仅在网关未发送文本时回退（#16391）。
    const rawText = (payload.text ?? payload.rendered ?? this.bufRef).trimStart()
    const split = splitReasoning(rawText)
    const finalText = finalTail(split.text, this.segmentMessages)
    const existingReasoning = this.reasoningText.trim() || String(payload.reasoning ?? '').trim()
    const savedReasoning = [existingReasoning, existingReasoning ? '' : split.reasoning].filter(Boolean).join('\n\n')
    const savedToolTokens = this.toolTokenAcc
    let tools = this.pendingSegmentTools
    const last = this.segmentMessages[this.segmentMessages.length - 1]

    if (tools.length && isToolShelfMessage(last)) {
      this.segmentMessages = [
        ...this.segmentMessages.slice(0, -1),
        { ...last, tools: [...(last.tools ?? []), ...tools] }
      ]
      this.pendingSegmentTools = []
      tools = []
    }

    // 删除智能体即将在最终回复中叙述的纯差异片段。否则收尾“这是差异……”消息会堆叠渲染两份
    // 相同补丁。仅处理 pushInlineDiffSegment 发出的 `kind: 'diff'` 片段，真实助手叙述保持不动。
    const finalHasOwnDiffFence = /```(?:diff|patch)\b/i.test(finalText)

    const segments = this.segmentMessages.filter(msg => {
      const body = diffSegmentBody(msg)

      return body === null || (!finalHasOwnDiffFence && !finalText.includes(body))
    })

    const hasReasoningSegment =
      this.reasoningSegmentIndex !== null || segments.some(msg => Boolean(msg.thinking?.trim()))

    const finalThinking = hasReasoningSegment ? '' : savedReasoning.trim()

    const finalDetails: Msg = {
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: finalThinking || undefined,
      thinkingTokens: finalThinking ? estimateTokensRough(finalThinking) : undefined,
      toolTokens: savedToolTokens || undefined,
      ...(tools.length && { tools })
    }

    // 将归档前置，使轨迹消息锚定在用户提示下，而不是思考/工具与最终助手文本之间。
    const finalMessages: Msg[] = [
      ...archiveDoneTodos(),
      ...segments,
      ...(hasDetails(finalDetails) ? [finalDetails] : [])
    ]

    if (finalText) {
      finalMessages.push({ role: 'assistant', text: finalText })
    }

    const wasInterrupted = this.interrupted

    // 在 idle() 从 turnState 丢弃子智能体前，将轮次 spawn 树归档到历史。这样 /replay 和浮层历史
    // 导航无需往返磁盘即可调出已完成的扇出。
    const finishedSubagents = getTurnState().subagents
    const sessionId = getUiState().sid

    if (finishedSubagents.length > 0) {
      pushSnapshot(finishedSubagents, { sessionId, startedAt: null })
      // 异步写入磁盘，使 /replay 在进程重启后仍可用。同一快照通过 spawnHistoryStore 保留在内存中
      // 供立即召回；磁盘用于长期归档。
      void this.persistSpawnTree?.(finishedSubagents, sessionId)
    }

    this.idle()
    this.clearReasoning()
    this.turnTools = []
    this.persistedToolLabels.clear()
    this.bufRef = ''
    this.interrupted = false
    patchTurnState({ activity: [], outcome: '' })

    return { finalMessages, finalText, wasInterrupted }
  }

  recordMessageDelta({ text }: { rendered?: string; text?: string }) {
    if (this.interrupted || !text) {
      return
    }

    this.pruneTransient()
    this.endReasoningPhase()

    // 始终累积原始文本增量。#16391 前的路径会用增量 Rich ANSI 片段 `rendered` 替换整个缓冲区，
    // 每次 tick 都丢弃此前已流出的全部内容；启用 `display.final_response_markdown: render` 时表现为
    // 彩色文本重叠和正文丢失。
    this.bufRef += text

    if (getUiState().streaming) {
      this.scheduleStreaming()
    }
  }

  recordReasoningAvailable(text: string) {
    if (this.interrupted || !getUiState().showReasoning) {
      return
    }

    const incoming = text.trim()

    if (!incoming || this.reasoningText.trim()) {
      return
    }

    this.reasoningText = incoming
    this.activeReasoningText = incoming
    this.scheduleReasoning()
    this.syncReasoningSegment()
    this.pulseReasoningStreaming()
  }

  recordReasoningDelta(text: string) {
    if (this.interrupted || !getUiState().showReasoning) {
      return
    }

    if (!this.activeReasoningText.trim() && this.pendingSegmentTools.length) {
      this.flushStreamingSegment()
    }

    this.reasoningText += text
    this.activeReasoningText += text

    if (this.reasoningText.length > 80_000) {
      this.reasoningText = this.reasoningText.slice(-60_000)
    }

    this.scheduleReasoning()
    this.syncReasoningSegment()
    this.pulseReasoningStreaming()
  }

  recordToolComplete(
    toolId: string,
    fallbackName?: string,
    error?: string,
    summary?: string,
    duration?: number,
    todos?: unknown
  ) {
    if (this.interrupted) {
      return
    }

    this.recordTodos(todos)
    const line = this.completeTool(toolId, fallbackName, error, summary, duration)

    this.pendingSegmentTools = [...this.pendingSegmentTools, line]
    this.flushPendingToolsIntoLastSegment()
    this.publishToolState()
  }

  recordInlineDiffToolComplete(
    diffText: string,
    toolId: string,
    fallbackName?: string,
    error?: string,
    duration?: number
  ) {
    if (this.interrupted) {
      return
    }

    this.flushStreamingSegment()
    this.pushInlineDiffSegment(diffText, [this.completeTool(toolId, fallbackName, error, '', duration)])
    this.publishToolState()
  }

  private completeTool(toolId: string, fallbackName?: string, error?: string, summary?: string, duration?: number) {
    const done = this.activeTools.find(tool => tool.id === toolId)
    const name = done?.name ?? fallbackName ?? 'tool'
    const label = toolTrailLabel(name)
    const fallbackDuration = done?.startedAt ? (Date.now() - done.startedAt) / 1000 : undefined

    const line = buildToolTrailLine(
      name,
      done?.context || '',
      Boolean(error),
      error || summary || '',
      duration ?? fallbackDuration
    )

    this.activeTools = this.activeTools.filter(tool => tool.id !== toolId)

    const next = this.turnTools.filter(item => !sameToolTrailGroup(label, item))

    if (!this.activeTools.length) {
      next.push('analyzing tool output…')
    }

    this.turnTools = next.slice(-TRAIL_LIMIT)

    return line
  }

  private publishToolState() {
    patchTurnState({
      streamPendingTools: this.pendingSegmentTools,
      tools: this.activeTools,
      turnTrail: this.turnTools
    })
  }

  recordToolProgress(toolName: string, preview: string) {
    if (this.interrupted) {
      return
    }

    const index = this.activeTools.findIndex(tool => tool.name === toolName)

    if (index < 0) {
      return
    }

    this.activeTools = this.activeTools.map((tool, i) => (i === index ? { ...tool, context: preview } : tool))

    if (this.toolProgressTimer) {
      return
    }

    this.toolProgressTimer = setTimeout(() => {
      this.toolProgressTimer = null
      patchTurnState({ tools: [...this.activeTools] })
    }, STREAM_BATCH_MS)
  }

  recordToolStart(toolId: string, name: string, context: string) {
    if (this.interrupted) {
      return
    }

    this.flushStreamingSegment()
    this.closeReasoningSegment()
    this.pruneTransient()
    this.endReasoningPhase()

    const sample = `${name} ${context}`.trim()

    this.toolTokenAcc += sample ? estimateTokensRough(sample) : 0
    this.activeTools = [...this.activeTools, { context, id: toolId, name, startedAt: Date.now() }]

    patchTurnState({ toolTokens: this.toolTokenAcc, tools: this.activeTools })
  }

  reset() {
    this.clearReasoning()
    this.clearStatusTimer()
    this.idle()
    this.bufRef = ''
    this.interrupted = false
    this.lastStatusNote = ''
    this.activeReasoningText = ''
    this.pendingSegmentTools = []
    this.protocolWarned = false
    this.reasoningSegmentIndex = null
    this.segmentMessages = []
    this.turnTools = []
    this.toolTokenAcc = 0
    this.persistedToolLabels.clear()
    patchTurnState({ activity: [], outcome: '' })
  }

  fullReset() {
    this.reset()
    resetTurnState()
  }

  scheduleReasoning() {
    if (this.reasoningTimer) {
      return
    }

    this.reasoningTimer = setTimeout(() => {
      this.reasoningTimer = null
      patchTurnState({
        reasoning: this.reasoningText,
        reasoningTokens: estimateTokensRough(this.reasoningText)
      })
    }, STREAM_BATCH_MS)
  }

  scheduleStreaming() {
    if (this.streamTimer) {
      return
    }

    this.streamTimer = setTimeout(() => {
      this.streamTimer = null
      const raw = this.bufRef.trimStart()
      const visible = hasReasoningTag(raw) ? splitReasoning(raw).text : raw
      patchTurnState({ streaming: boundedLiveRenderText(visible) })
    }, this.streamDelay)
  }

  startMessage() {
    this.endReasoningPhase()
    this.clearReasoning()
    this.activeTools = []
    this.activeReasoningText = ''
    this.reasoningSegmentIndex = null
    this.turnTools = []
    this.toolTokenAcc = 0
    this.interrupted = false
    this.persistedToolLabels.clear()
    patchUiState({ busy: true })
    patchTurnState({ activity: [], outcome: '', subagents: [], toolTokens: 0, tools: [], turnTrail: [] })
  }

  upsertSubagent(
    p: SubagentEventPayload,
    patch: (current: SubagentProgress) => Partial<SubagentProgress>,
    opts: { createIfMissing?: boolean } = { createIfMissing: true }
  ) {
    // 稳定 ID：优先使用服务器签发的 subagent_id，可跨嵌套孙节点和跨树关联保持稳定。旧网关省略
    // 该字段时回退到复合键；这些网关只产生扁平列表。
    const id = p.subagent_id || `sa:${p.task_index}:${p.goal || 'subagent'}`

    patchTurnState(state => {
      const existing = state.subagents.find(item => item.id === id)

      // 迟到事件（message.complete 已触发 idle() 后到达的 subagent.complete/tool/progress）原本会
      // 把已完成子智能体重新放入 turn.subagents，阻止 /agents 浮层显示“已完成”标题。
      // `createIfMissing` 为 false 时静默丢弃。
      if (!existing && !opts.createIfMissing) {
        return state
      }

      const base: SubagentProgress = existing ?? {
        depth: p.depth ?? 0,
        goal: p.goal,
        id,
        index: p.task_index,
        model: p.model,
        notes: [],
        parentId: p.parent_id ?? null,
        startedAt: Date.now(),
        status: 'running',
        taskCount: p.task_count ?? 1,
        thinking: [],
        toolCount: p.tool_count ?? 0,
        tools: [],
        toolsets: p.toolsets
      }

      // 将 snake_case 载荷键映射到 camelCase 状态。仅在事件实际携带字段时覆盖；`??` 会在只发出
      // 部分载荷的流式事件之间保留旧值。
      const outputTail = p.output_tail
        ? p.output_tail.map(e => ({
            isError: Boolean(e.is_error),
            preview: String(e.preview ?? ''),
            tool: String(e.tool ?? 'tool')
          }))
        : base.outputTail

      const next: SubagentProgress = {
        ...base,
        apiCalls: p.api_calls ?? base.apiCalls,
        costUsd: p.cost_usd ?? base.costUsd,
        depth: p.depth ?? base.depth,
        filesRead: p.files_read ?? base.filesRead,
        filesWritten: p.files_written ?? base.filesWritten,
        goal: p.goal || base.goal,
        inputTokens: p.input_tokens ?? base.inputTokens,
        iteration: p.iteration ?? base.iteration,
        model: p.model ?? base.model,
        outputTail,
        outputTokens: p.output_tokens ?? base.outputTokens,
        parentId: p.parent_id ?? base.parentId,
        reasoningTokens: p.reasoning_tokens ?? base.reasoningTokens,
        taskCount: p.task_count ?? base.taskCount,
        toolCount: p.tool_count ?? base.toolCount,
        toolsets: p.toolsets ?? base.toolsets,
        ...patch(base)
      }

      // 稳定顺序按 spawn 的深度、父节点和索引，而非插入时间。若无此规则，高并发下事件乱序到达时，
      // 孙节点会相对同级节点重新洗牌。
      const subagents = existing
        ? state.subagents.map(item => (item.id === id ? next : item))
        : [...state.subagents, next].sort((a, b) => a.depth - b.depth || a.index - b.index)

      return { ...state, subagents }
    })
  }
}

export const turnController = new TurnController()

export type { TurnController }
