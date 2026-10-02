// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { Msg } from '../types.js'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { estimateTokensRough } from '../lib/text.js'

const ref = <T>(current: T) => ({ current })

const buildCtx = (appended: Msg[]) =>
  ({
    composer: {
      dequeue: () => undefined,
      queueEditRef: ref<null | number>(null),
      sendQueued: vi.fn(),
      setInput: vi.fn()
    },
    gateway: {
      gw: { request: vi.fn() },
      rpc: vi.fn(async () => null)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: ref(80),
      newSession: vi.fn(),
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: {
      submitRef: { current: vi.fn() }
    },
    system: {
      bellOnComplete: false,
      sys: vi.fn()
    },
    transcript: {
      appendMessage: (msg: Msg) => appended.push(msg),
      panel: (title: string, sections: any[]) =>
        appended.push({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
      setHistoryItems: vi.fn()
    }
  }) as any

describe('createGatewayEventHandler', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ showReasoning: true })
  })

  it('clears a prior cost when completed usage omits unpriced cost', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    patchUiState(state => ({ ...state, usage: { ...state.usage, cost_usd: 0.25 } }))

    onEvent({
      payload: { text: 'done', usage: { calls: 1, input: 1, output: 1, total: 2 } },
      type: 'message.complete'
    } as any)

    expect(getUiState().usage.cost_usd).toBeUndefined()
  })

  it('archives incomplete todos into transcript flow at end of turn so they scroll up', () => {
    const appended: Msg[] = []

    const todos = [
      { content: 'Gather ingredients', id: 'prep', status: 'completed' },
      { content: 'Boil water', id: 'boil', status: 'in_progress' },
      { content: 'Make sauce', id: 'sauce', status: 'pending' }
    ]

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: {}, type: 'message.start' } as any)
    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    expect(getTurnState().todos).toEqual(todos)

    onEvent({ payload: { text: 'Started a todo list.' }, type: 'message.complete' } as any)

    const trail = appended.find(msg => msg.kind === 'trail' && msg.todos?.length)
    const finalText = appended.find(msg => msg.role === 'assistant' && msg.text === 'Started a todo list.')

    expect(finalText).toBeDefined()
    expect(trail).toMatchObject({ kind: 'trail', role: 'system', todos, todoIncomplete: true })
    // 待办归档必须位于最终助手文本上方，避免轮次结束时面板明显跳过最终答案。
    expect(appended.indexOf(trail!)).toBeLessThan(appended.indexOf(finalText!))
    expect(getTurnState().todos).toEqual([])
  })

  it('archives completed todos into transcript flow at end of turn', () => {
    const appended: Msg[] = []
    const todos = [{ content: 'Serve tiny latte', id: 'serve', status: 'completed' }]
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    expect(getTurnState().todos).toEqual([])
    expect(appended).toContainEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      todoCollapsedByDefault: true,
      todos
    })
  })

  it('releases clipboard images when a Session turn completes', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const releaseSessionImages = vi.fn()
    ctx.clipboard = { releaseSessionImages }
    patchUiState({ sid: 'tui:active' })
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    expect(releaseSessionImages).toHaveBeenCalledWith('tui:active')
  })

  it('releases clipboard images when a Session turn fails', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const releaseSessionImages = vi.fn()
    ctx.clipboard = { releaseSessionImages }
    patchUiState({ busy: true, sid: 'tui:active' })
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: { message: 'failed' }, type: 'error' } as any)

    expect(releaseSessionImages).toHaveBeenCalledWith('tui:active')
  })

  it('keeps unsent clipboard images on an idle subscription error', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const releaseSessionImages = vi.fn()
    ctx.clipboard = { releaseSessionImages }
    patchUiState({ busy: false, sid: 'tui:active' })
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: { message: 'subscription_capacity_exceeded' }, type: 'error' } as any)

    expect(releaseSessionImages).not.toHaveBeenCalled()
  })

  it('reports when a rejected Turn discards its pending image attachment', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { attachments_discarded: true, message: 'turn_failed' },
      type: 'error'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith('pending image attachment discarded; attach it again before retrying')
  })

  it('keeps the current todo list visible when the next message starts', () => {
    const appended: Msg[] = []
    const todos = [{ content: 'Boil water', id: 'boil', status: 'in_progress' }]

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    expect(getTurnState().todos).toEqual(todos)

    onEvent({ payload: {}, type: 'message.start' } as any)

    expect(getTurnState().todos).toEqual(todos)
  })

  it('prints compaction progress status into the transcript', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { kind: 'compressing', text: 'compressing 968 messages (~123,400 tok)…' },
      type: 'status.update'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith('compressing 968 messages (~123,400 tok)…')
  })

  it('surfaces self-improvement review summaries as a persistent system line', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({
      payload: { text: "💾 Self-improvement review: Skill 'codeflow-release' patched" },
      type: 'review.summary'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith("💾 Self-improvement review: Skill 'codeflow-release' patched")
  })

  it('ignores review.summary events with empty or missing text', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const onEvent = createGatewayEventHandler(ctx)

    onEvent({ payload: { text: '' }, type: 'review.summary' } as any)
    onEvent({ payload: { text: '   ' }, type: 'review.summary' } as any)
    onEvent({ payload: undefined, type: 'review.summary' } as any)

    expect(ctx.system.sys).not.toHaveBeenCalled()
  })

  it('clears the visible todo list when the todo tool returns an empty list', () => {
    const appended: Msg[] = []
    const todos = [{ content: 'Boil water', id: 'boil', status: 'in_progress' }]
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { name: 'todo', todos, tool_id: 'todo-1' }, type: 'tool.start' } as any)
    expect(getTurnState().todos).toEqual(todos)

    onEvent({ payload: { name: 'todo', todos: [], tool_id: 'todo-1' }, type: 'tool.complete' } as any)

    expect(getTurnState().todos).toEqual([])
  })

  it('persists completed tool rows when message.complete lands immediately after tool.complete', () => {
    const appended: Msg[] = []

    turnController.reasoningText = 'mapped the page'
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { context: 'home page', name: 'search', tool_id: 'tool-1' },
      type: 'tool.start'
    } as any)
    onEvent({
      payload: { name: 'search', preview: 'hero cards' },
      type: 'tool.progress'
    } as any)
    onEvent({
      payload: { summary: 'done', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({
      payload: { text: 'final answer' },
      type: 'message.complete'
    } as any)

    expect(appended).toHaveLength(2)
    expect(appended[0]).toMatchObject({ kind: 'trail', role: 'system', text: '', thinking: 'mapped the page' })
    expect(appended[0]?.tools).toHaveLength(1)
    expect(appended[0]?.tools?.[0]).toContain('hero cards')
    expect(appended[0]?.toolTokens).toBeGreaterThan(0)
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('groups sequential completed tools into one trail when the turn completes', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { context: 'alpha', name: 'search_files', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    onEvent({
      payload: { name: 'search_files', summary: 'first done', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({ payload: { context: 'beta', name: 'read_file', tool_id: 'tool-2' }, type: 'tool.start' } as any)
    onEvent({ payload: { name: 'read_file', summary: 'second done', tool_id: 'tool-2' }, type: 'tool.complete' } as any)

    expect(getTurnState().streamSegments.filter(msg => msg.kind === 'trail' && msg.tools?.length)).toHaveLength(1)
    expect(getTurnState().streamSegments[0]?.tools).toHaveLength(2)
    expect(getTurnState().streamPendingTools).toEqual([])

    onEvent({ payload: { text: '' }, type: 'message.complete' } as any)

    const toolTrails = appended.filter(msg => msg.kind === 'trail' && msg.tools?.length)
    expect(toolTrails).toHaveLength(1)
    expect(toolTrails[0]?.tools).toHaveLength(2)
    expect(toolTrails[0]?.tools?.[0]).toContain('Search Files')
    expect(toolTrails[0]?.tools?.[1]).toContain('Read File')
  })

  it('keeps tool tokens across handler recreation mid-turn', () => {
    const appended: Msg[] = []

    turnController.reasoningText = 'mapped the page'

    createGatewayEventHandler(buildCtx(appended))({
      payload: { context: 'home page', name: 'search', tool_id: 'tool-1' },
      type: 'tool.start'
    } as any)

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: { name: 'search', preview: 'hero cards' },
      type: 'tool.progress'
    } as any)
    onEvent({
      payload: { summary: 'done', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({
      payload: { text: 'final answer' },
      type: 'message.complete'
    } as any)

    expect(appended).toHaveLength(2)
    expect(appended[0]?.tools).toHaveLength(1)
    expect(appended[0]?.toolTokens).toBeGreaterThan(0)
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('streams legacy thinking.delta into visible reasoning state', () => {
    vi.useFakeTimers()
    const appended: Msg[] = []
    const streamed = 'short streamed reasoning'

    createGatewayEventHandler(buildCtx(appended))({ payload: { text: streamed }, type: 'thinking.delta' } as any)
    vi.runOnlyPendingTimers()

    expect(getTurnState().reasoning).toBe(streamed)
    expect(getTurnState().reasoningActive).toBe(true)
    expect(getTurnState().reasoningTokens).toBe(estimateTokensRough(streamed))
    vi.useRealTimers()
  })

  it('preserves streamed reasoning as one completed thinking panel after segment flushes', () => {
    const appended: Msg[] = []
    const streamed = 'first reasoning chunk\nsecond reasoning chunk'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: 'Before edit.' }, type: 'message.delta' } as any)
    turnController.flushStreamingSegment()
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended.map(msg => msg.thinking).filter(Boolean)).toEqual([streamed])
    expect(appended[appended.length - 1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('filters spinner/status-only reasoning noise from completed thinking', () => {
    const appended: Msg[] = []
    const streamed = '(¬_¬) synthesizing...\nactual plan\n( ͡° ͜ʖ ͡°) pondering...\nnext step'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended[0]?.thinking).toBe(streamed)
    expect(appended[0]?.text).toBe('')
    expect(appended[appended.length - 1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('ignores fallback reasoning.available when streamed reasoning already exists', () => {
    const appended: Msg[] = []
    const streamed = 'short streamed reasoning'
    const fallback = 'x'.repeat(400)

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { text: streamed }, type: 'reasoning.delta' } as any)
    onEvent({ payload: { text: fallback }, type: 'reasoning.available' } as any)
    onEvent({ payload: { text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(2)
    expect(appended[0]?.thinking).toBe(streamed)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(streamed))
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('uses message.complete reasoning when no streamed reasoning ref', () => {
    const appended: Msg[] = []
    const fromServer = 'recovered from last_reasoning'

    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { reasoning: fromServer, text: 'final answer' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(2)
    expect(appended[0]?.thinking).toBe(fromServer)
    expect(appended[0]?.thinkingTokens).toBe(estimateTokensRough(fromServer))
    expect(appended[1]).toMatchObject({ role: 'assistant', text: 'final answer' })
  })

  it('renders browser.progress events as system transcript lines as they stream in', () => {
    const appended: Msg[] = []
    const ctx = buildCtx(appended)
    const handler = createGatewayEventHandler(ctx)

    handler({
      payload: { message: 'Chrome launched and listening on port 9222' },
      type: 'browser.progress'
    } as any)

    expect(ctx.system.sys).toHaveBeenCalledWith('Chrome launched and listening on port 9222')
  })

  it('annotates gateway.start_timeout with stderr tail lines so users can diagnose without /logs', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: {
        cwd: '/repo',
        python: '/opt/venv/bin/python',
        stderr_tail:
          '[startup] timed out\nModuleNotFoundError: No module named openai\nFileNotFoundError: ~/.codeflow/config.yaml'
      },
      type: 'gateway.start_timeout'
    } as any)

    const messages = getTurnState().activity.map(a => a.text)

    expect(messages.some(m => m.includes('gateway startup timed out'))).toBe(true)
    expect(messages.some(m => m.includes('ModuleNotFoundError'))).toBe(true)
    expect(messages.some(m => m.includes('FileNotFoundError'))).toBe(true)
  })

  it('prefers raw text over Rich-rendered ANSI on message.complete (#16391)', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const raw = 'CodeFlow here.\n\nLine two.'
    // Rich 渲染的 ANSI（`final_response_markdown: render`）此前会优先，导致 Ink
    // 输出出现可见转义码；必须以原始文本优先。
    const rendered = '\u001b[33mCodeFlow here.\u001b[0m\n\n\u001b[2mLine two.\u001b[0m'

    onEvent({ payload: { rendered, text: raw }, type: 'message.complete' } as any)

    const assistant = appended.find(msg => msg.role === 'assistant')
    expect(assistant?.text).toBe(raw)
    expect(assistant?.text).not.toContain('\u001b[')
  })

  it('falls back to payload.rendered when text is missing on message.complete', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const rendered = 'fallback when gateway omitted text'

    onEvent({ payload: { rendered }, type: 'message.complete' } as any)

    const assistant = appended.find(msg => msg.role === 'assistant')
    expect(assistant?.text).toBe(rendered)
  })

  it('always accumulates raw text in message.delta and ignores `rendered` (#16391)', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    // 部分文本增量流中每个增量都携带 Rich-ANSI 片段。修复前代码会用最新片段
    // 替换整个 bufRef，丢失先前文本。
    onEvent({ payload: { rendered: '\u001b[33mFi\u001b[0m', text: 'Fi' }, type: 'message.delta' } as any)
    onEvent({ payload: { rendered: '\u001b[33mrst.\u001b[0m', text: 'rst.' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: ' second.' }, type: 'message.delta' } as any)
    onEvent({ payload: {}, type: 'message.complete' } as any)

    const assistant = appended.find(msg => msg.role === 'assistant')
    expect(assistant?.text).toBe('First. second.')
  })

  it('anchors inline_diff as its own segment where the edit happened', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '\u001b[31m--- a/foo.ts\u001b[0m\n\u001b[32m+++ b/foo.ts\u001b[0m\n@@\n-old\n+new'
    const cleaned = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const block = `\`\`\`diff\n${cleaned}\n\`\`\``

    // 顺序为叙述、工具、工具完成、更多叙述、消息完成。差异必须落在两个叙述片段
    // 之间，不能附加到最后一段。
    onEvent({ payload: { text: 'Editing the file' }, type: 'message.delta' } as any)
    onEvent({ payload: { context: 'foo.ts', name: 'patch', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    onEvent({ payload: { inline_diff: diff, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)

    // 差异已作为独立片段写入 segmentMessages。
    expect(appended).toHaveLength(0)
    expect(turnController.segmentMessages).toEqual([
      { role: 'assistant', text: 'Editing the file' },
      {
        kind: 'diff',
        role: 'assistant',
        text: block,
        tools: [expect.stringMatching(/^Patch\("foo\.ts"\)(?: \([^)]+\))? ✓$/)]
      }
    ])

    onEvent({ payload: { text: 'patch applied' }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(4)
    expect(appended[0]?.text).toBe('Editing the file')
    expect(appended[1]).toMatchObject({ kind: 'diff', text: block })
    expect(appended[1]?.tools?.[0]).toContain('Patch')
    expect(appended[3]?.text).toBe('patch applied')
    expect(appended[3]?.text).not.toContain('```diff')
  })

  it('keeps full final responses from duplicating flushed pre-diff narration', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const block = `\`\`\`diff\n${diff}\n\`\`\``

    onEvent({ payload: { text: 'Before edit. ' }, type: 'message.delta' } as any)
    onEvent({ payload: { context: 'foo.ts', name: 'patch', tool_id: 'tool-1' }, type: 'tool.start' } as any)
    onEvent({ payload: { inline_diff: diff, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: 'After edit.' }, type: 'message.delta' } as any)
    onEvent({ payload: { text: 'Before edit. After edit.' }, type: 'message.complete' } as any)

    expect(appended.map(msg => msg.text.trim()).filter(Boolean)).toEqual(['Before edit.', block, 'After edit.'])
    expect(appended[1]?.tools?.[0]).toContain('Patch')
  })

  it('drops the diff segment when the final assistant text narrates the same diff', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const cleaned = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const assistantText = `Done. Here's the inline diff:\n\n\`\`\`diff\n${cleaned}\n\`\`\``

    onEvent({ payload: { inline_diff: cleaned, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: assistantText }, type: 'message.complete' } as any)

    // 只保留最终消息；丢弃仅含差异的片段，避免上下重复渲染同一补丁。
    expect(appended).toHaveLength(1)
    expect(appended[0]?.text).toBe(assistantText)
    expect((appended[0]?.text.match(/```diff/g) ?? []).length).toBe(1)
  })

  it('strips the CLI "┊ review diff" header from inline diff segments', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const raw = '  \u001b[33m┊ review diff\u001b[0m\n--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'

    onEvent({ payload: { inline_diff: raw, summary: 'patched', tool_id: 'tool-1' }, type: 'tool.complete' } as any)
    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    // 顺序应为工具轨迹、差异片段（kind='diff'）、最终叙述。
    expect(appended).toHaveLength(2)
    expect(appended[0]?.kind).toBe('diff')
    expect(appended[0]?.text).not.toContain('┊ review diff')
    expect(appended[0]?.text).toContain('--- a/foo.ts')
    expect(appended[0]?.tools?.[0]).toContain('Tool')
    expect(appended[1]?.text).toBe('done')
  })

  it('drops the diff segment when assistant writes its own ```diff fence', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const inlineDiff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'
    const assistantText = 'Done. Clean swap:\n\n```diff\n-old\n+new\n```'

    onEvent({
      payload: { inline_diff: inlineDiff, summary: 'patched', tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({ payload: { text: assistantText }, type: 'message.complete' } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]?.text).toBe(assistantText)
    expect((appended[0]?.text.match(/```diff/g) ?? []).length).toBe(1)
  })

  it('keeps tool trail terse when inline_diff is present', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))
    const diff = '--- a/foo.ts\n+++ b/foo.ts\n@@\n-old\n+new'

    onEvent({
      payload: { inline_diff: diff, name: 'review_diff', summary: diff, tool_id: 'tool-1' },
      type: 'tool.complete'
    } as any)
    onEvent({ payload: { text: 'done' }, type: 'message.complete' } as any)

    // 工具行现在位于差异之前，因此遥测信息不会渲染到该工具产生的补丁下方。
    expect(appended).toHaveLength(2)
    expect(appended[0]?.kind).toBe('diff')
    expect(appended[0]?.text).toContain('```diff')
    expect(appended[0]?.tools?.[0]).toContain('Review Diff')
    expect(appended[0]?.tools?.[0]).not.toContain('--- a/foo.ts')
    expect(appended[1]?.text).toBe('done')
    expect(appended[1]?.tools ?? []).toEqual([])
  })

  it('shows setup panel for missing provider startup error', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({
      payload: {
        message:
          'agent init failed: No LLM provider configured. Run `codeflow model` to select a provider, or run `codeflow setup` for first-time configuration.'
      },
      type: 'error'
    } as any)

    expect(appended).toHaveLength(1)
    expect(appended[0]).toMatchObject({
      kind: 'panel',
      panelData: { title: 'Setup Required' },
      role: 'system'
    })
  })

  it('on gateway.ready with no STARTUP_RESUME_ID, forges a new session without config RPCs', () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = ''

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    expect(newSession).toHaveBeenCalled()
    expect(resumeById).not.toHaveBeenCalled()
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
  })

  it('on gateway.ready with STARTUP_RESUME_ID set, resumes that Session', () => {
    const appended: Msg[] = []
    const newSession = vi.fn()
    const resumeById = vi.fn()
    const ctx = buildCtx(appended)

    ctx.session.newSession = newSession
    ctx.session.resumeById = resumeById
    ctx.session.STARTUP_RESUME_ID = 'env-explicit'

    createGatewayEventHandler(ctx)({ payload: {}, type: 'gateway.ready' } as any)

    expect(resumeById).toHaveBeenCalledWith('env-explicit')
    expect(newSession).not.toHaveBeenCalled()
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
  })

  it('still surfaces terminal turn failures as errors', () => {
    const appended: Msg[] = []
    const onEvent = createGatewayEventHandler(buildCtx(appended))

    onEvent({ payload: { message: 'boom' }, type: 'error' } as any)

    expect(getTurnState().activity).toMatchObject([{ text: 'boom', tone: 'error' }])
  })
})
