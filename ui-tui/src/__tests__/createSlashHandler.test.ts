// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'

const buildCtx = () => ({
  composer: {
    attachImage: vi.fn(async () => ({ name: 'example.png', remainder: '' })),
    enqueue: vi.fn(),
    hasSelection: false,
    paste: vi.fn(),
    queueRef: { current: [] },
    selection: { copySelection: vi.fn(async () => '') },
    setInput: vi.fn()
  },
  gateway: {
    gw: { getLogTail: vi.fn(() => ''), request: vi.fn() },
    rpc: vi.fn(() => Promise.resolve({}))
  },
  local: {
    getHistoryItems: vi.fn(() => []),
    getLastUserMsg: vi.fn(() => null),
    maybeWarn: vi.fn()
  },
  session: {
    closeSession: vi.fn(() => Promise.resolve(null)),
    deleteSessionWithFallback: vi.fn(() => Promise.resolve(true)),
    die: vi.fn(),
    guardBusySessionSwitch: vi.fn(() => false),
    newSession: vi.fn(),
    releaseSessionImages: vi.fn(),
    resetVisibleHistory: vi.fn(),
    resumeById: vi.fn(),
    runSessionMutation: vi.fn(async (_what: string, operation: () => unknown) => operation()),
    setSessionStartedAt: vi.fn()
  },
  slashFlightRef: { current: 0 },
  transcript: {
    page: vi.fn(),
    panel: vi.fn(),
    dispatchSubmission: vi.fn(() => true),
    send: vi.fn(),
    setHistoryItems: vi.fn(),
    sys: vi.fn(),
    trimLastExchange: vi.fn((items: unknown[]) => items)
  }
})

describe('createSlashHandler retained contract', () => {
  beforeEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('handles status locally without an RPC compatibility fallback', () => {
    patchUiState({
      info: { memory: 'enabled', model: 'deepseek-chat', provider: 'deepseek', skills: {}, tools: {} },
      sid: 'tui:active',
      usage: { calls: 2, input: 10, output: 5, total: 15 }
    })
    const ctx = buildCtx()

    expect(createSlashHandler(ctx as any)('/status')).toBe(true)
    expect(ctx.transcript.panel).toHaveBeenCalledWith('Runtime status', expect.any(Array))
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('keeps unknown commands local and never dispatches them to a removed catalog', () => {
    const ctx = buildCtx()

    expect(createSlashHandler(ctx as any)('/reload-mcp')).toBe(true)
    expect(ctx.transcript.sys).toHaveBeenCalledWith('unknown command: /reload-mcp; type /help')
    expect(ctx.gateway.rpc).not.toHaveBeenCalled()
    expect(ctx.gateway.gw.request).not.toHaveBeenCalled()
  })

  it('opens retained Session and model overlays locally', () => {
    const ctx = buildCtx()
    const handle = createSlashHandler(ctx as any)

    handle('/resume')
    expect(getOverlayState().picker).toBe(true)

    resetOverlayState()
    handle('/model')
    expect(getOverlayState().modelPicker).toBe(true)
  })

  it('normalizes a bare Session id for /resume', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx as any)('/resume 20260612_061956_b9e391')

    expect(ctx.session.resumeById).toHaveBeenCalledWith('tui:20260612_061956_b9e391')
  })

  it('routes model changes through registered config.set', () => {
    patchUiState({ sid: 'tui:active' })
    const ctx = buildCtx()

    createSlashHandler(ctx as any)('/model deepseek-chat --provider deepseek')

    expect(ctx.gateway.rpc).toHaveBeenCalledWith('config.set', {
      key: 'model',
      provider: 'deepseek',
      session_id: 'tui:active',
      value: 'deepseek-chat'
    })
  })

  it('routes attachment input through registered image.attach', () => {
    patchUiState({ sid: 'tui:active' })
    const ctx = buildCtx()

    createSlashHandler(ctx as any)('/image /tmp/example.png')

    expect(ctx.composer.attachImage).toHaveBeenCalledWith('/tmp/example.png', 'tui:active')
  })

  it('routes /paste through the clipboard image attachment path', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx as any)('/paste')

    expect(ctx.composer.paste).toHaveBeenCalledOnce()
  })

  it('blocks image attachment commands during a Session Switch Flight', () => {
    patchUiState({ sessionSwitching: true, sid: 'tui:active' })
    const ctx = buildCtx()

    createSlashHandler(ctx as any)('/image /tmp/example.png')
    createSlashHandler(ctx as any)('/paste')

    expect(ctx.composer.attachImage).not.toHaveBeenCalled()
    expect(ctx.composer.paste).not.toHaveBeenCalled()
    expect(ctx.transcript.sys).toHaveBeenCalledTimes(2)
  })

  it('routes title and branch through registered Session methods', async () => {
    patchUiState({ sid: 'tui:active' })
    const titleCtx = buildCtx()

    createSlashHandler(titleCtx as any)('/title interview prep')
    expect(titleCtx.gateway.rpc).toHaveBeenCalledWith('session.title', {
      session_id: 'tui:active',
      title: 'interview prep'
    })

    const branchCtx = buildCtx()

    branchCtx.gateway.rpc.mockResolvedValue({ session_id: 'tui:branch' })
    createSlashHandler(branchCtx as any)('/branch alternate plan')
    await vi.waitFor(() => {
      expect(branchCtx.gateway.rpc).toHaveBeenCalledWith('session.branch', {
        name: 'alternate plan',
        session_id: 'tui:active'
      })
    })
  })

  it('releases parent attachments only after the branch commits', async () => {
    patchUiState({ sid: 'tui:active' })
    const ctx = buildCtx()

    ctx.gateway.rpc.mockResolvedValue({ session_id: 'tui:branch' })
    createSlashHandler(ctx as any)('/branch alternate plan')

    expect(ctx.session.runSessionMutation).toHaveBeenCalledWith('branch sessions', expect.any(Function))
    expect(getUiState().sessionSwitching).toBe(true)
    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('forked session: tui:branch')
    })
    expect(ctx.session.closeSession).not.toHaveBeenCalled()
    expect(ctx.session.releaseSessionImages).toHaveBeenCalledWith('tui:active')
    expect(getUiState()).toMatchObject({ sessionSwitching: false, sid: 'tui:branch', status: 'ready' })
  })

  it('keeps parent attachments when the branch does not commit', async () => {
    patchUiState({ sid: 'tui:active' })
    const ctx = buildCtx()

    ctx.gateway.rpc.mockResolvedValue({ session_id: null })
    createSlashHandler(ctx as any)('/branch alternate plan')

    await vi.waitFor(() => {
      expect(ctx.transcript.sys).toHaveBeenCalledWith('session could not be branched')
    })
    expect(ctx.session.releaseSessionImages).not.toHaveBeenCalled()
    expect(getUiState()).toMatchObject({ sessionSwitching: false, sid: 'tui:active', status: 'ready' })
  })

  it('confirms a new Session before invoking the lifecycle', () => {
    const ctx = buildCtx()

    createSlashHandler(ctx as any)('/new interview prep')
    expect(ctx.session.newSession).not.toHaveBeenCalled()

    getOverlayState().confirm?.onConfirm()
    expect(ctx.session.newSession).toHaveBeenCalledWith('New session started', 'interview prep')
  })

  it('runs undo under the shared Session mutation guard', async () => {
    patchUiState({ sid: 'tui:active' })
    const ctx = buildCtx()

    ctx.gateway.rpc.mockResolvedValue({ removed: 2 })
    createSlashHandler(ctx as any)('/undo')

    await vi.waitFor(() => {
      expect(ctx.session.runSessionMutation).toHaveBeenCalledWith('undo the last exchange', expect.any(Function))
      expect(ctx.transcript.sys).toHaveBeenCalledWith('undid 2 messages')
    })
  })

  it('keeps the retry mutation guard and preserves expanded paste payloads', async () => {
    patchUiState({ sid: 'tui:active' })
    const ctx = buildCtx()
    const events: string[] = []
    let resolveUndo: (value: { removed: number }) => void = () => undefined

    const pasteSnips = [{ label: '[[Pasted Content 1]]', text: 'the complete pasted implementation' }]
    ctx.local.getLastUserMsg.mockReturnValue({
      pasteSnips,
      submitText: 'review the complete pasted implementation',
      text: 'review [[Pasted Content 1]]'
    })
    ctx.gateway.rpc.mockReturnValue(
      new Promise(resolve => {
        resolveUndo = resolve
      })
    )
    ctx.session.runSessionMutation.mockImplementation(async (_what: string, operation: () => unknown) => {
      events.push('acquire')
      try {
        return await operation()
      } finally {
        events.push('release')
      }
    })
    ctx.transcript.dispatchSubmission.mockImplementation(() => {
      events.push('dispatch')

      return true
    })

    const slash = createSlashHandler(ctx as any)
    slash('/retry')

    expect(ctx.gateway.rpc).toHaveBeenCalledWith('session.undo', { session_id: 'tui:active' })
    expect(ctx.session.runSessionMutation).toHaveBeenCalledWith('retry the last user message', expect.any(Function))
    expect(events).toEqual(['acquire'])

    slash('/status')
    resolveUndo({ removed: 2 })

    await vi.waitFor(() => {
      expect(ctx.transcript.dispatchSubmission).toHaveBeenCalledWith(
        'review [[Pasted Content 1]]',
        true,
        'review the complete pasted implementation',
        pasteSnips
      )
    })
    expect(ctx.transcript.send).not.toHaveBeenCalled()
    expect(events).toEqual(['acquire', 'dispatch', 'release'])
  })
})
