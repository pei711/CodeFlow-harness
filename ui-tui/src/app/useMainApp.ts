// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { type ScrollBoxHandle, useApp, useHasSelection, useSelection, useStdout, useTerminalTitle } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type {
  ClarifyRespondResponse,
  GatewayEvent,
  ImageAttachResponse,
  TerminalResizeResponse
} from '../gatewayTypes.js'
import type { Msg, PanelSection } from '../types.js'

import { STARTUP_RESUME_ID } from '../config/env.js'
import { FULL_RENDER_TAIL_ITEMS, MAX_HISTORY, WHEEL_SCROLL_STEP } from '../config/limits.js'
import { SECTION_NAMES, sectionMode } from '../domain/details.js'
import { attachedImageNotice } from '../domain/messages.js'
import { fmtCwdBranch, shortCwd } from '../domain/paths.js'
import { useGitBranch } from '../hooks/useGitBranch.js'
import { useVirtualHistory } from '../hooks/useVirtualHistory.js'
import {
  claimClipboardImages,
  readClipboardImage,
  releaseClipboardImage,
  releaseClipboardImages,
  releaseSubmittedClipboardImages,
  retainClipboardImage,
  unclaimClipboardImages
} from '../lib/clipboard.js'
import { composerPromptWidth } from '../lib/inputMetrics.js'
import { appendTranscriptMessage } from '../lib/messages.js'
import { isMac } from '../lib/platform.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { terminalParityHints } from '../lib/terminalParity.js'
import { buildToolTrailLine, sameToolTrailGroup, toolTrailLabel } from '../lib/text.js'
import { estimatedMsgHeight, messageHeightKey } from '../lib/virtualHeights.js'
import { type TuiRpcClient } from '../tuiRpcClient.js'
import { createChatStream, type ChatStreamHandle, type ChatStreamRpcClient } from './chatStream.js'
import { answerConfirmRequest } from './confirmResponse.js'
import { createGatewayEventHandler } from './createGatewayEventHandler.js'
import { createSlashHandler } from './createSlashHandler.js'
import { getInputSelection } from './inputSelectionStore.js'
import { type GatewayRpc, type LastUserSubmission, type TranscriptRow } from './interfaces.js'
import { $overlayState, patchOverlayState } from './overlayStore.js'
import { scrollWithSelectionBy } from './scroll.js'
import { turnController } from './turnController.js'
import { patchTurnState, useTurnSelector } from './turnStore.js'
import { $uiState, getUiState, patchUiState } from './uiStore.js'
import { trackPasteFlight, useComposerState } from './useComposerState.js'
import { useInputHandlers } from './useInputHandlers.js'
import { useLongRunToolCharms } from './useLongRunToolCharms.js'
import { useSessionLifecycle } from './useSessionLifecycle.js'
import { dispatchNextQueuedSubmission, useSubmission } from './useSubmission.js'

const GOOD_VIBES_RE = /\b(good bot|thanks|thank you|thx|ty|ily|love you)\b/i
const BRACKET_PASTE_ON = '\x1b[?2004h'
const BRACKET_PASTE_OFF = '\x1b[?2004l'
const MAX_HEIGHT_CACHE_BUCKETS = 12

const capHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0]!, ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

// 带类型聊天路径的会话键接缝：流句柄必须使用 session.create / session.resume 签发的 ui.sid，
// 绝不能使用硬编码默认值。将其导出，使测试能固定此行为，防止回归。
export const buildChatStreamHandle = (
  rpcClient: ChatStreamRpcClient | undefined,
  sid: null | string,
  sys: (text: string) => void,
  appendMessage: (msg: Msg) => void,
  releaseSessionImages: (sessionId: string, claimId?: string) => void = releaseSubmittedClipboardImages
): ChatStreamHandle | null =>
  rpcClient && sid
    ? createChatStream({
        appendMessage,
        releaseSessionImages,
        rollbackSessionImages: unclaimClipboardImages,
        rpcClient,
        sessionKey: sid,
        sys
      })
    : null

type ClipboardImagePasteResult =
  | { info: ImageAttachResponse; status: 'attached' }
  | { status: 'busy' | 'empty' | 'failed' | 'no-session' | 'stale' }

export async function pasteClipboardImage(
  rpc: GatewayRpc,
  sessionId: null | string,
  readImage: () => Promise<null | string> = readClipboardImage,
  currentSession: () => null | string = () => getUiState().sid,
  releaseImage: (path: string) => void = releaseClipboardImage,
  retainImage: (path: string, sessionId: string) => boolean = retainClipboardImage,
  canAttach: () => boolean = () => {
    const ui = getUiState()

    return !ui.busy && !ui.sessionMutating && !ui.sessionSwitching
  }
): Promise<ClipboardImagePasteResult> {
  if (!sessionId) {
    return { status: 'no-session' }
  }

  if (!canAttach()) {
    return { status: 'busy' }
  }

  const path = await readImage()

  if (!path) {
    return { status: 'empty' }
  }

  if (currentSession() !== sessionId) {
    releaseImage(path)

    return { status: 'stale' }
  }

  if (!canAttach()) {
    releaseImage(path)

    return { status: 'busy' }
  }

  retainImage(path, sessionId)

  let info

  try {
    info = await rpc<ImageAttachResponse>('image.attach', { path, session_id: sessionId })
  } catch {
    releaseImage(path)

    return { status: 'failed' }
  }

  if (!info) {
    releaseImage(path)

    return { status: 'failed' }
  }

  if (currentSession() !== sessionId || !canAttach()) {
    releaseImage(path)

    return { status: 'stale' }
  }

  return { info, status: 'attached' }
}

const statusColorOf = (status: string, t: { error: string; muted: string; ok: string; warn: string }) => {
  if (status === 'ready') {
    return t.ok
  }

  if (status.startsWith('error')) {
    return t.error
  }

  if (status === 'interrupted') {
    return t.warn
  }

  return t.muted
}

export function useMainApp(gw: TuiRpcClient, rpcClient?: ChatStreamRpcClient) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const [cols, setCols] = useState(stdout?.columns ?? 80)

  useEffect(() => {
    if (!stdout) {
      return
    }

    const sync = () => setCols(stdout.columns ?? 80)

    stdout.on('resize', sync)

    if (stdout.isTTY) {
      stdout.write(BRACKET_PASTE_ON)
    }

    return () => {
      stdout.off('resize', sync)

      if (stdout.isTTY) {
        stdout.write(BRACKET_PASTE_OFF)
      }
    }
  }, [stdout])

  const [historyItems, setHistoryItems] = useState<Msg[]>(() => [{ kind: 'intro', role: 'system', text: '' }])
  const [lastUserMsg, setLastUserMsg] = useState<LastUserSubmission | null>(null)
  const [stickyPrompt, setStickyPrompt] = useState('')
  const [sessionStartedAt, setSessionStartedAt] = useState(() => Date.now())
  const [turnStartedAt, setTurnStartedAt] = useState<null | number>(null)
  const [goodVibesTick, setGoodVibesTick] = useState(0)
  const [bellOnComplete, setBellOnComplete] = useState(false)
  const [clipboardPasteTick, setClipboardPasteTick] = useState(0)

  const ui = useStore($uiState)
  const overlay = useStore($overlayState)

  const turnLiveTailActive = useTurnSelector(state =>
    Boolean(
      state.streaming ||
      state.streamPendingTools.length ||
      state.streamSegments.length ||
      state.reasoning.trim() ||
      state.reasoningActive ||
      state.tools.length ||
      state.subagents.length ||
      state.todos.length
    )
  )

  const slashFlightRef = useRef(0)
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const colsRef = useRef(cols)
  const scrollRef = useRef<null | ScrollBoxHandle>(null)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  const clipboardPasteRef = useRef<(quiet?: boolean) => Promise<void> | void>(() => {})
  const clipboardPasteFlightRef = useRef<Promise<void> | null>(null)
  const submitRef = useRef<(value: string) => void>(() => {})
  const terminalHintsShownRef = useRef(new Set<string>())
  const historyItemsRef = useRef(historyItems)
  const lastUserMsgRef = useRef(lastUserMsg)
  const msgIdsRef = useRef(new WeakMap<Msg, string>())
  const msgIdSeqRef = useRef(0)
  const heightCachesRef = useRef(new Map<string, Map<string, number>>())

  colsRef.current = cols
  historyItemsRef.current = historyItems
  lastUserMsgRef.current = lastUserMsg

  const hasSelection = useHasSelection()
  const selection = useSelection()
  const lastCopiedVersionRef = useRef(-1)

  useEffect(() => {
    selection.setSelectionBgColor(ui.theme.color.selectionBg)
  }, [selection, ui.theme.color.selectionBg])

  // macOS Terminal.app 不会把 Cmd+C 转发给启用鼠标追踪的全屏 TUI，因此唯一可靠且符合原生体验的
  // 路径是 iTerm 式选中即复制：拖拽产生稳定 TUI 选区后写入系统剪贴板，同时保留高亮。
  //
  // 直接通过 Ink 选择总线订阅，而非 useSyncExternalStore，避免 React 在每次拖拽移动 tick 时
  // 重渲染 MainApp。版本 Ref 用于去除重入通知。
  useEffect(() => {
    if (!isMac) {
      return
    }

    return selection.subscribe(() => {
      if (!selection.hasSelection()) {
        return
      }

      const state = selection.getState() as { isDragging?: boolean } | null

      if (state?.isDragging) {
        return
      }

      const version = selection.version()

      if (version === lastCopiedVersionRef.current) {
        return
      }

      lastCopiedVersionRef.current = version
      void selection.copySelectionNoClear()
    })
  }, [selection])

  const clearSelection = useCallback(() => {
    selection.clearSelection()
    getInputSelection()?.collapseToEnd()
  }, [selection])

  const composer = useComposerState({
    gw,
    onClipboardPaste: quiet => clipboardPasteRef.current(quiet),
    onImageAttached: info => {
      sys(attachedImageNotice(info))
    },
    onPasteSettled: () => setClipboardPasteTick(value => value + 1),
    pasteFlightRef: clipboardPasteFlightRef,
    submitRef
  })

  const { actions: composerActions, refs: composerRefs, state: composerState } = composer
  const empty = !historyItems.some(msg => msg.kind !== 'intro')

  useEffect(() => {
    void terminalParityHints()
      .then(hints => {
        for (const hint of hints) {
          if (terminalHintsShownRef.current.has(hint.key)) {
            continue
          }

          terminalHintsShownRef.current.add(hint.key)
          turnController.pushActivity(hint.message, hint.tone)
        }
      })
      .catch(() => {})
  }, [])

  const messageId = useCallback((msg: Msg) => {
    const hit = msgIdsRef.current.get(msg)

    if (hit) {
      return hit
    }

    const next = `${messageHeightKey(msg)}:${++msgIdSeqRef.current}`

    msgIdsRef.current.set(msg, next)

    return next
  }, [])

  const virtualRows = useMemo<TranscriptRow[]>(
    () => historyItems.map((msg, index) => ({ index, key: messageId(msg), msg })),
    [historyItems, messageId]
  )

  const detailsLayoutKey = useMemo(() => {
    const thinking = sectionMode('thinking', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride)
    const tools = sectionMode('tools', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride)

    return `${thinking}:${tools}`
  }, [ui.detailsMode, ui.detailsModeCommandOverride, ui.sections])

  const detailsVisible = detailsLayoutKey !== 'hidden:hidden'
  const userPromptWidth = composerPromptWidth(ui.theme.brand.prompt)
  const heightCacheKey = `${ui.sid ?? 'draft'}:${cols}:${userPromptWidth}:${ui.compact ? '1' : '0'}:${detailsLayoutKey}`

  const heightCache = useMemo(() => {
    let cache = heightCachesRef.current.get(heightCacheKey)

    if (!cache) {
      cache = new Map()
      heightCachesRef.current.set(heightCacheKey, cache)

      if (heightCachesRef.current.size > MAX_HEIGHT_CACHE_BUCKETS) {
        heightCachesRef.current.delete(heightCachesRef.current.keys().next().value!)
      }
    }

    return cache
  }, [heightCacheKey])

  // 第一条用户角色消息的索引。appLayout.tsx 的分隔符渲染会跳过该行，高度估算器也必须跳过。
  // 尚无用户消息时为 -1，此时不会有行通过门控。
  const firstUserIdx = useMemo(() => virtualRows.findIndex(r => r.msg.role === 'user'), [virtualRows])

  const estimateRowHeight = useCallback(
    (index: number) =>
      estimatedMsgHeight(virtualRows[index]!.msg, cols, {
        compact: ui.compact,
        details: detailsVisible,
        limitHistory: index < virtualRows.length - FULL_RENDER_TAIL_ITEMS,
        userPrompt: ui.theme.brand.prompt,
        withSeparator: virtualRows[index]!.msg.role === 'user' && firstUserIdx >= 0 && index > firstUserIdx
      }),
    [cols, detailsVisible, firstUserIdx, ui.compact, ui.theme.brand.prompt, virtualRows]
  )

  const syncHeightCache = useCallback(
    (heights: ReadonlyMap<string, number>) => {
      for (const row of virtualRows) {
        const h = heights.get(row.key)

        if (h) {
          heightCache.set(row.key, h)
        }
      }
    },
    [heightCache, virtualRows]
  )

  const virtualHistory = useVirtualHistory(scrollRef, virtualRows, cols, {
    estimateHeight: estimateRowHeight,
    initialHeights: heightCache,
    liveTailActive: turnLiveTailActive,
    onHeightsChange: syncHeightCache
  })

  const scrollWithSelection = useCallback(
    (delta: number) => scrollWithSelectionBy(delta, { scrollRef, selection }),
    [selection]
  )

  const appendMessage = useCallback(
    (msg: Msg) => setHistoryItems(prev => capHistory(appendTranscriptMessage(prev, msg))),
    []
  )

  const sys = useCallback((text: string) => appendMessage({ role: 'system', text }), [appendMessage])

  const page = useCallback(
    (text: string, title?: string) => patchOverlayState({ pager: { lines: text.split('\n'), offset: 0, title } }),
    []
  )

  const panel = useCallback(
    (title: string, sections: PanelSection[]) =>
      appendMessage({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
    [appendMessage]
  )

  const maybeWarn = useCallback(
    (value: unknown) => {
      const warning = (value as { warning?: unknown } | null)?.warning

      if (typeof warning === 'string' && warning) {
        sys(`warning: ${warning}`)
      }
    },
    [sys]
  )

  const maybeGoodVibes = useCallback((text: string) => {
    if (GOOD_VIBES_RE.test(text)) {
      setGoodVibesTick(v => v + 1)
    }
  }, [])

  const rpc: GatewayRpc = useCallback(
    async <T extends object = Record<string, unknown>>(method: string, params: Record<string, unknown> = {}) => {
      try {
        const result = asRpcResult<T>(await gw.request<T>(method, params))

        if (result) {
          return result
        }

        sys(`error: invalid response: ${method}`)
      } catch (e) {
        sys(`error: ${rpcErrorMessage(e)}`)
      }

      return null
    },
    [gw, sys]
  )

  // 第 6 阶段带类型聊天路径脚手架，见 design.md 第 D7 节。每次 `sid` 变化构造一次
  // `ChatStreamHandle`，并通过 `chatStreamRef` 暴露，使 useInputHandlers 中的 Ctrl+C 处理器
  // 可在带类型轮次进行时路由到 `turn.cancel`。实时 `attach()` 延迟到 Python `turn.*` 处理器
  // 交付（独立 L2 阶段）；此前句柄保持分离且为空操作，但接线已就绪，切换只需改一行。
  const chatStreamRef = useRef<ChatStreamHandle | null>(null)

  const gateway = useMemo(() => ({ gw, rpc, rpcClient }), [gw, rpc, rpcClient])

  const die = useCallback(() => {
    gw.kill()
    exit()
    // Ink 的 exit() 调用 unmount() 重置终端模式，但不会调用 process.exit()。若不显式退出，
    // 标准输入监听器会保持事件循环，Node 进程继续存活，entry.tsx 中发送最终
    // resetTerminalModes() 的 process.on('exit') 处理器永不触发，使父 Shell 仍启用 kitty 键盘
    // 协议、鼠标模式等。见问题 #19194。
    process.exit(0)
  }, [exit, gw])

  const session = useSessionLifecycle({
    colsRef,
    composerActions,
    gw,
    panel,
    pendingPasteRef: clipboardPasteFlightRef,
    releaseSessionImages: releaseClipboardImages,
    rpc,
    scrollRef,
    setHistoryItems,
    setLastUserMsg,
    setSessionStartedAt,
    setStickyPrompt,
    sys
  })

  useEffect(() => {
    if (ui.busy) {
      setTurnStartedAt(prev => prev ?? Date.now())
    } else {
      setTurnStartedAt(null)
    }
  }, [ui.busy])

  // 会话标识变化时安装或替换带类型聊天流句柄。句柄会立即创建并 attach()，使 turn.subscribe
  // 通过带类型路径流式传递 token.delta 事件，即 design.md 第 D7 节的第 4 阶段轮次流。
  // useInputHandlers 中的 Ctrl+C 处理器读取 `chatStreamRef.current`，轮次进行时优先使用带类型
  // `turn.cancel`。
  useEffect(() => {
    const handle = buildChatStreamHandle(rpcClient, ui.sid, sys, appendMessage, releaseSubmittedClipboardImages)

    if (!handle) {
      chatStreamRef.current = null

      return
    }

    chatStreamRef.current = handle
    void handle.attach().catch(err => {
      // 将订阅失败显示到系统消息日志，不让 React 树崩溃；用户输入时 turn.cancel/send 仍可调用。
      sys(`chat stream attach failed: ${String(err)}`)
    })

    return () => {
      chatStreamRef.current = null
      void handle.detach().catch(() => {})
    }
  }, [rpcClient, ui.sid, sys, appendMessage])

  const model = ui.info?.model?.replace(/^.*\//, '') ?? ''

  const marker = overlay.clarify || overlay.confirm ? '⚠' : ui.busy ? '⏳' : '✓'

  const tabCwd = ui.info?.cwd

  useTerminalTitle(model ? `CodeFlow · ${marker} ${model}${tabCwd ? ` · ${shortCwd(tabCwd, 24)}` : ''}` : 'CodeFlow')

  useEffect(() => {
    if (!ui.sid || !stdout) {
      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    const onResize = () => {
      clearTimeout(timer)
      timer = setTimeout(() => {
        timer = undefined
        void rpc<TerminalResizeResponse>('terminal.resize', { cols: stdout.columns ?? 80, session_id: ui.sid })
      }, 100)
    }

    stdout.on('resize', onResize)

    return () => {
      clearTimeout(timer)
      stdout.off('resize', onResize)
    }
  }, [rpc, stdout, ui.sid])

  const answerClarify = useCallback(
    (answer: string) => {
      const clarify = overlay.clarify

      if (!clarify) {
        return
      }

      const label = toolTrailLabel('clarify')

      turnController.turnTools = turnController.turnTools.filter(line => !sameToolTrailGroup(label, line))
      patchTurnState({ turnTrail: turnController.turnTools })

      rpc<ClarifyRespondResponse>('clarify.respond', { answer, request_id: clarify.requestId }).then(r => {
        if (!r) {
          return
        }

        if (answer) {
          turnController.persistedToolLabels.add(label)
          appendMessage({
            kind: 'trail',
            role: 'system',
            text: '',
            tools: [buildToolTrailLine('clarify', clarify.question)]
          })
          appendMessage({ role: 'user', text: answer })
          patchUiState({ status: 'running…' })
        } else {
          sys('prompt cancelled')
        }

        patchOverlayState({ clarify: null })
      })
    },
    [appendMessage, overlay.clarify, rpc, sys]
  )

  const paste = useCallback(
    (quiet = false) => {
      const tracked = trackPasteFlight(
        clipboardPasteFlightRef,
        (async () => {
          const result = await pasteClipboardImage(rpc, ui.sid)

          if (result.status === 'attached') {
            sys(attachedImageNotice(result.info))
          } else if (result.status === 'no-session') {
            sys('no active session')
          } else if (result.status === 'empty' && !quiet) {
            sys('No image found in clipboard')
          } else if (result.status === 'failed' && !quiet) {
            sys('Failed to attach clipboard image')
          } else if (result.status === 'busy' && !quiet) {
            sys('Wait for the current turn to finish before attaching an image')
          }
        })(),
        () => setClipboardPasteTick(value => value + 1)
      )

      return tracked
    },
    [rpc, sys, ui.sid]
  )

  clipboardPasteRef.current = paste

  const { dispatchSubmission, send, submit } = useSubmission({
    appendMessage,
    chatStreamRef,
    claimSessionImages: claimClipboardImages,
    composerActions,
    composerRefs,
    composerState,
    maybeGoodVibes,
    pendingPasteRef: clipboardPasteFlightRef,
    setLastUserMsg,
    slashRef,
    submitRef,
    sys
  })

  // 会话稳定后排空一条排队消息。
  useEffect(() => {
    if (
      !ui.sid ||
      ui.busy ||
      ui.sessionMutating ||
      ui.sessionSwitching ||
      clipboardPasteFlightRef.current ||
      composerRefs.queueEditRef.current !== null ||
      composerActions.isQueueHeadPaused() ||
      composerRefs.queueRef.current.length === 0
    ) {
      return
    }

    dispatchNextQueuedSubmission(composerActions, dispatchSubmission)
  }, [
    clipboardPasteTick,
    composerActions,
    composerRefs,
    composerState.queueEditIdx,
    composerState.queuedDisplay,
    dispatchSubmission,
    ui.busy,
    ui.sessionMutating,
    ui.sessionSwitching,
    ui.sid
  ])

  const { pagerPageSize } = useInputHandlers({
    actions: {
      answerClarify,
      appendMessage,
      die,
      dispatchSubmission,
      guardBusySessionSwitch: session.guardBusySessionSwitch,
      newSession: session.newSession,
      sys
    },
    chatStreamRef,
    composer: { actions: composerActions, refs: composerRefs, state: composerState },
    gateway,
    terminal: { hasSelection, scrollRef, scrollWithSelection, selection, stdout },
    wheelStep: WHEEL_SCROLL_STEP
  })

  const onEvent = useMemo(
    () =>
      createGatewayEventHandler({
        clipboard: { releaseSessionImages: releaseSubmittedClipboardImages },
        gateway,
        session: {
          STARTUP_RESUME_ID,
          newSession: session.newSession,
          resumeById: session.resumeById
        },
        submission: { submitRef },
        system: { bellOnComplete, stdout, sys },
        transcript: { appendMessage, panel, setHistoryItems }
      }),
    [
      appendMessage,
      bellOnComplete,
      clearSelection,
      gateway,
      panel,
      session.newSession,
      session.resumeById,
      stdout,
      submitRef,
      sys
    ]
  )

  onEventRef.current = onEvent

  useEffect(() => {
    const handler = (ev: GatewayEvent) => onEventRef.current(ev)

    const exitHandler = () => {
      turnController.reset()
      patchUiState({ busy: false, sid: null, status: 'gateway exited' })
      turnController.pushActivity('gateway exited · /logs to inspect', 'error')
      sys('error: gateway exited')
    }

    gw.on('event', handler)
    gw.on('exit', exitHandler)
    gw.drain()

    // 真正退出时由 entry.tsx 的 setupGracefulExit 处理进程清理。
    return () => {
      gw.off('event', handler)
      gw.off('exit', exitHandler)
    }
  }, [gw, sys])

  useLongRunToolCharms()

  const slash = useMemo(
    () =>
      createSlashHandler({
        composer: {
          attachImage: (path, sessionId) =>
            trackPasteFlight(
              clipboardPasteFlightRef,
              rpc<ImageAttachResponse>('image.attach', { path, session_id: sessionId }),
              () => setClipboardPasteTick(value => value + 1)
            ),
          enqueue: composerActions.enqueue,
          hasSelection,
          paste,
          queueRef: composerRefs.queueRef,
          selection,
          setInput: composerActions.setInput
        },
        gateway,
        local: {
          getHistoryItems: () => historyItemsRef.current,
          getLastUserMsg: () => lastUserMsgRef.current,
          maybeWarn
        },
        session: {
          closeSession: session.closeSession,
          deleteSessionWithFallback: session.deleteSessionWithFallback,
          die,
          guardBusySessionSwitch: session.guardBusySessionSwitch,
          newSession: session.newSession,
          releaseSessionImages: releaseClipboardImages,
          resetVisibleHistory: session.resetVisibleHistory,
          resumeById: session.resumeById,
          runSessionMutation: session.runSessionMutation,
          setSessionStartedAt
        },
        slashFlightRef,
        transcript: {
          dispatchSubmission,
          page,
          panel,
          send,
          setHistoryItems,
          sys,
          trimLastExchange: session.trimLastExchange
        }
      }),
    [
      composerActions,
      composerRefs,
      die,
      dispatchSubmission,
      gateway,
      hasSelection,
      maybeWarn,
      page,
      panel,
      paste,
      rpc,
      selection,
      send,
      session,
      sys
    ]
  )

  slashRef.current = slash

  const answerConfirm = useCallback(
    (answer: boolean) => {
      const requestId = overlay.confirm?.requestId

      if (!requestId) {
        return
      }

      return answerConfirmRequest(rpc, requestId, answer)
    },
    [overlay.confirm, rpc]
  )

  const onModelSelect = useCallback((model: string, providerSlug: string) => {
    patchOverlayState({ modelPicker: false })
    slashRef.current(`/model ${model} --provider ${providerSlug}`)
  }, [])

  const hasReasoning = useTurnSelector(state => Boolean(state.reasoning.trim()))

  // 逐区段覆盖优先于全局模式。所有区段最终均隐藏时，ToolTrail 只会显示浮动警报兜底
  // （错误/警告）。同步该行为，避免安静模式下在流式区域上方渲染空包装 Box。
  const anyPanelVisible = SECTION_NAMES.some(
    s => sectionMode(s, ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'
  )

  const thinkingPanelVisible =
    sectionMode('thinking', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const toolsPanelVisible =
    sectionMode('tools', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const activityPanelVisible =
    sectionMode('activity', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const showProgressArea = useTurnSelector(state =>
    anyPanelVisible
      ? Boolean(
          ui.busy ||
          state.outcome ||
          state.streamPendingTools.length ||
          state.streamSegments.some(segment => {
            const hasThinking = Boolean(segment.thinking?.trim())
            const hasTrailTools = Boolean(segment.tools?.length)

            if (segment.kind === 'trail' && !segment.text) {
              return (
                (thinkingPanelVisible && hasThinking) || ((toolsPanelVisible || activityPanelVisible) && hasTrailTools)
              )
            }

            return (
              Boolean(segment.text?.trim()) ||
              (thinkingPanelVisible && hasThinking) ||
              ((toolsPanelVisible || activityPanelVisible) && hasTrailTools)
            )
          }) ||
          state.subagents.length ||
          state.tools.length ||
          state.todos.length ||
          state.turnTrail.length ||
          (thinkingPanelVisible && hasReasoning) ||
          state.activity.length
        )
      : state.activity.some(item => item.tone !== 'info')
  )

  const appActions = useMemo(
    () => ({
      answerClarify,
      answerConfirm,
      clearSelection,
      deleteSessionWithFallback: session.deleteSessionWithFallback,
      onModelSelect,
      resumeById: session.resumeById,
      setStickyPrompt
    }),
    [answerClarify, answerConfirm, clearSelection, onModelSelect, session.deleteSessionWithFallback, session.resumeById]
  )

  const appComposer = useMemo(
    () => ({
      cols,
      compIdx: composerState.compIdx,
      completions: composerState.completions,
      empty,
      handleTextPaste: composerActions.handleTextPaste,
      input: composerState.input,
      inputBuf: composerState.inputBuf,
      pagerPageSize,
      queueEditIdx: composerState.queueEditIdx,
      queuedDisplay: composerState.queuedDisplay,
      submit,
      updateInput: composerActions.setInput
    }),
    [cols, composerActions, composerState, empty, pagerPageSize, submit]
  )

  // 当前进度不冻结地传递。流式更新节流负责交互负载；进度必须保持真实，避免实时尾部滚出视口时
  // 面板随机消失。
  const appProgress = useMemo(() => ({ showProgressArea }), [showProgressArea])

  const cwd = ui.info?.cwd || process.env.CODEFLOW_CWD || process.cwd()
  const gitBranch = useGitBranch(cwd)

  const appStatus = useMemo(
    () => ({
      cwdLabel: fmtCwdBranch(cwd, gitBranch),
      goodVibesTick,
      sessionStartedAt: ui.sid ? sessionStartedAt : null,
      showStickyPrompt: !!stickyPrompt,
      statusColor: statusColorOf(ui.status, ui.theme.color),
      stickyPrompt,
      turnStartedAt: ui.sid ? turnStartedAt : null
    }),
    [cwd, gitBranch, goodVibesTick, sessionStartedAt, stickyPrompt, turnStartedAt, ui]
  )

  const appTranscript = useMemo(
    () => ({ historyItems, scrollRef, virtualHistory, virtualRows }),
    [historyItems, virtualHistory, virtualRows]
  )

  return { appActions, appComposer, appProgress, appStatus, appTranscript, gateway }
}
