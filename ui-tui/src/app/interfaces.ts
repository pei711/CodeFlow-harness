// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { ScrollBoxHandle } from '@hermes/ink'
import type { MutableRefObject, ReactNode, RefObject, SetStateAction } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { ImageAttachResponse } from '../gatewayTypes.js'
import type { RpcResult } from '../lib/rpc.js'
import type { Theme } from '../theme.js'
import type { TuiRpcClient } from '../tuiRpcClient.js'
import type {
  ClarifyReq,
  ConfirmReq,
  DetailsMode,
  Msg,
  PanelSection,
  SectionVisibility,
  SessionInfo,
  Usage
} from '../types.js'
import type { ChatStreamRpcClient } from './chatStream.js'

export interface StateSetter<T> {
  (value: SetStateAction<T>): void
}

export type StatusBarMode = 'bottom' | 'off' | 'top'

export type BusyInputMode = 'interrupt' | 'queue'

// 指示器样式名称的唯一事实来源。
export const INDICATOR_STYLES = ['ascii', 'emoji', 'kaomoji', 'unicode'] as const
export type IndicatorStyle = (typeof INDICATOR_STYLES)[number]
export const DEFAULT_INDICATOR_STYLE: IndicatorStyle = 'unicode'

export interface SelectionApi {
  captureScrolledRows: (firstRow: number, lastRow: number, side: 'above' | 'below') => void
  clearSelection: () => void
  copySelection: () => Promise<string>
  copySelectionNoClear: () => Promise<string>
  getState: () => unknown
  version: () => number
  shiftAnchor: (dRow: number, minRow: number, maxRow: number) => void
  shiftSelection: (dRow: number, minRow: number, maxRow: number) => void
}

export interface CompletionItem {
  display: string
  meta?: string
  text: string
}

export interface GatewayRpc {
  <T extends object = RpcResult>(method: string, params?: Record<string, unknown>): Promise<null | T>
}

export interface GatewayServices {
  gw: TuiRpcClient
  rpc: GatewayRpc
  /**
   * 聊天路径使用的带类型 RpcClient 句柄（参见 design.md §D7）。测试使用的
   * 网关桩夹具不拥有真实套接字，因此此项可选；生产装配始终通过 entry.tsx
   * 填充它。
   */
  rpcClient?: ChatStreamRpcClient
}

export interface GatewayProviderProps {
  children: ReactNode
  value: GatewayServices
}

export interface OverlayState {
  agents: boolean
  agentsInitialHistoryIndex: number
  clarify: ClarifyReq | null
  confirm: ConfirmReq | null
  modelPicker: boolean
  pager: null | PagerState
  picker: boolean
}

export interface PagerState {
  lines: string[]
  offset: number
  title?: string
}

export interface TranscriptRow {
  index: number
  key: string
  msg: Msg
}

export interface UiState {
  bgTasks: Set<string>
  busy: boolean
  busyInputMode: BusyInputMode
  compact: boolean
  /**
   * 在进行中的带类型轮次首次收到 Ctrl+C 后设置：常规取消已发出，第二次
   * Ctrl+C 将在本地强制重置提示符（Ctrl+C 逃生路径），并切换忙碌占位提示。
   * 轮次结束时清除，参见 `turnController.idle`。
   */
  escapeArmed: boolean
  detailsMode: DetailsMode
  detailsModeCommandOverride: boolean
  info: null | SessionInfo
  inlineDiffs: boolean
  mouseTracking: boolean
  sections: SectionVisibility
  sessionMutating: boolean
  sessionSwitching: boolean
  showCost: boolean
  showReasoning: boolean
  indicatorStyle: IndicatorStyle
  sid: null | string
  status: string
  statusBar: StatusBarMode
  streaming: boolean
  theme: Theme
  usage: Usage
}

export interface VirtualHistoryState {
  bottomSpacer: number
  end: number
  measureRef: (key: string) => (el: unknown) => void
  offsets: ArrayLike<number>
  start: number
  topSpacer: number
}

export interface ComposerPasteResult {
  cursor: number
  fallbackText?: string
  value: string
}

export type MaybePromise<T> = Promise<T> | T

export type SessionMutationRunner = <T>(what: string, operation: () => MaybePromise<T>) => Promise<T | null>

export interface QueuedSubmission {
  alreadyDisplayed: boolean
  pasteSnips?: PasteSnippet[]
  paused: boolean
  submitText?: string
  text: string
}

export interface LastUserSubmission {
  pasteSnips?: PasteSnippet[]
  submitText: string
  text: string
}

export interface ComposerActions {
  appendQueue: (entry: QueuedSubmission) => void
  clearIn: () => void
  dequeue: () => QueuedSubmission | undefined
  enqueue: (text: string) => void
  getQueueEntry: (index: number) => QueuedSubmission | undefined
  handleTextPaste: (event: PasteEvent) => MaybePromise<ComposerPasteResult | null>
  isQueueHeadPaused: () => boolean
  openEditor: () => Promise<void>
  prependQueue: (entry: QueuedSubmission) => void
  pushHistory: (text: string) => void
  removeQueue: (index: number) => void
  replaceQueue: (index: number, text: string, submitText?: string, pasteSnips?: PasteSnippet[]) => void
  setCompIdx: StateSetter<number>
  setHistoryIdx: StateSetter<null | number>
  setInput: StateSetter<string>
  setInputBuf: StateSetter<string[]>
  setPasteSnips: StateSetter<PasteSnippet[]>
  setQueueEdit: (index: null | number) => void
  syncQueue: () => void
  takeQueue: (index: number) => QueuedSubmission | undefined
}

export interface ComposerRefs {
  historyDraftRef: MutableRefObject<string>
  historyRef: MutableRefObject<string[]>
  queueEditRef: MutableRefObject<null | number>
  queueRef: MutableRefObject<string[]>
  submitRef: MutableRefObject<(value: string) => void>
}

export interface ComposerState {
  compIdx: number
  compReplace: number
  completions: CompletionItem[]
  historyIdx: null | number
  input: string
  inputBuf: string[]
  pasteSnips: PasteSnippet[]
  queueEditIdx: null | number
  queuedDisplay: string[]
}

export interface UseComposerStateOptions {
  gw: TuiRpcClient
  onClipboardPaste: (quiet?: boolean) => Promise<void> | void
  onImageAttached?: (info: ImageAttachResponse) => void
  onPasteSettled?: () => void
  pasteFlightRef?: MutableRefObject<Promise<void> | null>
  submitRef: MutableRefObject<(value: string) => void>
}

export interface UseComposerStateResult {
  actions: ComposerActions
  refs: ComposerRefs
  state: ComposerState
}

export interface InputHandlerActions {
  answerClarify: (answer: string) => void
  appendMessage: (msg: Msg) => void
  die: () => void
  dispatchSubmission: (full: string) => boolean
  guardBusySessionSwitch: (what?: string) => boolean
  newSession: (msg?: string, title?: string) => void
  sys: (text: string) => void
}

export interface InputHandlerContext {
  actions: InputHandlerActions
  chatStreamRef?: RefObject<{
    cancel: () => Promise<void>
    forceReset: () => void
    isTurnActive: () => boolean
  } | null>
  composer: {
    actions: ComposerActions
    refs: ComposerRefs
    state: ComposerState
  }
  gateway: GatewayServices
  terminal: {
    hasSelection: boolean
    scrollRef: RefObject<null | ScrollBoxHandle>
    scrollWithSelection: (delta: number) => void
    selection: SelectionApi
    stdout?: NodeJS.WriteStream
  }
  wheelStep: number
}

export interface InputHandlerResult {
  pagerPageSize: number
}

export interface GatewayEventHandlerContext {
  clipboard?: {
    releaseSessionImages: (sessionId: string) => void
  }
  gateway: GatewayServices
  session: {
    STARTUP_RESUME_ID: string
    newSession: (msg?: string, title?: string) => void
    resumeById: (id: string) => void
  }
  submission: {
    submitRef: MutableRefObject<(value: string) => void>
  }
  system: {
    bellOnComplete: boolean
    stdout?: NodeJS.WriteStream
    sys: (text: string) => void
  }
  transcript: {
    appendMessage: (msg: Msg) => void
    panel: (title: string, sections: PanelSection[]) => void
    setHistoryItems: StateSetter<Msg[]>
  }
}

export interface SlashHandlerContext {
  composer: {
    attachImage: (path: string, sessionId: string) => Promise<null | ImageAttachResponse>
    enqueue: (text: string) => void
    hasSelection: boolean
    paste: (quiet?: boolean) => Promise<void> | void
    queueRef: MutableRefObject<string[]>
    selection: SelectionApi
    setInput: StateSetter<string>
  }
  gateway: GatewayServices
  local: {
    getHistoryItems: () => Msg[]
    getLastUserMsg: () => LastUserSubmission | null
    maybeWarn: (value: unknown) => void
  }
  session: {
    closeSession: (targetSid?: null | string) => Promise<unknown>
    deleteSessionWithFallback: (targetId: string) => Promise<boolean | null>
    die: () => void
    guardBusySessionSwitch: (what?: string) => boolean
    newSession: (msg?: string, title?: string) => void
    releaseSessionImages: (sessionId: string) => void
    resetVisibleHistory: (info?: null | SessionInfo) => void
    resumeById: (id: string) => void
    runSessionMutation: SessionMutationRunner
    setSessionStartedAt: StateSetter<number>
  }
  slashFlightRef: MutableRefObject<number>
  transcript: {
    dispatchSubmission: (
      full: string,
      showUserMessage?: boolean,
      submitText?: string,
      pasteSnips?: PasteSnippet[]
    ) => boolean
    page: (text: string, title?: string) => void
    panel: (title: string, sections: PanelSection[]) => void
    send: (text: string) => void
    setHistoryItems: StateSetter<Msg[]>
    sys: (text: string) => void
    trimLastExchange: (items: Msg[]) => Msg[]
  }
}

export interface AppLayoutActions {
  answerClarify: (answer: string) => void
  answerConfirm: (answer: boolean) => void
  clearSelection: () => void
  deleteSessionWithFallback: (id: string) => Promise<boolean | null>
  onModelSelect: (model: string, providerSlug: string) => void
  resumeById: (id: string) => void
  setStickyPrompt: (value: string) => void
}

export interface AppLayoutComposerProps {
  cols: number
  compIdx: number
  completions: CompletionItem[]
  empty: boolean
  handleTextPaste: (event: PasteEvent) => MaybePromise<ComposerPasteResult | null>
  input: string
  inputBuf: string[]
  pagerPageSize: number
  queueEditIdx: null | number
  queuedDisplay: string[]
  submit: (value: string) => void
  updateInput: StateSetter<string>
}

export interface AppLayoutProgressProps {
  showProgressArea: boolean
}

export interface AppLayoutStatusProps {
  cwdLabel: string
  goodVibesTick: number
  sessionStartedAt: null | number
  showStickyPrompt: boolean
  statusColor: string
  stickyPrompt: string
  turnStartedAt: null | number
}

export interface AppLayoutTranscriptProps {
  historyItems: Msg[]
  scrollRef: RefObject<null | ScrollBoxHandle>
  virtualHistory: VirtualHistoryState
  virtualRows: TranscriptRow[]
}

export interface AppLayoutProps {
  actions: AppLayoutActions
  composer: AppLayoutComposerProps
  mouseTracking: boolean
  progress: AppLayoutProgressProps
  status: AppLayoutStatusProps
  transcript: AppLayoutTranscriptProps
}

export interface AppOverlaysProps {
  cols: number
  compIdx: number
  completions: CompletionItem[]
  onClarifyAnswer: (value: string) => void
  onConfirmAnswer: (answer: boolean) => void
  onModelSelect: (model: string, providerSlug: string) => void
  onPickerDeleteActive: (sessionId: string) => Promise<boolean | null>
  onPickerSelect: (sessionId: string) => void
  pagerPageSize: number
}

export interface PasteSnippet {
  label: string
  path?: string
  text: string
}
