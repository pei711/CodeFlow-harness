// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { Msg } from '../types.js'
import type {
  ComposerActions,
  ComposerRefs,
  ComposerState,
  LastUserSubmission,
  PasteSnippet,
  QueuedSubmission
} from './interfaces.js'

import { TYPING_IDLE_MS } from '../config/timing.js'
import { looksLikeSlashCommand } from '../domain/slash.js'
import { PASTE_SNIPPET_RE } from '../protocol/paste.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

const DOUBLE_ENTER_MS = 450
const SESSION_BUSY_RE = /session busy|turn[_ ]already[_ ]in[_ ]progress|turn_in_progress|waiting for model response/i

const isSessionBusyError = (e: unknown) => e instanceof Error && SESSION_BUSY_RE.test(e.message)

export function submitUnlessPastePending(pendingPaste: Promise<void> | null, submit: () => void): boolean {
  if (pendingPaste) {
    return false
  }

  submit()

  return true
}

export async function continueAfterCancel(
  cancel: Promise<void>,
  send: () => void,
  requeue: (error: unknown) => void
): Promise<void> {
  try {
    await cancel
  } catch (error) {
    requeue(error)

    return
  }

  send()
}

export function dispatchNextQueuedSubmission(
  actions: Pick<ComposerActions, 'dequeue' | 'prependQueue' | 'setQueueEdit'>,
  dispatchSubmission: (
    full: string,
    showUserMessage?: boolean,
    submitText?: string,
    pasteSnips?: PasteSnippet[]
  ) => boolean
): boolean {
  const ui = getUiState()

  if (ui.busy || ui.sessionSwitching || ui.sessionMutating) {
    return false
  }

  const next = actions.dequeue()

  if (!next) {
    return false
  }

  actions.setQueueEdit(null)
  if (dispatchSubmission(next.text, !next.alreadyDisplayed, next.submitText, next.pasteSnips)) {
    return true
  }

  actions.prependQueue(next)

  return false
}

export function pauseBusySubmission(
  actions: Pick<ComposerActions, 'prependQueue'>,
  displayText: string,
  submitText: string,
  pasteSnips?: PasteSnippet[]
): void {
  actions.prependQueue({
    alreadyDisplayed: true,
    pasteSnips,
    paused: true,
    submitText,
    text: displayText
  })
}

const expandSnips = (snips: PasteSnippet[]) => {
  const byLabel = new Map<string, string[]>()

  for (const { label, text } of snips) {
    const hit = byLabel.get(label)
    hit ? hit.push(text) : byLabel.set(label, [text])
  }

  return (value: string) => value.replace(PASTE_SNIPPET_RE, tok => byLabel.get(tok)?.shift() ?? tok)
}

export function resolveQueuedEdit(
  entry: QueuedSubmission,
  editedText: string,
  currentSnips: PasteSnippet[]
): Pick<QueuedSubmission, 'pasteSnips' | 'submitText'> {
  const pasteSnips = currentSnips.length ? currentSnips : (entry.pasteSnips ?? [])

  return {
    pasteSnips,
    submitText:
      entry.text === editedText && entry.submitText !== undefined
        ? entry.submitText
        : expandSnips(pasteSnips)(editedText)
  }
}

export function useSubmission(opts: UseSubmissionOptions) {
  const {
    appendMessage,
    chatStreamRef,
    claimSessionImages,
    composerActions,
    composerRefs,
    composerState,
    maybeGoodVibes,
    setLastUserMsg,
    slashRef,
    submitRef,
    sys
  } = opts

  const lastEmptyAt = useRef(0)
  const typingIdleTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (typingIdleTimer.current) {
      clearTimeout(typingIdleTimer.current)
      typingIdleTimer.current = null
    }

    if (!composerState.input && !composerState.inputBuf.length) {
      turnController.relaxStreaming()

      return
    }

    if (getUiState().busy) {
      turnController.boostStreamingForTyping()
    }

    typingIdleTimer.current = setTimeout(() => {
      typingIdleTimer.current = null
      turnController.relaxStreaming()
    }, TYPING_IDLE_MS)

    return () => {
      if (typingIdleTimer.current) {
        clearTimeout(typingIdleTimer.current)
        typingIdleTimer.current = null
      }
    }
  }, [composerState.input, composerState.inputBuf])

  const send = useCallback(
    (text: string, showUserMessage = true, submitTextOverride?: string, pasteSnipsOverride?: PasteSnippet[]) => {
      const pasteSnips = pasteSnipsOverride ?? composerState.pasteSnips
      const expand = expandSnips(pasteSnips)

      const startSubmit = (displayText: string, submitText: string, showUserMessage = true): void => {
        const live = getUiState()
        const sid = live.sid

        if (!sid) {
          return sys('session not ready yet')
        }

        if (live.sessionSwitching || live.sessionMutating) {
          composerActions.prependQueue({
            alreadyDisplayed: !showUserMessage,
            pasteSnips,
            paused: false,
            submitText,
            text: displayText
          })
          patchUiState({ busy: false, status: 'ready' })
          sys(`${live.sessionSwitching ? 'session switch' : 'session mutation'} in progress; message queued`)

          return
        }

        const typed = chatStreamRef?.current

        if (!typed || typed.sessionKey !== sid) {
          composerActions.prependQueue({
            alreadyDisplayed: !showUserMessage,
            pasteSnips,
            paused: false,
            submitText,
            text: displayText
          })
          patchUiState({ busy: false, status: 'ready' })
          sys(
            typed
              ? 'chat transport is switching sessions; message queued'
              : 'chat transport is not ready; message queued'
          )

          return
        }

        turnController.clearStatusTimer()
        maybeGoodVibes(submitText)
        setLastUserMsg({ pasteSnips, submitText, text: displayText })

        if (showUserMessage) {
          appendMessage({ role: 'user', text: displayText })
        }

        patchUiState({ busy: true, status: 'running…' })
        turnController.bufRef = ''
        turnController.interrupted = false

        const imageClaimId = claimSessionImages?.(sid)
        void typed
          .send(submitText, imageClaimId)
          .then(result => {
            if (result.accepted) {
              return
            }

            pauseBusySubmission(composerActions, displayText, submitText, pasteSnips)
            patchUiState({ busy: false, status: 'ready' })
            sys('turn was not accepted; submission queued for manual retry')
          })
          .catch((e: Error) => {
            if (isSessionBusyError(e)) {
              pauseBusySubmission(composerActions, displayText, submitText, pasteSnips)
              patchUiState({ busy: false, status: 'ready' })
              return sys('session still busy; submission queued for manual retry')
            }
            sys(`error: ${e.message}`)
            patchUiState({ busy: false, status: 'ready' })
          })
      }

      const sid = getUiState().sid

      if (!sid) {
        return sys('session not ready yet')
      }

      startSubmit(text, submitTextOverride ?? expand(text), showUserMessage)
    },
    [
      appendMessage,
      chatStreamRef,
      composerActions,
      composerState.pasteSnips,
      maybeGoodVibes,
      claimSessionImages,
      setLastUserMsg,
      sys
    ]
  )

  const sendQueued = useCallback(
    (entry: QueuedSubmission) => send(entry.text, !entry.alreadyDisplayed, entry.submitText, entry.pasteSnips),
    [send]
  )

  // 排队会保留进行中的轮次；中断会先取消它，再发送新文本。编辑队列的回退路径
  // 会将所选项保留在队首。
  const handleBusyInput = useCallback(
    (full: string, opts: { fallbackEntry?: QueuedSubmission; fallbackToFront?: boolean } = {}) => {
      const live = getUiState()
      const mode = live.busyInputMode
      const entry =
        opts.fallbackEntry ??
        ({
          alreadyDisplayed: false,
          pasteSnips: composerState.pasteSnips,
          paused: false,
          text: full
        } satisfies QueuedSubmission)

      const fallback = (note: string) => {
        if (opts.fallbackToFront) {
          composerActions.prependQueue(entry)
        } else {
          composerActions.appendQueue(entry)
        }

        sys(note)
      }

      if (mode === 'queue') {
        return composerActions.appendQueue(entry)
      }

      const chat = chatStreamRef?.current
      if (!chat) {
        fallback('turn transport unavailable; message queued')
        return
      }

      void continueAfterCancel(
        chat.cancel(),
        () => (opts.fallbackEntry ? sendQueued(entry) : send(full)),
        error =>
          fallback(`turn cancel failed: ${error instanceof Error ? error.message : String(error)}; message queued`)
      )
    },
    [chatStreamRef, composerActions, composerState.pasteSnips, send, sendQueued, sys]
  )

  const dispatchSubmissionNow = useCallback(
    (
      full: string,
      showUserMessage = true,
      submitTextOverride?: string,
      pasteSnipsOverride: PasteSnippet[] = composerState.pasteSnips
    ) => {
      if (!full.trim()) {
        return
      }

      if (looksLikeSlashCommand(full)) {
        appendMessage({ kind: 'slash', role: 'system', text: full })
        composerActions.pushHistory(full)
        slashRef.current(full)
        composerActions.clearIn()

        return
      }

      const live = getUiState()

      if (live.sessionSwitching || live.sessionMutating) {
        composerActions.pushHistory(full)
        composerActions.appendQueue({
          alreadyDisplayed: false,
          pasteSnips: pasteSnipsOverride,
          paused: false,
          submitText: submitTextOverride,
          text: full
        })
        composerActions.clearIn()
        sys(`${live.sessionSwitching ? 'session switch' : 'session mutation'} in progress; message queued`)

        return
      }

      if (!live.sid) {
        composerActions.pushHistory(full)
        composerActions.appendQueue({
          alreadyDisplayed: false,
          pasteSnips: pasteSnipsOverride,
          paused: false,
          submitText: submitTextOverride,
          text: full
        })
        composerActions.clearIn()

        return
      }

      const editIdx = composerRefs.queueEditRef.current
      composerActions.clearIn()

      if (editIdx !== null) {
        const existing = composerActions.getQueueEntry(editIdx)

        if (!existing) {
          composerActions.setQueueEdit(null)

          return
        }

        const edited = resolveQueuedEdit(existing, full, pasteSnipsOverride)
        composerActions.replaceQueue(editIdx, full, edited.submitText, edited.pasteSnips)
        const picked = composerActions.takeQueue(editIdx)
        composerActions.setQueueEdit(null)

        if (!picked || !live.sid) {
          return
        }

        if (getUiState().busy) {
          // 中断应作用于活动轮次，不能静默地把所选项放回队列。
          if (getUiState().busyInputMode === 'queue') {
            return composerActions.prependQueue({ ...picked, paused: false })
          }

          return handleBusyInput(picked.text, {
            fallbackEntry: { ...picked, paused: false },
            fallbackToFront: true
          })
        }

        return sendQueued(picked)
      }

      composerActions.pushHistory(full)

      if (getUiState().busy) {
        return handleBusyInput(full, {
          fallbackEntry: {
            alreadyDisplayed: false,
            pasteSnips: pasteSnipsOverride,
            paused: false,
            submitText: submitTextOverride,
            text: full
          }
        })
      }

      send(full, showUserMessage, submitTextOverride, pasteSnipsOverride)
    },
    [
      appendMessage,
      composerActions,
      composerRefs,
      composerState.pasteSnips,
      handleBusyInput,
      send,
      sendQueued,
      slashRef,
      sys
    ]
  )

  const dispatchSubmission = useCallback(
    (full: string, showUserMessage = true, submitText?: string, pasteSnips?: PasteSnippet[]) => {
      const resolvedPasteSnips = pasteSnips ?? composerState.pasteSnips
      const resolvedSubmitText = submitText ?? expandSnips(resolvedPasteSnips)(full)
      const dispatched = submitUnlessPastePending(opts.pendingPasteRef?.current ?? null, () =>
        dispatchSubmissionNow(full, showUserMessage, resolvedSubmitText, resolvedPasteSnips)
      )

      if (!dispatched) {
        sys('wait for the clipboard paste to finish, then submit again')
      }

      return dispatched
    },
    [composerState.pasteSnips, dispatchSubmissionNow, opts.pendingPasteRef, sys]
  )

  const submit = useCallback(
    (value: string) => {
      if (composerState.completions.length) {
        const row = composerState.completions[composerState.compIdx]

        if (row?.text) {
          const text = value.startsWith('/') && row.text.startsWith('/') ? row.text.slice(1) : row.text
          const next = value.slice(0, composerState.compReplace) + text

          if (next !== value) {
            return composerActions.setInput(next)
          }
        }
      }

      if (!value.trim() && !composerState.inputBuf.length) {
        const live = getUiState()
        const now = Date.now()
        const doubleTap = now - lastEmptyAt.current < DOUBLE_ENTER_MS
        lastEmptyAt.current = now

        if (doubleTap && live.busy && live.sid) {
          return void chatStreamRef?.current?.cancel()
        }

        if (doubleTap && live.sid && composerRefs.queueRef.current.length) {
          dispatchNextQueuedSubmission(composerActions, dispatchSubmission)
        }

        return
      }

      lastEmptyAt.current = 0

      if (value.endsWith('\\')) {
        composerActions.setInputBuf(prev => [...prev, value.slice(0, -1)])

        return composerActions.setInput('')
      }

      dispatchSubmission([...composerState.inputBuf, value].join('\n'))
    },
    [chatStreamRef, composerActions, composerRefs, composerState, dispatchSubmission]
  )

  submitRef.current = submit

  return { dispatchSubmission, send, sendQueued, submit }
}

export interface UseSubmissionOptions {
  appendMessage: (msg: Msg) => void
  claimSessionImages?: (sessionId: string) => string
  composerActions: ComposerActions
  composerRefs: ComposerRefs
  composerState: ComposerState
  maybeGoodVibes: (text: string) => void
  pendingPasteRef?: MutableRefObject<Promise<void> | null>
  setLastUserMsg: (value: LastUserSubmission) => void
  slashRef: MutableRefObject<(cmd: string) => boolean>
  submitRef: MutableRefObject<(value: string) => void>
  sys: (text: string) => void
  /**
   * 带类型的聊天流句柄。已附加时，用户提交通过 `chatStream.send()` 路由到
   * `turn.send` RPC。句柄缺失或已分离时，提交会明确失败，而不是调用未注册的
   * 回退方法。
   */
  chatStreamRef?: MutableRefObject<{
    cancel: () => Promise<void>
    isTurnActive: () => boolean
    sessionKey: string
    send: (content: string, imageClaimId?: string) => Promise<{ accepted: boolean }>
  } | null>
}
