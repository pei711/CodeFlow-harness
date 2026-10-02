// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { useStdin, withInkSuspended } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { useCallback, useMemo, useState } from 'react'

import type { PasteEvent } from '../components/textInput.js'
import type { ImageAttachResponse } from '../gatewayTypes.js'
import type {
  ComposerPasteResult,
  MaybePromise,
  PasteSnippet,
  UseComposerStateOptions,
  UseComposerStateResult
} from './interfaces.js'

import { LARGE_PASTE } from '../config/limits.js'
import { useCompletion } from '../hooks/useCompletion.js'
import { useInputHistory } from '../hooks/useInputHistory.js'
import { useQueue } from '../hooks/useQueue.js'
import { isUsableClipboardText, readClipboardText } from '../lib/clipboard.js'
import { resolveEditor } from '../lib/editor.js'
import { readOsc52Clipboard } from '../lib/osc52.js'
import { isRemoteShellSession } from '../lib/terminalSetup.js'
import { pasteTokenLabel, stripTrailingPasteNewlines } from '../lib/text.js'
import { $isBlocked } from './overlayStore.js'
import { getUiState } from './uiStore.js'

const PASTE_SNIP_MAX_COUNT = 32
const PASTE_SNIP_MAX_TOTAL_BYTES = 4 * 1024 * 1024

export function trackPasteFlight<T>(
  ref: { current: Promise<void> | null } | undefined,
  flight: Promise<T>,
  onIdle?: () => void
): Promise<T> {
  if (!ref) {
    return flight
  }

  const settled = flight.then(
    () => undefined,
    () => undefined
  )
  const aggregate = ref.current ? Promise.all([ref.current, settled]).then(() => undefined) : settled
  ref.current = aggregate
  void aggregate.finally(() => {
    if (ref.current === aggregate) {
      ref.current = null
      onIdle?.()
    }
  })

  return flight
}

const trimSnips = (snips: PasteSnippet[]): PasteSnippet[] => {
  let total = 0
  const out: PasteSnippet[] = []

  for (let i = snips.length - 1; i >= 0; i--) {
    const snip = snips[i]!
    const size = snip.text.length

    if (out.length >= PASTE_SNIP_MAX_COUNT || total + size > PASTE_SNIP_MAX_TOTAL_BYTES) {
      break
    }

    total += size
    out.unshift(snip)
  }

  return out.length === snips.length ? snips : out
}

/** 在光标位置插入文本，并添加空格将其与相邻非空白字符分开。 */
function insertAtCursor(value: string, cursor: number, text: string): { cursor: number; value: string } {
  const lead = cursor > 0 && !/\s/.test(value[cursor - 1] ?? '') ? ' ' : ''
  const tail = cursor < value.length && !/\s/.test(value[cursor] ?? '') ? ' ' : ''
  const insert = `${lead}${text}${tail}`

  return {
    cursor: cursor + insert.length,
    value: value.slice(0, cursor) + insert + value.slice(cursor)
  }
}

/**
 * 在客户端快速启发式检测形似拖入文件路径的文本。返回 true 后，编辑框会向
 * 服务端发送 RPC 做实际验证。须与 cli.py 中的 _detect_file_drop() 保持同步；
 * 规范前缀列表以该函数为准。
 */
export function looksLikeDroppedPath(text: string): boolean {
  const trimmed = text.trim()

  if (!trimmed || trimmed.includes('\n')) {
    return false
  }

  // file:// URI、相对路径、主目录相对路径、带引号路径以及 Windows 盘符路径。
  if (
    trimmed.startsWith('file://') ||
    trimmed.startsWith('~/') ||
    trimmed.startsWith('./') ||
    trimmed.startsWith('../') ||
    trimmed.startsWith('"/') ||
    trimmed.startsWith("'/") ||
    trimmed.startsWith('"~') ||
    trimmed.startsWith("'~") ||
    /^[A-Za-z]:[/\\]/.test(trimmed) ||
    /^["'][A-Za-z]:[/\\]/.test(trimmed)
  ) {
    return true
  }

  // 对以 / 开头的裸绝对路径，要求再出现一个 / 或一个点，避免把 "/api"、
  // "/help" 等短字符串误判为路径并触发不必要的 RPC 往返。
  if (trimmed.startsWith('/')) {
    const rest = trimmed.slice(1)

    return rest.includes('/') || rest.includes('.')
  }

  return false
}

export function useComposerState({
  gw,
  onClipboardPaste,
  onImageAttached,
  onPasteSettled,
  pasteFlightRef,
  submitRef
}: UseComposerStateOptions): UseComposerStateResult {
  const [input, setInput] = useState('')
  const [inputBuf, setInputBuf] = useState<string[]>([])
  const [pasteSnips, setPasteSnips] = useState<PasteSnippet[]>([])
  const isBlocked = useStore($isBlocked)
  const { querier } = useStdin() as { querier: Parameters<typeof readOsc52Clipboard>[0] }

  const {
    appendQueue,
    queueRef,
    queueEditRef,
    queuedDisplay,
    queueEditIdx,
    enqueue,
    getQueueEntry,
    dequeue,
    isQueueHeadPaused,
    prependQueue,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue,
    takeQueue
  } = useQueue()

  const { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory } = useInputHistory()
  const { completions, compIdx, setCompIdx, compReplace } = useCompletion(input, isBlocked)

  const clearIn = useCallback(() => {
    setInput('')
    setInputBuf([])
    setPasteSnips([])
    setQueueEdit(null)
    setHistoryIdx(null)
    historyDraftRef.current = ''
  }, [historyDraftRef, setQueueEdit, setHistoryIdx])

  const handleResolvedPaste = useCallback(
    async ({ bracketed, cursor, text, value }: Omit<PasteEvent, 'hotkey'>): Promise<ComposerPasteResult | null> => {
      const cleanedText = stripTrailingPasteNewlines(text)

      if (!cleanedText || !/[^\n]/.test(cleanedText)) {
        if (bracketed) {
          void onClipboardPaste(true)
        }

        return null
      }

      const sid = getUiState().sid

      if (sid && !getUiState().sessionMutating && !getUiState().sessionSwitching && looksLikeDroppedPath(cleanedText)) {
        try {
          const attached = await gw.request<ImageAttachResponse>('image.attach', {
            path: cleanedText,
            session_id: sid
          })

          if (attached?.name) {
            if (getUiState().sid !== sid || getUiState().sessionMutating || getUiState().sessionSwitching) {
              return null
            }

            onImageAttached?.(attached)
            const remainder = attached.remainder?.trim() ?? ''

            if (!remainder) {
              return null
            }

            return { ...insertAtCursor(value, cursor, remainder), fallbackText: remainder }
          }
        } catch {
          // 回退到普通文本粘贴行为。
        }
      }

      const lineCount = cleanedText.split('\n').length

      if (cleanedText.length < LARGE_PASTE.chars && lineCount < LARGE_PASTE.lines) {
        return {
          cursor: cursor + cleanedText.length,
          fallbackText: cleanedText,
          value: value.slice(0, cursor) + cleanedText + value.slice(cursor)
        }
      }

      const label = pasteTokenLabel(cleanedText, lineCount)
      const inserted = insertAtCursor(value, cursor, label)

      setPasteSnips(prev => trimSnips([...prev, { label, text: cleanedText }]))

      return { ...inserted, fallbackText: label }
    },
    [gw, onClipboardPaste, onImageAttached]
  )

  const handleTextPaste = useCallback(
    ({
      bracketed,
      cursor,
      hotkey,
      text,
      value
    }: PasteEvent): MaybePromise<null | { cursor: number; value: string }> => {
      if (hotkey) {
        const preferOsc52 = isRemoteShellSession(process.env)

        const readPreferredText = preferOsc52
          ? readOsc52Clipboard(querier).then(async osc52Text => {
              if (isUsableClipboardText(osc52Text)) {
                return osc52Text
              }

              return readClipboardText()
            })
          : readClipboardText().then(async clipText => {
              if (isUsableClipboardText(clipText)) {
                return clipText
              }

              return readOsc52Clipboard(querier)
            })

        const flight = readPreferredText.then(async preferredText => {
          if (isUsableClipboardText(preferredText)) {
            return handleResolvedPaste({ bracketed: false, cursor, text: preferredText, value })
          }

          await onClipboardPaste(false)

          return null
        })

        return trackPasteFlight(pasteFlightRef, flight, onPasteSettled)
      }

      return trackPasteFlight(
        pasteFlightRef,
        handleResolvedPaste({ bracketed: !!bracketed, cursor, text, value }),
        onPasteSettled
      )
    },
    [handleResolvedPaste, onClipboardPaste, onPasteSettled, pasteFlightRef, querier]
  )

  const openEditor = useCallback(async () => {
    const dir = mkdtempSync(join(tmpdir(), 'codeflow-'))
    const file = join(dir, 'prompt.md')
    const [cmd, ...args] = resolveEditor()

    writeFileSync(file, [...inputBuf, input].join('\n'))

    let exitCode: null | number = null

    await withInkSuspended(async () => {
      exitCode = spawnSync(cmd!, [...args, file], { stdio: 'inherit' }).status
    })

    try {
      if (exitCode !== 0) {
        return
      }

      const text = readFileSync(file, 'utf8').trimEnd()

      if (!text) {
        return
      }

      setInput('')
      setInputBuf([])
      submitRef.current(text)
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  }, [input, inputBuf, submitRef])

  const actions = useMemo(
    () => ({
      appendQueue,
      clearIn,
      dequeue,
      enqueue,
      getQueueEntry,
      handleTextPaste,
      isQueueHeadPaused,
      openEditor,
      prependQueue,
      pushHistory,
      removeQueue: removeQ,
      replaceQueue: replaceQ,
      setCompIdx,
      setHistoryIdx,
      setInput,
      setInputBuf,
      setPasteSnips,
      setQueueEdit,
      syncQueue,
      takeQueue
    }),
    [
      appendQueue,
      clearIn,
      dequeue,
      enqueue,
      getQueueEntry,
      handleTextPaste,
      isQueueHeadPaused,
      openEditor,
      prependQueue,
      pushHistory,
      removeQ,
      replaceQ,
      setCompIdx,
      setHistoryIdx,
      setQueueEdit,
      syncQueue,
      takeQueue
    ]
  )

  const refs = useMemo(
    () => ({
      historyDraftRef,
      historyRef,
      queueEditRef,
      queueRef,
      submitRef
    }),
    [historyDraftRef, historyRef, queueEditRef, queueRef, submitRef]
  )

  const state = useMemo(
    () => ({
      compIdx,
      compReplace,
      completions,
      historyIdx,
      input,
      inputBuf,
      pasteSnips,
      queueEditIdx,
      queuedDisplay
    }),
    [compIdx, compReplace, completions, historyIdx, input, inputBuf, pasteSnips, queueEditIdx, queuedDisplay]
  )

  return {
    actions,
    refs,
    state
  }
}
