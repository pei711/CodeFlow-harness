// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { useCallback, useRef, useState } from 'react'

import type { QueuedSubmission } from '../app/interfaces.js'

// 原地修改 `arr`，返回值仍是同一个输入数组以支持链式调用；需要副本时请使用
// `Array.prototype.toSpliced`。
export function removeAtInPlace<T>(arr: T[], i: number): T[] {
  if (i < 0 || i >= arr.length) {
    return arr
  }

  arr.splice(i, 1)

  return arr
}

export function replaceQueuedSubmissionAt(
  queue: string[],
  metadata: Array<Omit<QueuedSubmission, 'text'>>,
  i: number,
  text: string,
  submitText?: string,
  pasteSnips?: QueuedSubmission['pasteSnips']
): void {
  if (i < 0 || i >= queue.length) {
    return
  }

  const current = metadata[i] ?? { alreadyDisplayed: false, paused: false }

  metadata[i] =
    queue[i] === text
      ? { ...current, pasteSnips, submitText }
      : { alreadyDisplayed: false, pasteSnips, paused: false, submitText }
  queue[i] = text
}

export function useQueue() {
  const queueRef = useRef<string[]>([])
  const queueMetadataRef = useRef<Array<Omit<QueuedSubmission, 'text'>>>([])
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay([...queueRef.current]), [])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }, [])

  const appendQueue = useCallback(
    (entry: QueuedSubmission) => {
      const { text, ...metadata } = entry

      queueRef.current.push(text)
      queueMetadataRef.current.push(metadata)
      syncQueue()
    },
    [syncQueue]
  )

  const enqueue = useCallback(
    (text: string) =>
      appendQueue({
        alreadyDisplayed: false,
        paused: false,
        text
      }),
    [appendQueue]
  )

  const dequeue = useCallback(() => {
    const text = queueRef.current.shift()
    const metadata = queueMetadataRef.current.shift()
    syncQueue()

    return text === undefined
      ? undefined
      : {
          alreadyDisplayed: metadata?.alreadyDisplayed ?? false,
          pasteSnips: metadata?.pasteSnips,
          paused: metadata?.paused ?? false,
          submitText: metadata?.submitText,
          text
        }
  }, [syncQueue])

  const prependQueue = useCallback(
    (entry: QueuedSubmission) => {
      const { text, ...metadata } = entry

      queueRef.current.unshift(text)
      queueMetadataRef.current.unshift(metadata)
      syncQueue()
    },
    [syncQueue]
  )

  const getQueueEntry = useCallback((i: number): QueuedSubmission | undefined => {
    const text = queueRef.current[i]
    const metadata = queueMetadataRef.current[i]

    return text === undefined
      ? undefined
      : {
          alreadyDisplayed: metadata?.alreadyDisplayed ?? false,
          pasteSnips: metadata?.pasteSnips,
          paused: metadata?.paused ?? false,
          submitText: metadata?.submitText,
          text
        }
  }, [])

  const replaceQ = useCallback(
    (i: number, text: string, submitText?: string, pasteSnips?: QueuedSubmission['pasteSnips']) => {
      replaceQueuedSubmissionAt(queueRef.current, queueMetadataRef.current, i, text, submitText, pasteSnips)
      syncQueue()
    },
    [syncQueue]
  )

  const removeQ = useCallback(
    (i: number) => {
      const before = queueRef.current.length

      removeAtInPlace(queueRef.current, i)
      removeAtInPlace(queueMetadataRef.current, i)

      if (queueRef.current.length !== before) {
        syncQueue()
      }
    },
    [syncQueue]
  )

  const takeQueue = useCallback(
    (i: number): QueuedSubmission | undefined => {
      const text = queueRef.current.splice(i, 1)[0]
      const metadata = queueMetadataRef.current.splice(i, 1)[0]

      syncQueue()
      if (text === undefined) {
        return undefined
      }

      return {
        alreadyDisplayed: metadata?.alreadyDisplayed ?? false,
        pasteSnips: metadata?.pasteSnips,
        paused: metadata?.paused ?? false,
        submitText: metadata?.submitText,
        text
      }
    },
    [syncQueue]
  )

  const isQueueHeadPaused = useCallback(() => queueMetadataRef.current[0]?.paused === true, [])

  return {
    appendQueue,
    dequeue,
    enqueue,
    getQueueEntry,
    isQueueHeadPaused,
    prependQueue,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue,
    takeQueue
  }
}
