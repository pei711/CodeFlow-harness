// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { forceRedraw, useInput } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import type { InputHandlerContext, InputHandlerResult } from './interfaces.js'

import { TYPING_IDLE_MS } from '../config/timing.js'
import { isAction, isCopyShortcut, isMac } from '../lib/platform.js'
import { computePrecisionWheelStep, initPrecisionWheel } from '../lib/precisionWheel.js'
import { computeWheelStep, initWheelAccelForHost } from '../lib/wheelAccel.js'
import { answerConfirmFromInput, cancelConfirmRequest } from './confirmResponse.js'
import { getInputSelection } from './inputSelectionStore.js'
import { $isBlocked, $overlayState, patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'
import { dispatchNextQueuedSubmission } from './useSubmission.js'

const isCtrl = (key: { ctrl: boolean }, ch: string, target: string) => key.ctrl && ch.toLowerCase() === target

export function useInputHandlers(ctx: InputHandlerContext): InputHandlerResult {
  const { actions, chatStreamRef, composer, gateway, terminal, wheelStep } = ctx
  const { actions: cActions, refs: cRefs, state: cState } = composer

  const overlay = useStore($overlayState)
  const isBlocked = useStore($isBlocked)
  const pagerPageSize = Math.max(5, (terminal.stdout?.rows ?? 24) - 6)
  const scrollIdleTimer = useRef<null | ReturnType<typeof setTimeout>>(null)

  // 从 Claude Code 移植的滚轮加速：事件间时间决定步长，方向翻转时重置。wheelStep
  // （WHEEL_SCROLL_STEP）是基数，最终行数为 wheelStep × accelMult。状态跨渲染原地修改。
  const wheelAccelRef = useRef(initWheelAccelForHost())

  const precisionWheelRef = useRef(initPrecisionWheel())

  useEffect(() => () => clearTimeout(scrollIdleTimer.current ?? undefined), [])

  const scrollTranscript = (delta: number) => {
    if (getUiState().busy) {
      turnController.boostStreamingForScroll()
      clearTimeout(scrollIdleTimer.current ?? undefined)
      scrollIdleTimer.current = setTimeout(() => {
        scrollIdleTimer.current = null
        turnController.relaxStreaming()
      }, TYPING_IDLE_MS)
    }

    terminal.scrollWithSelection(delta)
  }

  const copySelection = () => {
    // Ink 的 copySelection() 已调用 setClipboard()，可处理 macOS 的 pbcopy、Linux 的
    // wl-copy/xclip、tmux 和 OSC 52 回退。
    terminal.selection.copySelection()
  }

  const clearSelection = () => {
    terminal.selection.clearSelection()
  }

  const cancelOverlayFromCtrlC = () => {
    if (overlay.clarify) {
      return actions.answerClarify('')
    }

    if (overlay.confirm) {
      const requestId = overlay.confirm.requestId
      if (!requestId) {
        return patchOverlayState({ confirm: null })
      }

      return cancelConfirmRequest(gateway.rpc, requestId)
    }

    if (overlay.modelPicker) {
      return patchOverlayState({ modelPicker: false })
    }

    if (overlay.picker) {
      return patchOverlayState({ picker: false })
    }

    if (overlay.agents) {
      return patchOverlayState({ agents: false })
    }
  }

  const cycleQueue = (dir: 1 | -1) => {
    const len = cRefs.queueRef.current.length

    if (!len) {
      return false
    }

    const index = cState.queueEditIdx === null ? (dir > 0 ? 0 : len - 1) : (cState.queueEditIdx + dir + len) % len
    const entry = cActions.getQueueEntry(index)

    cActions.setQueueEdit(index)
    cActions.setHistoryIdx(null)
    cActions.setInput(entry?.text ?? '')
    cActions.setPasteSnips(entry?.pasteSnips ?? [])

    return true
  }

  const cycleHistory = (dir: 1 | -1) => {
    const h = cRefs.historyRef.current
    const cur = cState.historyIdx

    if (dir < 0) {
      if (!h.length) {
        return
      }

      if (cur === null) {
        cRefs.historyDraftRef.current = cState.input
      }

      const index = cur === null ? h.length - 1 : Math.max(0, cur - 1)

      cActions.setHistoryIdx(index)
      cActions.setQueueEdit(null)
      cActions.setInput(h[index] ?? '')

      return
    }

    if (cur === null) {
      return
    }

    const next = cur + 1

    if (next >= h.length) {
      cActions.setHistoryIdx(null)
      cActions.setInput(cRefs.historyDraftRef.current)
    } else {
      cActions.setHistoryIdx(next)
      cActions.setInput(h[next] ?? '')
    }
  }

  useInput((ch, key, event) => {
    const live = getUiState()

    if (isBlocked) {
      // 澄清/确认浮层活动时，其自身 useInput 处理器必须接收方向键、数字和 Enter。这里只拦截
      // Ctrl+C，让用户拒绝/关闭；其他按键继续传给组件级处理器。
      if (overlay.clarify || overlay.confirm) {
        if (isCtrl(key, ch, 'c')) {
          if (overlay.confirm) {
            return answerConfirmFromInput(event, cancelOverlayFromCtrlC)
          }

          cancelOverlayFromCtrlC()
        }

        return
      }

      if (overlay.pager) {
        if (key.escape || isCtrl(key, ch, 'c') || ch === 'q') {
          return patchOverlayState({ pager: null })
        }

        const move = (delta: number | 'top' | 'bottom') =>
          patchOverlayState(prev => {
            if (!prev.pager) {
              return prev
            }

            const { lines, offset } = prev.pager
            const max = Math.max(0, lines.length - pagerPageSize)
            const step = delta === 'top' ? -lines.length : delta === 'bottom' ? lines.length : delta
            const next = Math.max(0, Math.min(offset + step, max))

            return next === offset ? prev : { ...prev, pager: { ...prev.pager, offset: next } }
          })

        if (key.upArrow || ch === 'k') {
          return move(-1)
        }

        if (key.downArrow || ch === 'j') {
          return move(1)
        }

        if (key.pageUp || ch === 'b') {
          return move(-pagerPageSize)
        }

        if (ch === 'g') {
          return move('top')
        }

        if (ch === 'G') {
          return move('bottom')
        }

        if (key.return || ch === ' ' || key.pageDown) {
          patchOverlayState(prev => {
            if (!prev.pager) {
              return prev
            }

            const { lines, offset } = prev.pager
            const max = Math.max(0, lines.length - pagerPageSize)

            // 仅当已在最后一页时自动关闭；否则钳制到 `max`，使偏移与逐行/逐页回退处理器可达位置一致，
            // 避免下次 ↑/↓/PgUp 时突然回跳。
            return offset >= max
              ? { ...prev, pager: null }
              : { ...prev, pager: { ...prev.pager, offset: Math.min(offset + pagerPageSize, max) } }
          })
        }

        return
      }

      if (isCtrl(key, ch, 'c')) {
        cancelOverlayFromCtrlC()
      } else if (key.escape && overlay.picker) {
        patchOverlayState({ picker: false })
      }

      return
    }

    if (cState.completions.length && cState.input && cState.historyIdx === null && (key.upArrow || key.downArrow)) {
      const len = cState.completions.length

      cActions.setCompIdx(i => (key.upArrow ? (i - 1 + len) % len : (i + 1) % len))

      return
    }

    if (key.wheelUp || key.wheelDown) {
      const dir: -1 | 1 = key.wheelUp ? -1 : 1
      const now = Date.now()
      // 按住修饰键滚轮进入精确模式：每帧一行，不加速。顺滑鼠标/触控板会在同帧发出微小突发，
      // 将其合并，但不使用旧 80 毫秒节流；后者让 Option 滚动有台阶感。SGR/X10 鼠标编码只携带
      // shift/meta/ctrl 位；macOS 的 Cmd 被终端拦截，因此 Mac 使用 Option（meta），Windows/
      // Linux 使用 Alt（meta），Ctrl 作为便携回退。Shift 保留给扩展选择。
      const hasModifier = key.meta || key.ctrl
      const precision = computePrecisionWheelStep(precisionWheelRef.current, dir, hasModifier, now)

      if (precision.active) {
        // 进入精确模式必须丢弃加速滚轮状态，否则下一次普通滚轮事件会继承陈旧动量。
        if (precision.entered) {
          wheelAccelRef.current = initWheelAccelForHost()
        }

        return precision.rows ? scrollTranscript(dir * wheelStep) : undefined
      }

      // 0 表示方向翻转回弹已延迟，跳过无操作滚动。
      const rows = computeWheelStep(wheelAccelRef.current, dir, now)

      return rows ? scrollTranscript(dir * rows * wheelStep) : undefined
    }

    if (key.shift && key.upArrow) {
      return scrollTranscript(-1)
    }

    if (key.shift && key.downArrow) {
      return scrollTranscript(1)
    }

    if (key.pageUp || key.pageDown) {
      // 半视口可保留 50% 连续性，并保持在 Ink 的 `delta < innerHeight` DECSTBM 快速路径阈值内。
      const viewport = terminal.scrollRef.current?.getViewportHeight() ?? Math.max(6, (terminal.stdout?.rows ?? 24) - 8)
      const step = Math.max(4, Math.floor(viewport / 2))

      return scrollTranscript(key.pageUp ? -step : step)
    }

    // 单独按 Esc 时，取消队列编辑优先于清除选择；队列标题明确承诺“Esc 取消”，因此优先于隐式
    // 取消选区约定。没有活动编辑时继续回退。
    if (key.escape && cState.queueEditIdx !== null) {
      return cActions.clearIn()
    }

    if (key.escape && terminal.hasSelection) {
      return clearSelection()
    }

    if (key.upArrow && !cState.inputBuf.length) {
      const inputSel = getInputSelection()
      const cursor = inputSel && inputSel.start === inputSel.end ? inputSel.start : null

      const noLineAbove =
        !cState.input || (cursor !== null && cState.input.lastIndexOf('\n', Math.max(0, cursor - 1)) < 0)

      if (noLineAbove) {
        cycleQueue(1) || cycleHistory(-1)

        return
      }
    }

    if (key.downArrow && !cState.inputBuf.length) {
      const inputSel = getInputSelection()
      const cursor = inputSel && inputSel.start === inputSel.end ? inputSel.start : null
      const noLineBelow = !cState.input || (cursor !== null && cState.input.indexOf('\n', cursor) < 0)

      if (noLineBelow || cState.historyIdx !== null) {
        cycleQueue(-1) || cycleHistory(1)

        return
      }
    }

    if (isCopyShortcut(key, ch)) {
      if (terminal.hasSelection) {
        return copySelection()
      }

      const inputSel = getInputSelection()

      if (inputSel && inputSel.end > inputSel.start) {
        inputSel.clear()

        return
      }

      // macOS 上无选区时 Cmd+C 为空操作，中断由下方 Ctrl+C 处理。非 macOS 上 isAction 使用
      // Ctrl，因此继续进入中断/清除/退出逻辑。
      if (isMac) {
        return
      }
    }

    if (isCtrl(key, ch, 'x') && cState.queueEditIdx !== null) {
      cActions.removeQueue(cState.queueEditIdx)

      return cActions.clearIn()
    }

    if (key.ctrl && ch.toLowerCase() === 'c') {
      if (live.busy && live.sid) {
        const chat = chatStreamRef?.current ?? null

        if (chat?.isTurnActive()) {
          if (live.escapeArmed) {
            // 轮次仍进行时第二次按 Ctrl+C，说明第一次取消未产生终态事件，可能事件丢失或服务器卡死。
            // 在本地硬重置提示，避免用户永远等待不会到达的响应。
            patchUiState({ escapeArmed: false })
            chat.forceReset()

            return
          }

          // 第一次 Ctrl+C：发出正常服务器取消并启用逃生口。忙碌占位会切换为强制退出提示，
          // 第二次 Ctrl+C 回退到上方本地重置。
          patchUiState({ escapeArmed: true })
          chat.cancel().catch((err: Error) => actions.sys(`cancel failed: ${err.message}`))

          return
        }

        if (chat) {
          chat.cancel().catch((err: Error) => actions.sys(`cancel failed: ${err.message}`))
        }
        return
      }

      if (cState.input || cState.inputBuf.length) {
        return cActions.clearIn()
      }

      return actions.die()
    }

    if (isAction(key, ch, 'd')) {
      return actions.die()
    }

    if (isAction(key, ch, 'l')) {
      clearSelection()
      forceRedraw(terminal.stdout ?? process.stdout)

      return
    }

    // 支持 Cmd/Ctrl+G，并为 VSCode/Cursor 提供 Alt+G 回退；它们会在 TUI 收到主快捷键前将其
    // 绑定为“查找下一个”，而 Alt+G 在各平台均以 meta+g 到达。
    if (ch.toLowerCase() === 'g' && (isAction(key, ch, 'g') || key.meta)) {
      return void cActions.openEditor().catch((err: unknown) => {
        actions.sys(err instanceof Error ? `failed to open editor: ${err.message}` : 'failed to open editor')
      })
    }

    if (key.tab && cState.completions.length) {
      const row = cState.completions[cState.compIdx]

      if (row?.text) {
        const text =
          cState.input.startsWith('/') && row.text.startsWith('/') && cState.compReplace > 0
            ? row.text.slice(1)
            : row.text

        cActions.setInput(cState.input.slice(0, cState.compReplace) + text)
      }

      return
    }

    if (isAction(key, ch, 'k') && cRefs.queueRef.current.length && live.sid) {
      dispatchNextQueuedSubmission(cActions, actions.dispatchSubmission)
    }
  })

  return { pagerPageSize }
}
