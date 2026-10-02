// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { DOMElement, InputEvent, Key } from '@hermes/ink'

import * as Ink from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { type MutableRefObject, useEffect, useMemo, useRef, useState } from 'react'

import { setInputSelection } from '../app/inputSelectionStore.js'
import { $uiTheme } from '../app/uiStore.js'
import { readClipboardText, writeClipboardText } from '../lib/clipboard.js'
import { cursorLayout, offsetFromPosition } from '../lib/inputMetrics.js'
import { isActionMod, isMac, isMacActionFallback } from '../lib/platform.js'

type InkExt = typeof Ink & {
  stringWidth: (s: string) => number
  useDeclaredCursor: (a: { line: number; column: number; active: boolean }) => (el: DOMElement | null) => void
  useStdout: () => { stdout?: NodeJS.WriteStream }
  useTerminalFocus: () => boolean
}

const ink = Ink as unknown as InkExt
const { Box, Text, useStdin, useInput, useStdout, stringWidth, useDeclaredCursor, useTerminalFocus } = ink

const ESC = '\x1b'
const INV = `${ESC}[7m`
const INV_OFF = `${ESC}[27m`
const DIM = `${ESC}[2m`
const DIM_OFF = `${ESC}[22m`
const FWD_DEL_RE = new RegExp(`${ESC}\\[3(?:[~$^]|;)`)
const PRINTABLE = /^[ -~\u00a0-\uffff]+$/
const BRACKET_PASTE = new RegExp(`${ESC}?\\[20[01]~`, 'g')
const MULTI_CLICK_MS = 500

const invert = (s: string) => INV + s + INV_OFF
const dim = (s: string) => DIM + s + DIM_OFF

// 光标单元格：反色显示，可选着色。在反色下设置前景色会让反色块以 `color` 渲染，因为反色会把
// 前景交换为背景，从而得到主色块状光标并镂空字形；不设置时只是普通文本色反色块。
const cursorCell = (s: string, color?: string) =>
  color ? INV + Ink.colorize(s, color, 'foreground') + INV_OFF : invert(s)

let _seg: Intl.Segmenter | null = null
const seg = () => (_seg ??= new Intl.Segmenter(undefined, { granularity: 'grapheme' }))
const STOP_CACHE_MAX = 32
const stopCache = new Map<string, number[]>()

function graphemeStops(s: string) {
  const hit = stopCache.get(s)

  if (hit) {
    return hit
  }

  const stops = [0]

  for (const { index } of seg().segment(s)) {
    if (index > 0) {
      stops.push(index)
    }
  }

  if (stops.at(-1) !== s.length) {
    stops.push(s.length)
  }

  stopCache.set(s, stops)

  if (stopCache.size > STOP_CACHE_MAX) {
    const oldest = stopCache.keys().next().value

    if (oldest !== undefined) {
      stopCache.delete(oldest)
    }
  }

  return stops
}

function snapPos(s: string, p: number) {
  const pos = Math.max(0, Math.min(p, s.length))
  let last = 0

  for (const stop of graphemeStops(s)) {
    if (stop > pos) {
      break
    }

    last = stop
  }

  return last
}

function prevPos(s: string, p: number) {
  const pos = snapPos(s, p)
  let prev = 0

  for (const stop of graphemeStops(s)) {
    if (stop >= pos) {
      return prev
    }

    prev = stop
  }

  return prev
}

function nextPos(s: string, p: number) {
  const pos = snapPos(s, p)

  for (const stop of graphemeStops(s)) {
    if (stop > pos) {
      return stop
    }
  }

  return s.length
}

function wordLeft(s: string, p: number) {
  let i = snapPos(s, p) - 1

  while (i > 0 && /\s/.test(s[i]!)) {
    i--
  }

  while (i > 0 && !/\s/.test(s[i - 1]!)) {
    i--
  }

  return Math.max(0, i)
}

function wordRight(s: string, p: number) {
  let i = snapPos(s, p)

  while (i < s.length && !/\s/.test(s[i]!)) {
    i++
  }

  while (i < s.length && /\s/.test(s[i]!)) {
    i++
  }

  return i
}

/**
 * 在 `s` 中将光标上移或下移一个逻辑行，同时保留相对当前行首的列偏移。向上时
 * 已位于首行或向下时已位于末行则返回 `null`，调用方借此回退到历史记录循环，
 * 而不是吞掉方向键。
 */
export function lineNav(s: string, p: number, dir: -1 | 1): null | number {
  const pos = snapPos(s, p)
  const curStart = s.lastIndexOf('\n', pos - 1) + 1
  const col = pos - curStart

  if (dir < 0) {
    if (curStart === 0) {
      return null
    }

    const prevStart = s.lastIndexOf('\n', curStart - 2) + 1

    return snapPos(s, Math.min(prevStart + col, curStart - 1))
  }

  const nextBreak = s.indexOf('\n', pos)

  if (nextBreak < 0) {
    return null
  }

  const nextEnd = s.indexOf('\n', nextBreak + 1)
  const lineEnd = nextEnd < 0 ? s.length : nextEnd

  return snapPos(s, Math.min(nextBreak + 1 + col, lineEnd))
}

export { offsetFromPosition }

/**
 * 按键序列是单独的 LF 字节（`\n`，0x0A）时返回 true。在原始 stdin 中它始终
 * 表示 Ctrl+Enter 或 Ctrl+J，而普通回车在 xterm、WT、VS Code、iTerm 等的
 * 原始模式中都是 `\r`（0x0D）。此函数导出供单元测试使用；主回车键分派借此
 * 插入换行，无需依赖许多 SSH、zellij、VS Code 链路无法端到端支持的
 * Kitty/modifyOtherKeys 协议推送。
 */
export function isLfReturn(sequence: string | undefined): boolean {
  return sequence === '\n'
}

function renderWithCursor(value: string, cursor: number, cursorColor?: string) {
  const pos = Math.max(0, Math.min(cursor, value.length))

  let out = '',
    done = false

  for (const { segment, index } of seg().segment(value)) {
    if (!done && index >= pos) {
      out += cursorCell(index === pos && segment !== '\n' ? segment : ' ', cursorColor)
      done = true

      if (index === pos && segment !== '\n') {
        continue
      }
    }

    out += segment
  }

  return done ? out : out + cursorCell(' ', cursorColor)
}

function renderWithSelection(value: string, start: number, end: number) {
  if (start >= end) {
    return value
  }

  return value.slice(0, start) + invert(value.slice(start, end) || ' ') + value.slice(end)
}

function useFwdDelete(active: boolean) {
  const ref = useRef(false)
  const { inputEmitter: ee } = useStdin()

  useEffect(() => {
    if (!active) {
      return
    }

    const h = (d: string) => {
      ref.current = FWD_DEL_RE.test(d)
    }

    ee.prependListener('input', h)

    return () => {
      ee.removeListener('input', h)
    }
  }, [active, ee])

  return ref
}

type PasteResult = { cursor: number; fallbackText?: string; value: string } | null

export const asyncPasteFallbackText = (result: PasteResult, eventText: string): string | null =>
  result ? (result.fallbackText ?? eventText) : null

const isPasteResultPromise = (
  value: PasteResult | Promise<PasteResult> | null | undefined
): value is Promise<PasteResult> => !!value && typeof (value as PromiseLike<PasteResult>).then === 'function'

export function TextInput({
  columns = 80,
  value,
  onChange,
  onPaste,
  onSubmit,
  mask,
  mouseApiRef,
  placeholder = '',
  focus = true
}: TextInputProps) {
  const [cur, setCur] = useState(value.length)
  const [sel, setSel] = useState<null | { end: number; start: number }>(null)
  const fwdDel = useFwdDelete(focus)
  const termFocus = useTerminalFocus()
  const { stdout } = useStdout()
  const cursorColor = useStore($uiTheme).color.primary

  const curRef = useRef(cur)
  const selRef = useRef<null | { end: number; start: number }>(null)
  const vRef = useRef(value)
  const self = useRef(false)
  const pasteBuf = useRef('')
  const pasteEnd = useRef<null | number>(null)
  const pasteTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pastePos = useRef(0)
  const editVersionRef = useRef(0)
  const parentChangeTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingParentValue = useRef<string | null>(null)
  const localRenderTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lineWidthRef = useRef(stringWidth(value.includes('\n') ? value.slice(value.lastIndexOf('\n') + 1) : value))
  const mouseAnchorRef = useRef<null | number>(null)
  const lastClickRef = useRef<{ at: number; offset: number }>({ at: 0, offset: -1 })
  const undo = useRef<{ cursor: number; value: string }[]>([])
  const redo = useRef<{ cursor: number; value: string }[]>([])

  const cbChange = useRef(onChange)
  const cbSubmit = useRef(onSubmit)
  const cbPaste = useRef(onPaste)
  cbChange.current = onChange
  cbSubmit.current = onSubmit
  cbPaste.current = onPaste

  const raw = self.current ? vRef.current : value
  const display = mask ? raw.replace(/[^\n]/g, mask[0] ?? '*') : raw

  const selected = useMemo(
    () =>
      sel && sel.start !== sel.end ? { end: Math.max(sel.start, sel.end), start: Math.min(sel.start, sel.end) } : null,
    [sel]
  )

  const layout = useMemo(() => cursorLayout(display, cur, columns), [columns, cur, display])

  const boxRef = useDeclaredCursor({
    line: layout.line,
    column: layout.column,
    active: focus && termFocus && !selected
  })

  // 选区活动时隐藏硬件光标，避免反色文本恰好填满一列时自动换到下一行；终端失去焦点时也隐藏，
  // 抑制多数终端在停驻位置绘制的空心矩形残影。
  const hideHardwareCursor = focus && !!stdout?.isTTY && (!!selected || !termFocus)

  useEffect(() => {
    if (!hideHardwareCursor || !stdout) {
      return
    }

    stdout.write('\x1b[?25l')

    return () => {
      stdout.write('\x1b[?25h')
    }
  }, [hideHardwareCursor, stdout])

  const nativeCursor = focus && termFocus && !selected && !!stdout?.isTTY

  // 占位文本只是提示而非选区，应以暗色、非反色样式渲染。TTY 中硬件光标停在第 0 列，直观标记
  // 输入起点；非 TTY 表面仍需要合成的首字符反色才能绘制光标。
  const rendered = useMemo(() => {
    if (!focus) {
      return display || dim(placeholder)
    }

    if (!display && placeholder) {
      return nativeCursor
        ? dim(placeholder)
        : cursorCell(placeholder[0] ?? ' ', cursorColor) + dim(placeholder.slice(1))
    }

    if (selected) {
      return renderWithSelection(display, selected.start, selected.end)
    }

    return nativeCursor ? display || ' ' : renderWithCursor(display, cur, cursorColor)
  }, [cur, cursorColor, display, focus, nativeCursor, placeholder, selected])

  useEffect(() => {
    if (self.current) {
      self.current = false
    } else {
      setCur(value.length)
      setSel(null)
      curRef.current = value.length
      selRef.current = null
      vRef.current = value
      lineWidthRef.current = stringWidth(value.includes('\n') ? value.slice(value.lastIndexOf('\n') + 1) : value)
      undo.current = []
      redo.current = []
    }
  }, [value])

  useEffect(() => {
    if (!focus) {
      return
    }

    const dropSel = () => {
      if (!selRef.current) {
        return
      }

      selRef.current = null
      setSel(null)
    }

    setInputSelection({
      clear: dropSel,
      collapseToEnd: () => {
        dropSel()
        setCur(vRef.current.length)
        curRef.current = vRef.current.length
      },
      end: selected?.end ?? curRef.current,
      start: selected?.start ?? curRef.current,
      value: vRef.current
    })

    return () => setInputSelection(null)
  }, [cur, focus, selected])

  useEffect(
    () => () => {
      if (pasteTimer.current) {
        clearTimeout(pasteTimer.current)
      }

      if (parentChangeTimer.current) {
        clearTimeout(parentChangeTimer.current)
      }

      if (localRenderTimer.current) {
        clearTimeout(localRenderTimer.current)
      }
    },
    []
  )

  const flushParentChange = () => {
    if (parentChangeTimer.current) {
      clearTimeout(parentChangeTimer.current)
      parentChangeTimer.current = null
    }

    const next = pendingParentValue.current
    pendingParentValue.current = null

    if (next !== null) {
      self.current = true
      cbChange.current(next)
    }
  }

  const scheduleParentChange = (next: string) => {
    pendingParentValue.current = next

    if (parentChangeTimer.current) {
      return
    }

    parentChangeTimer.current = setTimeout(flushParentChange, 16)
  }

  const cancelLocalRender = () => {
    if (localRenderTimer.current) {
      clearTimeout(localRenderTimer.current)
      localRenderTimer.current = null
    }
  }

  const scheduleLocalRender = () => {
    if (localRenderTimer.current) {
      return
    }

    localRenderTimer.current = setTimeout(() => {
      localRenderTimer.current = null
      setCur(curRef.current)
    }, 16)
  }

  const canFastEchoBase = () => focus && termFocus && !selected && !mask && !!stdout?.isTTY

  const canFastAppend = (current: string, cursor: number, text: string) => {
    const sw = stringWidth(text)

    return (
      canFastEchoBase() &&
      cursor === current.length &&
      current.length > 0 &&
      !current.includes('\n') &&
      sw === text.length &&
      lineWidthRef.current + sw < Math.max(1, columns)
    )
  }

  const canFastBackspace = (current: string, cursor: number) => {
    if (!canFastEchoBase() || cursor !== current.length || cursor <= 0 || current.includes('\n')) {
      return false
    }

    return stringWidth(current.slice(prevPos(current, cursor), cursor)) === 1
  }

  const commit = (
    next: string,
    nextCur: number,
    track = true,
    syncParent = true,
    syncLocal = true,
    nextLineWidth?: number
  ) => {
    const prev = vRef.current
    const c = snapPos(next, nextCur)
    editVersionRef.current += 1

    if (selRef.current) {
      selRef.current = null
      setSel(null)
    }

    if (track && next !== prev) {
      undo.current.push({ cursor: curRef.current, value: prev })

      if (undo.current.length > 200) {
        undo.current.shift()
      }

      redo.current = []
    }

    if (syncLocal) {
      cancelLocalRender()
      setCur(c)
    } else {
      scheduleLocalRender()
    }

    curRef.current = c
    vRef.current = next
    lineWidthRef.current =
      nextLineWidth ?? stringWidth(next.includes('\n') ? next.slice(next.lastIndexOf('\n') + 1) : next)

    if (next !== prev) {
      if (syncParent) {
        flushParentChange()
        self.current = true
        cbChange.current(next)
      } else {
        self.current = true
        scheduleParentChange(next)
      }
    }
  }

  const swap = (from: typeof undo, to: typeof redo) => {
    const entry = from.current.pop()

    if (!entry) {
      return
    }

    to.current.push({ cursor: curRef.current, value: vRef.current })
    commit(entry.value, entry.cursor, false)
  }

  const emitPaste = (e: PasteEvent) => {
    const startVersion = editVersionRef.current
    const h = cbPaste.current?.(e)

    if (isPasteResultPromise(h)) {
      void h
        .then(result => {
          if (result && editVersionRef.current === startVersion) {
            commit(result.value, result.cursor)
          } else {
            const fallbackText = asyncPasteFallbackText(result, e.text)

            if (!fallbackText || !PRINTABLE.test(fallbackText)) {
              return
            }

            // 异步粘贴进行时用户又输入了内容；在当前光标处插入已解析的粘贴载荷，避免静默丢失。
            const cur = curRef.current
            const v = vRef.current
            commit(v.slice(0, cur) + fallbackText + v.slice(cur), cur + fallbackText.length)
          }
        })
        .catch(() => {})

      return true
    }

    if (h) {
      commit(h.value, h.cursor)
    }

    return !!h
  }

  const flushPaste = () => {
    const text = pasteBuf.current
    const at = pastePos.current
    const end = pasteEnd.current ?? at
    pasteBuf.current = ''
    pasteEnd.current = null
    pasteTimer.current = null

    if (!text) {
      return
    }

    if (!emitPaste({ cursor: at, text, value: vRef.current }) && PRINTABLE.test(text)) {
      commit(vRef.current.slice(0, at) + text + vRef.current.slice(end), at + text.length)
    }
  }

  const clearSel = () => {
    if (!selRef.current) {
      return
    }

    selRef.current = null
    setSel(null)
  }

  const selectAll = () => {
    const end = vRef.current.length

    if (!end) {
      return
    }

    const next = { end, start: 0 }
    selRef.current = next
    setSel(next)
    setCur(end)
    curRef.current = end
  }

  const moveCursor = (next: number, extend = false) => {
    const c = snapPos(vRef.current, next)
    const anchor = selRef.current?.start ?? curRef.current

    if (!extend || anchor === c) {
      clearSel()
    } else {
      const nextSel = { end: c, start: anchor }
      selRef.current = nextSel
      setSel(nextSel)
    }

    setCur(c)
    curRef.current = c
  }

  const selRange = () => {
    const range = selRef.current

    return range && range.start !== range.end
      ? { end: Math.max(range.start, range.end), start: Math.min(range.start, range.end) }
      : null
  }

  const ins = (v: string, c: number, s: string) => v.slice(0, c) + s + v.slice(c)

  const pastePlainText = (text: string) => {
    const cleaned = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

    if (!cleaned) {
      return
    }

    const range = selRange()

    const nextValue = range
      ? vRef.current.slice(0, range.start) + cleaned + vRef.current.slice(range.end)
      : vRef.current.slice(0, curRef.current) + cleaned + vRef.current.slice(curRef.current)

    const nextCursor = range ? range.start + cleaned.length : curRef.current + cleaned.length

    commit(nextValue, nextCursor)
  }

  const startMouseSelection = (next: number) => {
    const c = snapPos(vRef.current, next)

    mouseAnchorRef.current = c
    selRef.current = { end: c, start: c }
    setSel(null)
    setCur(c)
    curRef.current = c
  }

  const dragMouseSelection = (next: number) => {
    if (mouseAnchorRef.current === null) {
      return
    }

    const c = snapPos(vRef.current, next)
    const range = { end: c, start: mouseAnchorRef.current }
    selRef.current = range
    setSel(range.start === range.end ? null : range)
    setCur(c)
    curRef.current = c
  }

  const endMouseSelection = () => {
    mouseAnchorRef.current = null

    const range = selRef.current

    if (range && range.start === range.end) {
      selRef.current = null
      setSel(null)

      return
    }

    const normalized = selRange()

    if (isMac && normalized) {
      void writeClipboardText(vRef.current.slice(normalized.start, normalized.end))
    }
  }

  const offsetAt = (e: { localCol?: number; localRow?: number }) =>
    offsetFromPosition(display, e.localRow ?? 0, e.localCol ?? 0, columns)

  const isMultiClickAt = (offset: number) => {
    const now = Date.now()
    const last = lastClickRef.current
    lastClickRef.current = { at: now, offset }

    return now - last.at < MULTI_CLICK_MS && offset === last.offset
  }

  if (mouseApiRef) {
    mouseApiRef.current = {
      dragAt: (row, col) => dragMouseSelection(offsetFromPosition(display, row, col, columns)),
      end: endMouseSelection,
      startAtBeginning: () => startMouseSelection(0)
    }
  }

  useInput(
    (inp: string, k: Key, event: InputEvent) => {
      const eventRaw = event.keypress.raw

      if (shouldPassThroughToGlobalHandler(inp, k)) {
        return
      }

      if (
        eventRaw === '\x1bv' ||
        eventRaw === '\x1bV' ||
        eventRaw === '\x16' ||
        (isMac && isActionMod(k) && inp.toLowerCase() === 'v')
      ) {
        if (cbPaste.current) {
          return void emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })
        }

        if (isMac) {
          void readClipboardText().then(text => {
            if (text) {
              pastePlainText(text)
            }
          })
        }

        return
      }

      if (isMac && isActionMod(k) && inp.toLowerCase() === 'c') {
        const range = selRange()

        if (range) {
          const text = vRef.current.slice(range.start, range.end)

          void writeClipboardText(text)
        }

        return
      }

      if (k.upArrow || k.downArrow) {
        const next = lineNav(vRef.current, curRef.current, k.upArrow ? -1 : 1)

        if (next !== null) {
          moveCursor(next, k.shift)

          return
        }

        return
      }

      if (k.return) {
        // 原始标准输入中的 LF 字节（\n, 0x0a）普遍表示 Ctrl+Enter，历史上也表示 Ctrl+J；CR 字节
        // （\r, 0x0d）表示普通 Enter。字节级区别是跨技术栈可靠检测 Ctrl+Enter 的唯一方式；
        // 许多 SSH + zellij + VSCode 链路不遵守 Kitty 键盘/xterm modifyOtherKeys 协议推送。
        // 将 LF 视为 Ctrl+Enter 也会让 Ctrl+J 插入换行，语义兼容，因为聊天编辑器未给 Ctrl+J
        // 其他绑定。该路径独立且补充 CSI u 路径（Kitty/modifyOtherKeys 解析器的 k.shift/k.ctrl）；
        // 对遵守协议推送的终端仍优先使用后者。
        const lfByteAsCtrlEnter = isLfReturn(event.keypress.sequence)

        if (k.shift || k.ctrl || (isMac ? isActionMod(k) : k.meta) || lfByteAsCtrlEnter) {
          flushParentChange()
          commit(ins(vRef.current, curRef.current, '\n'), curRef.current + 1)
        } else {
          flushParentChange()
          cbSubmit.current?.(vRef.current)
        }

        return
      }

      let c = curRef.current
      let v = vRef.current
      const mod = isActionMod(k)
      const wordMod = mod || k.meta
      const actionHome = k.home || (!isMac && mod && inp === 'a') || isMacActionFallback(k, inp, 'a')
      const actionEnd = k.end || (mod && inp === 'e') || isMacActionFallback(k, inp, 'e')
      const actionDeleteToStart = (mod && inp === 'u') || isMacActionFallback(k, inp, 'u')
      const actionKillToEnd = (mod && inp === 'k') || isMacActionFallback(k, inp, 'k')
      const actionDeleteWord = (mod && inp === 'w') || isMacActionFallback(k, inp, 'w')
      const range = selRange()
      const delFwd = k.delete || fwdDel.current

      if (mod && inp === 'z') {
        return swap(undo, redo)
      }

      if ((mod && inp === 'y') || (mod && k.shift && inp === 'z')) {
        return swap(redo, undo)
      }

      if (isMac && mod && inp === 'a') {
        return selectAll()
      }

      if (actionHome) {
        c = 0
        moveCursor(c, k.shift)

        return
      } else if (actionEnd) {
        c = v.length
        moveCursor(c, k.shift)

        return
      } else if (k.leftArrow) {
        if (range && !wordMod && !k.shift) {
          clearSel()
          c = range.start
        } else {
          c = wordMod ? wordLeft(v, c) : prevPos(v, c)
        }

        moveCursor(c, k.shift)

        return
      } else if (k.rightArrow) {
        if (range && !wordMod && !k.shift) {
          clearSel()
          c = range.end
        } else {
          c = wordMod ? wordRight(v, c) : nextPos(v, c)
        }

        moveCursor(c, k.shift)

        return
      } else if (wordMod && inp === 'b') {
        clearSel()
        c = wordLeft(v, c)
      } else if (wordMod && inp === 'f') {
        clearSel()
        c = wordRight(v, c)
      } else if (range && (k.backspace || delFwd)) {
        v = v.slice(0, range.start) + v.slice(range.end)
        c = range.start
      } else if (k.backspace && c > 0) {
        if (wordMod) {
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else if (canFastBackspace(v, c)) {
          const t = prevPos(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
          stdout!.write('\b \b')
          commit(v, c, true, false, false, Math.max(0, lineWidthRef.current - 1))

          return
        } else {
          const t = prevPos(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        }
      } else if (delFwd && c < v.length) {
        if (wordMod) {
          const t = wordRight(v, c)
          v = v.slice(0, c) + v.slice(t)
        } else {
          v = v.slice(0, c) + v.slice(nextPos(v, c))
        }
      } else if (actionDeleteWord) {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else if (c > 0) {
          clearSel()
          const t = wordLeft(v, c)
          v = v.slice(0, t) + v.slice(c)
          c = t
        } else {
          return
        }
      } else if (actionDeleteToStart) {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else {
          v = v.slice(c)
          c = 0
        }
      } else if (actionKillToEnd) {
        if (range) {
          v = v.slice(0, range.start) + v.slice(range.end)
          c = range.start
        } else {
          v = v.slice(0, c)
        }
      } else if (event.keypress.isPasted || inp.length > 0) {
        const bracketed = event.keypress.isPasted || inp.includes('[200~')
        const text = inp.replace(BRACKET_PASTE, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        if (bracketed && emitPaste({ bracketed: true, cursor: c, text, value: v })) {
          return
        }

        if (!text) {
          return
        }

        if (text === '\n') {
          return commit(ins(v, c, '\n'), c + 1)
        }

        if (text.length > 1 || text.includes('\n')) {
          if (!pasteBuf.current) {
            pastePos.current = range ? range.start : c
            pasteEnd.current = range ? range.end : pastePos.current
          }

          pasteBuf.current += text

          if (pasteTimer.current) {
            clearTimeout(pasteTimer.current)
          }

          pasteTimer.current = setTimeout(flushPaste, 50)

          return
        }

        if (PRINTABLE.test(text)) {
          if (range) {
            v = v.slice(0, range.start) + text + v.slice(range.end)
            c = range.start + text.length
          } else {
            const simpleAppend = canFastAppend(v, c, text)

            v = v.slice(0, c) + text + v.slice(c)
            c += text.length

            if (simpleAppend) {
              stdout!.write(text)
              commit(v, c, true, false, false, lineWidthRef.current + stringWidth(text))

              return
            }
          }
        } else {
          return
        }
      } else {
        return
      }

      commit(v, c)
    },
    { isActive: focus }
  )

  return (
    <Box
      onClick={(e: MouseEventLite) => {
        if (!focus) {
          return
        }

        e.stopImmediatePropagation?.()
        clearSel()
        const next = offsetAt(e)
        setCur(next)
        curRef.current = next
      }}
      onMouseDown={(e: MouseEventLite) => {
        if (!focus) {
          return
        }

        // 右键：有活动选区时复制，否则粘贴。
        if (e.button === 2) {
          e.stopImmediatePropagation?.()
          const decision = decideRightClickAction(vRef.current, selRange())

          if (decision.action === 'copy') {
            void writeClipboardText(decision.text)

            return
          }

          emitPaste({ cursor: curRef.current, hotkey: true, text: '', value: vRef.current })

          return
        }

        if (e.button !== 0) {
          return
        }

        e.stopImmediatePropagation?.()
        const offset = offsetAt(e)

        if (isMultiClickAt(offset)) {
          mouseAnchorRef.current = null
          selectAll()

          return
        }

        startMouseSelection(offset)
      }}
      onMouseDrag={(e: MouseEventLite) => {
        if (!focus || e.button !== 0 || mouseAnchorRef.current === null) {
          return
        }

        e.stopImmediatePropagation?.()
        dragMouseSelection(offsetAt(e))
      }}
      onMouseUp={(e: MouseEventLite) => {
        e.stopImmediatePropagation?.()
        endMouseSelection()
      }}
      ref={boxRef}
      width={columns}
    >
      <Text wrap="wrap">{rendered}</Text>
    </Box>
  )
}

type MouseEventLite = {
  button?: number
  localCol?: number
  localRow?: number
  stopImmediatePropagation?: () => void
}

export interface PasteEvent {
  bracketed?: boolean
  cursor: number
  hotkey?: boolean
  text: string
  value: string
}

interface TextInputProps {
  columns?: number
  focus?: boolean
  mask?: string
  mouseApiRef?: MutableRefObject<null | TextInputMouseApi>
  onChange: (v: string) => void
  onPaste?: (e: PasteEvent) => PasteResult | Promise<PasteResult>
  onSubmit?: (v: string) => void
  placeholder?: string
  value: string
}

export type RightClickDecision = { action: 'copy'; text: string } | { action: 'paste' }

/**
 * 决定在编辑框中右键时执行的操作：
 *   - 非空选择范围：将文本复制到剪贴板；
 *   - 无选择范围或范围为空、已折叠：回退到粘贴。
 *
 * 与 xterm、iTerm、gnome-terminal 等终端原生行为一致：只有没有可复制的选中
 * 内容时，右键才粘贴。
 *
 * 调用方传入 `selRange()` 已归一化的范围（start <= end，折叠时为 null），
 * 因此本辅助函数无需再次归一化。
 */
export function decideRightClickAction(
  value: string,
  range: { end: number; start: number } | null
): RightClickDecision {
  if (range && range.end > range.start) {
    const text = value.slice(range.start, range.end)

    if (text) {
      return { action: 'copy', text }
    }
  }

  return { action: 'paste' }
}

export const shouldPassThroughToGlobalHandler = (input: string, key: Key): boolean =>
  Boolean(
    (key.ctrl && input === 'c') ||
    (key.ctrl && input === 'x') ||
    key.tab ||
    (key.shift && key.tab) ||
    key.pageUp ||
    key.pageDown ||
    key.escape
  )

export interface TextInputMouseApi {
  dragAt: (row: number, col: number) => void
  end: () => void
  startAtBeginning: () => void
}
