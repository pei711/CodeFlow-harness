// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it } from 'vitest'

import { isLfReturn } from '../components/textInput.js'

// 原始 stdin 中的 LF 字节（\n，0x0a）视为 Ctrl+Enter（插入换行），无需依赖许多
// SSH、zellij、VS Code 链路无法支持的 Kitty/modifyOtherKeys 协议推送。CR 字节
// （\r）仍表示普通回车（提交）；粘贴的 CRLF 不能匹配，因为粘贴走独立路径。
describe('isLfReturn (ctrl+enter detection via LF byte)', () => {
  it('returns true for bare LF (ctrl+enter / ctrl+j)', () => {
    expect(isLfReturn('\n')).toBe(true)
  })

  it('returns false for bare CR (plain enter — submit path)', () => {
    expect(isLfReturn('\r')).toBe(false)
  })

  it('returns false for ESC+CR (alt+enter — handled by text fall-through)', () => {
    expect(isLfReturn('\x1b\r')).toBe(false)
  })

  it('returns false for ESC+LF (alt+ctrl+j variant — handled elsewhere)', () => {
    expect(isLfReturn('\x1b\n')).toBe(false)
  })

  it('returns false for CSI u return sequences (kitty keyboard protocol)', () => {
    expect(isLfReturn('\x1b[13;5u')).toBe(false)
    expect(isLfReturn('\x1b[13;2u')).toBe(false)
    expect(isLfReturn('\x1b[13;3u')).toBe(false)
  })

  it('returns false for CRLF (pasted line ending, not a single keystroke)', () => {
    expect(isLfReturn('\r\n')).toBe(false)
  })

  it('returns false for empty / undefined / whitespace', () => {
    expect(isLfReturn(undefined)).toBe(false)
    expect(isLfReturn('')).toBe(false)
    expect(isLfReturn(' ')).toBe(false)
    expect(isLfReturn('\n\n')).toBe(false)
  })

  it('returns false for printable characters', () => {
    expect(isLfReturn('a')).toBe(false)
    expect(isLfReturn('n')).toBe(false)
    expect(isLfReturn('\\n')).toBe(false)
  })
})
