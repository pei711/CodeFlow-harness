// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it } from 'vitest'

import { lineNav } from '../components/textInput.js'

describe('lineNav', () => {
  it('returns null for single-line input (up)', () => {
    expect(lineNav('hello world', 6, -1)).toBeNull()
  })

  it('returns null for single-line input (down)', () => {
    expect(lineNav('hello world', 6, 1)).toBeNull()
  })

  it('returns null when cursor already on first line of a multiline block', () => {
    expect(lineNav('one\ntwo\nthree', 2, -1)).toBeNull()
  })

  it('returns null when cursor on last line of a multiline block', () => {
    expect(lineNav('one\ntwo\nthree', 10, 1)).toBeNull()
  })

  it('moves cursor up one line preserving column', () => {
    // "hello\nworld"：光标从第 1 行第 3 列（world 中的 l）移到第 0 行第 3 列（hello 中的 l）。
    expect(lineNav('hello\nworld', 9, -1)).toBe(3)
  })

  it('moves cursor down one line preserving column', () => {
    // 光标从第 0 行第 2 列移到第 1 行第 2 列。
    expect(lineNav('hello\nworld', 2, 1)).toBe(8)
  })

  it('clamps to end of shorter destination line on up', () => {
    // 长行第 10 列会限制到短行 "abc" 的末尾。
    const s = 'abc\nlong long text'
    const from = 14

    expect(lineNav(s, from, -1)).toBe(3)
  })

  it('clamps to end of shorter destination line on down', () => {
    // 第 0 行第 10 列会限制到第 1 行 "abc" 的末尾。
    const s = 'long long text\nabc'

    expect(lineNav(s, 10, 1)).toBe(18)
  })

  it('handles empty lines correctly', () => {
    // "a\n\nb"：光标从第 2 行的 b 上移到空的第 1 行。
    expect(lineNav('a\n\nb', 3, -1)).toBe(2)
  })

  it('handles leading newline without crashing', () => {
    expect(lineNav('\nfoo', 2, -1)).toBe(0)
  })
})
