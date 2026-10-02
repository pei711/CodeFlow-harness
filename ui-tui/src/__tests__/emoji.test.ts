// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it } from 'vitest'

import { ensureEmojiPresentation } from '../lib/emoji.js'

const VS16 = '\uFE0F'

describe('ensureEmojiPresentation', () => {
  it('passes through ASCII unchanged', () => {
    expect(ensureEmojiPresentation('hello world')).toBe('hello world')
    expect(ensureEmojiPresentation('')).toBe('')
  })

  it('passes through emoji that already defaults to emoji presentation', () => {
    expect(ensureEmojiPresentation('🚀 rocket')).toBe('🚀 rocket')
    expect(ensureEmojiPresentation('😀')).toBe('😀')
  })

  it('injects VS16 after text-default emoji codepoints', () => {
    expect(ensureEmojiPresentation('⚠ careful')).toBe(`⚠${VS16} careful`)
    expect(ensureEmojiPresentation('ℹ info')).toBe(`ℹ${VS16} info`)
    expect(ensureEmojiPresentation('love ❤ you')).toBe(`love ❤${VS16} you`)
    expect(ensureEmojiPresentation('✔ done')).toBe(`✔${VS16} done`)
  })

  it('is idempotent when VS16 is already present', () => {
    const already = `⚠${VS16} ℹ${VS16} ❤${VS16}`

    expect(ensureEmojiPresentation(already)).toBe(already)
    expect(ensureEmojiPresentation(ensureEmojiPresentation('⚠'))).toBe(`⚠${VS16}`)
  })

  it('leaves keycap sequences alone when the base is not a text-default emoji', () => {
    expect(ensureEmojiPresentation('1\u20e3')).toBe('1\u20e3')
  })

  it('injects VS16 before ZWJ so text-default bases participate in emoji sequences', () => {
    // ❤ + ZWJ + 🔥 会变成 ❤️‍🔥（燃烧的心）。若心形与 ZWJ 之间没有 VS16，
    // 终端会以文本或单色形式渲染心形，ZWJ 连字也可能无法形成。
    const heartFire = '\u2764\u200d\ud83d\udd25'

    expect(ensureEmojiPresentation(heartFire)).toBe(`\u2764\uFE0F\u200d\ud83d\udd25`)
  })

  it('leaves explicit text-presentation selector (VS15) alone', () => {
    // `❤︎`（U+2764 + U+FE0E）要求文本呈现；注入 VS16 会产生无效的双变体序列。
    const explicitText = '\u2764\ufe0e'

    expect(ensureEmojiPresentation(explicitText)).toBe(explicitText)
  })

  it('returns the original reference when no change is needed', () => {
    const already = `⚠${VS16} ℹ${VS16} ❤${VS16}`

    // 验证引用相等：无需注入时，延迟分配器应直接返回输入。
    expect(ensureEmojiPresentation(already)).toBe(already)
  })

  it('handles mixed content', () => {
    expect(ensureEmojiPresentation('⚠ path: /tmp/x ❤ done')).toBe(`⚠${VS16} path: /tmp/x ❤${VS16} done`)
  })
})
