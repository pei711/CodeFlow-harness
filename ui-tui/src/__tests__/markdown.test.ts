// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, renderSync } from '@hermes/ink'
import React from 'react'
import { PassThrough } from 'stream'
import { describe, expect, it } from 'vitest'

import { AUDIO_DIRECTIVE_RE, INLINE_RE, Md, MEDIA_LINE_RE, stripInlineMarkup } from '../components/markdown.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const matches = (text: string) => [...text.matchAll(INLINE_RE)].map(m => m[0])
const BEL = String.fromCharCode(7)
const ESC = String.fromCharCode(27)
const CSI_RE = new RegExp(`${ESC}\\[[0-?]*[ -/]*[@-~]`, 'g')
const OSC_RE = new RegExp(`${ESC}\\][\\s\\S]*?(?:${BEL}|${ESC}\\\\)`, 'g')

const renderPlain = (node: React.ReactNode) => {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 80, isTTY: false, rows: 24 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(node, {
    patchConsole: false,
    stderr: stderr as NodeJS.WriteStream,
    stdin: stdin as NodeJS.ReadStream,
    stdout: stdout as NodeJS.WriteStream
  })

  instance.unmount()
  instance.cleanup()

  return output
    .replace(OSC_RE, '')
    .split('\n')
    .map(line => stripAnsi(line).replace(CSI_RE, '').trimEnd())
}

describe('INLINE_RE emphasis', () => {
  it('matches word-boundary italic/bold', () => {
    expect(matches('say _hi_ there')).toEqual(['_hi_'])
    expect(matches('very __bold__ move')).toEqual(['__bold__'])
    expect(matches('(_paren_) and [_bracket_]')).toEqual(['_paren_', '_bracket_'])
  })

  it('keeps intraword underscores literal', () => {
    const path = '/home/me/.codeflow/cache/screenshots/browser_screenshot_ecc1c3feab.png'

    expect(matches(path)).toEqual([])
    expect(matches('snake_case_var and MY_CONST')).toEqual([])
    expect(matches('foo__bar__baz')).toEqual([])
  })

  it('still matches asterisk emphasis intraword', () => {
    expect(matches('a*b*c')).toEqual(['*b*'])
    expect(matches('a**bold**c')).toEqual(['**bold**'])
  })

  it('matches short alphanumeric subscript (H~2~O, CO~2~, X~n~)', () => {
    expect(matches('H~2~O')).toEqual(['~2~'])
    expect(matches('CO~2~ levels')).toEqual(['~2~'])
    expect(matches('the X~n~ term')).toEqual(['~n~'])
  })

  it('ignores kaomoji-style ~! and ~? punctuation', () => {
    // Kimi、Qwen、GLM 会将这些符号作为装饰器输出；此前两个波浪号之间的整段
    // 内容会被折叠成一个暗淡文本块。
    expect(matches('Aww ~! Building step by step, I love it ~!')).toEqual([])
    expect(matches('cool ~? yeah ~?')).toEqual([])
    expect(matches('mixed ~! and ~? flow')).toEqual([])
  })

  it('ignores tilde spans that contain spaces or punctuation', () => {
    // 真正的下标不含空格；波浪号后接单词再接波浪号几乎总是对话文本，匹配它会
    // 吞掉正文。
    expect(matches('hello ~good idea~ there')).toEqual([])
    expect(matches('x ~oh no!~ y')).toEqual([])
  })

  it('does not let strikethrough eat subscript', () => {
    expect(matches('~~strike~~ and H~2~O')).toEqual(['~~strike~~', '~2~'])
  })
})

describe('stripInlineMarkup', () => {
  it('strips word-boundary emphasis only', () => {
    expect(stripInlineMarkup('say _hi_ there')).toBe('say hi there')
    expect(stripInlineMarkup('browser_screenshot_ecc.png')).toBe('browser_screenshot_ecc.png')
    expect(stripInlineMarkup('__bold__ and foo__bar__')).toBe('bold and foo__bar__')
  })

  it('leaves ~!/~? kaomoji alone and still handles real subscript', () => {
    expect(stripInlineMarkup('Yay ~! nice work ~!')).toBe('Yay ~! nice work ~!')
    expect(stripInlineMarkup('H~2~O and CO~2~')).toBe('H_2O and CO_2')
  })

  it('strips inline math delimiters but keeps the formula text', () => {
    expect(stripInlineMarkup('$\\mathbb{Z}$ is a ring')).toBe('\\mathbb{Z} is a ring')
    expect(stripInlineMarkup('see \\(a + b\\) ok')).toBe('see a + b ok')
  })
})

describe('INLINE_RE inline math', () => {
  it('matches single-dollar math and beats emphasis at the same start', () => {
    // 若不处理数学公式，`*b*` 会被识别为斜体并破坏公式。将数学规则加入
    // INLINE_RE 后，第 0 列最靠左的匹配 `$P=a*b*c$` 会优先。
    expect(matches('$P=a*b*c$')).toEqual(['$P=a*b*c$'])
    expect(matches('see $\\mathbb{Z}$ here')).toEqual(['$\\mathbb{Z}$'])
  })

  it('does not match currency-style prose', () => {
    expect(matches('it costs $5 and $10')).toEqual([])
    expect(matches('paid $5')).toEqual([])
  })

  it('does not let inline math swallow a $$ display fence', () => {
    // `$$x$$` 是展示块，不是两个相邻的行内数学片段。
    expect(matches('$$x$$')).toEqual([])
  })

  it('matches \\(...\\) inline math', () => {
    expect(matches('foo \\(x + y\\) bar')).toEqual(['\\(x + y\\)'])
  })

  it('does not corrupt subscripts/superscripts inside math', () => {
    // `_n` 与 `^r` 在正文中是 Markdown 强调或上标标记，但位于 `$...$` 内时，
    // 整个公式会作为单个行内数学 token 捕获，内部正则不会看到这些字符。
    expect(matches('$P=a_n x^n + a_0$')).toEqual(['$P=a_n x^n + a_0$'])
    expect(matches('$\\beta_1,\\dots,\\beta_r$')).toEqual(['$\\beta_1,\\dots,\\beta_r$'])
  })

  it('places math content in the correct capture group (regression: m[16] is bare URL)', () => {
    // 由于裸 URL 模式缺少自身捕获括号，`m[16]` 曾同时充当裸 URL 组和行内数学
    // `$...$` 组，导致 MdInline 将 `$\\mathbb{R}$` 渲染成带下划线的自动链接，
    // 而不是琥珀色斜体数学文本。这里固定编号：数学使用 m[17]/m[18]，URL 使用
    // m[16]。
    const url = [...'see https://example.com here'.matchAll(INLINE_RE)][0]!
    const dollarMath = [...'$\\mathbb{R}$'.matchAll(INLINE_RE)][0]!
    const parenMath = [...'\\(\\pi\\)'.matchAll(INLINE_RE)][0]!

    expect(url[16]).toBe('https://example.com')
    expect(url[17]).toBeUndefined()
    expect(url[18]).toBeUndefined()

    expect(dollarMath[16]).toBeUndefined()
    expect(dollarMath[17]).toBe('\\mathbb{R}')
    expect(dollarMath[18]).toBeUndefined()

    expect(parenMath[16]).toBeUndefined()
    expect(parenMath[17]).toBeUndefined()
    expect(parenMath[18]).toBe('\\pi')
  })
})

describe('protocol sentinels', () => {
  it('captures MEDIA: paths with surrounding quotes or backticks', () => {
    expect('MEDIA:/tmp/a.png'.match(MEDIA_LINE_RE)?.[1]).toBe('/tmp/a.png')
    expect('  MEDIA: /home/me/.codeflow/cache/screenshots/browser_screenshot_ecc.png  '.match(MEDIA_LINE_RE)?.[1]).toBe(
      '/home/me/.codeflow/cache/screenshots/browser_screenshot_ecc.png'
    )
    expect('`MEDIA:/tmp/a.png`'.match(MEDIA_LINE_RE)?.[1]).toBe('/tmp/a.png')
    expect('"MEDIA:C:\\files\\a.png"'.match(MEDIA_LINE_RE)?.[1]).toBe('C:\\files\\a.png')
  })

  it('ignores MEDIA: tokens embedded in prose', () => {
    expect('here is MEDIA:/tmp/a.png for you'.match(MEDIA_LINE_RE)).toBeNull()
    expect('the media: section is empty'.match(MEDIA_LINE_RE)).toBeNull()
  })

  it('matches the [[audio_as_voice]] directive', () => {
    expect(AUDIO_DIRECTIVE_RE.test('[[audio_as_voice]]')).toBe(true)
    expect(AUDIO_DIRECTIVE_RE.test('  [[audio_as_voice]]  ')).toBe(true)
    expect(AUDIO_DIRECTIVE_RE.test('audio_as_voice')).toBe(false)
  })
})

describe('Md wrapping', () => {
  it('trims spaces from word-wrap continuation lines', () => {
    const lines = renderPlain(
      React.createElement(Box, { width: 5 }, React.createElement(Md, { t: DEFAULT_THEME, text: 'Let me' }))
    )

    expect(lines).toContain('Let')
    expect(lines).toContain('me')
    expect(lines).not.toContain(' me')
  })

  it('keeps nested list and quote indentation out of trim-sensitive text', () => {
    const lines = renderPlain(
      React.createElement(
        Box,
        { flexDirection: 'column', width: 24 },
        React.createElement(Md, { t: DEFAULT_THEME, text: '  - nested bullet' }),
        React.createElement(Md, { t: DEFAULT_THEME, text: '>> nested quote' })
      )
    )

    expect(lines).toContain('  • nested bullet')
    expect(lines).toContain('  │ nested quote')
  })

  it('preserves original inline-code edge spaces', () => {
    const lines = renderPlain(
      React.createElement(Box, { width: 24 }, React.createElement(Md, { t: DEFAULT_THEME, text: '` hi ` ok' }))
    )

    expect(lines.some(line => line.startsWith(' hi  ok'))).toBe(true)
  })
})

describe('Md link labels', () => {
  it('renders bare URLs with readable slug labels', () => {
    const lines = renderPlain(
      React.createElement(
        Box,
        { width: 120 },
        React.createElement(Md, {
          t: DEFAULT_THEME,
          text: 'see https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure for details'
        })
      )
    )

    const rendered = lines.join('\n')

    expect(rendered).toContain('Puerto Rico El Yunque Rainforest Adventure')
    expect(rendered).not.toContain('https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure')
  })

  it('keeps explicit markdown labels as the immediate fallback', () => {
    const lines = renderPlain(
      React.createElement(
        Box,
        { width: 80 },
        React.createElement(Md, {
          t: DEFAULT_THEME,
          text: '[Trip details](https://www.expedia.com/things-to-do/puerto-rico-el-yunque-rainforest-adventure)'
        })
      )
    )

    expect(lines.join('\n')).toContain('Trip details')
  })
})

describe('renderTable CJK width alignment', () => {
  it('column starts share the same display offset across CJK rows', async () => {
    const { stringWidth } = await import('@hermes/ink')

    const md = [
      '| 配置 | Config | 状态 |',
      '|------|--------|------|',
      '| Vicuna (report) | dense | × |',
      '| ChatGLM | chat | ✓ |',
      '| 通义千问 | qwen | × |'
    ].join('\n')

    // 修复前，` `.repeat(w - stripInlineMarkup(...).length) 使用 UTF-16 代码单元，
    // 导致中日韩标题单元格补齐为 2 格、正文单元格补齐为 4 格，后续列每个中日韩
    // 字符偏移 2 格。
    //
    // 修复后的约定：第 N 列起点之前的前缀在标题与所有正文行中显示宽度相同；
    // 去重时跳过独立渲染的分隔线。
    const lines = renderPlain(
      React.createElement(Box, null, React.createElement(Md, { compact: true, t: DEFAULT_THEME, text: md }))
    ).filter(line => line.trim().length > 0)

    // 启发式规则：数据行要么含标题 'Config'，要么含某个正文标签；分隔线则全是
    // 方框绘制字符。使用 'Config'、'dense'、'chat'、'qwen' 子串作为各行第 2 列
    // 起点的唯一锚点。
    const colStarts = (line: string, anchor: string): number => {
      const idx = line.indexOf(anchor)

      return idx < 0 ? -1 : stringWidth(line.slice(0, idx))
    }

    const headerCol2 = lines.map(l => colStarts(l, 'Config')).find(v => v >= 0)
    const denseCol2 = lines.map(l => colStarts(l, 'dense')).find(v => v >= 0)
    const chatCol2 = lines.map(l => colStarts(l, 'chat')).find(v => v >= 0)
    const qwenCol2 = lines.map(l => colStarts(l, 'qwen')).find(v => v >= 0)

    expect(headerCol2).toBeDefined()
    expect(denseCol2).toBe(headerCol2)
    expect(chatCol2).toBe(headerCol2)
    // 中日韩文本行在修复前发生偏移，现在必须与其余行对齐。
    expect(qwenCol2).toBe(headerCol2)
  })
})
