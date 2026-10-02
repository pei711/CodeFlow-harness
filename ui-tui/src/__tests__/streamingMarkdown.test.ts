// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { describe, expect, it } from 'vitest'

import { findStableBoundary } from '../components/streamingMarkdown.js'
// 通过重复调用渲染组件引用行为来测试纯边界逻辑。这里未渲染 React，而纯辅助函数
// findStableBoundary 又未导出，因此改测组件的可观察输出：依次传入文本值，并验证
// 稳定前缀永不后退。
//
// 策略：独立挂载 StreamingMd，并通过 text 属性观察它渲染了哪些 <Md> 实例。
// 在没有 DOM 渲染器时成本较高，因此通过重新导出的接口直接调用围栏和边界逻辑，
// 验证辅助函数行为。
import { DEFAULT_THEME } from '../theme.js'

describe('findStableBoundary', () => {
  it('returns -1 when no blank line exists yet', () => {
    expect(findStableBoundary('partial line with no newline yet')).toBe(-1)
  })

  it('returns -1 when only single newlines exist', () => {
    expect(findStableBoundary('line one\nline two\nline three')).toBe(-1)
  })

  it('splits after the last blank line separator', () => {
    // 'first\n\nsecond\n\nthird' 的最后一个空行为 'third' 之前。
    const text = 'first paragraph\n\nsecond paragraph\n\nthird'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('first paragraph\n\nsecond paragraph\n\n')
    expect(text.slice(idx)).toBe('third')
  })

  it('refuses to split inside an open fenced block', () => {
    // 围栏已打开且代码内部含空行，但尚未闭合。
    const text = '```ts\nfn();\n\nmore code here'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('splits before an open fenced block but not inside', () => {
    const text = 'intro paragraph\n\n```ts\nfn();\n\nmore code'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('```ts')).toBe(true)
  })

  it('allows splitting after a fenced block closes', () => {
    const text = '```ts\nfn();\n```\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```ts\nfn();\n```\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('walks backwards through nested fence boundaries safely', () => {
    // 两个闭合围栏加叙述文本后又打开一个新围栏；唯一合法切分点在新围栏之前，
    // 不能位于两个已闭合围栏之间。
    const text = '```js\na\n```\n\nmid text\n\n```python\nstill open'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('```js\na\n```\n\nmid text\n\n')
  })

  it('handles empty input', () => {
    expect(findStableBoundary('')).toBe(-1)
  })

  it('refuses to split inside an open $$ math block', () => {
    // 展示数学块已打开但未闭合，唯一空行位于开放块内部，因此尚无安全边界。
    const text = '$$\nx + y\n\nmore math'

    expect(findStableBoundary(text)).toBe(-1)
  })

  it('allows splitting after a $$ math block closes', () => {
    const text = '$$\nx + y = z\n$$\n\nnarration continues'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('$$\nx + y = z\n$$\n\n')
    expect(text.slice(idx)).toBe('narration continues')
  })

  it('splits before an open $$ block but not inside', () => {
    // 与现有围栏代码测试对称：正文后接未闭合数学块，唯一安全边界是 `$$` 前的空行。
    const text = 'intro paragraph\n\n$$\nx + y\n\nmore'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro paragraph\n\n')
    expect(text.slice(idx).startsWith('$$')).toBe(true)
  })

  it('treats single-line $$x$$ as zero net toggle', () => {
    // `$$x = y$$` 在同一行打开并闭合，因此允许在其后建立稳定边界。
    const text = 'intro\n\n$$x = y$$\n\nnarration'
    const idx = findStableBoundary(text)

    expect(text.slice(0, idx)).toBe('intro\n\n$$x = y$$\n\n')
    expect(text.slice(idx)).toBe('narration')
  })

  it('refuses to split inside an open \\[ math block', () => {
    const text = '\\[\nx + y\n\nmore'

    expect(findStableBoundary(text)).toBe(-1)
  })
})

describe('streaming theme assumption', () => {
  it('theme is exportable (component import sanity check)', () => {
    // 基线确认传入主题的结构不变。组件已在上方导入；此冒烟测试验证
    // streamingMarkdown 的模块图连接后没有循环依赖。
    expect(DEFAULT_THEME.color.accent).toBeTruthy()
  })
})
