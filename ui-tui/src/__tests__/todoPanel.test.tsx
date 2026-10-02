// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { render } from 'ink-testing-library'
import React from 'react'
import { describe, expect, it } from 'vitest'

import type { TodoItem } from '../types.js'

import { TodoPanel } from '../components/todoPanel.js'
import { stripAnsi } from '../lib/text.js'
import { DEFAULT_THEME } from '../theme.js'

const frameOf = (node: React.ReactElement): string => stripAnsi(render(node).lastFrame() ?? '')

const mixed: TodoItem[] = [
  { content: 'done item', id: '1', status: 'completed' },
  { content: 'active item', id: '2', status: 'in_progress' },
  { content: 'todo item', id: '3', status: 'pending' },
  { content: 'skip item', id: '4', status: 'cancelled' }
]

describe('TodoPanel', () => {
  it('renders each todo item text (R9)', () => {
    const frame = frameOf(<TodoPanel t={DEFAULT_THEME} todos={mixed} />)

    for (const todo of mixed) {
      expect(frame).toContain(todo.content)
    }
  })

  it('shows a completed / total count (R10)', () => {
    const frame = frameOf(<TodoPanel t={DEFAULT_THEME} todos={mixed} />)

    // 只有 `completed` 计入完成数；四项列表中恰好有一项。
    expect(frame).toContain('(1/4)')
  })

  it('renders the per-status glyph on each row (R11)', () => {
    const frame = frameOf(<TodoPanel t={DEFAULT_THEME} todos={mixed} />)

    expect(frame).toContain('[x] done item')
    expect(frame).toContain('[>] active item')
    expect(frame).toContain('[ ] todo item')
    expect(frame).toContain('[-] skip item')
  })

  it('renders nothing for an empty list', () => {
    expect(frameOf(<TodoPanel t={DEFAULT_THEME} todos={[]} />).trim()).toBe('')
  })

  it('hides rows when collapsed and shows them when expanded (R12)', () => {
    const collapsed = frameOf(<TodoPanel collapsed t={DEFAULT_THEME} todos={mixed} />)
    const expanded = frameOf(<TodoPanel collapsed={false} t={DEFAULT_THEME} todos={mixed} />)

    // 标题和计数在两种状态下都保留，只有各行切换显示。
    expect(collapsed).toContain('Todo')
    expect(collapsed).toContain('(1/4)')
    expect(collapsed).not.toContain('done item')

    expect(expanded).toContain('done item')
    expect(expanded).toContain('todo item')
  })

  it('respects defaultCollapsed for the uncontrolled path (R12)', () => {
    const frame = frameOf(<TodoPanel defaultCollapsed t={DEFAULT_THEME} todos={mixed} />)

    expect(frame).toContain('Todo')
    expect(frame).not.toContain('done item')
  })

  it('updates the completed count when item status changes (R13)', () => {
    const before: TodoItem[] = [
      { content: 'a', id: 'a', status: 'completed' },
      { content: 'b', id: 'b', status: 'pending' },
      { content: 'c', id: 'c', status: 'pending' }
    ]
    const { lastFrame, rerender } = render(<TodoPanel t={DEFAULT_THEME} todos={before} />)

    expect(stripAnsi(lastFrame() ?? '')).toContain('(1/3)')

    const after = before.map(todo => (todo.id === 'b' ? { ...todo, status: 'completed' as const } : todo))
    rerender(<TodoPanel t={DEFAULT_THEME} todos={after} />)

    expect(stripAnsi(lastFrame() ?? '')).toContain('(2/3)')
  })
})
