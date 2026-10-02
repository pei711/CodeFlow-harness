// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.
//
// 编辑框的上下历史循环（useInputHandlers.ts 中的 cycleHistory）是钩子内未导出的
// 闭包，因此它遍历 useInputHistory -> lib/history.ts 返回的后备存储。这些测试
// 固定该存储的约定：按从旧到新插入、连续项去重，以及磁盘上以 `+` 为前缀并由
// load() 解析回来的往返格式。cycleHistory 依赖此顺序：dir<0 从最后一个索引
// （最新）走向索引 0（最旧），dir>0 再向前移动。

import { existsSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

import type { append, load } from '../lib/history.js'

interface HistoryModule {
  append: typeof append
  load: typeof load
}

let history: HistoryModule
let codeflowDir: string

beforeAll(async () => {
  codeflowDir = mkdtempSync(join(tmpdir(), 'codeflow-hist-'))

  vi.stubEnv('CODEFLOW_HOME', codeflowDir)
  history = await import('../lib/history.js')
})

afterAll(() => {
  vi.unstubAllEnvs()
})

describe('input history store (backs cycleHistory)', () => {
  it('starts empty when no history file exists', () => {
    expect(history.load()).toEqual([])
  })

  it('appends entries in insertion order with newest last', () => {
    history.append('first command')
    history.append('second command')
    history.append('third command')

    const entries = history.load()

    expect(entries).toEqual(['first command', 'second command', 'third command'])
    // 向上循环（dir<0）先显示最新项……
    expect(entries.at(-1)).toBe('third command')
    // ……再走到索引 0 的最旧项。
    expect(entries[0]).toBe('first command')
    expect(existsSync(join(codeflowDir, '.codeflow_history'))).toBe(true)
  })

  it('deduplicates a repeat of the most recent entry', () => {
    history.append('third command')

    expect(history.load()).toEqual(['first command', 'second command', 'third command'])
  })

  it('trims leading/trailing whitespace and ignores blank input', () => {
    history.append('   ')
    history.append('  spaced command  ')

    const entries = history.load()

    expect(entries.at(-1)).toBe('spaced command')
    expect(entries).not.toContain('   ')
  })

  it('round-trips a multi-line entry through the on-disk format', async () => {
    history.append('line a\nline b')

    vi.resetModules()
    const reloaded: HistoryModule = await import('../lib/history.js')

    expect(reloaded.load().at(-1)).toBe('line a\nline b')
    expect(reloaded.load()).toEqual([
      'first command',
      'second command',
      'third command',
      'spaced command',
      'line a\nline b'
    ])
  })
})
