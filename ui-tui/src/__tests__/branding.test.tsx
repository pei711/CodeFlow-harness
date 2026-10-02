// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { render } from 'ink-testing-library'
import React from 'react'
import { describe, expect, it } from 'vitest'

import { codeflowLogo, codeflowLogoWord, CODEFLOW_WORD_WIDTH } from '../banner.js'
import { Branding, formatProvider, StartupLoader } from '../components/branding.js'
import { DEFAULT_THEME } from '../theme.js'

describe('Branding', () => {
  it('renders without throwing when invoked with no props', () => {
    // 布局会按终端宽度选择完整、堆叠或紧凑模式，三者都必须正确渲染。
    expect(() => render(<Branding />)).not.toThrow()
  })
})

describe('StartupLoader', () => {
  it('renders a spinner frame and the first startup message', () => {
    const { lastFrame } = render(<StartupLoader t={DEFAULT_THEME} />)
    expect(lastFrame()).toContain('starting codeflow')
    expect(lastFrame()).toMatch(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/)
  })
})

// 字标本身与宽度无关；直接测试纯构建器，避免依赖 ink-testing 的终端列数。
describe('banner wordmark', () => {
  const ramp = DEFAULT_THEME.yellow

  it('codeflowLogo renders the 6-row block wordmark', () => {
    const lines = codeflowLogo(ramp)
    expect(lines.length).toBe(6)
    expect(lines.map(([, text]) => text).join('')).toContain('█')
  })

  it('codeflowLogoWord renders just CODEFLOW within one-word width', () => {
    const lines = codeflowLogoWord(ramp)
    expect(lines.length).toBe(6)
    expect(lines.map(([, text]) => text).join('')).toContain('█')
    const maxWidth = Math.max(...lines.map(([, text]) => [...text].length))
    expect(maxWidth).toBeLessThanOrEqual(CODEFLOW_WORD_WIDTH)
  })

  it('CODEFLOW word is compact enough for a standard terminal', () => {
    expect(CODEFLOW_WORD_WIDTH).toBeLessThanOrEqual(40)
  })
})

describe('formatProvider', () => {
  it('returns LUT value for known anthropic slug', () => {
    expect(formatProvider('anthropic', 'claude-sonnet-4-6')).toBe('Anthropic')
  })

  it('parses model_id prefix when slug is "auto" (LiteLLM dispatch)', () => {
    expect(formatProvider('auto', 'openrouter/qwen/qwen3.6-plus')).toBe('OpenRouter')
  })

  it('returns LUT value for qwen slug', () => {
    expect(formatProvider('qwen', 'qwen-max')).toBe('Qwen')
  })

  it('returns em-dash fallback when slug empty and model_id has no slash prefix', () => {
    expect(formatProvider('', 'sonnet')).toBe('—')
  })

  it('returns em-dash fallback when slug is "auto" and model_id is empty', () => {
    expect(formatProvider('auto', '')).toBe('—')
  })

  it('returns canonical OpenAI (not Openai) for openai slug', () => {
    expect(formatProvider('openai', 'gpt-4')).toBe('OpenAI')
  })

  it('falls back to capitalize for unknown providers', () => {
    expect(formatProvider('xyz', 'xyz-foo')).toBe('Xyz')
  })
})
