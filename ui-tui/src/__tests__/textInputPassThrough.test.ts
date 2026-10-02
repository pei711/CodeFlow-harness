import { describe, expect, it } from 'vitest'

import { shouldPassThroughToGlobalHandler } from '../components/textInput.js'

const key = (overrides: Record<string, unknown> = {}) => ({ ctrl: false, meta: false, ...overrides }) as any

describe('shouldPassThroughToGlobalHandler', () => {
  it('passes terminal-level navigation and control keys through the composer', () => {
    expect(shouldPassThroughToGlobalHandler('c', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('x', key({ ctrl: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ escape: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ tab: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageUp: true }))).toBe(true)
    expect(shouldPassThroughToGlobalHandler('', key({ pageDown: true }))).toBe(true)
  })

  it('keeps ordinary typing in the composer', () => {
    expect(shouldPassThroughToGlobalHandler('h', key())).toBe(false)
    expect(shouldPassThroughToGlobalHandler('b', key({ ctrl: true }))).toBe(false)
  })
})
