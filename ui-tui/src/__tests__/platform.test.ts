// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { afterEach, describe, expect, it, vi } from 'vitest'

const originalPlatform = process.platform

async function importPlatform(platform: NodeJS.Platform) {
  vi.resetModules()
  Object.defineProperty(process, 'platform', { value: platform })

  return import('../lib/platform.js')
}

afterEach(() => {
  Object.defineProperty(process, 'platform', { value: originalPlatform })
  vi.resetModules()
})

describe('platform shortcuts', () => {
  it('uses Command on macOS and Control elsewhere', async () => {
    const mac = await importPlatform('darwin')

    expect(mac.isActionMod({ ctrl: false, meta: true })).toBe(true)
    expect(mac.isActionMod({ ctrl: true, meta: false })).toBe(false)

    const linux = await importPlatform('linux')

    expect(linux.isActionMod({ ctrl: true, meta: false })).toBe(true)
    expect(linux.isActionMod({ ctrl: false, meta: true })).toBe(false)
  })

  it('recognizes remote Command+C without treating local Alt+C as copy', async () => {
    const platform = await importPlatform('linux')

    expect(platform.isCopyShortcut({ ctrl: false, meta: false, super: true }, 'c', { SSH_CONNECTION: '1 2 3 4' })).toBe(
      true
    )
    expect(platform.isCopyShortcut({ ctrl: false, meta: true }, 'c', {})).toBe(false)
  })

  it('keeps readline fallbacks macOS-only', async () => {
    const mac = await importPlatform('darwin')

    expect(mac.isMacActionFallback({ ctrl: true, meta: false }, 'k', 'k')).toBe(true)
    expect(mac.isMacActionFallback({ ctrl: true, meta: true }, 'k', 'k')).toBe(false)

    const linux = await importPlatform('linux')

    expect(linux.isMacActionFallback({ ctrl: true, meta: false }, 'k', 'k')).toBe(false)
  })
})
