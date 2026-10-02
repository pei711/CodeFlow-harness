// SPDX-License-Identifier: MIT
// Copyright (c) 2026 EverMind.
// See NOTICES.md.

import { describe, expect, it } from 'vitest'

import { SLASH_COMMANDS } from '../app/slash/registry.js'

const RETAINED = [
  'help',
  'quit',
  'mouse',
  'new',
  'resume',
  'title',
  'compact',
  'details',
  'copy',
  'paste',
  'history',
  'status',
  'usage',
  'agents',
  'undo',
  'retry',
  'model',
  'sessions',
  'image',
  'branch',
  'export',
  'reasoning'
]

describe('slash command contract', () => {
  it('contains exactly the retained local commands', () => {
    expect(SLASH_COMMANDS.map(command => command.name)).toEqual(RETAINED)
  })

  it('does not expose deleted feature administration commands', () => {
    const names = new Set(SLASH_COMMANDS.flatMap(command => [command.name, ...(command.aliases ?? [])]))

    expect(names.has('voice')).toBe(false)
    expect(names.has('reload-mcp')).toBe(false)
    expect(names.has('browser')).toBe(false)
    expect(names.has('skills')).toBe(false)
    expect(names.has('setup')).toBe(false)
  })
})
