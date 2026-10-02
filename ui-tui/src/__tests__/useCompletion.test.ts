import { render } from 'ink-testing-library'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { SlashCommand } from '../app/slash/types.js'

import { SLASH_COMMANDS } from '../app/slash/registry.js'
import { slashCompletions, useCompletion } from '../hooks/useCompletion.js'

const REGISTRY: SlashCommand[] = [
  { aliases: ['exit', 'q'], help: 'exit codeflow', name: 'quit', run: vi.fn() },
  { help: 'show runtime status', name: 'status', run: vi.fn() },
  { name: 'sessions', run: vi.fn() }
]

describe('slashCompletions', () => {
  it('completes retained commands locally in registry order', () => {
    expect(slashCompletions('/', REGISTRY).map(item => item.display)).toEqual(['/quit', '/status', '/sessions'])
    expect(slashCompletions('/s', REGISTRY).map(item => item.display)).toEqual(['/status', '/sessions'])
    expect(slashCompletions('/ex', REGISTRY).map(item => item.display)).toEqual(['/quit'])
  })

  it('closes for arguments, non-slash input, and unknown commands', () => {
    expect(slashCompletions('/sessions list', REGISTRY)).toEqual([])
    expect(slashCompletions('hello', REGISTRY)).toEqual([])
    expect(slashCompletions('/unknown', REGISTRY)).toEqual([])
  })

  it('exposes only the contracted real registry', () => {
    const names = slashCompletions('/', SLASH_COMMANDS).map(item => item.display)

    expect(names).toContain('/help')
    expect(names).toContain('/model')
    expect(names).toContain('/sessions')
    expect(names).toContain('/image')
    expect(names).not.toContain('/voice')
    expect(names).not.toContain('/reload-mcp')
    expect(names).not.toContain('/skills')
  })
})

function HookSpy({ input, blocked, out }: { input: string; blocked: boolean; out: { completions: string[] } }) {
  out.completions = useCompletion(input, blocked).completions.map(item => item.display)

  return React.createElement(React.Fragment, null)
}

describe('useCompletion', () => {
  it('populates local slash completions without a gateway dependency', async () => {
    const out = { completions: [] as string[] }

    render(React.createElement(HookSpy, { blocked: false, input: '/s', out }))

    await vi.waitFor(() => {
      expect(out.completions).toContain('/status')
    })
  })

  it('suppresses completion while an overlay blocks input', async () => {
    const out = { completions: [] as string[] }

    render(React.createElement(HookSpy, { blocked: true, input: '/s', out }))

    await vi.waitFor(() => {
      expect(out.completions).toEqual([])
    })
  })
})
