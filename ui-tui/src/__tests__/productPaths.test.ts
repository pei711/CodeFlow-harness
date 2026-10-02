import { homedir } from 'node:os'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

import { getCodeFlowHome, getCodeFlowHomeLabel } from '../config/paths.js'

describe('CodeFlow product paths', () => {
  it('uses the CodeFlow home override for persistent TUI state', () => {
    expect(getCodeFlowHome({ CODEFLOW_HOME: '/tmp/codeflow-home' })).toBe('/tmp/codeflow-home')
    expect(getCodeFlowHomeLabel({ CODEFLOW_HOME: '/tmp/codeflow-home' })).toBe('/tmp/codeflow-home')
  })

  it('uses ~/.codeflow when the override is empty', () => {
    expect(getCodeFlowHome({ CODEFLOW_HOME: '  ' })).toBe(join(homedir(), '.codeflow'))
    expect(getCodeFlowHomeLabel({ CODEFLOW_HOME: '  ' })).toBe('~/.codeflow')
  })
})
