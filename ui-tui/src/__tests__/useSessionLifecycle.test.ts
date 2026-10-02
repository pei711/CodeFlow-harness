import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { GatewayRpc } from '../app/interfaces.js'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import {
  closeCurrentSessionForSwitch,
  closeSessionWithCleanup,
  performDeleteWithFallback,
  runSessionSwitchFlight,
  sessionSwitchBlockMessage,
  tryAcquireSessionMutation,
  writeActiveSessionFile
} from '../app/useSessionLifecycle.js'

describe('sessionSwitchBlockMessage', () => {
  it('blocks Session switches while a clipboard image attachment is pending', () => {
    expect(sessionSwitchBlockMessage(false, true, 'switch sessions')).toBe(
      'wait for the clipboard image attachment before trying to switch sessions'
    )
  })

  it('retains the active Turn guard and allows an idle Session switch', () => {
    expect(sessionSwitchBlockMessage(true, false, 'switch sessions')).toBe(
      'interrupt the current turn before trying to switch sessions'
    )
    expect(sessionSwitchBlockMessage(false, false, 'switch sessions')).toBeNull()
  })

  it('blocks another switch while a Session Switch Flight is active', () => {
    expect(sessionSwitchBlockMessage(false, false, 'switch sessions', true)).toBe(
      'wait for the current session switch before trying to switch sessions'
    )
  })

  it('blocks a switch while a Session mutation is active', () => {
    expect(sessionSwitchBlockMessage(false, false, 'switch sessions', false, true)).toBe(
      'wait for the current session mutation before trying to switch sessions'
    )
  })
})

describe('tryAcquireSessionMutation', () => {
  it('serializes mutations until the current owner releases the guard', () => {
    const activeRef = { current: false }
    const release = tryAcquireSessionMutation(activeRef, false)

    expect(release).not.toBeNull()
    expect(activeRef.current).toBe(true)
    expect(tryAcquireSessionMutation(activeRef, false)).toBeNull()

    release?.()
    release?.()

    expect(activeRef.current).toBe(false)
    expect(tryAcquireSessionMutation(activeRef, false)).not.toBeNull()
  })

  it('does not start a mutation during a Session Switch Flight', () => {
    const activeRef = { current: false }

    expect(tryAcquireSessionMutation(activeRef, true)).toBeNull()
    expect(activeRef.current).toBe(false)
  })
})

describe('writeActiveSessionFile', () => {
  let dir = ''

  afterEach(() => {
    if (dir) {
      rmSync(dir, { force: true, recursive: true })
      dir = ''
    }
  })

  it('writes the actual resumed session id for the shell exit summary', () => {
    dir = mkdtempSync(join(tmpdir(), 'codeflow-tui-active-'))
    const path = join(dir, 'active.json')

    writeActiveSessionFile('actual_session', path)

    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ session_id: 'actual_session' })
  })
})

describe('performDeleteWithFallback', () => {
  beforeEach(() => {
    resetOverlayState()
  })

  const makeDeps = (
    mostRecent: { session_id?: null | string } | null = null,
    activeSid: null | string = 'tui:active'
  ) => {
    const calls: { method: string; params: unknown }[] = []

    const rpc = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      if (method === 'session.most_recent') {
        return mostRecent
      }

      if (method === 'session.delete') {
        return { deleted: params?.session_id }
      }

      return {}
    })

    return {
      calls,
      deps: {
        activeSid,
        newSession: vi.fn(async () => {}),
        releaseSessionImages: vi.fn(),
        resumeById: vi.fn(),
        rpc: rpc as unknown as GatewayRpc
      }
    }
  }

  it('non-active delete: deletes and stops (no fresh session)', async () => {
    const { calls, deps } = makeDeps()

    await performDeleteWithFallback('tui:other', deps)

    expect(calls.map(c => c.method)).toEqual(['session.delete'])
    expect(deps.resumeById).not.toHaveBeenCalled()
    expect(deps.newSession).not.toHaveBeenCalled()
    expect(deps.releaseSessionImages).toHaveBeenCalledWith('tui:other')
  })

  it('active delete always mints a fresh session, even when a survivor exists', async () => {
    const { calls, deps } = makeDeps({ session_id: 'tui:survivor' })

    await performDeleteWithFallback('tui:active', deps)

    expect(calls.map(c => c.method)).toEqual(['session.delete'])
    expect(deps.resumeById).not.toHaveBeenCalled()
    expect(deps.newSession).toHaveBeenCalledTimes(1)
  })

  it('closes the picker overlay before minting the fresh session', async () => {
    patchOverlayState({ picker: true })
    const { deps } = makeDeps()

    await performDeleteWithFallback('tui:active', deps)

    expect(getOverlayState().picker).toBe(false)
    expect(deps.newSession).toHaveBeenCalledTimes(1)
  })

  it('releases the mutation guard immediately before the active-delete switch', async () => {
    const { deps } = makeDeps()
    const order: string[] = []

    await performDeleteWithFallback('tui:active', {
      ...deps,
      beforeNewSession: () => order.push('release'),
      newSession: () => order.push('switch')
    })

    expect(order).toEqual(['release', 'switch'])
  })

  it('resolves true when the server confirms the removal', async () => {
    const { deps } = makeDeps()

    await expect(performDeleteWithFallback('tui:other', deps)).resolves.toBe(true)
  })

  it('resolves false when the server returns deleted: null (no such session)', async () => {
    const { deps } = makeDeps()
    const rpc = vi.fn(async (method: string) => (method === 'session.delete' ? { deleted: null } : {}))
    deps.rpc = rpc as unknown as GatewayRpc

    await expect(performDeleteWithFallback('tui:other', deps)).resolves.toBe(false)
    expect(deps.releaseSessionImages).not.toHaveBeenCalled()
  })

  it('resolves null and keeps the active session when confirmation is declined', async () => {
    const { deps } = makeDeps()
    const rpc = vi.fn(async (method: string) => (method === 'session.delete' ? { cancelled: true, deleted: null } : {}))
    deps.rpc = rpc as unknown as GatewayRpc

    await expect(performDeleteWithFallback('tui:active', deps)).resolves.toBeNull()
    expect(deps.newSession).not.toHaveBeenCalled()
    expect(deps.releaseSessionImages).not.toHaveBeenCalled()
  })
})

describe('closeSessionWithCleanup', () => {
  it('releases unsent images after the backend closes a Session', async () => {
    const rpc = vi.fn(async () => ({ ok: true })) as unknown as GatewayRpc
    const releaseSessionImages = vi.fn()

    await expect(closeSessionWithCleanup('tui:active', rpc, releaseSessionImages)).resolves.toEqual({ ok: true })

    expect(rpc).toHaveBeenCalledWith('session.close', { session_id: 'tui:active' })
    expect(releaseSessionImages).toHaveBeenCalledWith('tui:active')
  })

  it('retains images when the close request fails before quiescence is proven', async () => {
    const rpc = vi.fn(async () => {
      throw new Error('gateway unavailable')
    }) as unknown as GatewayRpc
    const releaseSessionImages = vi.fn()

    await expect(closeSessionWithCleanup('tui:active', rpc, releaseSessionImages)).rejects.toThrow(
      'gateway unavailable'
    )
    expect(releaseSessionImages).not.toHaveBeenCalled()
  })
})

describe('closeCurrentSessionForSwitch', () => {
  it('aborts a switch when close returns no quiescence proof', async () => {
    const closeSession = vi.fn(async () => null)

    await expect(closeCurrentSessionForSwitch('tui:active', 'tui:next', closeSession)).resolves.toBe(false)
    expect(closeSession).toHaveBeenCalledWith('tui:active')
  })

  it('aborts a switch when close rejects', async () => {
    const closeSession = vi.fn(async () => {
      throw new Error('turn in progress')
    })

    await expect(closeCurrentSessionForSwitch('tui:active', null, closeSession)).resolves.toBe(false)
  })

  it('allows a same-session resume without closing it', async () => {
    const closeSession = vi.fn()

    await expect(closeCurrentSessionForSwitch('tui:active', 'tui:active', closeSession)).resolves.toBe(true)
    expect(closeSession).not.toHaveBeenCalled()
  })
})

describe('runSessionSwitchFlight', () => {
  const deferred = <T>() => {
    let resolve!: (value: T) => void
    const promise = new Promise<T>(done => {
      resolve = done
    })

    return { promise, resolve }
  }

  it('keeps submission blocking active until the switch response settles', async () => {
    const response = deferred<void>()
    const epochRef = { current: 0 }
    const sid: null | string = 'tui:old'
    let switching = false
    let committed = false
    const flight = runSessionSwitchFlight(
      epochRef,
      sid,
      () => sid,
      active => {
        switching = active
      },
      async isCurrent => {
        await response.promise
        committed = isCurrent()
      }
    )

    expect(switching).toBe(true)
    response.resolve()
    await flight

    expect(committed).toBe(true)
    expect(switching).toBe(false)
  })

  it('prevents an older out-of-order response from overwriting the newer switch', async () => {
    const firstResponse = deferred<string>()
    const secondResponse = deferred<string>()
    const epochRef = { current: 0 }
    const sid: null | string = 'tui:old'
    let switching = false
    const committed: string[] = []
    const run = (response: Promise<string>) =>
      runSessionSwitchFlight(
        epochRef,
        sid,
        () => sid,
        active => {
          switching = active
        },
        async isCurrent => {
          const result = await response

          if (isCurrent()) {
            committed.push(result)
          }
        }
      )

    const first = run(firstResponse.promise)
    const second = run(secondResponse.promise)
    secondResponse.resolve('newer')
    await second
    firstResponse.resolve('older')
    await first

    expect(committed).toEqual(['newer'])
    expect(switching).toBe(false)
  })
})
