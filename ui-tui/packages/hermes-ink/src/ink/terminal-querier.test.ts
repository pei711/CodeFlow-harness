import { afterEach, describe, expect, it, vi } from 'vitest'

import { oscColor, TerminalQuerier } from './terminal-querier.js'

afterEach(() => {
  vi.useRealTimers()
})

describe('TerminalQuerier', () => {
  it('disables future queries after a timed-out barrier', async () => {
    vi.useFakeTimers()

    const stdout = { write: vi.fn() } as unknown as NodeJS.WriteStream
    const querier = new TerminalQuerier(stdout)
    const firstResult = vi.fn()
    const firstFlushResult = vi.fn()
    const secondResult = vi.fn()
    const secondFlushResult = vi.fn()

    void querier.send(oscColor(52)).then(firstResult)
    void querier.flush(100).then(firstFlushResult)
    void querier.send(oscColor(52)).then(secondResult)
    void querier.flush(100).then(secondFlushResult)

    await vi.advanceTimersByTimeAsync(100)

    expect(firstResult).toHaveBeenCalledWith(undefined)
    expect(firstFlushResult).toHaveBeenCalledOnce()
    expect(secondResult).toHaveBeenCalledWith(undefined)
    expect(secondFlushResult).toHaveBeenCalledOnce()
    expect(stdout.write).toHaveBeenCalledTimes(4)

    const thirdResult = vi.fn()
    const thirdFlushResult = vi.fn()
    void querier.send(oscColor(52)).then(thirdResult)
    void querier.flush(25).then(thirdFlushResult)

    querier.onResponse({ type: 'osc', code: 52, data: 'c;c2Vjb25k' })
    querier.onResponse({ type: 'da1', params: [1, 2] })
    await Promise.resolve()

    expect(thirdResult).toHaveBeenCalledWith(undefined)
    expect(thirdFlushResult).toHaveBeenCalledOnce()
    expect(stdout.write).toHaveBeenCalledTimes(4)
  })
})
