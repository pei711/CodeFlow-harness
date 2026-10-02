import { describe, expect, it } from 'vitest'

import { removeAtInPlace, replaceQueuedSubmissionAt } from '../hooks/useQueue.js'

describe('removeAtInPlace', () => {
  it('removes the item at the given index in place', () => {
    const arr = ['a', 'b', 'c']

    removeAtInPlace(arr, 1)
    expect(arr).toEqual(['a', 'c'])
  })

  it('is a no-op when the index is out of bounds', () => {
    const arr = ['a', 'b']

    removeAtInPlace(arr, -1)
    removeAtInPlace(arr, 5)
    expect(arr).toEqual(['a', 'b'])
  })

  it('returns the same reference (mutates in place)', () => {
    const arr = ['x']
    const same = removeAtInPlace(arr, 0)

    expect(same).toBe(arr)
    expect(arr).toEqual([])
  })
})

describe('replaceQueuedSubmissionAt', () => {
  it('keeps the expanded payload when editing a queued paste placeholder', () => {
    const queue = ['review [[Pasted Content 1]]']
    const metadata = [{ alreadyDisplayed: true, paused: true, submitText: 'old payload' }]

    replaceQueuedSubmissionAt(
      queue,
      metadata,
      0,
      'review this [[Pasted Content 2]]',
      'review this complete replacement payload'
    )

    expect(queue).toEqual(['review this [[Pasted Content 2]]'])
    expect(metadata).toEqual([
      {
        alreadyDisplayed: false,
        pasteSnips: undefined,
        paused: false,
        submitText: 'review this complete replacement payload'
      }
    ])
  })
})
