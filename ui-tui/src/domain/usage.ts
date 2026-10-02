import type { Usage } from '../types.js'

export const ZERO: Usage = { calls: 0, input: 0, output: 0, total: 0 }

export const mergeUsage = (current: Usage, update: Partial<Usage>): Usage => {
  const next = { ...current, ...update }
  if (!Object.prototype.hasOwnProperty.call(update, 'cost_usd')) {
    delete next.cost_usd
  }
  return next
}
