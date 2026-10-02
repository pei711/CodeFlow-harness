// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { ConfirmRespondResult } from '../rpc/index.js'
import type { GatewayRpc } from './interfaces.js'

import { buildConfirmRespond } from '../lib/confirmCountdown.js'
import { patchOverlayState } from './overlayStore.js'
import { getUiState, patchUiState } from './uiStore.js'

export function answerConfirmFromInput<T>(event: { stopImmediatePropagation: () => void }, answer: () => T): T {
  event.stopImmediatePropagation()

  return answer()
}

export async function answerConfirmRequest(rpc: GatewayRpc, requestId: string, answer: boolean): Promise<boolean> {
  const result = await rpc<ConfirmRespondResult>('confirm.respond', buildConfirmRespond(requestId, answer))

  if (result?.ok !== true) {
    patchOverlayState({ confirm: null })
    patchUiState({ status: getUiState().busy ? 'running…' : 'ready' })

    return false
  }

  patchOverlayState({ confirm: null })
  patchUiState({ status: getUiState().busy ? 'running…' : 'ready' })

  return true
}

export const cancelConfirmRequest = (rpc: GatewayRpc, requestId: string) => answerConfirmRequest(rpc, requestId, false)
