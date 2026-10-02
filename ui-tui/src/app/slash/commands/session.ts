// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type {
  ConfigSetResponse,
  ImageAttachResponse,
  SessionBranchResponse,
  SessionExportResponse
} from '../../../gatewayTypes.js'
import type { SlashCommand } from '../types.js'

import { attachedImageNotice } from '../../../domain/messages.js'
import { toTuiSessionKey } from '../../../domain/session.js'
import { patchOverlayState } from '../../overlayStore.js'
import { getUiState, patchUiState } from '../../uiStore.js'

const parseModelArg = (arg: string): { provider?: string; value: string } => {
  const match = arg.trim().match(/^(.*?)\s+--provider\s+(\S+)\s*$/)
  return match ? { provider: match[2], value: match[1]!.trim() } : { value: arg.trim() }
}

export const sessionCommands: SlashCommand[] = [
  {
    help: 'change or show model',
    name: 'model',
    run: (arg, ctx) => {
      if (ctx.session.guardBusySessionSwitch('change models')) {
        return
      }

      if (!arg.trim()) {
        return patchOverlayState({ modelPicker: true })
      }

      const { provider, value } = parseModelArg(arg)
      ctx.gateway
        .rpc<ConfigSetResponse>('config.set', {
          key: 'model',
          session_id: ctx.sid,
          value,
          ...(provider ? { provider } : {})
        })
        .then(
          ctx.guarded<ConfigSetResponse>(response => {
            if (!response.value) {
              return ctx.transcript.sys('error: invalid model response')
            }

            patchUiState(state => ({
              ...state,
              info: state.info
                ? { ...state.info, model: response.value! }
                : { model: response.value!, skills: {}, tools: {} }
            }))
            ctx.transcript.sys(`model: ${response.value}`)
            ctx.local.maybeWarn(response)
          })
        )
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'browse, create, resume, or delete sessions',
    name: 'sessions',
    run: (arg, ctx) => {
      const [sub = 'list', ...restParts] = arg.trim() ? arg.trim().split(/\s+/) : ['list']
      const rest = restParts.join(' ').trim()

      if (sub === 'list') {
        if (!ctx.session.guardBusySessionSwitch('switch sessions')) {
          patchOverlayState({ picker: true })
        }
        return
      }

      if (sub === 'new') {
        if (!ctx.session.guardBusySessionSwitch('switch sessions')) {
          ctx.session.newSession('New session started', rest || undefined)
        }
        return
      }

      if (sub === 'resume') {
        if (ctx.session.guardBusySessionSwitch('switch sessions')) {
          return
        }
        rest ? ctx.session.resumeById(toTuiSessionKey(rest)) : patchOverlayState({ picker: true })
        return
      }

      if (sub === 'delete') {
        const target = rest === 'current' || !rest ? ctx.sid : toTuiSessionKey(rest)

        if (!target) {
          return ctx.transcript.sys('no active session to delete')
        }

        const deletingActive = target === ctx.sid

        const guardDelete = deletingActive || ctx.ui.sessionSwitching
        const deleteDescription = deletingActive ? 'delete the active session' : 'delete a session'

        if (guardDelete && ctx.session.guardBusySessionSwitch(deleteDescription)) {
          return
        }

        void ctx.session
          .deleteSessionWithFallback(target)
          .then(deleted => {
            if (deleted === false) {
              ctx.transcript.sys(`no such session: ${target}`)
            } else if (deleted && !deletingActive) {
              ctx.transcript.sys(`deleted session: ${target}`)
            }
          })
          .catch(error => ctx.transcript.sys(`error: ${error instanceof Error ? error.message : String(error)}`))
        return
      }

      ctx.transcript.sys('usage: /sessions [list|new [title]|resume [id]|delete [id|current]]')
    }
  },
  {
    help: 'attach an image',
    name: 'image',
    run: (arg, ctx) => {
      if (ctx.ui.sessionMutating || ctx.ui.sessionSwitching) {
        return ctx.transcript.sys(
          `wait for the current session ${ctx.ui.sessionSwitching ? 'switch' : 'mutation'} before attaching an image`
        )
      }

      if (!ctx.sid || !arg.trim()) {
        return ctx.transcript.sys('usage: /image <path>')
      }

      ctx.composer
        .attachImage(arg.trim(), ctx.sid)
        .then(
          ctx.guarded<ImageAttachResponse>(response => {
            ctx.transcript.sys(attachedImageNotice(response))
            if (response.remainder) {
              ctx.composer.setInput(response.remainder)
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },
  {
    aliases: ['fork'],
    help: 'branch the current session',
    name: 'branch',
    run: (arg, ctx) => {
      if (!ctx.sid || ctx.session.guardBusySessionSwitch('branch sessions')) {
        return
      }

      const parent = ctx.sid
      void ctx.session
        .runSessionMutation('branch sessions', async () => {
          patchUiState({ sessionSwitching: true, status: 'branching…' })
          try {
            const response = await ctx.gateway.rpc<SessionBranchResponse>('session.branch', {
              name: arg.trim(),
              session_id: parent
            })

            if (!response || getUiState().sid !== parent) {
              return
            }

            if (!response.session_id) {
              return ctx.transcript.sys('session could not be branched')
            }

            ctx.session.releaseSessionImages(parent)
            patchUiState({ sid: response.session_id, status: 'ready' })
            ctx.session.setSessionStartedAt(Date.now())
            ctx.transcript.sys(`forked session: ${response.session_id}`)
          } finally {
            const state = getUiState()
            patchUiState({
              sessionSwitching: false,
              ...(state.sid === parent && state.status === 'branching…' ? { status: 'ready' } : {})
            })
          }
        })
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'export a session transcript',
    name: 'export',
    usage: '/export [id]',
    run: (arg, ctx) => {
      const target = arg.trim() || ctx.sid

      if (!target) {
        return ctx.transcript.sys('no active session to export')
      }

      ctx.gateway
        .rpc<SessionExportResponse>('session.export', { session_id: target })
        .then(
          ctx.guarded<SessionExportResponse>(response => {
            if (response.exported && response.path) {
              return ctx.transcript.sys(`exported to ${response.path}`)
            }

            if (response.reason === 'ambiguous') {
              return ctx.transcript.sys(`ambiguous session id: ${(response.candidates ?? []).join(', ')}`)
            }

            ctx.transcript.sys(`session export failed: ${response.reason ?? 'not found'}`)
          })
        )
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'show or hide reasoning',
    name: 'reasoning',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()

      if (mode && mode !== 'on' && mode !== 'off' && mode !== 'toggle') {
        return ctx.transcript.sys('usage: /reasoning [on|off|toggle]')
      }

      const next = mode === 'on' ? true : mode === 'off' ? false : !ctx.ui.showReasoning
      patchUiState({ showReasoning: next })
      ctx.transcript.sys(`reasoning ${next ? 'on' : 'off'}`)
    }
  }
]
