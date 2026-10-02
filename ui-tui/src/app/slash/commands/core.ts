// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { SessionTitleResponse, SessionUndoResponse } from '../../../gatewayTypes.js'
import type { Msg } from '../../../types.js'
import type { SlashCommand } from '../types.js'

import { isSectionName, nextDetailsMode, parseDetailsMode, SECTION_NAMES } from '../../../domain/details.js'
import { toTuiSessionKey } from '../../../domain/session.js'
import { writeClipboardText } from '../../../lib/clipboard.js'
import { writeOsc52Clipboard } from '../../../lib/osc52.js'
import { patchOverlayState } from '../../overlayStore.js'
import { getUiState, patchUiState } from '../../uiStore.js'

const flagFromArg = (arg: string, current: boolean): boolean | null => {
  if (!arg || arg === 'toggle') {
    return !current
  }

  if (arg === 'on') {
    return true
  }

  if (arg === 'off') {
    return false
  }

  return null
}

const HELP_ROWS: [string, string][] = [
  ['/help', 'list TUI commands'],
  ['/new [title]', 'start a new Session'],
  ['/resume [id]', 'resume a Session'],
  ['/sessions', 'browse, create, resume, or delete Sessions'],
  ['/model [name]', 'select the active model'],
  ['/image <path>', 'attach an image'],
  ['/status', 'show Session, Context, Memory, and usage'],
  ['/usage', 'show token and cost usage'],
  ['/details', 'control reasoning and Tool detail visibility'],
  ['/agents', 'inspect the current subagent tree'],
  ['/undo', 'remove the last exchange'],
  ['/retry', 'retry the last user message'],
  ['/branch [title]', 'fork the current Session'],
  ['/export [id]', 'export a Session transcript'],
  ['/quit', 'exit CodeFlow']
]

export const coreCommands: SlashCommand[] = [
  {
    help: 'list TUI commands',
    name: 'help',
    run: (_arg, ctx) => ctx.transcript.panel(ctx.ui.theme.brand.helpHeader, [{ rows: HELP_ROWS }])
  },
  {
    aliases: ['exit', 'q'],
    help: 'exit codeflow',
    name: 'quit',
    run: (_arg, ctx) => ctx.session.die()
  },
  {
    aliases: ['scroll'],
    help: 'toggle mouse and wheel tracking',
    name: 'mouse',
    run: (arg, ctx) => {
      const next = flagFromArg(arg.trim().toLowerCase(), ctx.ui.mouseTracking)

      if (next === null) {
        return ctx.transcript.sys('usage: /mouse [on|off|toggle]')
      }

      patchUiState({ mouseTracking: next })
      ctx.transcript.sys(`mouse tracking ${next ? 'on' : 'off'}`)
    }
  },
  {
    aliases: ['clear'],
    help: 'start a new session',
    name: 'new',
    run: (arg, ctx) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      const title = arg.trim()
      const commit = () => ctx.session.newSession('New session started', title || undefined)

      patchOverlayState({
        confirm: {
          cancelLabel: 'No, keep going',
          confirmLabel: 'Yes, start a new session',
          danger: true,
          detail: 'This ends the current conversation and starts a fresh session.',
          onConfirm: commit,
          title: 'Start a new session?'
        }
      })
    }
  },
  {
    help: 'resume a prior session',
    name: 'resume',
    run: (arg, ctx) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      arg.trim() ? ctx.session.resumeById(toTuiSessionKey(arg.trim())) : patchOverlayState({ picker: true })
    }
  },
  {
    help: 'set or show the current session title',
    name: 'title',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('no active session')
      }

      const title = arg.trim()
      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', {
          session_id: ctx.sid,
          ...(title ? { title } : {})
        })
        .then(
          ctx.guarded<SessionTitleResponse>(r => {
            const current = (r.title ?? title).trim()
            ctx.transcript.sys(current ? `title: ${current}` : 'no title set')
          })
        )
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'toggle compact transcript',
    name: 'compact',
    run: (arg, ctx) => {
      const next = flagFromArg(arg.trim().toLowerCase(), ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys('usage: /compact [on|off|toggle]')
      }

      patchUiState({ compact: next })
      ctx.transcript.sys(`compact mode ${next ? 'on' : 'off'}`)
    }
  },
  {
    help: 'control reasoning and tool detail visibility',
    name: 'details',
    run: (arg, ctx) => {
      const [first = '', second = ''] = arg.trim().toLowerCase().split(/\s+/)

      if (isSectionName(first)) {
        const mode = second ? parseDetailsMode(second) : undefined

        if (second && !mode) {
          return ctx.transcript.sys(`usage: /details <${SECTION_NAMES.join('|')}> [hidden|collapsed|expanded]`)
        }

        const next = mode ?? nextDetailsMode(ctx.ui.sections[first] ?? ctx.ui.detailsMode)
        patchUiState({ sections: { ...ctx.ui.sections, [first]: next } })
        return ctx.transcript.sys(`${first} details: ${next}`)
      }

      const parsed = first ? parseDetailsMode(first) : undefined
      const next = parsed ?? nextDetailsMode(ctx.ui.detailsMode)
      patchUiState({ detailsMode: next })
      ctx.transcript.sys(`details: ${next}`)
    }
  },
  {
    help: 'copy selection or assistant message',
    name: 'copy',
    run: async (arg, ctx) => {
      if (!arg && ctx.composer.hasSelection) {
        const text = await ctx.composer.selection.copySelection()
        return ctx.transcript.sys(text ? `copied ${text.length} characters` : 'clipboard copy failed')
      }

      const index = arg ? Number.parseInt(arg, 10) : 0
      if (arg && Number.isNaN(index)) {
        return ctx.transcript.sys('usage: /copy [number]')
      }

      const messages = ctx.local.getHistoryItems().filter(message => message.role === 'assistant')
      const target = index ? messages[Math.min(index, messages.length) - 1] : messages.at(-1)

      if (!target) {
        return ctx.transcript.sys('nothing to copy')
      }

      const copied = await writeClipboardText(target.text)
      if (copied) {
        return ctx.transcript.sys('copied to clipboard')
      }

      writeOsc52Clipboard(target.text)
      ctx.transcript.sys('sent OSC52 copy sequence')
    }
  },
  {
    help: 'attach a clipboard image',
    name: 'paste',
    run: (arg, ctx) =>
      arg
        ? ctx.transcript.sys('usage: /paste')
        : ctx.ui.sessionMutating || ctx.ui.sessionSwitching
          ? ctx.transcript.sys(
              `wait for the current session ${ctx.ui.sessionSwitching ? 'switch' : 'mutation'} before attaching an image`
            )
          : ctx.composer.paste()
  },
  {
    help: 'view the current transcript',
    name: 'history',
    run: (_arg, ctx) => {
      const items = ctx.local
        .getHistoryItems()
        .filter(message => message.role === 'user' || message.role === 'assistant')

      if (!items.length) {
        return ctx.transcript.sys('no conversation yet')
      }

      const body = items
        .map((message, index) => `[${message.role === 'user' ? 'You' : 'CodeFlow'} #${index + 1}]\n${message.text}`)
        .join('\n\n')
      ctx.transcript.page(body, 'History')
    }
  },
  {
    help: 'show session, context, memory, and usage',
    name: 'status',
    run: (_arg, ctx) => {
      const info = ctx.ui.info
      const usage = ctx.ui.usage
      const context = usage.context_max
        ? `${usage.context_used ?? 0}/${usage.context_max} (${usage.context_percent ?? 0}%)`
        : 'not reported'

      ctx.transcript.panel('Runtime status', [
        {
          rows: [
            ['Session', ctx.sid ?? 'not ready'],
            ['Model', info?.model ?? 'not configured'],
            ['Provider', info?.provider ?? 'not configured'],
            ['Context', context],
            ['Memory', info?.memory ?? 'not configured'],
            ['Tokens', usage.total.toLocaleString()],
            ['Cost', typeof usage.cost_usd === 'number' ? `$${usage.cost_usd.toFixed(4)}` : 'not reported']
          ]
        }
      ])
    }
  },
  {
    help: 'show token and cost usage',
    name: 'usage',
    run: (_arg, ctx) => {
      const usage = ctx.ui.usage
      ctx.transcript.panel('Usage', [
        {
          rows: [
            ['Input tokens', usage.input.toLocaleString()],
            ['Output tokens', usage.output.toLocaleString()],
            ['Total tokens', usage.total.toLocaleString()],
            ['API calls', usage.calls.toLocaleString()],
            ['Cost', typeof usage.cost_usd === 'number' ? `$${usage.cost_usd.toFixed(4)}` : 'not reported']
          ]
        }
      ])
    }
  },
  {
    help: 'inspect the current subagent tree',
    name: 'agents',
    run: (_arg, _ctx) => patchOverlayState({ agents: true, agentsInitialHistoryIndex: 0 })
  },
  {
    help: 'undo the last exchange',
    name: 'undo',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('nothing to undo')
      }

      void ctx.session
        .runSessionMutation('undo the last exchange', async () => {
          const r = await ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid })

          if (!r || getUiState().sid !== ctx.sid) {
            return
          }

          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys('nothing to undo')
          }

          ctx.transcript.setHistoryItems((previous: Msg[]) => ctx.transcript.trimLastExchange(previous))
          ctx.transcript.sys(`undid ${r.removed} messages`)
        })
        .catch(ctx.guardedErr)
    }
  },
  {
    help: 'retry the last user message',
    name: 'retry',
    run: (_arg, ctx) => {
      const last = ctx.local.getLastUserMsg()

      if (!last || !ctx.sid) {
        return ctx.transcript.sys('nothing to retry')
      }

      void ctx.session
        .runSessionMutation('retry the last user message', async () => {
          const r = await ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid })

          if (!r || getUiState().sid !== ctx.sid) {
            return
          }

          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys('nothing to retry')
          }

          ctx.transcript.setHistoryItems((previous: Msg[]) => ctx.transcript.trimLastExchange(previous))
          ctx.transcript.dispatchSubmission(last.text, true, last.submitText, last.pasteSnips)
        })
        .catch(ctx.guardedErr)
    }
  }
]
