// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { ScrollBoxHandle } from '@hermes/ink'

import { evictInkCaches } from '@hermes/ink'
import { writeFileSync } from 'node:fs'
import { type RefObject, useCallback, useRef } from 'react'

import type {
  SessionCloseResponse,
  SessionCreateResponse,
  SessionDeleteResponse,
  SessionResumeResponse,
  SessionTitleResponse,
  SetupStatusResponse
} from '../gatewayTypes.js'
import type { Msg, PanelSection, SessionInfo, Usage } from '../types.js'
import type {
  ComposerActions,
  GatewayRpc,
  LastUserSubmission,
  SessionMutationRunner,
  StateSetter
} from './interfaces.js'

import { buildSetupRequiredSections, SETUP_REQUIRED_TITLE } from '../content/setup.js'
import { introMsg, toTranscriptMessages } from '../domain/messages.js'
import { ZERO } from '../domain/usage.js'
import { asRpcResult } from '../lib/rpc.js'
import { type TuiRpcClient } from '../tuiRpcClient.js'
import { patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { patchTurnState } from './turnStore.js'
import { getUiState, patchUiState } from './uiStore.js'

const usageFrom = (info: null | SessionInfo): Usage => (info?.usage ? { ...ZERO, ...info.usage } : ZERO)

export const writeActiveSessionFile = (sessionId: null | string, file = process.env.CODEFLOW_TUI_ACTIVE_SESSION_FILE) => {
  if (!file || !sessionId) {
    return
  }

  try {
    writeFileSync(file, JSON.stringify({ session_id: sessionId }), { mode: 0o600 })
  } catch {
    // 仅作为尽力而为的 shell 收尾提示，不能破坏实时会话变更。
  }
}

const trimTail = (items: Msg[]) => {
  const q = [...items]

  while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
    q.pop()
  }

  if (q.at(-1)?.role === 'user') {
    q.pop()
  }

  return q
}

export interface DeleteFallbackDeps {
  activeSid: null | string
  beforeNewSession?: () => void
  newSession: () => Promise<unknown> | unknown
  releaseSessionImages?: (sessionId: string) => void
  rpc: GatewayRpc
}

export const closeSessionWithCleanup = async (
  targetSid: null | string | undefined,
  rpc: GatewayRpc,
  releaseSessionImages?: (sessionId: string) => void
): Promise<null | SessionCloseResponse> => {
  if (!targetSid) {
    return null
  }

  const result = await rpc<SessionCloseResponse>('session.close', { session_id: targetSid })
  if (result?.ok) {
    releaseSessionImages?.(targetSid)
  }

  return result
}

export const closeCurrentSessionForSwitch = async (
  currentSid: null | string,
  targetSid: null | string,
  closeSession: (targetSid: string) => Promise<null | SessionCloseResponse>
): Promise<boolean> => {
  if (!currentSid || currentSid === targetSid) {
    return true
  }

  try {
    return (await closeSession(currentSid))?.ok === true
  } catch {
    return false
  }
}

export const runSessionSwitchFlight = async (
  epochRef: { current: number },
  expectedSid: null | string,
  getCurrentSid: () => null | string,
  setSwitching: (active: boolean) => void,
  run: (isCurrent: () => boolean) => Promise<void>
): Promise<void> => {
  epochRef.current += 1
  const epoch = epochRef.current
  const isCurrent = () => epochRef.current === epoch && getCurrentSid() === expectedSid

  setSwitching(true)
  try {
    await run(isCurrent)
  } finally {
    if (epochRef.current === epoch) {
      setSwitching(false)
    }
  }
}

export const tryAcquireSessionMutation = (
  activeRef: { current: boolean },
  sessionSwitching: boolean
): null | (() => void) => {
  if (sessionSwitching || activeRef.current) {
    return null
  }

  activeRef.current = true
  let released = false

  return () => {
    if (released) {
      return
    }

    released = true
    activeRef.current = false
  }
}

// 删除目标会话；若它是活动会话，则始终创建新会话而不恢复其他存留会话，确保
// 界面不会继续绑定已删除的键。返回服务端是否删除了匹配的逻辑会话世代；
// deleted 为 null 表示当前没有匹配项。
export const performDeleteWithFallback = async (
  targetId: string,
  deps: DeleteFallbackDeps
): Promise<boolean | null> => {
  const isActive = deps.activeSid === targetId

  const r = await deps.rpc<SessionDeleteResponse>('session.delete', { session_id: targetId })
  if (r?.cancelled) {
    return null
  }

  const removed = r?.deleted === targetId

  if (removed) {
    deps.releaseSessionImages?.(targetId)
  }

  if (!isActive || !removed) {
    return removed
  }

  // 切换前关闭选择器，避免由选择器发起的删除在新会话上残留浮层；resumeById
  // 采用同样处理。
  patchOverlayState({ picker: false })
  deps.beforeNewSession?.()
  await deps.newSession()

  return removed
}

export interface UseSessionLifecycleOptions {
  colsRef: { current: number }
  composerActions: ComposerActions
  gw: TuiRpcClient
  panel: (title: string, sections: PanelSection[]) => void
  pendingPasteRef?: RefObject<Promise<void> | null>
  releaseSessionImages?: (sessionId: string) => void
  rpc: GatewayRpc
  scrollRef: RefObject<null | ScrollBoxHandle>
  setHistoryItems: StateSetter<Msg[]>
  setLastUserMsg: StateSetter<LastUserSubmission | null>
  setSessionStartedAt: StateSetter<number>
  setStickyPrompt: StateSetter<string>
  sys: (text: string) => void
}

export const sessionSwitchBlockMessage = (
  busy: boolean,
  pendingPaste: boolean,
  what: string,
  switching = false,
  mutating = false
): string | null => {
  if (switching) {
    return `wait for the current session switch before trying to ${what}`
  }

  if (mutating) {
    return `wait for the current session mutation before trying to ${what}`
  }

  if (pendingPaste) {
    return `wait for the clipboard image attachment before trying to ${what}`
  }

  return busy ? `interrupt the current turn before trying to ${what}` : null
}

export function useSessionLifecycle(opts: UseSessionLifecycleOptions) {
  const {
    colsRef,
    composerActions,
    gw,
    panel,
    pendingPasteRef,
    releaseSessionImages,
    rpc,
    scrollRef,
    setHistoryItems,
    setLastUserMsg,
    setSessionStartedAt,
    setStickyPrompt,
    sys
  } = opts
  const sessionSwitchEpochRef = useRef(0)
  const sessionMutationActiveRef = useRef(false)

  const closeSession = useCallback(
    (targetSid?: null | string) => closeSessionWithCleanup(targetSid, rpc, releaseSessionImages),
    [releaseSessionImages, rpc]
  )

  const resetSession = useCallback(() => {
    turnController.fullReset()
    patchUiState({ bgTasks: new Set(), info: null, sid: null, usage: ZERO })
    setHistoryItems([])
    setLastUserMsg(null)
    setStickyPrompt('')
    composerActions.setPasteSnips([])
    // 半量裁剪：新会话使用新键，同时保留热缓存池以便用户恢复到先前会话。
    evictInkCaches('half')
  }, [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt])

  const resetVisibleHistory = useCallback(
    (info: null | SessionInfo = null) => {
      turnController.idle()
      turnController.clearReasoning()
      turnController.turnTools = []
      turnController.persistedToolLabels.clear()

      setHistoryItems(info ? [introMsg(info)] : [])
      setStickyPrompt('')
      setLastUserMsg(null)
      composerActions.setPasteSnips([])
      patchTurnState({ activity: [] })
      patchUiState({ info, usage: usageFrom(info) })
    },
    [composerActions, setHistoryItems, setLastUserMsg, setStickyPrompt]
  )

  const acquireSessionMutation = useCallback(
    (what: string): null | (() => void) => {
      if (pendingPasteRef?.current) {
        sys(`wait for the clipboard image attachment before trying to ${what}`)

        return null
      }

      const switching = getUiState().sessionSwitching
      const release = tryAcquireSessionMutation(sessionMutationActiveRef, switching)

      if (release) {
        patchUiState({ sessionMutating: true })
        let mutationReleased = false

        return () => {
          if (mutationReleased) {
            return
          }

          mutationReleased = true
          release()
          patchUiState({ sessionMutating: false })
        }
      }

      sys(
        switching
          ? `wait for the current session switch before trying to ${what}`
          : `wait for the current session mutation before trying to ${what}`
      )

      return null
    },
    [pendingPasteRef, sys]
  )

  const runSessionMutation = useCallback<SessionMutationRunner>(
    async (what, operation) => {
      const release = acquireSessionMutation(what)

      if (!release) {
        return null
      }

      try {
        return await operation()
      } finally {
        release()
      }
    },
    [acquireSessionMutation]
  )

  const newSession = useCallback(
    async (msg?: string, title?: string) => {
      if (sessionMutationActiveRef.current) {
        sys('wait for the current session mutation before switching sessions')

        return
      }

      if (pendingPasteRef?.current) {
        sys('wait for the clipboard image attachment before switching sessions')

        return
      }

      const currentSid = getUiState().sid

      return runSessionSwitchFlight(
        sessionSwitchEpochRef,
        currentSid,
        () => getUiState().sid,
        active => patchUiState({ sessionSwitching: active }),
        async isCurrent => {
          const setup = await rpc<SetupStatusResponse>('setup.status', {})

          if (!isCurrent()) {
            return
          }

          if (setup?.provider_configured === false) {
            panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
            patchUiState({ status: 'setup required' })

            return
          }

          const closed = await closeCurrentSessionForSwitch(currentSid, null, closeSession)

          if (!isCurrent()) {
            return
          }

          if (!closed) {
            sys('session switch aborted because the current session could not be closed safely')
            patchUiState({ status: 'ready' })

            return
          }

          const r = await rpc<SessionCreateResponse>('session.create', { cols: colsRef.current })

          if (!isCurrent()) {
            return
          }

          if (!r) {
            patchUiState({ status: 'ready' })

            return
          }

          const info = r.info ?? null
          const requestedTitle = title?.trim() ?? ''

          resetSession()
          setSessionStartedAt(Date.now())

          writeActiveSessionFile(r.session_id)
          patchUiState({
            info,
            sid: r.session_id,
            status: info?.version ? 'ready' : 'starting agent…',
            usage: usageFrom(info)
          })

          if (info) {
            setHistoryItems([introMsg(info)])
          }

          if (info?.credential_warning) {
            sys(`warning: ${info.credential_warning}`)
          }

          if (info?.config_warning) {
            sys(`warning: ${info.config_warning}`)
          }

          if (msg) {
            const bareId = r.session_id.includes(':') ? r.session_id.slice(r.session_id.indexOf(':') + 1) : r.session_id
            sys(`${msg}, new session id = ${bareId}`)
          }

          if (requestedTitle) {
            rpc<SessionTitleResponse>('session.title', {
              session_id: r.session_id,
              title: requestedTitle
            })
              .then(result => {
                if (!result || getUiState().sid !== r.session_id) {
                  return
                }

                const nextTitle = (result.title ?? requestedTitle).trim()
                const suffix = result.pending ? ' (queued while session initializes)' : ''
                sys(`session title set: ${nextTitle}${suffix}`)
              })
              .catch((err: unknown) => {
                if (getUiState().sid !== r.session_id) {
                  return
                }

                const message = err instanceof Error ? err.message : String(err)
                sys(`warning: failed to set session title: ${message}`)
              })
          }
        }
      )
    },
    [closeSession, colsRef, panel, pendingPasteRef, resetSession, rpc, setHistoryItems, setSessionStartedAt, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      if (sessionMutationActiveRef.current) {
        sys('wait for the current session mutation before switching sessions')

        return
      }

      if (pendingPasteRef?.current) {
        sys('wait for the clipboard image attachment before switching sessions')

        return
      }

      const currentSid = getUiState().sid

      patchOverlayState({ picker: false })
      patchUiState({ status: 'resuming…' })

      void runSessionSwitchFlight(
        sessionSwitchEpochRef,
        currentSid,
        () => getUiState().sid,
        active => patchUiState({ sessionSwitching: active }),
        async isCurrent => {
          try {
            const setup = await rpc<SetupStatusResponse>('setup.status', {})

            if (!isCurrent()) {
              return
            }

            if (setup?.provider_configured === false) {
              panel(SETUP_REQUIRED_TITLE, buildSetupRequiredSections())
              patchUiState({ status: 'setup required' })

              return
            }

            const closed = await closeCurrentSessionForSwitch(currentSid, id, closeSession)

            if (!isCurrent()) {
              return
            }

            if (!closed) {
              sys('session switch aborted because the current session could not be closed safely')
              patchUiState({ status: 'ready' })

              return
            }

            const raw = await gw.request<SessionResumeResponse>('session.resume', { session_id: id })

            if (!isCurrent()) {
              return
            }

            const r = asRpcResult<SessionResumeResponse>(raw)

            if (!r) {
              sys('error: invalid response: session.resume')
              patchUiState({ status: 'ready' })

              return
            }

            resetSession()
            setSessionStartedAt(Date.now())

            const resumed = toTranscriptMessages(r.messages)

            setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
            writeActiveSessionFile(r.resumed ?? r.session_id)
            patchUiState({
              info: r.info ?? null,
              sid: r.session_id,
              status: 'ready',
              usage: usageFrom(r.info ?? null)
            })
            setTimeout(() => scrollRef.current?.scrollToBottom(), 0)
          } catch (error) {
            if (isCurrent()) {
              const message = error instanceof Error ? error.message : String(error)
              sys(`error: ${message}`)
              patchUiState({ status: 'ready' })
            }
          }
        }
      )
    },
    [closeSession, gw, panel, pendingPasteRef, resetSession, rpc, scrollRef, setHistoryItems, setSessionStartedAt, sys]
  )

  const guardBusySessionSwitch = useCallback(
    (what = 'switch sessions') => {
      const message = sessionSwitchBlockMessage(
        getUiState().busy,
        Boolean(pendingPasteRef?.current),
        what,
        getUiState().sessionSwitching,
        sessionMutationActiveRef.current
      )

      if (!message) {
        return false
      }

      sys(message)

      return true
    },
    [pendingPasteRef, sys]
  )

  const deleteSessionWithFallback = useCallback(
    async (targetId: string) => {
      let release = acquireSessionMutation('delete a session')

      if (!release) {
        return null
      }

      const releaseMutation = () => {
        release?.()
        release = null
      }

      try {
        return await performDeleteWithFallback(targetId, {
          activeSid: getUiState().sid,
          beforeNewSession: () => {
            resetSession()
            patchUiState({ status: 'ready' })
            releaseMutation()
          },
          newSession,
          releaseSessionImages,
          rpc
        })
      } finally {
        releaseMutation()
      }
    },
    [acquireSessionMutation, newSession, releaseSessionImages, resetSession, rpc]
  )

  return {
    closeSession,
    deleteSessionWithFallback,
    guardBusySessionSwitch,
    newSession,
    resetSession,
    resetVisibleHistory,
    resumeById,
    runSessionMutation,
    trimLastExchange: trimTail
  }
}
