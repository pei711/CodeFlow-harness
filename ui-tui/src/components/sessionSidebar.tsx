// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo, useEffect, useState } from 'react'

import type { SessionListItem, SessionListResponse } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'
import type { TuiRpcClient } from '../tuiRpcClient.js'

import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { $uiSessionBusy, $uiTurnBusy } from '../app/uiStore.js'

const MAX_VISIBLE = 8

export const SessionSidebar = memo(function SessionSidebar({
  activeSid,
  gw,
  onSelect,
  t,
  width
}: {
  activeSid?: null | string
  gw: TuiRpcClient
  onSelect: (id: string) => void
  t: Theme
  width: number
}) {
  const [items, setItems] = useState<SessionListItem[]>([])
  const [error, setError] = useState('')
  const switchBlocked = useStore($uiSessionBusy)
  const turnBusy = useStore($uiTurnBusy)

  useEffect(() => {
    if (turnBusy) {
      return
    }

    let active = true

    gw.request<SessionListResponse>('session.list', { limit: 200 })
      .then(raw => {
        const response = asRpcResult<SessionListResponse>(raw)

        if (!active) {
          return
        }

        if (!response) {
          setError('invalid session list')

          return
        }

        setItems(response.sessions ?? [])
        setError('')
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(rpcErrorMessage(reason))
        }
      })

    return () => {
      active = false
    }
  }, [activeSid, gw, turnBusy])

  return (
    <Box
      borderColor={t.color.border}
      borderStyle="single"
      flexDirection="column"
      flexShrink={0}
      paddingX={1}
      width={width}
    >
      <Box justifyContent="space-between" marginBottom={1}>
        <Text bold color={t.color.accent}>
          History
        </Text>
        <Text color={t.color.muted}>{items.length}</Text>
      </Box>

      {items.slice(0, MAX_VISIBLE).map(session => {
        const selected = session.id === activeSid
        const label = session.title || session.preview || '(untitled)'
        const ageDays = Math.max(0, Math.floor((Date.now() / 1000 - session.started_at) / 86400))
        const age = ageDays === 0 ? 'today' : ageDays === 1 ? 'yesterday' : `${ageDays}d ago`

        return (
          <Box
            flexDirection="column"
            key={session.id}
            marginBottom={1}
            onClick={switchBlocked ? undefined : () => onSelect(session.id)}
          >
            <Text bold={selected} color={selected ? t.color.accent : t.color.text} wrap="truncate-end">
              {selected ? '› ' : '  '}
              {label}
            </Text>
            <Text color={t.color.muted} wrap="truncate-end">
              {'  '}
              {age} · {session.message_count} msgs
            </Text>
          </Box>
        )
      })}

      {!items.length && <Text color={t.color.muted}>{error || 'No conversations yet'}</Text>}

      {items.length > MAX_VISIBLE && <Text color={t.color.muted}>+{items.length - MAX_VISIBLE} more</Text>}

      <Box flexGrow={1} />
      {switchBlocked ? (
        <Text color={t.color.label}>wait for turn to finish</Text>
      ) : (
        <>
          <Text color={t.color.muted}>/sessions</Text>
          <Text color={t.color.muted}>browse · create · delete</Text>
        </>
      )}
    </Box>
  )
})
