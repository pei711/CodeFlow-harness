// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'

import type { SessionDeleteResponse, SessionListItem, SessionListResponse } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'
import type { TuiRpcClient } from '../tuiRpcClient.js'

import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { OverlayHint, useOverlayKeys, windowOffset } from './overlayControls.js'

const VISIBLE = 15
const MIN_WIDTH = 60
const MAX_WIDTH = 120

const age = (ts: number) => {
  const d = (Date.now() / 1000 - ts) / 86400

  if (d < 1) {
    return 'today'
  }

  if (d < 2) {
    return 'yesterday'
  }

  return `${Math.floor(d)}d ago`
}

export function SessionPicker({ activeSid, gw, onCancel, onDeleteActive, onSelect, t }: SessionPickerProps) {
  const [items, setItems] = useState<SessionListItem[]>([])
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(0)
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState(false)

  const { stdout } = useStdout()
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useOverlayKeys({ onClose: onCancel })

  useEffect(() => {
    gw.request<SessionListResponse>('session.list', { limit: 200 })
      .then(raw => {
        const r = asRpcResult<SessionListResponse>(raw)

        if (!r) {
          setErr('invalid response: session.list')
          setLoading(false)

          return
        }

        setItems(r.sessions ?? [])
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw])

  const performDelete = (index: number) => {
    const target = items[index]

    if (!target || deleting) {
      return
    }

    setDeleting(true)

    if (onDeleteActive && activeSid === target.id) {
      // 回退路径会切换会话并关闭选择器，导致本组件卸载，因此直接跳过本地列表记账。
      onDeleteActive(target.id).catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setDeleting(false)
      })

      return
    }

    gw.request<SessionDeleteResponse>('session.delete', { session_id: target.id })
      .then(raw => {
        const r = asRpcResult<SessionDeleteResponse>(raw)

        if (!r) {
          setErr('invalid response: session.delete')
          setDeleting(false)

          return
        }

        if (r.cancelled) {
          setErr('')
          setDeleting(false)

          return
        }

        if (r.deleted === null) {
          setErr(`no such session: ${target.id}`)
          setDeleting(false)

          return
        }

        if (r.deleted !== target.id) {
          setErr('invalid response: session.delete')
          setDeleting(false)

          return
        }

        setItems(prev => {
          const next = prev.filter((_, i) => i !== index)
          setSel(s => Math.max(0, Math.min(s, next.length - 1)))

          return next
        })
        setErr('')
        setDeleting(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setDeleting(false)
      })
  }

  useInput((ch, key) => {
    if (deleting) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < items.length - 1) {
      setSel(s => s + 1)
    }

    if (key.return && items[sel]) {
      onSelect(items[sel]!.id)

      return
    }

    if (ch?.toLowerCase() === 'd' && items[sel]) {
      performDelete(sel)

      return
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= Math.min(9, items.length)) {
      onSelect(items[n - 1]!.id)
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading sessions…</Text>
  }

  if (err && !items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!items.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>no previous sessions</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  const offset = windowOffset(items.length, sel, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent}>
        Resume Session
      </Text>

      {offset > 0 && <Text color={t.color.muted}> ↑ {offset} more</Text>}

      {items.slice(offset, offset + VISIBLE).map((s, vi) => {
        const i = offset + vi
        const selected = sel === i

        return (
          <Box key={s.id}>
            <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
              {selected ? '▸ ' : '  '}
            </Text>

            <Box width={30}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
                {String(i + 1).padStart(2)}. [{s.id}]
              </Text>
            </Box>

            <Box width={30}>
              <Text bold={selected} color={selected ? t.color.accent : t.color.muted} inverse={selected}>
                ({s.message_count} msgs, {age(s.started_at)}, {s.source || 'tui'})
              </Text>
            </Box>

            <Text
              bold={selected}
              color={selected ? t.color.accent : t.color.muted}
              inverse={selected}
              wrap="truncate-end"
            >
              {s.title || s.preview || '(untitled)'}
            </Text>
          </Box>
        )
      })}

      {offset + VISIBLE < items.length && <Text color={t.color.muted}> ↓ {items.length - offset - VISIBLE} more</Text>}
      {err && <Text color={t.color.label}>error: {err}</Text>}
      {deleting ? (
        <OverlayHint t={t}>deleting…</OverlayHint>
      ) : (
        <OverlayHint t={t}>↑/↓ select · Enter resume · 1-9 quick · d delete · Esc/q cancel</OverlayHint>
      )}
    </Box>
  )
}

interface SessionPickerProps {
  activeSid?: null | string
  gw: TuiRpcClient
  onCancel: () => void
  onDeleteActive?: (id: string) => Promise<unknown>
  onSelect: (id: string) => void
  t: Theme
}
