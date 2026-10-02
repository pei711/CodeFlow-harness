// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text, useInput } from '@hermes/ink'
import { useEffect, useRef, useState } from 'react'

import type { Theme } from '../theme.js'
import type { ClarifyReq, ConfirmReq } from '../types.js'

import { answerConfirmFromInput } from '../app/confirmResponse.js'
import { CONFIRM_COUNTDOWN_SECONDS, tickCountdown } from '../lib/confirmCountdown.js'
import { isMac } from '../lib/platform.js'
import { TextInput } from './textInput.js'

export function ClarifyPrompt({ cols = 80, onAnswer, onCancel, req, t }: ClarifyPromptProps) {
  const [sel, setSel] = useState(0)
  const [custom, setCustom] = useState('')
  const [typing, setTyping] = useState(false)
  const choices = req.choices ?? []

  const heading = (
    <Text bold>
      <Text color={t.color.accent}>ask</Text>
      <Text color={t.color.text}> {req.question}</Text>
    </Text>
  )

  useInput((ch, key) => {
    if (key.escape) {
      typing && choices.length ? setTyping(false) : onCancel()

      return
    }

    if (typing || !choices.length) {
      return
    }

    if (key.upArrow && sel > 0) {
      setSel(s => s - 1)
    }

    if (key.downArrow && sel < choices.length) {
      setSel(s => s + 1)
    }

    if (key.return) {
      sel === choices.length ? setTyping(true) : choices[sel] && onAnswer(choices[sel]!)
    }

    const n = parseInt(ch)

    if (n >= 1 && n <= choices.length) {
      onAnswer(choices[n - 1]!)
    }
  })

  if (typing || !choices.length) {
    return (
      <Box flexDirection="column">
        {heading}

        <Box>
          <Text color={t.color.label}>{'> '}</Text>
          <TextInput columns={Math.max(20, cols - 6)} onChange={setCustom} onSubmit={onAnswer} value={custom} />
        </Box>

        <Text color={t.color.muted}>
          Enter send · Esc {choices.length ? 'back' : 'cancel'} ·{' '}
          {isMac ? 'Cmd+C copy · Cmd+V paste · Ctrl+C cancel' : 'Ctrl+C cancel'}
        </Text>
      </Box>
    )
  }

  return (
    <Box flexDirection="column">
      {heading}

      {[...choices, 'Other (type your answer)'].map((c, i) => (
        <Text key={i}>
          <Text bold={sel === i} color={sel === i ? t.color.label : t.color.muted} inverse={sel === i}>
            {sel === i ? '▸ ' : '  '}
            {i + 1}. {c}
          </Text>
        </Text>
      ))}

      <Text color={t.color.muted}>↑/↓ select · Enter confirm · 1-{choices.length} quick pick · Esc/Ctrl+C cancel</Text>
    </Box>
  )
}

export function ConfirmPrompt({ onCancel, onConfirm, req, t }: ConfirmPromptProps) {
  const [sel, setSel] = useState(0)

  // 30 秒倒计时只驱动 RPC 路径，即服务端代理等待响应的情况。进程内确认没有
  // 远端截止时间，因此以暂停状态启动（remaining = null）且不会自动取消。
  const isRpc = Boolean(req.requestId)
  const [remaining, setRemaining] = useState<null | number>(isRpc ? CONFIRM_COUNTDOWN_SECONDS : null)
  const intervalRef = useRef<null | ReturnType<typeof setInterval>>(null)

  const suspend = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }

    setRemaining(null)
  }

  useEffect(() => {
    if (!isRpc) {
      return
    }

    intervalRef.current = setInterval(() => {
      setRemaining(prev => {
        if (prev === null) {
          return prev
        }

        const { autoCancel, remaining: next } = tickCountdown(prev)

        if (autoCancel) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }

          onCancel()
        }

        return next
      })
    }, 1000)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
    // onCancel 在给定确认浮层的整个生命周期内保持稳定。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRpc])

  useInput((ch, key, event) => {
    const lower = ch.toLowerCase()
    const answer = (callback: () => void) => {
      suspend()

      return answerConfirmFromInput(event, callback)
    }

    if (key.escape || (key.ctrl && lower === 'c') || lower === 'n') {
      return answer(onCancel)
    }

    if (lower === 'y') {
      return answer(onConfirm)
    }

    if (key.upArrow) {
      setSel(0)
      suspend()

      return
    }

    if (key.downArrow) {
      setSel(1)
      suspend()

      return
    }

    if (key.return) {
      return answer(sel === 0 ? onCancel : onConfirm)
    }

    // 其他任意按键会暂停倒计时，但不作答。
    suspend()
  })

  const accent = req.danger ? t.color.error : t.color.warn
  const countdownLabel = remaining === null ? '' : ` (${remaining}s)`

  const rows = [
    { color: t.color.text, label: `${req.cancelLabel ?? 'No'}${countdownLabel}` },
    { color: req.danger ? t.color.error : t.color.text, label: req.confirmLabel ?? 'Yes' }
  ]

  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" paddingX={1}>
      <Text bold color={accent}>
        {req.danger ? '⚠' : '?'} {req.title ?? req.prompt ?? 'Continue?'}
      </Text>

      {req.detail ? (
        <Box paddingLeft={1}>
          <Text color={t.color.text} wrap="truncate-end">
            {req.detail}
          </Text>
        </Box>
      ) : null}

      <Text />

      {rows.map((row, i) => (
        <Text key={row.label}>
          <Text color={sel === i ? accent : t.color.muted}>{sel === i ? '▸ ' : '  '}</Text>
          <Text color={sel === i ? row.color : t.color.muted}>{row.label}</Text>
        </Text>
      ))}

      <Text color={t.color.muted}>↑/↓ select · Enter confirm · Y/N quick · Esc cancel</Text>
    </Box>
  )
}

interface ClarifyPromptProps {
  cols?: number
  onAnswer: (s: string) => void
  onCancel: () => void
  req: ClarifyReq
  t: Theme
}

interface ConfirmPromptProps {
  onCancel: () => void
  onConfirm: () => void
  req: ConfirmReq
  t: Theme
}
