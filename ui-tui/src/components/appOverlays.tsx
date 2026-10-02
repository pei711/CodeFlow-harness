// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import type { AppOverlaysProps } from '../app/interfaces.js'
import type { Theme } from '../theme.js'

import { useGateway } from '../app/gatewayContext.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme } from '../app/uiStore.js'
import { FloatBox } from './appChrome.js'
import { ModelPicker } from './modelPicker.js'
import { OverlayHint } from './overlayControls.js'
import { ClarifyPrompt, ConfirmPrompt } from './prompts.js'
import { SessionPicker } from './sessionPicker.js'

const COMPLETION_WINDOW = 16

export function PromptZone({
  cols,
  onClarifyAnswer,
  onConfirmAnswer
}: Pick<AppOverlaysProps, 'cols' | 'onClarifyAnswer' | 'onConfirmAnswer'>) {
  const overlay = useStore($overlayState)
  const theme = useStore($uiTheme)

  if (overlay.confirm) {
    const req = overlay.confirm
    const isRpc = Boolean(req.requestId)

    const onConfirm = () => {
      if (isRpc) {
        onConfirmAnswer(true)

        return
      }

      patchOverlayState({ confirm: null })
      req.onConfirm?.()
    }

    const onCancel = () => {
      if (isRpc) {
        onConfirmAnswer(false)

        return
      }

      patchOverlayState({ confirm: null })
    }

    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ConfirmPrompt onCancel={onCancel} onConfirm={onConfirm} req={req} t={theme} />
      </Box>
    )
  }

  if (overlay.clarify) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ClarifyPrompt
          cols={cols}
          onAnswer={onClarifyAnswer}
          onCancel={() => onClarifyAnswer('')}
          req={overlay.clarify}
          t={theme}
        />
      </Box>
    )
  }

  return null
}

export function FloatingOverlays({
  cols,
  compIdx,
  completions,
  onModelSelect,
  onPickerDeleteActive,
  onPickerSelect,
  pagerPageSize
}: Pick<
  AppOverlaysProps,
  'cols' | 'compIdx' | 'completions' | 'onModelSelect' | 'onPickerDeleteActive' | 'onPickerSelect' | 'pagerPageSize'
>) {
  const { gw } = useGateway()
  const overlay = useStore($overlayState)
  const sid = useStore($uiSessionId)
  const theme = useStore($uiTheme)

  const hasAny = overlay.modelPicker || overlay.pager || overlay.picker || completions.length

  if (!hasAny) {
    return null
  }

  // 使用以 compIdx 为中心的固定视口。此前切片终点为 compIdx + 8，导致用户向下
  // 滚动时下拉框从 8 行增长到 16 行，并在每次按键时跳动高度。
  const viewportSize = Math.min(COMPLETION_WINDOW, completions.length)

  const start = Math.max(0, Math.min(compIdx - Math.floor(COMPLETION_WINDOW / 2), completions.length - viewportSize))

  return (
    <Box alignItems="flex-start" bottom="100%" flexDirection="column" left={0} position="absolute" right={0}>
      {overlay.picker && (
        <FloatBox color={theme.color.border}>
          <SessionPicker
            activeSid={sid}
            gw={gw}
            onCancel={() => patchOverlayState({ picker: false })}
            onDeleteActive={onPickerDeleteActive}
            onSelect={onPickerSelect}
            t={theme}
          />
        </FloatBox>
      )}

      {overlay.modelPicker && (
        <FloatBox color={theme.color.border}>
          <ModelPicker
            gw={gw}
            onCancel={() => patchOverlayState({ modelPicker: false })}
            onSelect={onModelSelect}
            sessionId={sid}
            t={theme}
          />
        </FloatBox>
      )}

      {overlay.pager && (
        <FloatBox color={theme.color.border}>
          <Box flexDirection="column" paddingX={1} paddingY={1}>
            {overlay.pager.title && (
              <Box justifyContent="center" marginBottom={1}>
                <Text bold color={theme.color.primary}>
                  {overlay.pager.title}
                </Text>
              </Box>
            )}

            {overlay.pager.lines.slice(overlay.pager.offset, overlay.pager.offset + pagerPageSize).map((line, i) => (
              <Text key={i}>{line}</Text>
            ))}

            <Box marginTop={1}>
              <OverlayHint t={theme}>
                {overlay.pager.offset + pagerPageSize < overlay.pager.lines.length
                  ? `↑↓/jk line · Enter/Space/PgDn page · b/PgUp back · g/G top/bottom · Esc/q close (${Math.min(overlay.pager.offset + pagerPageSize, overlay.pager.lines.length)}/${overlay.pager.lines.length})`
                  : `end · ↑↓/jk · b/PgUp back · g top · Esc/q close (${overlay.pager.lines.length} lines)`}
              </OverlayHint>
            </Box>
          </Box>
        </FloatBox>
      )}

      {!!completions.length && (
        <CompletionPalette
          compIdx={compIdx}
          completions={completions.slice(start, start + viewportSize)}
          start={start}
          t={theme}
          total={completions.length}
          width={Math.max(8, cols - 4)}
        />
      )}
    </Box>
  )
}

function CompletionPalette({
  compIdx,
  completions,
  start,
  t,
  total,
  width
}: {
  compIdx: number
  completions: AppOverlaysProps['completions']
  start: number
  t: Theme
  total: number
  width: number
}) {
  const labelWidth = Math.max(1, Math.min(32, width - 8, Math.max(...completions.map(item => item.display.length))))

  return (
    <Box
      backgroundColor={t.color.completionBg}
      borderColor={t.color.border}
      borderStyle="single"
      flexDirection="column"
      marginTop={1}
      opaque
      paddingX={1}
      width={width}
    >
      <Text color={t.color.muted} wrap="truncate-end">
        {total} matches
      </Text>
      {completions.map((item, i) => {
        const active = start + i === compIdx
        const display =
          item.display.length > labelWidth
            ? `${item.display.slice(0, Math.max(0, labelWidth - 1))}…`
            : item.display.padEnd(labelWidth)

        return (
          <Box
            backgroundColor={active ? t.color.completionCurrentBg : t.color.completionBg}
            key={`${start + i}:${item.text}:${item.display}:${item.meta ?? ''}`}
            width="100%"
          >
            <Text bold={active} color={active ? t.color.info : t.color.text} wrap="truncate-end">
              {active ? `${t.brand.prompt} ` : '  '}
              {display}
            </Text>
            {item.meta ? (
              <Text color={t.color.muted} wrap="truncate-end">
                {' '}
                {item.meta}
              </Text>
            ) : null}
          </Box>
        )
      })}
      <Text color={t.color.muted} wrap="truncate-end">
        ↑↓ move · tab apply · esc close
      </Text>
    </Box>
  )
}
