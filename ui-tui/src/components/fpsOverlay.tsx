// FPS 计数浮层（CODEFLOW_TUI_FPS=1），禁用时没有额外开销。

import { Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import type { Theme } from '../theme.js'

import { SHOW_FPS } from '../config/env.js'
import { $fpsState } from '../lib/fpsStore.js'

const fpsColor = (fps: number, t: Theme) =>
  fps >= 50 ? t.color.statusGood : fps >= 30 ? t.color.statusWarn : t.color.error

export function FpsOverlay({ t }: { t: Theme }) {
  if (!SHOW_FPS) {
    return null
  }

  return <FpsOverlayInner t={t} />
}

function FpsOverlayInner({ t }: { t: Theme }) {
  const { fps, lastDurationMs, totalFrames } = useStore($fpsState)

  // 用零补齐宽度，避免数字变化导致角落抖动。
  return (
    <Text color={fpsColor(fps, t)}>
      {fps.toFixed(1).padStart(5)}fps · {lastDurationMs.toFixed(1).padStart(5)}ms · #{totalFrames}
    </Text>
  )
}
