// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text, useInput, useStdout } from '@hermes/ink'
import { useEffect, useMemo, useState } from 'react'

import type { ModelOptionProvider, ModelOptionsResponse } from '../gatewayTypes.js'
import type { Theme } from '../theme.js'
import type { TuiRpcClient } from '../tuiRpcClient.js'

import { getCodeFlowHomeLabel } from '../config/paths.js'
import { providerDisplayNames } from '../domain/providers.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { OverlayHint, useOverlayKeys, windowItems } from './overlayControls.js'

const VISIBLE = 12
const MIN_WIDTH = 40
const MAX_WIDTH = 90

type Stage = 'provider' | 'key' | 'model' | 'addModel' | 'disconnect'
type KeyField = 'api_key' | 'api_base'

export function ModelPicker({ gw, onCancel, onSelect, sessionId, t }: ModelPickerProps) {
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [currentModel, setCurrentModel] = useState('')
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(true)
  const [providerIdx, setProviderIdx] = useState(0)
  const [modelIdx, setModelIdx] = useState(0)
  const [stage, setStage] = useState<Stage>('provider')
  const [keyInput, setKeyInput] = useState('')
  const [baseInput, setBaseInput] = useState('')
  const [keyField, setKeyField] = useState<KeyField>('api_key')
  const [keySaving, setKeySaving] = useState(false)
  const [keyError, setKeyError] = useState('')
  const [modelNameInput, setModelNameInput] = useState('')

  const { stdout } = useStdout()
  // 将选择器宽度固定，避免采用 alignSelf="flex-start" 自适应收缩的 FloatBox
  // 父容器在较长的供应商或模型名称滚入视野时改变宽度，同时为各行的
  // `wrap="truncate-end"` 提供实际可用的截断约束。
  const width = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, (stdout?.columns ?? 80) - 6))

  useEffect(() => {
    gw.request<ModelOptionsResponse>('model.options', sessionId ? { session_id: sessionId } : {})
      .then(raw => {
        const r = asRpcResult<ModelOptionsResponse>(raw)

        if (!r) {
          setErr('invalid response: model.options')
          setLoading(false)

          return
        }

        const next = r.providers ?? []
        setProviders(next)
        setCurrentModel(String(r.model ?? ''))
        setProviderIdx(
          Math.max(
            0,
            next.findIndex(p => p.is_current)
          )
        )
        setModelIdx(0)
        setStage('provider')
        setErr('')
        setLoading(false)
      })
      .catch((e: unknown) => {
        setErr(rpcErrorMessage(e))
        setLoading(false)
      })
  }, [gw, sessionId])

  const provider = providers[providerIdx]
  const models = provider?.models ?? []
  const names = useMemo(() => providerDisplayNames(providers), [providers])

  const back = () => {
    if (stage === 'addModel') {
      setStage('model')
      setModelNameInput('')
      setKeyError('')

      return
    }

    if (stage === 'model' || stage === 'key' || stage === 'disconnect') {
      setStage('provider')
      setModelIdx(0)
      setKeyInput('')
      setBaseInput('')
      setKeyField('api_key')
      setKeyError('')
      setKeySaving(false)

      return
    }

    onCancel()
  }

  useOverlayKeys({ onBack: back, onClose: onCancel })

  useInput((ch, key) => {
    // 密钥输入阶段自行处理输入（api_key 以及可选的 api_base）。
    if (stage === 'key') {
      if (keySaving) {
        return
      }

      const showBase = provider?.auth_type === 'api_key'
      const focusBase = showBase && keyField === 'api_base'

      // 显示 api_base 时，Tab 用于在两个字段之间切换。
      if (key.tab && showBase) {
        setKeyField(f => (f === 'api_key' ? 'api_base' : 'api_key'))

        return
      }

      if (key.return) {
        // 在 api_key 字段按回车会移至 api_base 而非提交，使用户仅靠单键导航
        // 即可填写两个字段。
        if (showBase && keyField === 'api_key') {
          setKeyField('api_base')

          return
        }

        const apiKey = keyInput.trim()
        const apiBase = baseInput.trim()

        if (!apiKey) {
          setKeyError('API key is required')

          return
        }

        if (provider?.needs_api_base && !apiBase) {
          setKeyError('API base URL is required for this provider')

          return
        }

        setKeySaving(true)
        setKeyError('')
        gw.request<{ provider?: ModelOptionProvider }>('model.save_key', {
          slug: provider?.slug,
          api_key: apiKey,
          ...(apiBase ? { api_base: apiBase } : {}),
          ...(sessionId ? { session_id: sessionId } : {})
        })
          .then(raw => {
            const r = asRpcResult<{ provider?: ModelOptionProvider }>(raw)

            if (!r?.provider) {
              setKeyError('failed to save key')
              setKeySaving(false)

              return
            }

            // 用最新数据更新列表中的供应商。
            setProviders(prev => prev.map(p => (p.slug === r.provider!.slug ? r.provider! : p)))
            setKeyInput('')
            setBaseInput('')
            setKeyField('api_key')
            setKeySaving(false)
            setStage('model')
            setModelIdx(0)
          })
          .catch((e: unknown) => {
            setKeyError(rpcErrorMessage(e))
            setKeySaving(false)
          })

        return
      }

      if (key.backspace || key.delete) {
        if (focusBase) {
          setBaseInput(v => v.slice(0, -1))
        } else {
          setKeyInput(v => v.slice(0, -1))
        }

        return
      }

      // Ctrl+U 清空当前聚焦的字段。
      if (ch === '\u0015') {
        if (focusBase) {
          setBaseInput('')
        } else {
          setKeyInput('')
        }

        return
      }

      if (ch && !key.ctrl && !key.meta) {
        if (focusBase) {
          setBaseInput(v => v + ch)
        } else {
          setKeyInput(v => v + ch)
        }
      }

      return
    }

    // 添加模型名称的子输入框。
    if (stage === 'addModel') {
      if (keySaving) {
        return
      }

      if (key.return) {
        const model = modelNameInput.trim()

        if (!model || !provider) {
          return
        }

        setKeySaving(true)
        setKeyError('')
        gw.request<{ provider?: ModelOptionProvider }>('model.add_model', {
          slug: provider.slug,
          model,
          ...(sessionId ? { session_id: sessionId } : {})
        })
          .then(raw => {
            const r = asRpcResult<{ provider?: ModelOptionProvider }>(raw)

            if (!r?.provider) {
              setKeyError('failed to add model')
              setKeySaving(false)

              return
            }

            setProviders(prev => prev.map(p => (p.slug === r.provider!.slug ? r.provider! : p)))
            const idx = (r.provider.models ?? []).indexOf(model)
            setModelNameInput('')
            setKeySaving(false)
            setStage('model')
            setModelIdx(idx >= 0 ? idx : 0)
          })
          .catch((e: unknown) => {
            setKeyError(rpcErrorMessage(e))
            setKeySaving(false)
          })

        return
      }

      if (key.backspace || key.delete) {
        setModelNameInput(v => v.slice(0, -1))

        return
      }

      if (ch && !key.ctrl && !key.meta) {
        setModelNameInput(v => v + ch)
      }

      return
    }

    // 断开连接确认阶段。
    if (stage === 'disconnect') {
      if (ch.toLowerCase() === 'y' || key.return) {
        if (!provider) {
          setStage('provider')

          return
        }

        setKeySaving(true)
        gw.request<{ disconnected?: boolean }>('model.disconnect', {
          slug: provider.slug,
          ...(sessionId ? { session_id: sessionId } : {})
        })
          .then(raw => {
            const r = asRpcResult<{ disconnected?: boolean }>(raw)

            if (r?.disconnected) {
              // 在本地状态中将供应商标记为未认证。
              setProviders(prev =>
                prev.map(p =>
                  p.slug === provider.slug
                    ? {
                        ...p,
                        authenticated: false,
                        models: [],
                        total_models: 0,
                        warning: p.key_env ? `paste ${p.key_env} to activate` : 'run `codeflow model` to configure'
                      }
                    : p
                )
              )
            }

            setKeySaving(false)
            setStage('provider')
          })
          .catch(() => {
            setKeySaving(false)
            setStage('provider')
          })

        return
      }

      if (ch.toLowerCase() === 'n' || key.escape) {
        setStage('provider')

        return
      }

      return
    }

    const count = stage === 'provider' ? providers.length : models.length
    const sel = stage === 'provider' ? providerIdx : modelIdx
    const setSel = stage === 'provider' ? setProviderIdx : setModelIdx

    if (key.upArrow && sel > 0) {
      setSel(v => v - 1)

      return
    }

    if (key.downArrow && sel < count - 1) {
      setSel(v => v + 1)

      return
    }

    if (key.return) {
      if (stage === 'provider') {
        if (!provider) {
          return
        }

        if (provider.authenticated === false) {
          // api_key 类型的供应商会在界面内提示输入密钥，即使 key_env 为 null；
          // custom 和 azure 使用通用密钥，并要求填写 api_base。
          if (provider.auth_type === 'api_key') {
            setStage('key')
            setKeyInput('')
            setBaseInput('')
            setKeyField('api_key')
            setKeyError('')
          }

          // OAuth 或其他认证类型无需处理，警告信息会提示用户运行 codeflow model。
          return
        }

        setStage('model')
        setModelIdx(0)

        return
      }

      if (stage === 'model' && keySaving) {
        return
      }

      const model = models[modelIdx]

      if (provider && model) {
        onSelect(model, provider.slug)
      } else {
        setStage('provider')
      }

      return
    }

    // 模型阶段：向供应商列表添加模型名称。
    if (ch.toLowerCase() === 'a' && stage === 'model' && provider && !keySaving) {
      setStage('addModel')
      setModelNameInput('')
      setKeyError('')

      return
    }

    // 模型阶段：从供应商列表删除高亮的模型名称。
    if ((ch.toLowerCase() === 'd' || ch.toLowerCase() === 'x') && stage === 'model' && !keySaving) {
      const model = models[modelIdx]

      if (!provider || !model) {
        return
      }

      setKeySaving(true)
      setKeyError('')
      gw.request<{ provider?: ModelOptionProvider }>('model.remove_model', {
        slug: provider.slug,
        model,
        ...(sessionId ? { session_id: sessionId } : {})
      })
        .then(raw => {
          const r = asRpcResult<{ provider?: ModelOptionProvider }>(raw)

          if (r?.provider) {
            setProviders(prev => prev.map(p => (p.slug === r.provider!.slug ? r.provider! : p)))
            setModelIdx(idx => Math.max(0, Math.min(idx, (r.provider!.models?.length ?? 1) - 1)))
          }

          setKeySaving(false)
        })
        .catch((e: unknown) => {
          setKeyError(rpcErrorMessage(e))
          setKeySaving(false)
        })

      return
    }

    // 仅在供应商阶段且供应商已认证时允许断开连接。
    if (ch.toLowerCase() === 'd' && stage === 'provider' && provider?.authenticated !== false) {
      setStage('disconnect')

      return
    }
  })

  if (loading) {
    return <Text color={t.color.muted}>loading models…</Text>
  }

  if (err) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.label}>error: {err}</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  if (!providers.length) {
    return (
      <Box flexDirection="column">
        <Text color={t.color.muted}>no providers available</Text>
        <OverlayHint t={t}>Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  // ── 密钥输入阶段 ──────────────────────────────────────────────────
  if (stage === 'key' && provider) {
    const showBase = provider.auth_type === 'api_key'
    const focusBase = showBase && keyField === 'api_base'
    const masked = keyInput ? '•'.repeat(Math.min(keyInput.length, 40)) : ''
    const keyLabel = provider.key_env ?? 'API key'
    const baseLabel = `API base${provider.needs_api_base ? ' (required)' : ' (optional)'}`
    const caret = keySaving ? '' : '▎'

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Configure {provider.name}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Saved to {getCodeFlowHomeLabel()}/.env{showBase ? ' · Tab switches field' : ''}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={focusBase ? t.color.muted : t.color.accent} wrap="truncate-end">
          {focusBase ? '  ' : '▸ '}
          {keyLabel}:
        </Text>

        <Text color={t.color.accent} wrap="truncate-end">
          {'  '}
          {masked || '(empty)'}
          {focusBase ? '' : caret}
        </Text>

        {showBase ? (
          <>
            <Text color={focusBase ? t.color.accent : t.color.muted} wrap="truncate-end">
              {focusBase ? '▸ ' : '  '}
              {baseLabel}:
            </Text>

            <Text color={t.color.accent} wrap="truncate-end">
              {'  '}
              {baseInput || '(empty)'}
              {focusBase ? caret : ''}
            </Text>
          </>
        ) : null}

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keyError ? (
          <Text color={t.color.label} wrap="truncate-end">
            error: {keyError}
          </Text>
        ) : keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            saving…
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {' '}
          </Text>
        )}

        <OverlayHint t={t}>
          {showBase ? 'Enter next/save · Tab field · Ctrl+U clear · Esc back' : 'Enter save · Ctrl+U clear · Esc back'}
        </OverlayHint>
      </Box>
    )
  }

  // ── 添加模型名称阶段 ─────────────────────────────────────────────
  if (stage === 'addModel' && provider) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Add model to {provider.name}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Type the full model id
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={t.color.accent} wrap="truncate-end">
          {'  '}
          {modelNameInput || '(empty)'}
          {keySaving ? '' : '▎'}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keyError ? (
          <Text color={t.color.label} wrap="truncate-end">
            error: {keyError}
          </Text>
        ) : keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            saving…
          </Text>
        ) : (
          <Text color={t.color.muted} wrap="truncate-end">
            {' '}
          </Text>
        )}

        <OverlayHint t={t}>Enter add · Ctrl+U clear · Esc back</OverlayHint>
      </Box>
    )
  }

  // ── 断开连接确认阶段 ─────────────────────────────────────
  if (stage === 'disconnect' && provider) {
    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Disconnect {provider.name}?
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          This removes saved credentials for {provider.name}.
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          You can re-authenticate later by selecting it again.
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          {' '}
        </Text>

        {keySaving ? (
          <Text color={t.color.muted} wrap="truncate-end">
            disconnecting…
          </Text>
        ) : (
          <OverlayHint t={t}>y/Enter confirm · n/Esc cancel</OverlayHint>
        )}
      </Box>
    )
  }

  // ── 供应商选择阶段 ─────────────────────────────────────────
  if (stage === 'provider') {
    const rows = providers.map((p, i) => {
      const authMark = p.authenticated === false ? '○' : p.is_current ? '*' : '●'
      const modelCount = p.total_models ?? p.models?.length ?? 0

      const suffix =
        p.authenticated === false ? (p.auth_type === 'api_key' ? '(no key)' : '(needs setup)') : `${modelCount} models`

      return `${authMark} ${names[i]} · ${suffix}`
    })

    const { items, offset } = windowItems(rows, providerIdx, VISIBLE)

    return (
      <Box flexDirection="column" width={width}>
        <Text bold color={t.color.accent} wrap="truncate-end">
          Select provider (step 1/2)
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Full model IDs on the next step · Enter to continue
        </Text>

        <Text color={t.color.muted} wrap="truncate-end">
          Current: {currentModel || '(unknown)'}
        </Text>
        <Text color={t.color.label} wrap="truncate-end">
          {provider?.warning ? `warning: ${provider.warning}` : ' '}
        </Text>
        <Text color={t.color.muted} wrap="truncate-end">
          {offset > 0 ? ` ↑ ${offset} more` : ' '}
        </Text>

        {Array.from({ length: VISIBLE }, (_, i) => {
          const row = items[i]
          const idx = offset + i
          const p = providers[idx]
          const dimmed = p?.authenticated === false

          return row ? (
            <Text
              bold={providerIdx === idx}
              color={providerIdx === idx ? t.color.accent : dimmed ? t.color.label : t.color.muted}
              inverse={providerIdx === idx}
              key={providers[idx]?.slug ?? `row-${idx}`}
              wrap="truncate-end"
            >
              {providerIdx === idx ? '▸ ' : '  '}
              {idx + 1}. {row}
            </Text>
          ) : (
            <Text color={t.color.muted} key={`pad-${i}`} wrap="truncate-end">
              {' '}
            </Text>
          )
        })}

        <Text color={t.color.muted} wrap="truncate-end">
          {offset + VISIBLE < rows.length ? ` ↓ ${rows.length - offset - VISIBLE} more` : ' '}
        </Text>

        <OverlayHint t={t}>↑/↓ select · Enter choose · d disconnect · Esc/q cancel</OverlayHint>
      </Box>
    )
  }

  // ── 模型选择阶段 ────────────────────────────────────────────
  const { items, offset } = windowItems(models, modelIdx, VISIBLE)

  return (
    <Box flexDirection="column" width={width}>
      <Text bold color={t.color.accent} wrap="truncate-end">
        Select model (step 2/2)
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        {names[providerIdx] || '(unknown provider)'} · Esc back
      </Text>
      <Text color={t.color.label} wrap="truncate-end">
        {provider?.warning ? `warning: ${provider.warning}` : ' '}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {offset > 0 ? ` ↑ ${offset} more` : ' '}
      </Text>

      {Array.from({ length: VISIBLE }, (_, i) => {
        const row = items[i]
        const idx = offset + i

        if (!row) {
          return !models.length && i === 0 ? (
            <Text color={t.color.muted} key="empty" wrap="truncate-end">
              no models listed for this provider
            </Text>
          ) : (
            <Text color={t.color.muted} key={`pad-${i}`} wrap="truncate-end">
              {' '}
            </Text>
          )
        }

        const prefix = modelIdx === idx ? '▸ ' : row === currentModel ? '* ' : '  '

        return (
          <Text
            bold={modelIdx === idx}
            color={modelIdx === idx ? t.color.accent : t.color.muted}
            inverse={modelIdx === idx}
            key={`${provider?.slug ?? 'prov'}:${idx}:${row}`}
            wrap="truncate-end"
          >
            {prefix}
            {idx + 1}. {row}
          </Text>
        )
      })}

      <Text color={t.color.muted} wrap="truncate-end">
        {offset + VISIBLE < models.length ? ` ↓ ${models.length - offset - VISIBLE} more` : ' '}
      </Text>

      <Text color={t.color.muted} wrap="truncate-end">
        scope: global
      </Text>
      <OverlayHint t={t}>
        {models.length
          ? '↑/↓ select · Enter switch · a add · d/x delete · Esc back · q close'
          : 'a add model · Enter/Esc back · q close'}
      </OverlayHint>
    </Box>
  )
}

interface ModelPickerProps {
  gw: TuiRpcClient
  onCancel: () => void
  onSelect: (model: string, providerSlug: string) => void
  sessionId: string | null
  t: Theme
}
