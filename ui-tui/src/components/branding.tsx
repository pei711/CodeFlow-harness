// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { Box, Text, useStdout } from '@hermes/ink'
import { useEffect, useState } from 'react'
import unicodeSpinners from 'unicode-animations'

import type { PanelSection, SessionInfo } from '../types.js'

import { flat } from '../lib/text.js'
import { DEFAULT_THEME, type Theme } from '../theme.js'

const LOADER_TICK_MS = 120

function InlineLoader({ label, t }: { label: string; t: Theme }) {
  const [tick, setTick] = useState(0)
  const spinner = unicodeSpinners.braille
  const frame = spinner.frames[tick % spinner.frames.length] ?? '⠋'

  useEffect(() => {
    const id = setInterval(() => setTick(n => n + 1), Math.max(LOADER_TICK_MS, spinner.interval))

    return () => clearInterval(id)
  }, [spinner.interval])

  return (
    <Text color={t.color.muted} wrap="truncate">
      <Text color={t.color.accent}>{frame}</Text> {label}
    </Text>
  )
}

const STARTUP_MESSAGES = ['starting codeflow…', 'building agent loop…', 'loading tools & skills…']
const STARTUP_LABEL_MS = 900

// 后端构建智能体循环期间、session.info 握手填充 SessionPanel 前，在介绍行显示的占位符。
export function StartupLoader({ t }: { t: Theme }) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setStep(n => n + 1), STARTUP_LABEL_MS)

    return () => clearInterval(id)
  }, [])

  const label = STARTUP_MESSAGES[Math.min(step, STARTUP_MESSAGES.length - 1)] ?? STARTUP_MESSAGES[0]

  return (
    <Box backgroundColor={t.color.surfaceRaised} marginBottom={1} paddingX={1}>
      <InlineLoader label={label} t={t} />
    </Box>
  )
}

export function ArtLines({ lines }: { lines: [string, string][] }) {
  return (
    <>
      {lines.map(([c, text], i) => (
        // 使用 `truncate`，使宽横幅图案在 Box 边缘裁剪，而不是在窄终端中把每行换成难以阅读的散块。
        <Text color={c} key={i} wrap="truncate">
          {text}
        </Text>
      ))}
    </>
  )
}

// 与 ArtLines 类似，但每行是行内渲染的 `[颜色, 片段]` 对数组，用于横向渐变图案（CodeFlow 主视觉）。
export function ArtRows({ rows }: { rows: [string, string][][] }) {
  return (
    <>
      {rows.map((segs, i) => (
        <Text key={i}>
          {segs.map(([c, text], j) => (
            <Text color={c} key={j}>
              {text}
            </Text>
          ))}
        </Text>
      ))}
    </>
  )
}

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic',
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
  qwen: 'Qwen',
  google: 'Google',
  mistral: 'Mistral'
}

// formatProvider——解析面向用户的提供商标签。
//
// 使用模型短名称回退的原因：用户配置可能设置 `provider="auto"`，即 LiteLLM 自动路由分发模式，
// 它不是真实提供商名称。此时解析模型前缀，例如 "openrouter/qwen/..." → "openrouter"，
// 以找出真实提供商。
//
// 使用查找表的原因：单纯首字母大写会得到视觉错误的 "Openai" / "Openrouter"。
// PROVIDER_LABELS 为已知提供商保留规范大小写，未知提供商回退到普通首字母大写。
export function formatProvider(slug?: string, modelId?: string): string {
  let effective = slug ?? ''
  if (!effective || effective === 'auto') {
    // 仅当模型带 `/` 前缀（如 "openrouter/qwen/qwen3.6-plus"）时才视为携带提供商信息。
    // 裸 "sonnet" 是模型名而非提供商，应回退到 '—'。
    const id = modelId ?? ''
    effective = id.includes('/') ? (id.split('/')[0] ?? '') : ''
  }
  if (!effective) {
    return '—'
  }
  const key = effective.toLowerCase()
  return PROVIDER_LABELS[key] ?? effective.charAt(0).toUpperCase() + effective.slice(1)
}

export function Branding({ t }: { t?: Theme } = {}) {
  const theme = t ?? DEFAULT_THEME

  return (
    <Box marginBottom={1}>
      <Text bold color={theme.color.text}>
        {theme.brand.name}
      </Text>
      <Text color={theme.color.muted}> · agent harness</Text>
    </Box>
  )
}

// 旧调用方 appLayout.tsx 导入 `Banner`，保留该名称可用。
export const Banner = Branding

// ── 可折叠辅助组件 ───────────────────────────────────────────────────

function CollapseToggle({
  count,
  open,
  suffix,
  t,
  title,
  onToggle
}: {
  count?: number
  open: boolean
  suffix?: string
  t: Theme
  title: string
  onToggle: () => void
}) {
  return (
    <Box onClick={onToggle}>
      <Text color={t.color.accent}>{open ? '▾ ' : '▸ '}</Text>
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      {typeof count === 'number' ? <Text color={t.color.muted}> ({count})</Text> : null}
      {suffix ? <Text color={t.color.muted}> {suffix}</Text> : null}
    </Box>
  )
}

// ── 会话面板 ─────────────────────────────────────────────────────────

const SKILLS_MAX = 8
const TOOLSETS_MAX = 8
export function SessionPanel({ info, maxCols, sid, t }: SessionPanelProps) {
  // 面板实际宽度。嵌入更窄容器（如演示画廊侧栏）时，完整终端宽度会超出，因此调用方可传入
  // `maxCols`；否则假定为终端宽度。
  const stdoutCols = useStdout().stdout?.columns ?? 100
  const cols = maxCols ?? stdoutCols
  const w = Math.max(20, cols - 6)
  const lineBudget = Math.max(12, w - 2)
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

  const footerMeta = `${info.model.split('/').pop()} · ${formatProvider(info.provider, info.model)}${sid ? ` · ${sid}` : ''}`

  // ── 各区段的本地折叠状态 ──
  const [toolsOpen, setToolsOpen] = useState(false)
  const [skillsOpen, setSkillsOpen] = useState(false)
  const [systemOpen, setSystemOpen] = useState(false)
  const [mcpOpen, setMcpOpen] = useState(false)

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''
    let shown = 0

    for (const item of [...items].sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > lineBudget) {
        return line ? `${line}, …+${items.length - shown}` : `${item}, …`
      }

      line = next
      shown++
    }

    return line
  }

  // ── 可折叠技能区段 ──
  const skillEntries = Object.entries(info.skills).sort()
  const skillsTotal = flat(info.skills).length
  const skillsCatCount = skillEntries.length

  const skillsBody = () => {
    if (info.lazy && skillEntries.length === 0) {
      return <InlineLoader label="scanning skills" t={t} />
    }

    const shown = skillEntries.slice(0, SKILLS_MAX)
    const overflow = skillEntries.length - SKILLS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && <Text color={t.color.muted}>(and {overflow} more categories…)</Text>}
      </>
    )
  }

  // ── 可折叠工具区段 ──
  const toolEntries = Object.entries(info.tools).sort()
  const toolsTotal = flat(info.tools).length

  const toolsBody = () => {
    const shown = toolEntries.slice(0, TOOLSETS_MAX)
    const overflow = toolEntries.length - TOOLSETS_MAX

    return (
      <>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && <Text color={t.color.muted}>(and {overflow} more toolsets…)</Text>}
      </>
    )
  }

  // ── 可折叠 MCP 区段 ──
  const mcpBody = () => (
    <>
      {(info.mcp_servers ?? []).map(s => (
        <Text key={s.name} wrap="truncate">
          <Text color={t.color.muted}>{`  ${s.name} `}</Text>
          <Text color={t.color.muted}>{`[${s.transport}]`}</Text>
          <Text color={t.color.muted}>: </Text>
          {s.connected ? (
            <Text color={t.color.text}>
              {s.tools} tool{s.tools === 1 ? '' : 's'}
            </Text>
          ) : (
            <Text color={t.color.error}>failed</Text>
          )}
        </Text>
      ))}
    </>
  )

  // ── 系统提示正文 ──
  const sysPromptLen = (info.system_prompt ?? '').length

  const systemBody = () => {
    if (sysPromptLen === 0) {
      return <Text color={t.color.muted}>No system prompt loaded.</Text>
    }

    return <Text color={t.color.muted}>{info.system_prompt}</Text>
  }

  return (
    <Box flexDirection="column" marginBottom={1} width={w}>
      <Box backgroundColor={t.color.surfaceRaised} justifyContent="space-between" paddingX={1}>
        <Text bold color={t.color.text}>
          {t.brand.name}
          {info.version ? ` v${info.version}` : ''}
          {info.release_date ? ` (${info.release_date})` : ''}
        </Text>
        <Text color={t.color.muted}>{footerMeta}</Text>
      </Box>

      <Box paddingX={1}>
        <Text color={t.color.muted} wrap="truncate-end">
          {info.cwd || process.cwd()}
        </Text>
      </Box>

      <Box flexDirection="column" paddingX={1}>
        <Box flexDirection="column">
          <CollapseToggle
            count={toolsTotal}
            onToggle={() => setToolsOpen(v => !v)}
            open={toolsOpen}
            t={t}
            title="Tools"
          />
          {toolsOpen && toolsBody()}
        </Box>

        <Box flexDirection="column">
          <CollapseToggle
            count={skillsTotal}
            onToggle={() => setSkillsOpen(v => !v)}
            open={skillsOpen}
            suffix={
              skillsCatCount > 0 ? `in ${skillsCatCount} categor${skillsCatCount === 1 ? 'y' : 'ies'}` : undefined
            }
            t={t}
            title="Skills"
          />
          {skillsOpen && skillsBody()}
        </Box>

        {sysPromptLen > 0 && (
          <Box flexDirection="column">
            <CollapseToggle
              onToggle={() => setSystemOpen(v => !v)}
              open={systemOpen}
              suffix={`${sysPromptLen.toLocaleString()} chars`}
              t={t}
              title="System Prompt"
            />
            {systemOpen && systemBody()}
          </Box>
        )}

        {info.mcp_servers && info.mcp_servers.length > 0 && (
          <Box flexDirection="column">
            <CollapseToggle
              count={info.mcp_servers.length}
              onToggle={() => setMcpOpen(v => !v)}
              open={mcpOpen}
              suffix="connected"
              t={t}
              title="MCP Servers"
            />
            {mcpOpen && mcpBody()}
          </Box>
        )}

        <Text color={t.color.muted}>
          <Text color={t.color.info}>/help</Text> commands
        </Text>
      </Box>
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box flexDirection="column" paddingX={1} paddingY={1}>
      <Box marginBottom={1}>
        <Text bold color={t.color.heading}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.accent}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.muted}>{k.padEnd(20)}</Text>
              <Text color={t.color.text}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.text} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.muted}>{sec.text}</Text>}
        </Box>
      ))}
    </Box>
  )
}

interface PanelProps {
  sections: PanelSection[]
  t: Theme
  title: string
}

interface SessionPanelProps {
  info: SessionInfo
  // 布局所依据的容器宽度，默认为完整终端。将面板嵌入更窄区域（如演示画廊）时传入。
  maxCols?: number
  sid?: string | null
  t: Theme
}
