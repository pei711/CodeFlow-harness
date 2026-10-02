// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import type { DetailsMode, SectionName, SectionVisibility } from '../types.js'

const MODES = ['hidden', 'collapsed', 'expanded'] as const

export const SECTION_NAMES = ['thinking', 'tools', 'subagents', 'activity'] as const

// 开箱即用的逐区段默认值：用户未固定显式覆盖时应用，层级高于全局 details_mode：
//
//   - thinking / tools：展开——保持流打开，使轮次像实时转录（推理与工具调用并列），而不是每轮
//     都要点击的一墙箭头。
//   - activity：隐藏——环境元数据（网关提示、终端一致性状态 ping 和后台通知）对典型使用是噪声。
//     工具失败仍在失败工具行内渲染；所有面板均隐藏时，环境错误/警告通过浮动警报兜底显示。
//   - subagents：未设置——回退到全局 details_mode，使 spawn 树在真正委派发生前保持在箭头下。
//
// 可在 config.yaml 中用 `display.sections.<name>`，或运行时用
// `/details <name> collapsed|hidden` 退出任一默认值。
const SECTION_DEFAULTS: SectionVisibility = {
  thinking: 'expanded',
  tools: 'expanded',
  activity: 'hidden'
}

const THINKING_FALLBACK: Record<string, DetailsMode> = {
  collapsed: 'collapsed',
  full: 'expanded',
  truncated: 'collapsed'
}

const norm = (v: unknown) =>
  String(v ?? '')
    .trim()
    .toLowerCase()

export const parseDetailsMode = (v: unknown): DetailsMode | null => MODES.find(m => m === norm(v)) ?? null

export const isSectionName = (v: unknown): v is SectionName =>
  typeof v === 'string' && (SECTION_NAMES as readonly string[]).includes(v)

export const resolveDetailsMode = (d?: { details_mode?: unknown; thinking_mode?: unknown } | null): DetailsMode =>
  parseDetailsMode(d?.details_mode) ?? THINKING_FALLBACK[norm(d?.thinking_mode)] ?? 'collapsed'

// 从自由格式数据构建 SectionVisibility。未知区段名和无效模式静默丢弃；部分覆盖是有意设计，
// 缺失键在查找时回退到 SECTION_DEFAULTS 或全局值。
export const resolveSections = (raw: unknown): SectionVisibility =>
  raw && typeof raw === 'object' && !Array.isArray(raw)
    ? (Object.fromEntries(
        Object.entries(raw as Record<string, unknown>)
          .map(([k, v]) => [k, parseDetailsMode(v)] as const)
          .filter(([k, m]) => !!m && isSectionName(k))
      ) as SectionVisibility)
    : {}

// 单个区段的有效模式：显式覆盖 → 全局命令模式 → 内置实时流默认值 → 全局配置模式。
//
// 会话内 `/details <mode>` 变更会设置 `commandOverride`。该命令应立即应用到所有区段，包括
// thinking/tools=expanded、activity=hidden 等内置默认值区段。启动/配置同步时，这些默认值仍
// 位于持久化全局配置之上，使 TUI 默认打开实时推理/工具，除非用户固定显式逐区段覆盖。
export const sectionMode = (
  name: SectionName,
  global: DetailsMode,
  sections?: SectionVisibility,
  commandOverride = false
): DetailsMode => sections?.[name] ?? (commandOverride ? global : (SECTION_DEFAULTS[name] ?? global))

export const nextDetailsMode = (m: DetailsMode): DetailsMode => MODES[(MODES.indexOf(m) + 1) % MODES.length]!
