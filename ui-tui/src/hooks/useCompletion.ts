// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { useEffect, useRef, useState } from 'react'

import type { CompletionItem } from '../app/interfaces.js'
import type { SlashCommand } from '../app/slash/types.js'

import { SLASH_COMMANDS } from '../app/slash/registry.js'
import { looksLikeSlashCommand } from '../domain/slash.js'

export function slashCompletions(input: string, commands: SlashCommand[]): CompletionItem[] {
  if (!looksLikeSlashCommand(input)) {
    return []
  }

  // 用户越过命令名开始输入参数（空格后）时已无内容可补全，应关闭面板而不是继续
  // 悬浮在 `/cmd arg` 上方。
  if (/\s/.test(input.slice(1))) {
    return []
  }

  const token = input.slice(1).split(/\s/)[0]!.toLowerCase()
  const seen = new Set<string>()
  const items: CompletionItem[] = []

  for (const cmd of commands) {
    if (seen.has(cmd.name)) {
      continue
    }

    const nameMatch = cmd.name.startsWith(token)
    const aliasMatch = !nameMatch && (cmd.aliases ?? []).some(a => a.startsWith(token))

    if (nameMatch || aliasMatch) {
      seen.add(cmd.name)

      const item: CompletionItem = { display: `/${cmd.name}`, text: `/${cmd.name}` }

      if (cmd.help) {
        item.meta = cmd.help
      }

      items.push(item)
    }
  }

  return items
}

export function useCompletion(input: string, blocked: boolean) {
  const [completions, setCompletions] = useState<CompletionItem[]>([])
  const [compIdx, setCompIdx] = useState(0)
  const [compReplace, setCompReplace] = useState(0)
  const ref = useRef('')

  useEffect(() => {
    const clear = () => {
      setCompletions(prev => (prev.length ? [] : prev))
      setCompIdx(prev => (prev ? 0 : prev))
      setCompReplace(prev => (prev ? 0 : prev))
    }

    if (blocked) {
      ref.current = ''
      clear()

      return
    }

    if (input === ref.current) {
      return
    }

    ref.current = input

    if (looksLikeSlashCommand(input)) {
      const items = slashCompletions(input, SLASH_COMMANDS)

      if (items.length) {
        setCompletions(items)
        setCompIdx(0)
        setCompReplace(1)
      } else {
        clear()
      }

      return
    }

    clear()
  }, [blocked, input])

  return { completions, compIdx, setCompIdx, compReplace }
}
