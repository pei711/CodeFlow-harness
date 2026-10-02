import type { SlashCommand } from './types.js'

import { coreCommands } from './commands/core.js'
import { sessionCommands } from './commands/session.js'

export const SLASH_COMMANDS: SlashCommand[] = [...coreCommands, ...sessionCommands]

const byName = new Map<string, SlashCommand>(
  SLASH_COMMANDS.flatMap(cmd => [cmd.name, ...(cmd.aliases ?? [])].map(name => [name, cmd] as const))
)

export const findSlashCommand = (name: string) => byName.get(name.toLowerCase())
