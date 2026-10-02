import { homedir } from 'node:os'
import { join } from 'node:path'

export const getCodeFlowHome = (env: NodeJS.ProcessEnv = process.env) => env.CODEFLOW_HOME?.trim() || join(homedir(), '.codeflow')

export const getCodeFlowHomeLabel = (env: NodeJS.ProcessEnv = process.env) => env.CODEFLOW_HOME?.trim() || '~/.codeflow'
