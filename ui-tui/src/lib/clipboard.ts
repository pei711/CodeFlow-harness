// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { execFile, spawn } from 'node:child_process'
import { readdirSync, rmSync } from 'node:fs'
import { mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)
const CLIPBOARD_MAX_BUFFER = 4 * 1024 * 1024
const CLIPBOARD_IMAGE_MAX_BUFFER = 32 * 1024 * 1024
const CLIPBOARD_READ_TIMEOUT_MS = 5000
const CLIPBOARD_IMAGE_DIR_RE = /^codeflow-clipboard-(\d+)-/
const POWERSHELL_ARGS = ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'] as const
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
const MACOS_IMAGE_SCRIPT = [
  "ObjC.import('AppKit')",
  'function run() {',
  '  const image = $.NSImage.alloc.initWithPasteboard($.NSPasteboard.generalPasteboard)',
  "  if (!image || image.isNil()) throw new Error('no clipboard image')",
  '  const bitmap = $.NSBitmapImageRep.imageRepWithData(image.TIFFRepresentation)',
  "  if (!bitmap || bitmap.isNil()) throw new Error('clipboard image conversion failed')",
  '  const data = bitmap.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $({}))',
  '  return ObjC.unwrap(data.base64EncodedStringWithOptions(0))',
  '}'
].join('\n')
const WINDOWS_IMAGE_SCRIPT = [
  'Add-Type -AssemblyName System.Windows.Forms',
  'Add-Type -AssemblyName System.Drawing',
  '$image = [Windows.Forms.Clipboard]::GetImage()',
  'if ($null -eq $image) { exit 1 }',
  '$stream = New-Object IO.MemoryStream',
  'try {',
  '  $image.Save($stream, [Drawing.Imaging.ImageFormat]::Png)',
  '  [Convert]::ToBase64String($stream.ToArray())',
  '} finally {',
  '  $stream.Dispose()',
  '  $image.Dispose()',
  '}'
].join('; ')
const clipboardImageDirs = new Map<string, string>()
const clipboardImagesBySession = new Map<string, Set<string>>()
const clipboardImageSessions = new Map<string, string>()
const submittedClipboardImagesBySession = new Map<string, Map<string, Set<string>>>()
let clipboardClaimSequence = 0

type ClipboardRun = typeof execFileAsync
type ClipboardImageRun = (
  command: string,
  args: string[],
  options: {
    encoding: 'buffer'
    maxBuffer: number
    timeout: number
    windowsHide: boolean
  }
) => Promise<{ stdout: Buffer | string }>

interface ClipboardImageAttempt {
  args: string[]
  base64: boolean
  cmd: string
}

function pngCrc32(parts: Buffer[]): number {
  let crc = 0xffffffff

  for (const part of parts) {
    for (const byte of part) {
      crc ^= byte

      for (let bit = 0; bit < 8; bit += 1) {
        crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
      }
    }
  }

  return (crc ^ 0xffffffff) >>> 0
}

export function isUsableClipboardText(text: null | string): text is string {
  if (!text || !/[^\s]/.test(text)) {
    return false
  }

  if (text.includes('\u0000')) {
    return false
  }

  let suspicious = 0

  for (const ch of text) {
    const code = ch.charCodeAt(0)
    const isControl = code < 0x20 && ch !== '\n' && ch !== '\r' && ch !== '\t'

    if (isControl || ch === '\ufffd') {
      suspicious += 1
    }
  }

  return suspicious <= Math.max(2, Math.floor(text.length * 0.02))
}

function readClipboardImageCommands(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): ClipboardImageAttempt[] {
  if (platform === 'darwin') {
    return [{ args: ['-l', 'JavaScript', '-e', MACOS_IMAGE_SCRIPT], base64: true, cmd: 'osascript' }]
  }

  if (platform === 'win32') {
    return [
      {
        args: ['-NoProfile', '-NonInteractive', '-Command', WINDOWS_IMAGE_SCRIPT],
        base64: true,
        cmd: 'powershell'
      }
    ]
  }

  const attempts: ClipboardImageAttempt[] = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({
      args: ['-NoProfile', '-NonInteractive', '-Command', WINDOWS_IMAGE_SCRIPT],
      base64: true,
      cmd: 'powershell.exe'
    })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ args: ['--no-newline', '--type', 'image/png'], base64: false, cmd: 'wl-paste' })
  }

  attempts.push({
    args: ['-selection', 'clipboard', '-t', 'image/png', '-o'],
    base64: false,
    cmd: 'xclip'
  })

  return attempts
}

function decodeClipboardImage(stdout: Buffer | string, base64: boolean): Buffer | null {
  const raw = Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout)
  const image = base64 ? Buffer.from(raw.toString('utf8').trim(), 'base64') : raw

  if (image.length > CLIPBOARD_IMAGE_MAX_BUFFER || !image.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    return null
  }

  let offset = PNG_SIGNATURE.length
  let sawHeader = false
  let sawImageData = false
  let finishedImageData = false
  let imageDataSize = 0

  while (offset + 12 <= image.length) {
    const dataLength = image.readUInt32BE(offset)
    const chunkEnd = offset + 12 + dataLength

    if (chunkEnd > image.length) {
      return null
    }

    const type = image.subarray(offset + 4, offset + 8)
    const data = image.subarray(offset + 8, offset + 8 + dataLength)
    const expectedCrc = image.readUInt32BE(offset + 8 + dataLength)

    if (pngCrc32([type, data]) !== expectedCrc) {
      return null
    }

    const chunkType = type.toString('ascii')

    if (!sawHeader) {
      if (chunkType !== 'IHDR' || dataLength !== 13 || data.readUInt32BE(0) === 0 || data.readUInt32BE(4) === 0) {
        return null
      }

      sawHeader = true
    } else if (chunkType === 'IHDR') {
      return null
    }

    if (chunkType === 'IDAT') {
      if (finishedImageData) {
        return null
      }

      imageDataSize += dataLength

      sawImageData = true
    } else if (sawImageData) {
      finishedImageData = true
    }

    if (chunkType === 'IEND') {
      if (dataLength !== 0 || !sawHeader || !sawImageData || imageDataSize === 0 || chunkEnd !== image.length) {
        return null
      }

      return image
    }

    offset = chunkEnd
  }

  return null
}

export async function readClipboardImage(
  platform: NodeJS.Platform = process.platform,
  run: ClipboardImageRun = execFileAsync as unknown as ClipboardImageRun,
  env: NodeJS.ProcessEnv = process.env
): Promise<string | null> {
  for (const attempt of readClipboardImageCommands(platform, env)) {
    try {
      const result = await run(attempt.cmd, attempt.args, {
        encoding: 'buffer',
        maxBuffer: CLIPBOARD_IMAGE_MAX_BUFFER,
        timeout: CLIPBOARD_READ_TIMEOUT_MS,
        windowsHide: true
      })
      const image = decodeClipboardImage(result.stdout, attempt.base64)

      if (!image) {
        continue
      }

      const dir = await mkdtemp(join(tmpdir(), `codeflow-clipboard-${process.pid}-`))
      const path = join(dir, 'clipboard.png')

      try {
        await writeFile(path, image)
      } catch (error) {
        rmSync(dir, { force: true, recursive: true })
        throw error
      }

      clipboardImageDirs.set(path, dir)

      return path
    } catch {
      continue
    }
  }

  return null
}

export function retainClipboardImage(path: string, sessionId: string): boolean {
  if (!clipboardImageDirs.has(path)) {
    return false
  }

  const previousSession = clipboardImageSessions.get(path)

  if (previousSession && previousSession !== sessionId) {
    const previousPaths = clipboardImagesBySession.get(previousSession)
    previousPaths?.delete(path)

    if (!previousPaths?.size) {
      clipboardImagesBySession.delete(previousSession)
    }
  }

  clipboardImageSessions.set(path, sessionId)

  const paths = clipboardImagesBySession.get(sessionId) ?? new Set<string>()
  paths.add(path)
  clipboardImagesBySession.set(sessionId, paths)

  return true
}

export function releaseClipboardImage(path: string): void {
  const dir = clipboardImageDirs.get(path)

  if (!dir) {
    return
  }

  clipboardImageDirs.delete(path)

  const sessionId = clipboardImageSessions.get(path)
  clipboardImageSessions.delete(path)

  if (sessionId) {
    const paths = clipboardImagesBySession.get(sessionId)
    paths?.delete(path)
    const submittedPaths = submittedClipboardImagesBySession.get(sessionId)
    submittedPaths?.forEach(claim => claim.delete(path))

    if (!paths?.size) {
      clipboardImagesBySession.delete(sessionId)
    }
  }

  try {
    rmSync(dir, { force: true, recursive: true })
  } catch {
    // 清理操作不能改变 TUI 进程的退出状态。
  }
}

export function releaseClipboardImages(sessionId: string): void {
  const paths = clipboardImagesBySession.get(sessionId)

  if (paths) {
    for (const path of [...paths]) {
      releaseClipboardImage(path)
    }
  }

  submittedClipboardImagesBySession.delete(sessionId)
}

export function claimClipboardImages(sessionId: string): string {
  const paths = clipboardImagesBySession.get(sessionId)
  const claims = submittedClipboardImagesBySession.get(sessionId) ?? new Map<string, Set<string>>()
  const alreadyClaimed = new Set([...claims.values()].flatMap(claim => [...claim]))
  const claim = new Set([...(paths ?? [])].filter(path => !alreadyClaimed.has(path)))
  const claimId = `clipboard-claim-${++clipboardClaimSequence}`

  claims.set(claimId, claim)
  submittedClipboardImagesBySession.set(sessionId, claims)

  return claimId
}

export function unclaimClipboardImages(sessionId: string, claimId?: string): void {
  const claims = submittedClipboardImagesBySession.get(sessionId)

  if (!claims) {
    return
  }

  if (claimId) {
    claims.delete(claimId)
  } else {
    const newestClaimId = [...claims.keys()].at(-1)
    if (newestClaimId) {
      claims.delete(newestClaimId)
    }
  }

  if (!claims.size) {
    submittedClipboardImagesBySession.delete(sessionId)
  }
}

export function releaseSubmittedClipboardImages(sessionId: string, claimId?: string): void {
  const claims = submittedClipboardImagesBySession.get(sessionId)
  const targetClaimId = claimId ?? claims?.keys().next().value

  if (!targetClaimId) {
    return
  }

  const paths = claims?.get(targetClaimId)

  if (!paths) {
    return
  }

  claims?.delete(targetClaimId)

  for (const path of [...paths]) {
    releaseClipboardImage(path)
  }

  if (!claims?.size) {
    submittedClipboardImagesBySession.delete(sessionId)
  }
}

export function cleanupClipboardImages(): void {
  for (const path of [...clipboardImageDirs.keys()]) {
    releaseClipboardImage(path)
  }

  clipboardImageDirs.clear()
  clipboardImagesBySession.clear()
  clipboardImageSessions.clear()
  submittedClipboardImagesBySession.clear()
  clipboardClaimSequence = 0
}

const isProcessAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0)

    return true
  } catch (error) {
    return !(error instanceof Error && 'code' in error && error.code === 'ESRCH')
  }
}

export function cleanupStaleClipboardImages(
  root: string = tmpdir(),
  processAlive: (pid: number) => boolean = isProcessAlive
): void {
  let entries

  try {
    entries = readdirSync(root, { withFileTypes: true })
  } catch {
    return
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }

    const match = CLIPBOARD_IMAGE_DIR_RE.exec(entry.name)

    if (!match || processAlive(Number(match[1]))) {
      continue
    }

    try {
      rmSync(join(root, entry.name), { force: true, recursive: true })
    } catch {
      // 清理操作不能改变 TUI 进程的启动状态。
    }
  }
}

function readClipboardCommands(
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv
): Array<{ args: readonly string[]; cmd: string }> {
  if (platform === 'darwin') {
    return [{ cmd: 'pbpaste', args: [] }]
  }

  if (platform === 'win32') {
    return [{ cmd: 'powershell', args: POWERSHELL_ARGS }]
  }

  const attempts: Array<{ args: readonly string[]; cmd: string }> = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({ cmd: 'powershell.exe', args: POWERSHELL_ARGS })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: 'wl-paste', args: ['--type', 'text'] })
  }

  attempts.push({ cmd: 'xclip', args: ['-selection', 'clipboard', '-out'] })

  return attempts
}

/**
 * 从系统剪贴板读取纯文本。
 *
 * 按以下回退顺序使用平台原生工具：
 * - macOS: pbpaste
 * - Windows: PowerShell Get-Clipboard -Raw
 * - WSL: powershell.exe Get-Clipboard -Raw
 * - Linux Wayland: wl-paste --type text
 * - Linux X11: xclip -selection clipboard -out
 */
export async function readClipboardText(
  platform: NodeJS.Platform = process.platform,
  run: ClipboardRun = execFileAsync,
  env: NodeJS.ProcessEnv = process.env
): Promise<string | null> {
  for (const attempt of readClipboardCommands(platform, env)) {
    try {
      const result = await run(attempt.cmd, [...attempt.args], {
        encoding: 'utf8',
        maxBuffer: CLIPBOARD_MAX_BUFFER,
        timeout: CLIPBOARD_READ_TIMEOUT_MS,
        windowsHide: true
      })

      if (typeof result.stdout === 'string') {
        return result.stdout
      }
    } catch {
      // 回退到下一个剪贴板后端。
    }
  }

  return null
}

function writeClipboardCommands(
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv
): Array<{ args: readonly string[]; cmd: string }> {
  if (platform === 'darwin') {
    return [{ cmd: 'pbcopy', args: [] }]
  }

  if (platform === 'win32') {
    return [{ cmd: 'powershell', args: ['-NoProfile', '-NonInteractive', '-Command', 'Set-Clipboard -Value $input'] }]
  }

  const attempts: Array<{ args: readonly string[]; cmd: string }> = []

  if (env.WSL_INTEROP || env.WSL_DISTRO_NAME) {
    attempts.push({
      cmd: 'powershell.exe',
      args: ['-NoProfile', '-NonInteractive', '-Command', 'Set-Clipboard -Value $input']
    })
  }

  if (env.WAYLAND_DISPLAY) {
    attempts.push({ cmd: 'wl-copy', args: ['--type', 'text/plain'] })
  }

  attempts.push({ cmd: 'xclip', args: ['-selection', 'clipboard', '-in'] })
  attempts.push({ cmd: 'xsel', args: ['--clipboard', '--input'] })

  return attempts
}

/**
 * 向系统剪贴板写入纯文本。
 *
 * 按以下回退顺序尝试平台原生工具：
 * - macOS: pbcopy
 * - Windows: PowerShell Set-Clipboard
 * - WSL: powershell.exe Set-Clipboard
 * - Linux Wayland: wl-copy --type text/plain
 * - Linux X11: xclip -selection clipboard -in
 * - Linux X11 备用方案：xsel --clipboard --input
 *
 * 至少一个后端成功时返回 true，否则返回 false；返回 false 时调用方应回退到
 * OSC52。
 */
export async function writeClipboardText(
  text: string,
  platform: NodeJS.Platform = process.platform,
  start: typeof spawn = spawn,
  env: NodeJS.ProcessEnv = process.env
): Promise<boolean> {
  const candidates = writeClipboardCommands(platform, env)

  for (const { cmd, args } of candidates) {
    try {
      const ok = await new Promise<boolean>(resolve => {
        const child = start(cmd, [...args], { stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })

        child.once('error', () => resolve(false))
        child.once('close', code => resolve(code === 0))
        child.stdin?.end(text)
      })

      if (ok) {
        return true
      }
    } catch {
      // 回退到下一个剪贴板后端。
    }
  }

  return false
}
