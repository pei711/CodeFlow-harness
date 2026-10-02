// SPDX-License-Identifier: MIT
// Portions Copyright (c) 2025 Nous Research (hermes-agent, MIT).
// Modifications Copyright (c) 2026 EverMind.
// See NOTICES.md and LICENSES/MIT-hermes-agent.txt.

import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { trackPasteFlight } from '../app/useComposerState.js'
import { pasteClipboardImage } from '../app/useMainApp.js'
import {
  continueAfterCancel,
  dispatchNextQueuedSubmission,
  pauseBusySubmission,
  resolveQueuedEdit,
  submitUnlessPastePending
} from '../app/useSubmission.js'
import { asyncPasteFallbackText } from '../components/textInput.js'
import {
  claimClipboardImages,
  cleanupClipboardImages,
  cleanupStaleClipboardImages,
  isUsableClipboardText,
  readClipboardImage,
  readClipboardText,
  releaseClipboardImage,
  releaseClipboardImages,
  releaseSubmittedClipboardImages,
  retainClipboardImage,
  unclaimClipboardImages,
  writeClipboardText
} from '../lib/clipboard.js'

const PNG_BYTES = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64'
)
const EMPTY_IDAT_PNG_BYTES = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAElEQVQ1rwYeAAAAAElFTkSuQmCC',
  'base64'
)

afterEach(() => {
  vi.useRealTimers()
  resetUiState()
})

describe('readClipboardImage', () => {
  it('materializes a macOS clipboard PNG for image.attach', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: Buffer.from(PNG_BYTES.toString('base64')) })
    const path = await readClipboardImage('darwin', run)

    expect(path).not.toBeNull()

    try {
      await expect(readFile(path!)).resolves.toEqual(PNG_BYTES)
      expect(run).toHaveBeenCalledWith(
        'osascript',
        expect.arrayContaining(['-l', 'JavaScript']),
        expect.objectContaining({
          encoding: 'buffer',
          maxBuffer: 32 * 1024 * 1024,
          timeout: 5000,
          windowsHide: true
        })
      )
    } finally {
      cleanupClipboardImages()
    }

    await expect(readFile(path!)).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('falls back from Wayland to X11 for a clipboard PNG', async () => {
    const run = vi
      .fn()
      .mockRejectedValueOnce(new Error('wl-paste unavailable'))
      .mockResolvedValueOnce({ stdout: PNG_BYTES })
    const path = await readClipboardImage('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)

    expect(path).not.toBeNull()

    try {
      await expect(readFile(path!)).resolves.toEqual(PNG_BYTES)
      expect(run).toHaveBeenNthCalledWith(
        1,
        'wl-paste',
        ['--no-newline', '--type', 'image/png'],
        expect.objectContaining({ encoding: 'buffer' })
      )
      expect(run).toHaveBeenNthCalledWith(
        2,
        'xclip',
        ['-selection', 'clipboard', '-t', 'image/png', '-o'],
        expect.objectContaining({ encoding: 'buffer' })
      )
    } finally {
      cleanupClipboardImages()
    }
  })

  it('materializes a Windows clipboard PNG from PowerShell', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: Buffer.from(PNG_BYTES.toString('base64')) })
    const path = await readClipboardImage('win32', run)

    expect(path).not.toBeNull()

    try {
      await expect(readFile(path!)).resolves.toEqual(PNG_BYTES)
      expect(run).toHaveBeenCalledWith(
        'powershell',
        expect.arrayContaining(['-NoProfile', '-NonInteractive', '-Command']),
        expect.objectContaining({ encoding: 'buffer' })
      )
    } finally {
      cleanupClipboardImages()
    }
  })

  it('returns null when clipboard backends provide no PNG', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: Buffer.from('not an image') })

    await expect(readClipboardImage('linux', run, {})).resolves.toBeNull()
  })

  it('rejects a truncated payload that only carries the PNG signature', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: Buffer.concat([PNG_BYTES.subarray(0, 8), Buffer.from('broken')]) })

    await expect(readClipboardImage('linux', run, {})).resolves.toBeNull()
  })

  it('rejects a CRC-valid PNG with an empty IDAT payload', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: EMPTY_IDAT_PNG_BYTES })

    try {
      await expect(readClipboardImage('linux', run, {})).resolves.toBeNull()
    } finally {
      cleanupClipboardImages()
    }
  })

  it('releases retained clipboard images for one Session', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: PNG_BYTES })
    const path = await readClipboardImage('linux', run, {})

    expect(path).not.toBeNull()
    expect(retainClipboardImage(path!, 'tui:active')).toBe(true)

    releaseClipboardImages('tui:active')

    await expect(readFile(path!)).rejects.toMatchObject({ code: 'ENOENT' })
    expect(retainClipboardImage(path!, 'tui:active')).toBe(false)
  })

  it('releases a single owned clipboard image', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: PNG_BYTES })
    const path = await readClipboardImage('linux', run, {})

    expect(path).not.toBeNull()
    releaseClipboardImage(path!)

    await expect(readFile(path!)).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('releases submitted clipboard images in Turn order', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: PNG_BYTES })
    const first = await readClipboardImage('linux', run, {})
    expect(first).not.toBeNull()
    expect(retainClipboardImage(first!, 'tui:active')).toBe(true)
    claimClipboardImages('tui:active')

    const second = await readClipboardImage('linux', run, {})
    expect(second).not.toBeNull()
    expect(retainClipboardImage(second!, 'tui:active')).toBe(true)
    claimClipboardImages('tui:active')

    releaseSubmittedClipboardImages('tui:active')
    await expect(readFile(first!)).rejects.toMatchObject({ code: 'ENOENT' })
    await expect(readFile(second!)).resolves.toEqual(PNG_BYTES)

    releaseSubmittedClipboardImages('tui:active')
    await expect(readFile(second!)).rejects.toMatchObject({ code: 'ENOENT' })
  })

  it('keeps images from a rejected submission available for retry', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: PNG_BYTES })
    const path = await readClipboardImage('linux', run, {})
    expect(path).not.toBeNull()
    expect(retainClipboardImage(path!, 'tui:active')).toBe(true)

    const claimId = claimClipboardImages('tui:active')
    unclaimClipboardImages('tui:active', claimId)
    releaseSubmittedClipboardImages('tui:active')

    await expect(readFile(path!)).resolves.toEqual(PNG_BYTES)
    releaseClipboardImages('tui:active')
  })

  it('settles clipboard claims by id when submissions finish out of order', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: PNG_BYTES })
    const first = await readClipboardImage('linux', run, {})
    expect(first).not.toBeNull()
    expect(retainClipboardImage(first!, 'tui:active')).toBe(true)
    const firstClaim = claimClipboardImages('tui:active')

    const second = await readClipboardImage('linux', run, {})
    expect(second).not.toBeNull()
    expect(retainClipboardImage(second!, 'tui:active')).toBe(true)
    const secondClaim = claimClipboardImages('tui:active')

    unclaimClipboardImages('tui:active', firstClaim)
    releaseSubmittedClipboardImages('tui:active', secondClaim)

    await expect(readFile(first!)).resolves.toEqual(PNG_BYTES)
    await expect(readFile(second!)).rejects.toMatchObject({ code: 'ENOENT' })

    releaseClipboardImages('tui:active')
  })

  it('cleans clipboard directories left by dead TUI processes', async () => {
    const root = await mkdtemp(join(tmpdir(), 'codeflow-clipboard-cleanup-test-'))
    const dead = join(root, 'codeflow-clipboard-12345-dead')
    const alive = join(root, 'codeflow-clipboard-67890-alive')

    try {
      await mkdir(dead)
      await mkdir(alive)

      cleanupStaleClipboardImages(root, pid => pid === 67890)

      await expect(readFile(join(dead, 'clipboard.png'))).rejects.toMatchObject({ code: 'ENOENT' })
      await expect(readFile(join(alive, 'clipboard.png'))).rejects.toMatchObject({ code: 'ENOENT' })
      await expect(mkdir(dead)).resolves.toBeUndefined()
      await expect(mkdir(alive)).rejects.toMatchObject({ code: 'EEXIST' })
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  })
})

describe('pasteClipboardImage', () => {
  it('routes a materialized clipboard image through image.attach', async () => {
    const rpc = vi.fn().mockResolvedValue({ name: 'clipboard.png', remainder: '' })
    const readImage = vi.fn().mockResolvedValue('/tmp/codeflow-clipboard-test/clipboard.png')

    await expect(pasteClipboardImage(rpc, 'tui:active', readImage, () => 'tui:active')).resolves.toEqual({
      info: { name: 'clipboard.png', remainder: '' },
      status: 'attached'
    })
    expect(rpc).toHaveBeenCalledWith('image.attach', {
      path: '/tmp/codeflow-clipboard-test/clipboard.png',
      session_id: 'tui:active'
    })
  })

  it('does not inspect the clipboard without an active Session', async () => {
    const rpc = vi.fn()
    const readImage = vi.fn()

    await expect(pasteClipboardImage(rpc, null, readImage)).resolves.toEqual({ status: 'no-session' })
    expect(readImage).not.toHaveBeenCalled()
    expect(rpc).not.toHaveBeenCalled()
  })

  it('does not call image.attach when the clipboard has no image', async () => {
    const rpc = vi.fn()
    const readImage = vi.fn().mockResolvedValue(null)

    await expect(pasteClipboardImage(rpc, 'tui:active', readImage, () => 'tui:active')).resolves.toEqual({
      status: 'empty'
    })
    expect(rpc).not.toHaveBeenCalled()
  })

  it('does not inspect the clipboard while the current Turn is active', async () => {
    const rpc = vi.fn()
    const readImage = vi.fn()

    await expect(
      pasteClipboardImage(
        rpc,
        'tui:active',
        readImage,
        () => 'tui:active',
        vi.fn(),
        vi.fn(),
        () => false
      )
    ).resolves.toEqual({ status: 'busy' })
    expect(readImage).not.toHaveBeenCalled()
    expect(rpc).not.toHaveBeenCalled()
  })

  it('discards an image when a Turn starts during clipboard materialization', async () => {
    let canAttach = true
    const rpc = vi.fn()
    const readImage = vi.fn().mockImplementation(async () => {
      canAttach = false

      return '/tmp/codeflow-clipboard-test/clipboard.png'
    })
    const releaseImage = vi.fn()

    await expect(
      pasteClipboardImage(
        rpc,
        'tui:active',
        readImage,
        () => 'tui:active',
        releaseImage,
        vi.fn(),
        () => canAttach
      )
    ).resolves.toEqual({ status: 'busy' })
    expect(releaseImage).toHaveBeenCalledWith('/tmp/codeflow-clipboard-test/clipboard.png')
    expect(rpc).not.toHaveBeenCalled()
  })

  it('discards a clipboard read when the active Session changes before attach', async () => {
    let resolveRead: (path: string) => void = () => undefined
    let activeSession = 'tui:active'
    const readImage = vi.fn(
      () =>
        new Promise<string>(resolve => {
          resolveRead = resolve
        })
    )
    const rpc = vi.fn()
    const releaseImage = vi.fn()
    const operation = pasteClipboardImage(rpc, 'tui:active', readImage, () => activeSession, releaseImage)

    activeSession = 'tui:other'
    resolveRead('/tmp/codeflow-clipboard-test/clipboard.png')

    await expect(operation).resolves.toEqual({ status: 'stale' })
    expect(releaseImage).toHaveBeenCalledWith('/tmp/codeflow-clipboard-test/clipboard.png')
    expect(rpc).not.toHaveBeenCalled()
  })

  it('releases the clipboard image when image.attach rejects', async () => {
    const rpc = vi.fn().mockRejectedValue(new Error('transport closed'))
    const readImage = vi.fn().mockResolvedValue('/tmp/codeflow-clipboard-test/clipboard.png')
    const releaseImage = vi.fn()
    const retainImage = vi.fn()

    await expect(
      pasteClipboardImage(rpc, 'tui:active', readImage, () => 'tui:active', releaseImage, retainImage)
    ).resolves.toEqual({ status: 'failed' })
    expect(retainImage).toHaveBeenCalledWith('/tmp/codeflow-clipboard-test/clipboard.png', 'tui:active')
    expect(releaseImage).toHaveBeenCalledWith('/tmp/codeflow-clipboard-test/clipboard.png')
  })

  it('releases a retained image when the Session changes during image.attach', async () => {
    let resolveAttach: (info: { name: string; remainder: string }) => void = () => undefined
    let activeSession = 'tui:active'
    const rpc = vi.fn(
      () =>
        new Promise<{ name: string; remainder: string }>(resolve => {
          resolveAttach = resolve
        })
    )
    const path = '/tmp/codeflow-clipboard-test/clipboard.png'
    const releaseImage = vi.fn()
    const retainImage = vi.fn()
    const operation = pasteClipboardImage(
      rpc,
      'tui:active',
      vi.fn().mockResolvedValue(path),
      () => activeSession,
      releaseImage,
      retainImage
    )

    await vi.waitFor(() => expect(rpc).toHaveBeenCalledOnce())
    activeSession = 'tui:other'
    resolveAttach({ name: 'clipboard.png', remainder: '' })

    await expect(operation).resolves.toEqual({ status: 'stale' })
    expect(retainImage).toHaveBeenCalledWith(path, 'tui:active')
    expect(releaseImage).toHaveBeenCalledWith(path)
  })

  it('blocks submission until an in-flight clipboard paste settles', async () => {
    const events: string[] = []
    let resolvePaste: () => void = () => undefined
    const paste = new Promise<void>(resolve => {
      resolvePaste = () => {
        events.push('attached')
        resolve()
      }
    })
    const pasteFlightRef = { current: null as Promise<void> | null }
    void trackPasteFlight(pasteFlightRef, paste)

    expect(events).toEqual([])
    expect(submitUnlessPastePending(pasteFlightRef.current, () => events.push('early'))).toBe(false)

    resolvePaste()
    await pasteFlightRef.current

    expect(submitUnlessPastePending(pasteFlightRef.current, () => events.push('sent'))).toBe(true)
    expect(events).toEqual(['attached', 'sent'])
  })

  it('restores a queued submission when clipboard paste blocks dispatch', () => {
    const queue = [
      {
        alreadyDisplayed: true,
        paused: true,
        submitText: 'expanded queued message',
        text: 'queued message'
      },
      { alreadyDisplayed: false, paused: false, text: 'later message' }
    ]
    const actions = {
      dequeue: vi.fn(() => queue.shift()),
      prependQueue: vi.fn(entry => queue.unshift(entry)),
      setQueueEdit: vi.fn()
    }
    const dispatchSubmission = vi.fn(() => false)

    expect(dispatchNextQueuedSubmission(actions, dispatchSubmission)).toBe(false)
    expect(queue.map(entry => entry.text)).toEqual(['queued message', 'later message'])
    expect(dispatchSubmission).toHaveBeenCalledWith('queued message', false, 'expanded queued message', undefined)
    expect(actions.setQueueEdit).toHaveBeenCalledWith(null)
    expect(actions.prependQueue).toHaveBeenCalledOnce()
  })

  it('requeues without sending when an interrupt cancellation fails', async () => {
    const send = vi.fn()
    const requeue = vi.fn()

    await continueAfterCancel(Promise.reject(new Error('cancel transport unavailable')), send, requeue)

    expect(send).not.toHaveBeenCalled()
    expect(requeue).toHaveBeenCalledWith(expect.objectContaining({ message: 'cancel transport unavailable' }))
  })

  it.each([
    ['a busy turn', { busy: true }],
    ['a Session Switch Flight', { sessionSwitching: true }],
    ['a Session Mutation Flight', { sessionMutating: true }]
  ])('does not dequeue while %s is active', (_label, state) => {
    patchUiState(state)
    const actions = {
      dequeue: vi.fn(),
      prependQueue: vi.fn(),
      setQueueEdit: vi.fn()
    }
    const dispatchSubmission = vi.fn(() => true)

    expect(dispatchNextQueuedSubmission(actions, dispatchSubmission)).toBe(false)
    expect(actions.dequeue).not.toHaveBeenCalled()
    expect(dispatchSubmission).not.toHaveBeenCalled()
  })

  it('pauses a busy submission without overwriting a newer composer draft', () => {
    const queue = [{ alreadyDisplayed: false, paused: false, text: 'later message' }]
    const actions = {
      prependQueue: vi.fn(entry => queue.unshift(entry))
    }

    pauseBusySubmission(actions, 'retry label', 'retry expanded')

    expect(queue).toEqual([
      {
        alreadyDisplayed: true,
        pasteSnips: undefined,
        paused: true,
        submitText: 'retry expanded',
        text: 'retry label'
      },
      { alreadyDisplayed: false, paused: false, text: 'later message' }
    ])
  })

  it('re-expands an edited queued paste from its stored snippet provenance', () => {
    const pasteSnips = [{ label: '[[Pasted Content 1]]', text: 'complete implementation payload' }]
    const entry = {
      alreadyDisplayed: false,
      pasteSnips,
      paused: false,
      submitText: 'review complete implementation payload',
      text: 'review [[Pasted Content 1]]'
    }

    expect(resolveQueuedEdit(entry, 'carefully review [[Pasted Content 1]]', [])).toEqual({
      pasteSnips,
      submitText: 'carefully review complete implementation payload'
    })
    expect(resolveQueuedEdit(entry, entry.text, [])).toEqual({
      pasteSnips,
      submitText: 'review complete implementation payload'
    })
  })

  it('keeps submission blocked until overlapping paste flights both settle', async () => {
    let resolveFirst: () => void = () => undefined
    let resolveSecond: () => void = () => undefined
    const first = new Promise<void>(resolve => {
      resolveFirst = resolve
    })
    const second = new Promise<void>(resolve => {
      resolveSecond = resolve
    })
    const pasteFlightRef = { current: null as Promise<void> | null }
    const onIdle = vi.fn()

    void trackPasteFlight(pasteFlightRef, first, onIdle)
    void trackPasteFlight(pasteFlightRef, second, onIdle)

    expect(submitUnlessPastePending(pasteFlightRef.current, () => undefined)).toBe(false)

    resolveSecond()
    await second

    expect(submitUnlessPastePending(pasteFlightRef.current, () => undefined)).toBe(false)
    expect(onIdle).not.toHaveBeenCalled()

    resolveFirst()
    await pasteFlightRef.current

    expect(submitUnlessPastePending(pasteFlightRef.current, () => undefined)).toBe(true)
    expect(onIdle).toHaveBeenCalledOnce()
  })
})

describe('asyncPasteFallbackText', () => {
  it('treats an attachment-only null result as consumed', () => {
    expect(asyncPasteFallbackText(null, '/tmp/image.png')).toBeNull()
  })

  it('uses the resolved remainder instead of the original dropped path', () => {
    expect(
      asyncPasteFallbackText(
        {
          cursor: 5,
          fallbackText: 'describe this image',
          value: 'draft describe this image'
        },
        '/tmp/image.png'
      )
    ).toBe('describe this image')
  })
})

describe('readClipboardText', () => {
  it('reads text from pbpaste on macOS', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'hello world\n' })

    await expect(readClipboardText('darwin', run)).resolves.toBe('hello world\n')
    expect(run).toHaveBeenCalledWith(
      'pbpaste',
      [],
      expect.objectContaining({
        encoding: 'utf8',
        maxBuffer: 4 * 1024 * 1024,
        timeout: 5000,
        windowsHide: true
      })
    )
  })

  it('reads text from PowerShell on Windows', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from windows\r\n' })

    await expect(readClipboardText('win32', run)).resolves.toBe('from windows\r\n')
    expect(run).toHaveBeenCalledWith(
      'powershell',
      ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('tries powershell.exe first on WSL', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from wsl\n' })

    await expect(readClipboardText('linux', run, { WSL_INTEROP: '/tmp/socket' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wsl\n'
    )
    expect(run).toHaveBeenCalledWith(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', 'Get-Clipboard -Raw'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('uses wl-paste on Wayland Linux', async () => {
    const run = vi.fn().mockResolvedValue({ stdout: 'from wayland\n' })

    await expect(readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from wayland\n'
    )
    expect(run).toHaveBeenCalledWith(
      'wl-paste',
      ['--type', 'text'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('falls back to xclip on Linux when wl-paste fails', async () => {
    const run = vi
      .fn()
      .mockRejectedValueOnce(new Error('wl-paste missing'))
      .mockResolvedValueOnce({ stdout: 'from xclip\n' })

    await expect(readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)).resolves.toBe(
      'from xclip\n'
    )
    expect(run).toHaveBeenNthCalledWith(
      1,
      'wl-paste',
      ['--type', 'text'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
    expect(run).toHaveBeenNthCalledWith(
      2,
      'xclip',
      ['-selection', 'clipboard', '-out'],
      expect.objectContaining({ encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, windowsHide: true })
    )
  })

  it('returns null when every clipboard backend fails', async () => {
    const run = vi.fn().mockRejectedValue(new Error('clipboard failed'))

    await expect(
      readClipboardText('linux', run, { WAYLAND_DISPLAY: 'wayland-1' } as NodeJS.ProcessEnv)
    ).resolves.toBeNull()
  })
})

describe('isUsableClipboardText', () => {
  it('accepts normal text', () => {
    expect(isUsableClipboardText('hello world\n')).toBe(true)
  })

  it('rejects empty or whitespace-only content', () => {
    expect(isUsableClipboardText('')).toBe(false)
    expect(isUsableClipboardText('  \n\t')).toBe(false)
  })

  it('rejects binary-looking clipboard payloads', () => {
    expect(isUsableClipboardText('PNG\u0000\u0001\u0002\u0003IHDR')).toBe(false)
    expect(isUsableClipboardText('TIFF\ufffd\ufffd\ufffdmetadata')).toBe(false)
  })
})

describe('writeClipboardText', () => {
  it('does nothing off macOS when no tools are available', async () => {
    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(1) // 非零退出码表示失败。
        }

        return child
      }),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    // Linux 未设置 WAYLAND_DISPLAY 和 WSL_INTEROP 时依次回退到 xclip、xsel，二者均失败。
    await expect(writeClipboardText('hello', 'linux', start, {})).resolves.toBe(false)
  })

  it('writes text to pbcopy on macOS', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'pbcopy',
      [],
      expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })
    )
    expect(stdin.end).toHaveBeenCalledWith('hello world')
  })

  it('returns false when pbcopy fails', async () => {
    const child = {
      once: vi.fn((event: string, cb: () => void) => {
        if (event === 'error') {
          cb()
        }

        return child
      }),
      stdin: { end: vi.fn() }
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('hello world', 'darwin', start as any)).resolves.toBe(false)
  })

  it('uses wl-copy on Wayland Linux', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('wayland text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'wl-copy',
      ['--type', 'text/plain'],
      expect.objectContaining({ stdio: ['pipe', 'ignore', 'ignore'], windowsHide: true })
    )
    expect(stdin.end).toHaveBeenCalledWith('wayland text')
  })

  it('falls back to xclip when wl-copy fails on Wayland', async () => {
    let callCount = 0
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          callCount++
          // wl-copy 失败，xclip 成功。
          cb(callCount === 1 ? 1 : 0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('x11 text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })).resolves.toBe(
      true
    )
    expect(start).toHaveBeenNthCalledWith(1, 'wl-copy', ['--type', 'text/plain'], expect.anything())
    expect(start).toHaveBeenNthCalledWith(2, 'xclip', ['-selection', 'clipboard', '-in'], expect.anything())
  })

  it('falls back to xsel when both wl-copy and xclip fail', async () => {
    let callCount = 0
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          callCount++
          cb(callCount < 3 ? 1 : 0) // 前两次失败，第三次由 xsel 成功。
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('xsel text', 'linux', start as any, { WAYLAND_DISPLAY: 'wayland-1' })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(3, 'xsel', ['--clipboard', '--input'], expect.anything())
  })

  it('uses PowerShell on WSL2 when WSL_DISTRO_NAME is set', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('wsl text', 'linux', start as any, { WSL_DISTRO_NAME: 'Ubuntu' })).resolves.toBe(
      true
    )
    expect(start).toHaveBeenCalledWith(
      'powershell.exe',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    expect(stdin.end).toHaveBeenCalledWith('wsl text')
  })

  it('prefers the Windows clipboard path over wl-copy inside WSLg', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(
      writeClipboardText('wslg text', 'linux', start as any, {
        WAYLAND_DISPLAY: 'wayland-0',
        WSL_DISTRO_NAME: 'Ubuntu'
      })
    ).resolves.toBe(true)
    expect(start).toHaveBeenNthCalledWith(
      1,
      'powershell.exe',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
    expect(stdin.end).toHaveBeenCalledWith('wslg text')
  })

  it('uses PowerShell on Windows', async () => {
    const stdin = { end: vi.fn() }

    const child = {
      once: vi.fn((event: string, cb: (code?: number) => void) => {
        if (event === 'close') {
          cb(0)
        }

        return child
      }),
      stdin
    }

    const start = vi.fn().mockReturnValue(child)

    await expect(writeClipboardText('windows text', 'win32', start as any)).resolves.toBe(true)
    expect(start).toHaveBeenCalledWith(
      'powershell',
      expect.arrayContaining(['-NoProfile', '-NonInteractive']),
      expect.anything()
    )
  })
})
