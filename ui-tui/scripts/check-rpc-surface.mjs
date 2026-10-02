#!/usr/bin/env node

import { readdir, readFile } from 'node:fs/promises'
import { extname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const SOURCE_ROOT = join(ROOT, 'src')
const SCHEMA_PATH = join(ROOT, 'rpc-schema', 'openrpc.json')
const CALL_NAMES = new Set(['request', 'rpc', 'subscribe'])

const sourceFiles = []

async function walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name)
    const rel = relative(SOURCE_ROOT, path)

    if (entry.isDirectory()) {
      if (entry.name !== '__tests__') {
        await walk(path)
      }
      continue
    }

    if (!['.ts', '.tsx'].includes(extname(entry.name))) {
      continue
    }
    if (entry.name.endsWith('.test.ts') || entry.name.endsWith('.test.tsx') || rel === 'rpc/generated.ts') {
      continue
    }
    sourceFiles.push(path)
  }
}

function literalMethod(node) {
  if (ts.isStringLiteralLike(node)) {
    return node.text
  }
  if (ts.isNoSubstitutionTemplateLiteral(node)) {
    return node.text
  }
  return null
}

await walk(SOURCE_ROOT)

const calls = []
for (const path of sourceFiles) {
  const text = await readFile(path, 'utf8')
  const source = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true)

  const visit = node => {
    if (ts.isCallExpression(node)) {
      const callee = node.expression
      const name = ts.isIdentifier(callee)
        ? callee.text
        : ts.isPropertyAccessExpression(callee)
          ? callee.name.text
          : null
      const method = node.arguments[0] ? literalMethod(node.arguments[0]) : null

      if (name && CALL_NAMES.has(name) && method) {
        const pos = source.getLineAndCharacterOfPosition(node.getStart(source))
        calls.push({ method, path: relative(ROOT, path), line: pos.line + 1 })
      }
    }
    ts.forEachChild(node, visit)
  }

  visit(source)
}

const schema = JSON.parse(await readFile(SCHEMA_PATH, 'utf8'))
const registered = new Set(schema.methods.map(method => method.name))
const unknown = calls.filter(call => !registered.has(call.method))

if (unknown.length) {
  for (const call of unknown) {
    console.error(`${call.path}:${call.line}: unregistered RPC method ${call.method}`)
  }
  process.exit(1)
}

const unique = [...new Set(calls.map(call => call.method))].sort()
console.log(`OK: ${calls.length} frontend RPC calls use ${unique.length} registered methods`)
