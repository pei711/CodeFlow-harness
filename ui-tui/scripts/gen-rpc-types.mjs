#!/usr/bin/env node
// 读取唯一事实来源 `ui-tui/rpc-schema/openrpc.json`（REQ-5），并将 TypeScript
// 类型输出到 `ui-tui/src/rpc/generated.ts`。
//
// 输出内容：
//   - 所有 `components/schemas/*`；
//   - 每个 RPC 方法各一个 `<MethodName>Params` 和 `<MethodName>Result` 接口；
//   - JSON-RPC 2.0 信封类型（手写尾部；它们属于协议层而非方法层，不在模式中）。
//
// F-C 分叉点（参见 research-findings.md 发现 2 与设计问题 Q1）：
//   `json-schema-to-typescript` 过去难以处理 OpenRPC 风格的 `discriminator`
//   （问题 bcherny/json-schema-to-typescript#239），因此必须验证输出的
//   `TurnEvent` 能正确收窄：
//
//       if (event.type === 'token.delta') { event.payload.text /* string */ }
//
//   下方 `verifyTurnEventNarrowing()` 探针运行代码生成、搜索输出，并在联合类型
//   疑似损坏时警告。探针触发时，文档约定的逃生方案是切换到
//   `@open-rpc/typings`，参见 schema tooling decision 文档。
//
// 用法：
//   node scripts/gen-rpc-types.mjs            # 写入 generated.ts
//   node scripts/gen-rpc-types.mjs --check    # 写入临时文件并与已检入的
//                                              # generated.ts 比较；发生漂移时
//                                              # 以 1 退出（CI lint 模式）

import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { compile } from 'json-schema-to-typescript';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');
const SCHEMA_PATH = resolve(ROOT, 'rpc-schema/openrpc.json');
const OUT_PATH = resolve(ROOT, 'src/rpc/generated.ts');

const HEADER = `// AUTO-GENERATED — DO NOT EDIT — run \`npm run gen:rpc\`
//
// Source of truth: ui-tui/rpc-schema/openrpc.json (OpenRPC 1.2.6).
// Regenerate via: cd ui-tui && npm run gen:rpc
// Lint (drift check) via: cd ui-tui && npm run lint:rpc
//
// CodeFlow method-scoped types + components/schemas + JSON-RPC 2.0 envelopes.

/* eslint-disable */
/* tslint:disable */
`;

// ---------------------------------------------------------------------------
// 参数名转 PascalCase，例如 "turn.send" 转为 "TurnSend"。
// ---------------------------------------------------------------------------
function methodToPascal(method) {
  return method
    .split(/[._]/)
    .map((part) => (part.length > 0 ? part[0].toUpperCase() + part.slice(1) : part))
    .join('');
}

// 将 OpenRPC 方法的 `params: [{name, required, schema}, ...]` 转为单个 JSON
// Schema 对象类型，以满足 json-schema-to-typescript 的输入要求。
function paramsToSchema(method) {
  const properties = {};
  const required = [];
  for (const p of method.params ?? []) {
    properties[p.name] = p.schema;
    if (p.required === true) required.push(p.name);
  }
  return {
    type: 'object',
    additionalProperties: false,
    properties,
    ...(required.length > 0 ? { required } : {}),
  };
}

// 构建 json-schema-to-typescript 可一次编译的单一复合根模式。原模式中的所有
// `$ref` 使用 `#/components/schemas/X`；将其提升到 JSON Schema 规范位置
// `#/definitions/X` 并改写引用，以保留结构。
function buildRootSchema(openrpcDoc) {
  const defs = {};
  const componentSchemas = openrpcDoc.components?.schemas ?? {};

  for (const [name, schema] of Object.entries(componentSchemas)) {
    defs[name] = rewriteRefs(schema);
  }

  // 各方法的 Params 与 Result 类型。
  for (const method of openrpcDoc.methods) {
    const pascal = methodToPascal(method.name);
    defs[`${pascal}Params`] = rewriteRefs(paramsToSchema(method));
    const resultSchema = method.result.schema;
    const resultRef = typeof resultSchema?.$ref === 'string' ? resultSchema.$ref.split('/').at(-1) : null;
    defs[`${pascal}Result`] = rewriteRefs(
      resultRef && componentSchemas[resultRef] ? componentSchemas[resultRef] : resultSchema,
    );
  }

  const rootProperties = {};
  for (const name of Object.keys(defs)) {
    rootProperties[name] = { $ref: `#/definitions/${name}` };
  }

  return {
    $schema: 'http://json-schema.org/draft-07/schema#',
    title: 'CodeFlowRpcRoot',
    type: 'object',
    properties: rootProperties,
    definitions: defs,
  };
}

function rewriteRefs(node) {
  if (Array.isArray(node)) return node.map(rewriteRefs);
  if (node && typeof node === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(node)) {
      if (k === '$ref' && typeof v === 'string') {
        out[k] = v.replace('#/components/schemas/', '#/definitions/');
      } else {
        out[k] = rewriteRefs(v);
      }
    }
    return out;
  }
  return node;
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 信封按原样追加，属于协议层而不在模式中。
// ---------------------------------------------------------------------------
const JSON_RPC_ENVELOPE = `
// ---------------------------------------------------------------------------
// JSON-RPC 2.0 envelope (specs/tui-ipc.md §2.1/2.2/2.3/2.4)
// ---------------------------------------------------------------------------

export interface JsonRpcRequest<P = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  method: string;
  params: P;
}

export interface JsonRpcSuccess<R = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  result: R;
}

export interface JsonRpcErrorObject {
  code: number;
  message: string;
  data?: unknown;
}

export interface JsonRpcErrorResponse {
  jsonrpc: '2.0';
  id: string | number;
  error: JsonRpcErrorObject;
}

export type JsonRpcResponse<R = unknown> = JsonRpcSuccess<R> | JsonRpcErrorResponse;

export interface JsonRpcNotification<P = unknown> {
  jsonrpc: '2.0';
  method: string;
  params: P;
}

export interface EventNotificationParams<E = unknown> {
  subscription_id: string;
  event: E;
}

export function isJsonRpcError<R>(
  resp: JsonRpcResponse<R>,
): resp is JsonRpcErrorResponse {
  return (resp as JsonRpcErrorResponse).error !== undefined;
}
`;

// ---------------------------------------------------------------------------
// F-C 检查：验证 TurnEvent 能正确收窄。
// ---------------------------------------------------------------------------
function verifyTurnEventNarrowing(emitted) {
  // 正确输出有两种形式：
  //   (a) 在 `export type TurnEvent = ...` 中内联类似
  //       `{ type: 'token.delta'; payload: ... }` 的带标签联合；
  //   (b) 将 TurnEvent 输出为 `TurnEventMessageStart | TurnEventTokenDelta | ...`，
  //       且每个变体都包含字面量 `type` 判别字段。
  // 错误输出会把 payload 折叠为 Record<string, any> 或丢弃 `type` 字面量，
  // 导致无法收窄。

  const turnEventBlock = emitted.match(/export type TurnEvent\s*=([\s\S]*?);/);
  if (!turnEventBlock) {
    return {
      ok: false,
      reason: 'TurnEvent type not found in output — codegen failed to emit it.',
    };
  }
  const body = turnEventBlock[1];
  // 启发式检查：确保每个判别字面量都以内联字面量类型
  // `type: 'token.delta'` 或包含该字面量的引用接口出现。通过检查完整文件中的
  // 字面量同时接受两种形式。
  const literals = [
    'message.start',
    'token.delta',
    'thinking.delta',
    'tool.start',
    'tool.progress',
    'tool.complete',
    'message.complete',
    'error',
    'cron.delivered',
    'subagent.delivered',
  ];
  const missing = literals.filter(
    (lit) => !body.includes(`"${lit}"`) && !body.includes(`'${lit}'`) && !emitted.match(
      new RegExp(`type:\\s*['"]${lit.replace('.', '\\.')}['"]`),
    ),
  );
  if (missing.length > 0) {
    return {
      ok: false,
      reason: `TurnEvent missing discriminator literals: ${missing.join(', ')}`,
    };
  }
  // 验证它是包含 `|` 且对象结构带 `type` 判别字段的联合类型。
  if (!body.includes('|')) {
    return {
      ok: false,
      reason: 'TurnEvent is not a union — narrowing impossible.',
    };
  }
  return { ok: true };
}

// ---------------------------------------------------------------------------
// 代码生成流水线。
// ---------------------------------------------------------------------------
async function generate() {
  const raw = await readFile(SCHEMA_PATH, 'utf-8');
  const openrpcDoc = JSON.parse(raw);
  const root = buildRootSchema(openrpcDoc);

  // 编译选项以减少运行间的虚假漂移：
  //   - declareExternallyReferenced: false（引用树由我们控制）；
  //   - unreachableDefinitions: true（即使方法范围类型仅由根对象引用也输出）；
  //   - bannerComment: empty（自行前置 HEADER）；
  //   - additionalProperties: false（保留模式的严格闭合）。
  const compiled = await compile(root, 'CodeFlowRpcRoot', {
    bannerComment: '',
    unreachableDefinitions: true,
    additionalProperties: false,
    style: { singleQuote: true, semi: true, trailingComma: 'all' },
  });

  // 丢弃合成根接口；它只是覆盖所有定义的 `Record`，对消费者无用。仅保留命名类型。
  const withoutRoot = compiled
    .replace(/export interface CodeFlowRpcRoot\s*\{[\s\S]*?\n\}\n?/, '')
    .trim();

  // json-schema-to-typescript 会把结构相同的定义合并为单个导出类型，例如六个仅限
  // Hermes 的桩结果都折叠为 `StubResult`，`CliDispatchResult` 折叠为
  // `CliResult`。为使每个模式命名类型都能按规范名称访问，对未得到独立声明的
  // 定义名称输出类型别名。
  //
  // 通过解析 json-schema-to-typescript 写在各合并声明上方的延迟注释轨迹，
  // 检测规范名称：
  //   /**
  //    * This interface was referenced by `CodeFlowRpcRoot`'s JSON-Schema
  //    * via the `definition` "CliResult".
  //    *
  //    * This interface was referenced by ... `definition` "CliDispatchResult".
  //    */
  //   export interface CliResult { ... }
  const allNames = new Set(Object.keys(root.definitions));
  const declared = new Set();
  const declRegex = /^export (?:interface|type) (\w+)\b/gm;
  let m;
  while ((m = declRegex.exec(withoutRoot)) !== null) declared.add(m[1]);

  // 遍历每个声明及其前置文档注释，将所有被引用的定义名称映射到已声明规范名称。
  const aliasFor = {}; // 缺失名称映射到规范名称。
  const blockRegex =
    /\/\*\*([\s\S]*?)\*\/\s*export (?:interface|type) (\w+)\b/g;
  while ((m = blockRegex.exec(withoutRoot)) !== null) {
    const doc = m[1];
    const canonical = m[2];
    const refNameRegex = /`definition` "(\w+)"/g;
    let r;
    while ((r = refNameRegex.exec(doc)) !== null) {
      const refName = r[1];
      if (refName !== canonical && !declared.has(refName)) {
        aliasFor[refName] = canonical;
      }
    }
  }

  // 双重保险：对仍未声明且没有别名的名称输出空对象回退，使下游导入仍通过类型检查。
  const aliasLines = [];
  for (const name of allNames) {
    if (declared.has(name)) continue;
    if (aliasFor[name]) {
      aliasLines.push(`export type ${name} = ${aliasFor[name]};`);
    } else {
      // 理论上不应发生，发生时明确失败。
      throw new Error(
        `codegen: definition "${name}" produced no declaration and no alias` +
          ' candidate was found in deferred-comments. Inspect compiled output.',
      );
    }
  }
  const aliasBlock = aliasLines.length
    ? '\n// ---- Schema-name aliases for structurally-deduplicated types ----\n' +
      aliasLines.sort().join('\n') +
      '\n'
    : '';

  const full = HEADER + '\n' + withoutRoot + '\n' + aliasBlock + JSON_RPC_ENVELOPE;

  // 执行 F-C 检查。
  const probe = verifyTurnEventNarrowing(full);
  if (!probe.ok) {
    console.error('');
    console.error('!! F-C fork point tripped: TurnEvent narrowing broken.');
    console.error(`   reason: ${probe.reason}`);
    console.error('   action: switch codegen tool to `@open-rpc/typings`.');
    console.error('   see: docs/RepoMem/temp/tui-ipc-bridge/01-schema-tooling-decision.md');
    process.exit(2);
  }

  return full;
}

function sha(s) {
  return createHash('sha256').update(s).digest('hex').slice(0, 12);
}

async function main() {
  const check = process.argv.includes('--check');
  const fresh = await generate();

  if (check) {
    let existing = '';
    try {
      existing = await readFile(OUT_PATH, 'utf-8');
    } catch {
      console.error(`!! ${OUT_PATH} does not exist — run \`npm run gen:rpc\` first.`);
      process.exit(1);
    }
    if (existing.trim() !== fresh.trim()) {
      console.error('!! generated.ts is out of sync with rpc-schema/openrpc.json');
      console.error(`   existing sha256: ${sha(existing)}`);
      console.error(`   fresh    sha256: ${sha(fresh)}`);
      console.error('   fix: cd ui-tui && npm run gen:rpc && git add src/rpc/generated.ts');
      process.exit(1);
    }
    console.log(`OK: generated.ts in sync (sha256: ${sha(fresh)})`);
    return;
  }

  await writeFile(OUT_PATH, fresh, 'utf-8');
  const methodCount = JSON.parse(await readFile(SCHEMA_PATH, 'utf-8')).methods.length;
  console.log(
    `Wrote ${OUT_PATH}\n  ${methodCount} methods × 2 (Params/Result) + components + envelope\n  sha256: ${sha(fresh)}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
