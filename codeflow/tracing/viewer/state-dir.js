'use strict';

const os = require('os');
const path = require('path');

// 将框架名称映射到追踪状态目录，即 logs/audit-*.log 所在位置。
function frameworkStateDir(name) {
  const home = os.homedir();
  switch (String(name || '').toLowerCase()) {
    case 'openclaw':
      return process.env.OPENCLAW_STATE_DIR || path.join(home, '.openclaw');
    case 'hermes':
      return process.env.HERMES_HOME || path.join(home, '.hermes');
    case 'codeflow':
      return path.join(home, '.codeflow', 'traces');
    default:
      return null;
  }
}

// 从 argv 解析 `--state-dir <path>` / `--framework <name>`，设置 TRACING_STATE_DIR，
// 并从 process.argv 删除已消费参数，使下游位置参数解析（如 trace-viewer 的 traceId）仍可工作。
function applyStateDirArg(argv = process.argv) {
  const out = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--state-dir' && argv[i + 1]) {
      process.env.TRACING_STATE_DIR = argv[i + 1];
      i += 1;
      continue;
    }
    if (arg.startsWith('--state-dir=')) {
      process.env.TRACING_STATE_DIR = arg.slice('--state-dir='.length);
      continue;
    }
    if (arg === '--framework' && argv[i + 1]) {
      const dir = frameworkStateDir(argv[i + 1]);
      if (dir) process.env.TRACING_STATE_DIR = dir;
      i += 1;
      continue;
    }
    if (arg.startsWith('--framework=')) {
      const dir = frameworkStateDir(arg.slice('--framework='.length));
      if (dir) process.env.TRACING_STATE_DIR = dir;
      continue;
    }
    out.push(arg);
  }
  argv.length = 0;
  argv.push(...out);
  return process.env.TRACING_STATE_DIR || null;
}

module.exports = { applyStateDirArg, frameworkStateDir };
