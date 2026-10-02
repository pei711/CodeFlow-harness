const fs = require('fs');
const os = require('os');
const path = require('path');

const DEFAULT_MAX_BYTES = 50 * 1024 * 1024;
const DEFAULT_VIEWER_MAX_BYTES = 8 * 1024 * 1024;

const KIND_FILES = {
  events: 'audit-events.log',
  spans: 'audit-spans.log'
};

function getStateDir() {
// 跨框架通用：优先使用入口脚本根据 --state-dir 设置的 TRACING_STATE_DIR，
// 其次使用旧 OpenClaw 变量，最后使用 ~/.openclaw。
  return (
    process.env.TRACING_STATE_DIR ||
    process.env.OPENCLAW_STATE_DIR ||
    path.join(os.homedir(), '.openclaw')
  );
}

function getLogsDir() {
  return path.join(getStateDir(), 'logs');
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function getActiveLogPath(kind) {
  return path.join(ensureDir(getLogsDir()), KIND_FILES[kind]);
}

function getArchiveDir() {
  return ensureDir(path.join(getLogsDir(), 'archive'));
}

function getDateKey(date = new Date()) {
  return date.toISOString().slice(0, 10);
}

function getMaxBytes() {
  const raw = Number(process.env.TRACE_LOG_MAX_BYTES || DEFAULT_MAX_BYTES);
  return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_MAX_BYTES;
}

function toJsonText(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function parseJsonl(text) {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function getViewerMaxBytes() {
  const raw = Number(process.env.TRACE_VIEWER_MAX_BYTES || DEFAULT_VIEWER_MAX_BYTES);
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : DEFAULT_VIEWER_MAX_BYTES;
}

function readJsonlFileWindow(filePath, maxBytes) {
  const stat = fs.statSync(filePath);
  const bytesToRead = Math.min(stat.size, maxBytes);
  const start = stat.size - bytesToRead;
  const fd = fs.openSync(filePath, 'r');
  let buffer;
  try {
    buffer = Buffer.allocUnsafe(bytesToRead);
    const actualBytes = fs.readSync(fd, buffer, 0, bytesToRead, start);
    buffer = buffer.subarray(0, actualBytes);
  } finally {
    fs.closeSync(fd);
  }

  if (start > 0) {
    const firstNewline = buffer.indexOf(0x0a);
    buffer = firstNewline === -1 ? Buffer.alloc(0) : buffer.subarray(firstNewline + 1);
  }

  return {
    records: parseJsonl(buffer.toString('utf8')),
    bytesRead: bytesToRead,
    truncated: start > 0
  };
}

function walkFiles(rootDir) {
  const result = [];
  if (!fs.existsSync(rootDir)) return result;
  const stack = [rootDir];
  while (stack.length) {
    const dir = stack.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
        continue;
      }
      result.push(full);
    }
  }
  return result;
}

function listLogFiles(kind) {
  const rootDir = getLogsDir();
  const baseName = KIND_FILES[kind];
  const activeFile = path.join(rootDir, baseName);
  const archiveDir = path.join(rootDir, 'archive');
  const candidates = [
    ...(fs.existsSync(activeFile) ? [activeFile] : []),
    ...walkFiles(archiveDir)
  ];
  const files = candidates.filter((filePath) => {
    const name = path.basename(filePath);
    if (name === baseName) return true;
    if (!name.startsWith(baseName.replace('.log', ''))) return false;
    return name.endsWith('.log');
  });
  files.sort((a, b) => {
    let statA = null;
    let statB = null;
    try {
      statA = fs.statSync(a);
    } catch {}
    try {
      statB = fs.statSync(b);
    } catch {}
    const timeA = statA ? statA.mtimeMs : 0;
    const timeB = statB ? statB.mtimeMs : 0;
    if (timeA !== timeB) return timeA - timeB;
    return a.localeCompare(b);
  });
  return files;
}

function readJsonlWindow(kind, maxBytes = getViewerMaxBytes()) {
  const files = listLogFiles(kind);
  const recordGroups = [];
  let bytesRead = 0;
  let filesRead = 0;
  let truncated = false;

  for (let index = files.length - 1; index >= 0; index -= 1) {
    const remaining = maxBytes - bytesRead;
    if (remaining <= 0) {
      truncated = true;
      break;
    }

    let result;
    try {
      result = readJsonlFileWindow(files[index], remaining);
    } catch {
      truncated = true;
      continue;
    }
    recordGroups.unshift(result.records);
    bytesRead += result.bytesRead;
    filesRead += 1;
    if (result.truncated) {
      truncated = true;
      break;
    }
    if (index > 0 && bytesRead >= maxBytes) truncated = true;
  }

  return {
    records: recordGroups.flat(),
    window: {
      truncated,
      bytesRead,
      maxBytes,
      filesRead,
      filesAvailable: files.length
    }
  };
}

function logFingerprint() {
  const entries = [];
  for (const kind of Object.keys(KIND_FILES)) {
    for (const filePath of listLogFiles(kind)) {
      try {
        const stat = fs.statSync(filePath);
        entries.push([kind, filePath, stat.size, stat.mtimeMs]);
      } catch {}
    }
  }
  return JSON.stringify(entries);
}

function rotateIfNeeded(kind, nextText = '') {
  const filePath = getActiveLogPath(kind);
  if (!fs.existsSync(filePath)) return filePath;

  const stat = fs.statSync(filePath);
  const currentDateKey = getDateKey(new Date(stat.mtimeMs));
  const todayKey = getDateKey(new Date());
  const maxBytes = getMaxBytes();
  const nextBytes = Buffer.byteLength(String(nextText), 'utf8');
  const shouldRotateByDate = currentDateKey !== todayKey;
  const shouldRotateBySize = stat.size + nextBytes > maxBytes;
  if (!shouldRotateByDate && !shouldRotateBySize) return filePath;

  const archiveDayDir = ensureDir(path.join(getArchiveDir(), currentDateKey));
  const suffix = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  const rotatedPath = path.join(archiveDayDir, `${KIND_FILES[kind].replace('.log', '')}-${currentDateKey}-${suffix}.log`);
  fs.renameSync(filePath, rotatedPath);
  return filePath;
}

function appendJsonl(kind, record) {
  const text = `${toJsonText(record)}\n`;
  const filePath = rotateIfNeeded(kind, text);
  fs.appendFileSync(filePath, text, { encoding: 'utf8' });
  return filePath;
}

module.exports = {
  getStateDir,
  getLogsDir,
  getActiveLogPath,
  getArchiveDir,
  ensureDir,
  readJsonlWindow,
  logFingerprint,
  appendJsonl,
  rotateIfNeeded,
  getDateKey
};
