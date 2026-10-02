#!/usr/bin/env bash
# CodeFlow 的 ClawBench 流式运行时执行器。
#
# 用法：
#   CLAW_BENCH_ROOT=/path/to/claw-bench ./benchmarks/clawbench/run.sh --limit 80

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PYTHON="${PYTHON:-python3}"

exec "$PYTHON" benchmarks/clawbench/stream.py "$@"
