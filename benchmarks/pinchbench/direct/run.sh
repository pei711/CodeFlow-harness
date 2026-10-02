#!/usr/bin/env bash
# CodeFlow 的 PinchBench 直接模式执行器。
#
# 用法：
#   ./benchmarks/pinchbench/direct/run.sh                                          # 运行所有任务
#   ./benchmarks/pinchbench/direct/run.sh --model deepseek-v4-flash                # 指定模型
#   ./benchmarks/pinchbench/direct/run.sh --provider custom --api-base "$OPENROUTER_API_BASE"
#   ./benchmarks/pinchbench/direct/run.sh --suite task_00_sanity                   # 运行单个任务
#   ./benchmarks/pinchbench/direct/run.sh --suite automated-only                   # 仅运行自动化任务
#   ./benchmarks/pinchbench/direct/run.sh --verbose                                # 输出详细信息

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 切换到项目根目录：direct -> pinchbench -> benchmarks -> 项目根目录。
cd "$SCRIPT_DIR/../../.."

echo "=================================================="
echo "  PinchBench for CodeFlow (DIRECT MODE)"
echo "=================================================="

# 使用 Anaconda Python 3.13（CodeFlow 要求 >=3.11）。
PYTHON="${PYTHON:-$HOME/anaconda3/bin/python3}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

# 确保 yaml 可用。
"$PYTHON" -c "import yaml" 2>/dev/null || "$PYTHON" -m pip install pyyaml -q

exec "$PYTHON" benchmarks/pinchbench/direct/benchmark.py "$@"
