#!/usr/bin/env bash
# CodeFlow 的 PinchBench 机器人模式执行器。
#
# 此执行器为每个任务运行完整机器人（AgentLoop run_turn），测试通过 Spine 的
# 完整轮次流程。
#
# 用法：
#   ./benchmarks/pinchbench/bot_runner/run.sh                                    # 运行所有任务
#   ./benchmarks/pinchbench/bot_runner/run.sh --model anthropic/claude-sonnet-4  # 指定模型
#   ./benchmarks/pinchbench/bot_runner/run.sh --suite task_00_sanity             # 运行单个任务
#   ./benchmarks/pinchbench/bot_runner/run.sh --suite automated-only             # 仅运行自动化任务
#   ./benchmarks/pinchbench/bot_runner/run.sh --verbose                          # 输出详细信息

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 切换到项目根目录：bot_runner -> pinchbench -> benchmarks -> 项目根目录。
cd "$SCRIPT_DIR/../../.."

echo "=================================================="
echo "  PinchBench for CodeFlow (BOT MODE)"
echo "=================================================="

# 使用 Anaconda Python 3.13（CodeFlow 要求 >=3.11）。
PYTHON="${PYTHON:-$HOME/anaconda3/bin/python3}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

# 确保 yaml 可用。
"$PYTHON" -c "import yaml" 2>/dev/null || "$PYTHON" -m pip install pyyaml -q

exec "$PYTHON" benchmarks/pinchbench/bot_runner/benchmark.py "$@"
