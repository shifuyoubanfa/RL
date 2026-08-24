#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 训练的唯一入口。
#
# 用法：
#   bash scripts/train.sh                      # 跑完整主线：sft → rft → dpo → grpo
#   bash scripts/train.sh --stages sft,rft     # 只跑前两个阶段
#   bash scripts/train.sh --no-baseline        # 跳过基座基线评测
#
# 这个脚本本身不做任何训练决策，只干三件事：
#   1. 把 Python 环境和路径准备好
#   2. 检查 key 在不在（训到一半才发现 key 没设，代价是整轮排队时间）
#   3. exec 到 Python 编排器
#
# 长跑请放进 tmux 或 screen，否则终端一断整条链路就没了：
#   tmux new -s rl
#   bash scripts/train.sh
#
# 想改超参、改路径、改门限，都去改 configs/train.yaml，不要改这个脚本。
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Python 解释器。训练机上通常是某个 conda 环境里的 python，用环境变量指过去。
PYTHON_BIN="${PYTHON_BIN:-python}"
# ms-swift 可执行文件。训练脚本会用到，这里导出让子脚本继承。
export SWIFT_BIN="${SWIFT_BIN:-swift}"
# vllm 可执行文件。推理服务和训练如果装在不同环境里，这里分别指。
export VLLM_BIN="${VLLM_BIN:-vllm}"
# 让 src.* 能被 import
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# 日志目录要在起服务之前就定下来：serve_vllm.sh 和 Python 编排器都靠它找 pid 文件，
# 两边不一致的话编排器停不掉服务，下一步训练必然 OOM。
export RL_LOG_DIR="${RL_LOG_DIR:-$REPO_ROOT/runs/logs}"
mkdir -p "$RL_LOG_DIR"

# key 只从环境变量读。没设就在这里停，别等模型加载完。
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "[train] 缺 DASHSCOPE_API_KEY。先 export，或参考 .env.example 配好。" >&2
  echo "[train] （改用别的裁判服务时，改 configs/train.yaml 的 judge.api_key_env）" >&2
  exit 1
fi

echo "===== 强化学习训练主线 ====="
echo "配置:     ${RL_CONFIG:-$REPO_ROOT/configs/train.yaml}"
echo "日志:     $RL_LOG_DIR/pipeline/"
echo "状态:     $RL_LOG_DIR/pipeline/state.json  （另开一个终端 cat 它看进度）"
echo ""

exec "$PYTHON_BIN" -X utf8 -m src.training.pipeline "$@"
