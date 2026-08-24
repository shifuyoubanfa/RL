#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 起一个常驻 vLLM 推理服务，供数据构建、rollout 采样、评测推理使用。
#
# 用法：
#   bash scripts/serve_vllm.sh <模型目录> <对外的模型名>
#
# 第二个参数是**语义开关**，不只是个标签：数据构建和评测脚本按这个名字决定套哪套 system 提示
# （见 src/data/prompts.py 的 system_for）。传成基座那个名字，训练后的模型就会被套上
# RAG 腔提示——分数会全错，而且从日志里完全看不出来。
#
# 关键环境变量：
#   VLLM_BIN     vllm 可执行文件
#   VLLM_GPUS    用哪几张卡，张数即张量并行度
#   VLLM_PORT    监听端口
#   RL_LOG_DIR   日志和 pid 文件落哪
#   VLLM_SERVE_GPU_UTIL  占多少显存比例
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL_DIR="$1"
SERVED_NAME="$2"

VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${RL_LOG_DIR:-./runs/logs}"
LOG_FILE="$LOG_DIR/vllm.log"
PID_FILE="$LOG_DIR/vllm.pid"
TP="$(echo "$VLLM_GPUS" | tr ',' '\n' | grep -c .)"
GPU_UTIL="${VLLM_SERVE_GPU_UTIL:-0.90}"
# 服务端上下文上限。给得比训练侧宽：评测时题面加参考资料可能比训练样本长，
# 卡在这里会直接拒绝请求，而不是截断。
MAX_MODEL_LEN="${VLLM_SERVE_MAX_LEN:-16384}"

[ -f "$MODEL_DIR/config.json" ] || { echo "[serve] 缺模型: $MODEL_DIR"; exit 1; }
command -v "$VLLM_BIN" >/dev/null 2>&1 || [ -x "$VLLM_BIN" ] \
  || { echo "[serve] 找不到 vllm: $VLLM_BIN"; exit 1; }

mkdir -p "$LOG_DIR"
echo "[serve] model=$MODEL_DIR served_name=$SERVED_NAME GPU=$VLLM_GPUS TP=$TP port=$PORT util=$GPU_UTIL"
echo "[serve] 日志: $LOG_FILE"

# 用 setsid 起新会话：这样 $! 同时是进程组 ID，停服务时 kill -TERM -<PID> 能把
# 张量并行拉起来的那一堆 worker 子进程一起收掉。不这么做，主进程退了 worker 还占着显存。
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
setsid "$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --port "$PORT" > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
PID="$(cat "$PID_FILE")"
echo "[serve] PID/PGID=$PID（停止：kill -TERM -$PID）"

# 等几秒看进程还在不在。秒退基本是三种原因：显存不够、端口被占、权重路径不对。
# 这时候把日志末尾打出来，比让调用方去等一个半小时的就绪超时强。
sleep 4
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[serve] vLLM 秒退，日志末尾："
  tail -n 30 "$LOG_FILE"
  exit 1
fi
echo "[serve] 进程存活，正在加载权重；调用方会轮询 /v1/models 直到就绪。"
