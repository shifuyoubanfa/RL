#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 在一个全量模型上做 LoRA 在线 GRPO。
#
# 和 DPO 的根本区别：DPO 的训练数据是事先构好的固定偏好对；GRPO 每一步都要现场采样——
# 拿当前策略对一批 prompt 各采 K 条，调奖励函数打分，用组内相对好坏当优势，再更新。
# 所以 GRPO 进程里同时住着一个训练器和一个推理引擎，两者抢同一批卡。
#
# 参考模型和 DPO 同理：起点全量模型 + 新建 LoRA，关掉 LoRA 就是冻结的参考。
#
# 用法：
#   bash scripts/swift/grpo.sh <训练 prompt jsonl> <基座目录> <输出目录>
#
# 关键环境变量：
#   GRPO_K               每个 prompt 采几条。组内优势就是在这 K 条之间算的
#   GRPO_STEPS           训练多少个优化步
#   GRPO_BETA            KL 系数，控制离参考模型多远
#   GRPO_REWARD_FUNC     用哪个奖励函数：rule_warmup（只看规则）或 online（接裁判）
#
# 显存那五个开关（colocate 模式下缺一不可，见下方注释）：
#   VLLM_GPU_UTIL / MOVE_MODEL_BATCHES / SLEEP_LEVEL / OFFLOAD_MODEL / OFFLOAD_OPTIMIZER
# ---------------------------------------------------------------------------
set -euo pipefail

DATA_FILE="$1"        # 训练 prompt jsonl，每行带题面和答案池
BASE_MODEL="$2"       # 起点模型
OUT_DIR="$3"          # LoRA adapter 落点

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
SWIFT_BIN="${SWIFT_BIN:-swift}"
DEEPSPEED="${DEEPSPEED:-zero3}"

K="${GRPO_K:-8}"
STEPS="${GRPO_STEPS:-90}"
BETA="${GRPO_BETA:-0.06}"
LR="${GRPO_LR:-7e-7}"
REWARD_FUNC="${GRPO_REWARD_FUNC:-online}"

[ -f "$DATA_FILE" ] || { echo "[grpo] 缺数据: $DATA_FILE"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[grpo] 缺基座: $BASE_MODEL"; exit 1; }
command -v "$SWIFT_BIN" >/dev/null 2>&1 || [ -x "$SWIFT_BIN" ] \
  || { echo "[grpo] 找不到 swift: $SWIFT_BIN"; exit 1; }

DS_ARG=()
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[grpo] 策略起点 = $BASE_MODEL + 新建 LoRA"
echo "[grpo] 参考模型 = $BASE_MODEL（把新建 LoRA 关掉即得）"
echo "[grpo] data=$DATA_FILE out=$OUT_DIR K=$K steps=$STEPS beta=$BETA lr=$LR reward=$REWARD_FUNC"

# --- colocate 显存五连关 ---------------------------------------------------
# colocate 表示推理引擎和训练器住在同一批卡上。好处是不用另外留一组卡专门做采样；
# 代价是显存必须精打细算，任何一关放松都会在几十步之后 OOM：
#   vllm_gpu_memory_utilization  推理引擎最多占一半显存，另一半留给训练
#   vllm_tensor_parallel_size    推理侧的并行度对齐训练进程数，避免两套并行切分打架
#   move_model_batches           权重同步分批搬，不要一次性拷整个模型
#   sleep_level                  采样间隙让推理引擎释放显存
#   offload_model/optimizer      非活跃时把模型和优化器状态挪到内存
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT_BIN" rlhf \
  --rlhf_type grpo \
  --model "$BASE_MODEL" \
  --model_type "${MODEL_TYPE:-qwen2}" \
  --template "${TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$DATA_FILE" \
  --external_plugins "$REPO_ROOT/src/rewards/grpo_reward.py" \
  --reward_funcs "$REWARD_FUNC" \
  --num_generations "$K" \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size "${VLLM_TP:-$NPROC}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_UTIL:-0.5}" \
  --vllm_max_model_len "${VLLM_MAX_LEN:-6144}" \
  --move_model_batches "${MOVE_MODEL_BATCHES:-16}" \
  --sleep_level "${SLEEP_LEVEL:-1}" \
  --offload_model "${OFFLOAD_MODEL:-true}" \
  --offload_optimizer "${OFFLOAD_OPTIMIZER:-true}" \
  "${DS_ARG[@]}" \
  --scale_rewards "${SCALE_REWARDS:-group}" \
  --beta "$BETA" \
  --temperature "${GRPO_TEMPERATURE:-1.0}" \
  --top_p "${GRPO_TOP_P:-0.95}" \
  --max_steps "$STEPS" \
  --learning_rate "$LR" \
  --lora_rank "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules ${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj} \
  --max_length "${MAX_LEN:-4096}" \
  --max_completion_length "${GRPO_MAX_COMPLETION:-1536}" \
  --per_device_train_batch_size "${PDBS:-1}" \
  --gradient_accumulation_steps "${GA:-8}" \
  --gradient_checkpointing true \
  --attn_impl "${ATTN_IMPL:-sdpa}" \
  --logging_steps 1 \
  --save_steps "${GRPO_SAVE_STEPS:-25}" \
  --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT:-8}" \
  --output_dir "$OUT_DIR"

echo done > "$OUT_DIR/.done"
echo "[grpo] 完成 -> $OUT_DIR"
