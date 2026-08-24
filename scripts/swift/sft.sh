#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 在一个全量模型上做 LoRA 监督微调。
# 冷启动（阶段一）和拒绝采样（阶段二）共用这一个脚本——两个阶段的训练方式完全一样，
# 只有数据来源和学习率不同，所以没必要写两份。
#
# 用法：
#   bash scripts/swift/sft.sh <训练集> <验证集> <基座目录> <输出目录> [学习率] [轮数]
#
# 依赖的环境变量（都有默认值，一般由 scripts/train.sh 统一注入）：
#   TRAIN_GPUS   用哪几张卡，逗号分隔；张数即分布式进程数
#   SWIFT_BIN    ms-swift 可执行文件路径
#   MODEL_TYPE   基座的模型类型标识
#   TEMPLATE     基座的对话模板标识
#   LORA_R / LORA_ALPHA / LORA_DROPOUT / TARGET_MODULES
#   MAX_LEN      单条样本最大 token 数，超出会被截断
#   PDBS         每张卡一次前向几条
#   GA           梯度累计步数
#   DEEPSPEED    DeepSpeed 配置名，空串则关掉
# ---------------------------------------------------------------------------
set -euo pipefail

TRAIN_FILE="$1"       # 训练集 jsonl
VAL_FILE="$2"         # 验证集 jsonl，用来选 best checkpoint
BASE_MODEL="$3"       # 起点模型目录（阶段一是原始基座，阶段二是阶段一合并出来的模型）
OUT_DIR="$4"          # LoRA adapter 落点
LR="${5:-5e-5}"       # 学习率
EPOCHS="${6:-3}"      # 训练轮数

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
SWIFT_BIN="${SWIFT_BIN:-swift}"
DEEPSPEED="${DEEPSPEED:-zero3}"

# 前置检查一次做完。缺文件就早点报出来，别等模型加载完几分钟后才失败。
[ -f "$TRAIN_FILE" ] || { echo "[sft] 缺训练集: $TRAIN_FILE"; exit 1; }
[ -f "$VAL_FILE" ]   || { echo "[sft] 缺验证集: $VAL_FILE"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[sft] 缺基座: $BASE_MODEL"; exit 1; }
command -v "$SWIFT_BIN" >/dev/null 2>&1 || [ -x "$SWIFT_BIN" ] \
  || { echo "[sft] 找不到 swift: $SWIFT_BIN"; exit 1; }

DS_ARG=()
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

echo "[sft] base=$BASE_MODEL train=$TRAIN_FILE val=$VAL_FILE out=$OUT_DIR"
echo "[sft] lr=$LR epochs=$EPOCHS gpus=$TRAIN_GPUS deepspeed=${DEEPSPEED:-off}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT_BIN" sft \
  --model "$BASE_MODEL" \
  --model_type "${MODEL_TYPE:-qwen2}" \
  --template "${TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$TRAIN_FILE" \
  --val_dataset "$VAL_FILE" \
  --torch_dtype bfloat16 \
  --num_train_epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --warmup_ratio 0.05 \
  --lora_rank "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules ${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj} \
  --max_length "${MAX_LEN:-4096}" \
  --per_device_train_batch_size "${PDBS:-1}" \
  --gradient_accumulation_steps "${GA:-8}" \
  --gradient_checkpointing true \
  "${DS_ARG[@]}" \
  --eval_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --output_dir "$OUT_DIR"

# 只有走到这里才写完成标记。set -e 保证前面任何一步失败都到不了这行——
# 这样下游"训完了没有"只需要看这一个文件，不用去猜一堆半成品 checkpoint 的状态。
echo done > "$OUT_DIR/.done"
echo "[sft] 完成 -> $OUT_DIR"
