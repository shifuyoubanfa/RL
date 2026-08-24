#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 在一个全量模型上做 LoRA 直接偏好优化（DPO）。
#
# 参考模型（pi_ref）怎么来的：起点是这个全量模型，训练的是一个全新的 LoRA。
# 把这个 LoRA 关掉，剩下的就正好是冻结的起点模型本身——所以参考模型不需要额外加载一份权重，
# 也不存在"参考用的是哪个 adapter"这种含糊。
#
# 用法：
#   bash scripts/swift/dpo.sh <偏好对 jsonl> <基座目录> <输出目录>
#
# 关键环境变量：
#   DPO_BETA       偏离参考模型的惩罚强度。越大越保守，越小越敢改
#   DPO_LR         学习率。偏好优化比监督微调敏感得多，量级要小一到两个数量级
#   DPO_EPOCHS     训练轮数
#   DPO_GA         梯度累计步数
#   DPO_RPO_ALPHA  在偏好损失上再加一份 chosen 的负对数似然，权重即此值
# ---------------------------------------------------------------------------
set -euo pipefail

PAIRS_FILE="$1"       # 偏好对 jsonl
BASE_MODEL="$2"       # 起点模型（上一阶段合并出来的）
OUT_DIR="$3"          # LoRA adapter 落点

TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(echo "$TRAIN_GPUS" | tr ',' '\n' | grep -c .)"
SWIFT_BIN="${SWIFT_BIN:-swift}"
DEEPSPEED="${DEEPSPEED:-zero3}"
DPO_BETA="${DPO_BETA:-0.1}"
DPO_LR="${DPO_LR:-5e-6}"
DPO_EPOCHS="${DPO_EPOCHS:-2}"
DPO_GA="${DPO_GA:-2}"
DPO_RPO_ALPHA="${DPO_RPO_ALPHA:-1.0}"

[ -f "$PAIRS_FILE" ] || { echo "[dpo] 缺偏好对: $PAIRS_FILE"; exit 1; }
[ -f "$BASE_MODEL/config.json" ] || { echo "[dpo] 缺基座: $BASE_MODEL"; exit 1; }
command -v "$SWIFT_BIN" >/dev/null 2>&1 || [ -x "$SWIFT_BIN" ] \
  || { echo "[dpo] 找不到 swift: $SWIFT_BIN"; exit 1; }

DS_ARG=()
[ -n "$DEEPSPEED" ] && DS_ARG=(--deepspeed "$DEEPSPEED")

# rpo_alpha 不是所有 swift 版本都有。要就必须真的有，不能悄悄降级成没有——
# 少了这一项，模型只学"把 chosen 和 rejected 的差距拉开"，可以靠把 rejected 概率压到极低达成，
# 而 chosen 本身写得好不好完全没人管，训久了 chosen 那一侧的语言质量会塌。
RPO_ARG=()
if [ -n "$DPO_RPO_ALPHA" ]; then
  if "$SWIFT_BIN" rlhf --help 2>/dev/null | grep -q -- "--rpo_alpha"; then
    RPO_ARG=(--rpo_alpha "$DPO_RPO_ALPHA")
  else
    echo "[dpo] 致命：要求 rpo_alpha=$DPO_RPO_ALPHA，但当前 swift 没有 --rpo_alpha"
    exit 2
  fi
fi

# 对数少的时候把梯度累计降到 1。累计步数太大会让一轮只剩几个优化步，
# 学习率调度器还没热身完训练就结束了。
PAIR_LINES="$(wc -l < "$PAIRS_FILE" | tr -d ' ')"
if [ "${DPO_AUTO_GA:-1}" = "1" ] && [ "$PAIR_LINES" -lt 400 ]; then
  DPO_GA=1
fi

# 让 PyTorch 用可扩展显存段。DPO 的 chosen/rejected 两侧长度不一，
# 固定分段会产生大量碎片，长跑之后明明有空闲显存却分配不出来。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[dpo] 策略起点 = $BASE_MODEL + 新建 LoRA"
echo "[dpo] 参考模型 = $BASE_MODEL（把新建 LoRA 关掉即得，无 adapter 歧义）"
echo "[dpo] pairs=$PAIRS_FILE 行数=$PAIR_LINES out=$OUT_DIR"
echo "[dpo] beta=$DPO_BETA lr=$DPO_LR epochs=$DPO_EPOCHS ga=$DPO_GA rpo=${DPO_RPO_ALPHA:-off}"

CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" NPROC_PER_NODE="$NPROC" \
"$SWIFT_BIN" rlhf \
  --rlhf_type dpo \
  --model "$BASE_MODEL" \
  --model_type "${MODEL_TYPE:-qwen2}" \
  --template "${TEMPLATE:-qwen2_5}" \
  --train_type lora \
  --dataset "$PAIRS_FILE" \
  --torch_dtype bfloat16 \
  --beta "$DPO_BETA" \
  --num_train_epochs "$DPO_EPOCHS" \
  --learning_rate "$DPO_LR" \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --lora_rank "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --target_modules ${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj} \
  --max_length "${MAX_LEN:-4096}" \
  --per_device_train_batch_size "${PDBS:-1}" \
  --gradient_accumulation_steps "$DPO_GA" \
  --gradient_checkpointing true \
  "${DS_ARG[@]}" \
  "${RPO_ARG[@]}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_strategy steps \
  --save_steps "${DPO_SAVE_STEPS:-5}" \
  --save_total_limit "${DPO_SAVE_TOTAL_LIMIT:-8}" \
  --output_dir "$OUT_DIR"

echo done > "$OUT_DIR/.done"
echo "[dpo] 完成 -> $OUT_DIR"
