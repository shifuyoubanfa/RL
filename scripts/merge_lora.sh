#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 把一个 LoRA adapter 合并回它的全量基座，产出一个新的全量模型。
#
# 为什么每个阶段结束都要合并，而不是一路挂着 adapter：
#   1. 下一阶段要在"上一阶段的结果"之上再训一个**新的** LoRA。基座是全量模型时，
#      参考模型就是"把新 LoRA 关掉"，干净利落；如果基座本身还挂着旧 adapter，
#      "参考模型到底包不包含旧 adapter"就成了一个要靠读框架源码才能确定的问题。
#   2. 采样和评测都走独立的推理服务，喂一个全量目录最省事。
#
# 合并在 CPU 上做（CUDA_VISIBLE_DEVICES 置空）。它是纯权重加减，不需要 GPU，
# 而这时候卡通常正被别的阶段占着。代价是慢几分钟。
#
# 用法：
#   bash scripts/merge_lora.sh <基座目录> <adapter 目录> <输出目录>
# ---------------------------------------------------------------------------
set -euo pipefail

BASE_MODEL="$1"
ADAPTER="$2"
OUT_DIR="$3"
SWIFT_BIN="${SWIFT_BIN:-swift}"
# 先写 .partial，校验通过才改名成正式目录。
# 中途被 kill 只会留下一个 .partial，不会留下一个"看起来是完整模型、其实少几个分片"的目录——
# 后者会被下游当成可用模型直接加载，报的错还全是权重形状不匹配。
TMP_DIR="${OUT_DIR}.partial"

[ -f "$BASE_MODEL/config.json" ] || { echo "[merge] 缺基座 config: $BASE_MODEL"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "[merge] 缺 adapter_config.json: $ADAPTER"; exit 1; }
command -v "$SWIFT_BIN" >/dev/null 2>&1 || [ -x "$SWIFT_BIN" ] \
  || { echo "[merge] 找不到 swift: $SWIFT_BIN"; exit 1; }

if [ -f "$OUT_DIR/.done" ] && [ -f "$OUT_DIR/config.json" ]; then
  echo "[merge] 已经合并过: $OUT_DIR"
  exit 0
fi
if [ -e "$OUT_DIR" ]; then
  # 存在但没有完成标记 = 上次合并没成。不自动删——几十 G 的东西，删错了要重跑几十分钟。
  echo "[merge] 输出目录已存在但不完整: $OUT_DIR"
  echo "[merge] 先自行挪走再重试；本脚本不做任何自动删除。"
  exit 1
fi
if [ -e "$TMP_DIR" ]; then
  STALE="${TMP_DIR}.interrupted-$(date +%Y%m%d-%H%M%S)"
  echo "[merge] 保留上次中断的半成品 -> $STALE"
  mv "$TMP_DIR" "$STALE"
fi

mkdir -p "$(dirname "$OUT_DIR")"
echo "[merge] base=$BASE_MODEL"
echo "[merge] adapter=$ADAPTER"
echo "[merge] output=$OUT_DIR"
echo "[merge] 在 CPU 上合并，不占训练卡；几十 G 的模型要几分钟。"

CUDA_VISIBLE_DEVICES="" "$SWIFT_BIN" export \
  --model "$BASE_MODEL" \
  --model_type "${MODEL_TYPE:-qwen2}" \
  --template "${TEMPLATE:-qwen2_5}" \
  --adapters "$ADAPTER" \
  --merge_lora true \
  --device_map cpu \
  --torch_dtype bfloat16 \
  --safe_serialization true \
  --max_shard_size 5GB \
  --output_dir "$TMP_DIR"

# 校验两样：config 在不在、权重分片有没有。缺任何一样都说明合并没写完。
[ -f "$TMP_DIR/config.json" ] || { echo "[merge] 合并结果缺 config.json: $TMP_DIR"; exit 1; }
find "$TMP_DIR" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . \
  || { echo "[merge] 合并结果缺权重分片: $TMP_DIR"; exit 1; }

echo done > "$TMP_DIR/.done"
mv "$TMP_DIR" "$OUT_DIR"
echo "[merge] 完成 -> $OUT_DIR"
