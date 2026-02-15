#!/usr/bin/env bash
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CHECKPOINT:=/vePFS-0x0d/home/cx/ptft/output_old/flagship_cross_attn/checkpoint_epoch_6.pth}"
: "${CONFIG:=configs/pretrain.yaml}"
: "${OUTPUT:=/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final.csv}"
: "${BATCH_SIZE:=256}"
: "${DEVICE:=cuda}"

python eval_features.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dataset TUEG \
  --split val

echo "Done. Metrics saved to $OUTPUT"
echo "Visualizations (Scatter Plots & Combined Figure) saved to ${OUTPUT%.*}_viz/"
