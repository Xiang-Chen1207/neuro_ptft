#!/usr/bin/env bash
# Evaluate feature prediction performance using the feature-prediction-only pretrained model
# (sanity_feat_only: only feature prediction task, no reconstruction)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

: "${CHECKPOINT:=/vepfs-0x0d/home/cx/ptft/output/sanity_feat_only_all_60s/checkpoint_epoch_25.pth}"
: "${CONFIG:=configs/pretrain.yaml}"
: "${OUTPUT:=/vepfs-0x0d/home/cx/ptft/experiments/feature_pred_validation/feat_only_final.csv}"
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
