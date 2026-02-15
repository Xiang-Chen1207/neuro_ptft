#!/bin/bash
set -e
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram
export PTFT_DATASET="BCIC2A"
# Restrict to GPUs with available memory (0 and 1 have some space)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Paths
BASELINE_WEIGHTS="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"
FLAGSHIP_WEIGHTS="/vePFS-0x0d/home/cx/ptft/output_old/flagship_cross_attn/checkpoint_epoch_6.pth"
FEATONLY_WEIGHTS="/vepfs-0x0d/home/cx/ptft/output/sanity_feat_only_all_60s/checkpoint_epoch_25.pth"

# Output Features Paths
BASELINE_FEAT="experiments/bcic2a_lr/features/recon_features.npz"
FLAGSHIP_FEAT="experiments/bcic2a_lr/features/neuro_ke_features.npz"
FEATONLY_FEAT="experiments/bcic2a_lr_feat_only/features/feat_only_features.npz"

OUTPUT_REPORT="experiments/bcic2a_lr/final.md"
DATASET_DIR="/vePFS-0x0d/home/downstream_data/BCIC2A"

echo "=== Starting Full Comparative Experiment (BCIC2A) (Baseline vs Flagship vs FeatOnly) ==="

# 1. Feature Extraction - Baseline
echo "Extracting Baseline features..."
python experiments/bcic2a_lr/extract_features.py \
    --model_type recon \
    --weights_path "$BASELINE_WEIGHTS" \
    --output_dir experiments/bcic2a_lr/features \
    --dataset_dir "$DATASET_DIR" \
    --device cuda

# 2. Feature Extraction - Flagship
echo "Extracting Flagship features..."
python experiments/bcic2a_lr/extract_features.py \
    --model_type neuro_ke \
    --weights_path "$FLAGSHIP_WEIGHTS" \
    --output_dir experiments/bcic2a_lr/features \
    --dataset_dir "$DATASET_DIR" \
    --device cuda

# 3. Feature Extraction - FeatOnly
echo "Extracting FeatOnly features..."
mkdir -p experiments/bcic2a_lr_feat_only/features
python experiments/bcic2a_lr/extract_features.py \
    --model_type feat_only \
    --weights_path "$FEATONLY_WEIGHTS" \
    --output_dir experiments/bcic2a_lr_feat_only/features \
    --dataset_dir "$DATASET_DIR" \
    --device cuda

# 4. Run Comparative Linear Probing
echo "Running Comparative Linear Probing (Incremental Subjects)..."
# Using tee to show output in terminal AND save to file
python experiments/bcic2a_lr/run_lp_compare.py \
    --baseline_path "$BASELINE_FEAT" \
    --flagship_path "$FLAGSHIP_FEAT" \
    --featonly_path "$FEATONLY_FEAT" \
    --seed 42 \
    | tee "$OUTPUT_REPORT"

echo "Done. Final results saved to $OUTPUT_REPORT"
