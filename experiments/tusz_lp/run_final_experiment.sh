#!/bin/bash
set -e
# Activate environment
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram

export PTFT_DATASET="TUSZ"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- Configuration ---

# Dataset Directory
DATASET_DIR="/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure"

# Model Weights Paths (PLEASE UPDATE THESE PATHS)
# Baseline: Usually the reconstruction-only pretrained model
BASELINE_WEIGHTS="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"
FLAGSHIP_WEIGHTS="/vePFS-0x0d/home/cx/ptft/output_old/flagship_cross_attn/checkpoint_epoch_6.pth"
FEATONLY_WEIGHTS="/vePFS-0x0d/home/cx/ptft/output_old/sanity_feat_only_all_60s/checkpoint_epoch_25.pth"

# Output Feature Files
BASELINE_FEAT="experiments/tusz_lp/features/recon_features.npz"
FLAGSHIP_FEAT="experiments/tusz_lp/features/neuro_ke_features.npz"
FEATONLY_FEAT="experiments/tusz_lp/features/feat_only_features.npz"

OUTPUT_REPORT="experiments/tusz_lp/final.md"

# Create output directory
mkdir -p experiments/tusz_lp/features

echo "=== Starting Full Comparative Experiment (TUSZ) (Baseline vs Flagship vs FeatOnly) ==="
echo "Dataset Dir: $DATASET_DIR"

# 1. Feature Extraction - Baseline (Recon)
echo "Extracting Baseline features..."
python experiments/tusz_lp/extract_features.py \
    --model_type recon \
    --weights_path "$BASELINE_WEIGHTS" \
    --output_dir experiments/tusz_lp/features \
    --dataset_dir "$DATASET_DIR" \
    --device cuda

# 2. Feature Extraction - Flagship (Neuro-KE)
echo "Extracting Flagship features..."
python experiments/tusz_lp/extract_features.py \
    --model_type neuro_ke \
    --weights_path "$FLAGSHIP_WEIGHTS" \
    --output_dir experiments/tusz_lp/features \
    --dataset_dir "$DATASET_DIR" \
    --device cuda

# 3. Feature Extraction - FeatOnly
echo "Extracting FeatOnly features..."
python experiments/tusz_lp/extract_features.py \
    --model_type feat_only \
    --weights_path "$FEATONLY_WEIGHTS" \
    --output_dir experiments/tusz_lp/features \
    --dataset_dir "$DATASET_DIR" \
    --device cuda

# 4. Run Comparative Linear Probing
echo "Running Comparative Linear Probing (Incremental Subjects)..."
# Using tee to show output in terminal AND save to file
python experiments/tusz_lp/run_lp.py \
    --baseline_path "$BASELINE_FEAT" \
    --flagship_path "$FLAGSHIP_FEAT" \
    --featonly_path "$FEATONLY_FEAT" \
    --seed 42 \
    | tee "$OUTPUT_REPORT"

echo "Done. Final results saved to $OUTPUT_REPORT"
