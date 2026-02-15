#!/bin/bash
set -e
set -o pipefail

# === Configuration ===
# Project & Environment
PROJECT_ROOT="/vePFS-0x0d/home/cx/ptft"
ENV_PATH="/vePFS-0x0d/home/cx/miniconda3/bin/activate"
ENV_NAME="labram"
GPU_IDS="0,1,2,3"

# Hyperparameters (Same as baseline)
BATCH_SIZE=128
LEARNING_RATE=1e-4
EPOCHS=20
NUM_WORKERS=16

# Paths
OUTPUT_DIR="experiments/tuev_full_ft/results_featonly_seed6_head"
CONFIG="configs/finetune_tuev.yaml"
DATASET_DIR="/vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Events"

# Weights Paths
FEATONLY_WEIGHTS="/vepfs-0x0d/home/cx/ptft/output/sanity_feat_only_all_60s/checkpoint_epoch_25.pth"

# === Initialization & Checks ===

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project root $PROJECT_ROOT does not exist."
    exit 1
fi
cd "$PROJECT_ROOT"

if [ ! -f "$ENV_PATH" ]; then
     source "$(conda info --base)/etc/profile.d/conda.sh" || true
     conda activate "$ENV_NAME"
else
     source "$ENV_PATH" "$ENV_NAME"
fi

ulimit -n 65535 || echo "Warning: Could not set ulimit -n 65535. Proceeding with default."

REQUIRED_FILES=("$CONFIG" "$FEATONLY_WEIGHTS")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run_featonly_$(date +%Y%m%d_%H%M%S).log"

# === Execution ===
echo "=== Starting Full Fine-tuning Experiment (FEATONLY - TUEV) ===" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python experiments/tuev_full_ft/run_full_ft_compare.py \
    --config "$CONFIG" \
    --featonly_path "$FEATONLY_WEIGHTS" \
    --run_models FeatOnly \
    --output_dir "$OUTPUT_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --seed 6 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success. Results saved to $OUTPUT_DIR" | tee -a "$LOG_FILE"
else
    echo "Experiment failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi
