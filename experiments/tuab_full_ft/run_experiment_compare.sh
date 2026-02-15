#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail # Return value of a pipeline is the status of the last command to exit with a non-zero status

# === Configuration ===
# Project & Environment
PROJECT_ROOT="/vePFS-0x0d/home/cx/ptft"
ENV_PATH="/vePFS-0x0d/home/cx/miniconda3/bin/activate"
ENV_NAME="labram"
GPU_IDS="0,1"

# Hyperparameters
BATCH_SIZE=512 # 128 per GPU * 4 GPUs
LEARNING_RATE=0.0004 # Scaled linearly from 0.0016 for BS=2048 (0.0016 / 4)
EPOCHS=10
NUM_WORKERS=16

# Paths
OUTPUT_DIR="experiments/tuab_full_ft/results_compare60s"
CONFIG="configs/finetune.yaml"
DATASET_DIR="/vePFS-0x0d/eeg-data/TUAB"

# Weights Paths
BASELINE_WEIGHTS="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"
FLAGSHIP_WEIGHTS="/vePFS-0x0d/home/cx/ptft/output_old/flagship_cross_attn/checkpoint_epoch_6.pth"
FEATONLY_WEIGHTS="/vepfs-0x0d/home/cx/ptft/output/sanity_feat_only_all_60s/checkpoint_epoch_27.pth" # Assuming relative path works or fix it if needed

# === Initialization & Checks ===

# 1. Check Project Root
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project root $PROJECT_ROOT does not exist."
    exit 1
fi
cd "$PROJECT_ROOT"

# 2. Activate Environment
if [ ! -f "$ENV_PATH" ]; then
     # Fallback: try sourcing conda directly if activate script not found at explicit path
     # But for now, we stick to the user's known path or just source it.
     # Note: 'source' might not work if script is run with sh instead of bash, but shebang says bash.
     echo "Warning: Activation script not found at $ENV_PATH. Trying generic conda activate."
     source "$(conda info --base)/etc/profile.d/conda.sh" || true
     conda activate "$ENV_NAME"
else
     source "$ENV_PATH" "$ENV_NAME"
fi

# Increase file descriptor limit for high concurrency data loading
ulimit -n 65535 || echo "Warning: Could not set ulimit -n 65535. Proceeding with default."

# 3. Check Critical Files
REQUIRED_FILES=("$CONFIG" "$BASELINE_WEIGHTS" "$FLAGSHIP_WEIGHTS" "$FEATONLY_WEIGHTS")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
done

# 4. Check Dataset Directory
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# 5. Setup Logging
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# === Execution ===
echo "=== Starting Full Fine-tuning Comparative Experiment ===" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Run Python Script
# Using 'python3' is safer if 'python' is ambiguous, but conda env usually maps 'python' correctly.
python experiments/tuab_full_ft/run_full_ft_compare.py \
    --config "$CONFIG" \
    --baseline_path "$BASELINE_WEIGHTS" \
    --flagship_path "$FLAGSHIP_WEIGHTS" \
    --featonly_path "$FEATONLY_WEIGHTS" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --seed 42 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success. Results saved to $OUTPUT_DIR" | tee -a "$LOG_FILE"
else
    echo "Experiment failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi
