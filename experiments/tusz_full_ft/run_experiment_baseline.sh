#!/bin/bash
#!/bin/bash

# Activate environment
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram
set -e
set -o pipefail

PROJECT_ROOT="/vePFS-0x0d/home/cx/ptft"
ENV_PATH="/vePFS-0x0d/home/cx/miniconda3/bin/activate"
ENV_NAME="labram"
GPU_IDS="0,1,2,3"

BATCH_SIZE=32
LEARNING_RATE=1e-4
EPOCHS=20
NUM_WORKERS=8
SPLIT="0.8 0.1 0.1"

OUTPUT_DIR="experiments/tusz_full_ft/results_baseline"
CONFIG="configs/finetune_tusz_dynamic.yaml"
DATASET_DIR="/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure"

BASELINE_WEIGHTS="/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth"

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

ulimit -n 65535 || echo "Warning: Could not set ulimit -n 65535."

REQUIRED_FILES=("$CONFIG" "$BASELINE_WEIGHTS")
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
LOG_FILE="$OUTPUT_DIR/run_baseline_$(date +%Y%m%d_%H%M%S).log"

echo "=== Starting Full Fine-tuning Baseline Experiment (TUSZ) ===" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Split (subject): $SPLIT" | tee -a "$LOG_FILE"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python experiments/tusz_full_ft/run_full_ft_compare.py \
  --config "$CONFIG" \
  --baseline_path "$BASELINE_WEIGHTS" \
  --run_models Baseline \
  --output_dir "$OUTPUT_DIR" \
  --dataset_dir "$DATASET_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LEARNING_RATE" \
  --num_workers "$NUM_WORKERS" \
  --split $SPLIT \
  --device cuda \
  --seed 42 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -eq 0 ]; then
  echo "Success. Results saved to $OUTPUT_DIR" | tee -a "$LOG_FILE"
else
  echo "Experiment failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
  exit $EXIT_CODE
fi
