#!/bin/bash
#!/bin/bash
set -e
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Reconstruction Only Experiment
# Reference: scripts/run_flagship_fixed.sh

OUTPUT_DIR="output/reconstruction_only"
mkdir -p $OUTPUT_DIR

echo "Starting Reconstruction Only Experiment"

RESUME_ARG=""
CHECKPOINT_PATH="${OUTPUT_DIR}/latest.pth"

# Auto-detect checkpoint logic
if [ "$1" == "resume" ]; then
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Resuming from checkpoint: $CHECKPOINT_PATH"
        RESUME_ARG="--resume $CHECKPOINT_PATH"
    else
        echo "Warning: Checkpoint not found at $CHECKPOINT_PATH, starting from scratch"
    fi
elif [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found existing checkpoint at $CHECKPOINT_PATH. Auto-resuming..."
    RESUME_ARG="--resume $CHECKPOINT_PATH"
elif [ -n "$RESUME_PATH" ]; then
    echo "Resuming from checkpoint: $RESUME_PATH"
    RESUME_ARG="--resume $RESUME_PATH"
fi

# Start Training
python3 main.py \
  --config configs/pretrain.yaml \
  $RESUME_ARG \
  --opts \
    model.pretrain_tasks=['reconstruction'] \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-reconstruction-only" \
    epochs=50 \
    dataset.batch_size=256 \
    dataset.num_workers=16 \
    dataset.persistent_workers=true \
    dataset.prefetch_factor=4 \
    optimizer.lr=1e-3 \
    val_freq_split=10 \
    dataset.input_size=12000 \
    model.seq_len=60 \
    dataset.cache_path="output/dataset_index_60s.json"

echo "Experiment Finished."
