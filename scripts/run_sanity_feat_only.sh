#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Ablation B: Feature Prediction Only (Sanity Check)
OUTPUT_DIR="output/sanity_feat_only_all_60s"
mkdir -p $OUTPUT_DIR

echo "Starting Ablation B: Feature Prediction Only"

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

python3 main.py \
  --config configs/pretrain.yaml \
  $RESUME_ARG \
  --opts \
    model.pretrain_tasks=['feature_pred'] \
    model.feature_token_strategy='single' \
    output_dir=$OUTPUT_DIR \
    enable_wandb=true \
    project="ptft-sanity"\
    epochs=50\
    dataset.batch_size=400 \
    dataset.input_size=12000 \
    optimizer.lr=3e-4 \
    val_freq_split=10 \
    model.seq_len=60 \
    dataset.num_workers=16 \
    dataset.prefetch_factor=2 \
    dataset.cache_path="output/dataset_index_60s.json"

echo "Ablation B Finished."
