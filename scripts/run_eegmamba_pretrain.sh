#!/bin/bash
set -e
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba
export CUDA_VISIBLE_DEVICES=2,3

# Optimization for HDF5 and PyTorch Dataloader
export OMP_NUM_THREADS=4
export HDF5_USE_FILE_LOCKING=FALSE
export ENABLE_WANDB=true

# Ensure we are in the project root
# cd /vePFS-0x0d/home/cx/ptft

OUTPUT_DIR="/vePFS-0x0d/home/cx/ptft/output/joint_pretrain_eegmamba"
mkdir -p $OUTPUT_DIR

echo "Starting EEGMamba Joint Pretraining Experiment"

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
  --config configs/pretrain_eegmamba.yaml \
  $RESUME_ARG \
  --opts \
    model.pretrain_tasks=['reconstruction'] \
    output_dir=$OUTPUT_DIR \
    enable_wandb=$ENABLE_WANDB \
    entity="cx2521-new-york-university" \
    project="ptft-eegmamba-reconstruction" \
    epochs=50 \
    dataset.name="JOINT" \
    dataset.data_csv="/vePFS-0x0d/home/cx/ptft/datasets/data.csv" \
    dataset.batch_size=256 \
    dataset.val_split=0.05 \
    dataset.num_workers=8 \
    dataset.persistent_workers=true \
    dataset.pin_memory=false \
    dataset.prefetch_factor=2 \
    optimizer.lr=1e-3 \
    val_freq_split=10 \
    dataset.input_size=12000 \
    model.seq_len=60 \
    dataset.cache_path="${OUTPUT_DIR}/dataset_index_joint.json"

echo "Experiment Finished."
