#!/bin/bash
set -e
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export ENABLE_WANDB=true
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-12}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-6}
IO_GROUP_CHUNK_SIZE=${IO_GROUP_CHUNK_SIZE:-2048}
H5_RDCC_NBYTES=${H5_RDCC_NBYTES:-67108864}
H5_CACHE_SIZE=${H5_CACHE_SIZE:-512}
SHARED_CACHE_DIR=${SHARED_CACHE_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/shared_cache}
SHARED_INDEX_PATH=${SHARED_INDEX_PATH:-$SHARED_CACHE_DIR/tueg_dataset_index.json}
SHARED_FEATURE_CACHE_PATH=${SHARED_FEATURE_CACHE_PATH:-$SHARED_CACHE_DIR/tuep_feature_map.pkl}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-512}

OUTPUT_DIR=${OUTPUT_DIR:-"output/tueg_dev/tech_joint"}
PROJECT_NAME=${PROJECT_NAME:-"ptft-tech-joint"}
export PROJECT_NAME

# Ensure output directory exists
mkdir -p $OUTPUT_DIR
mkdir -p $SHARED_CACHE_DIR

echo "Starting Experiment: $PROJECT_NAME"
echo "Using dataset.batch_size=$BATCH_SIZE"
echo "Using shared dataset index cache: $SHARED_INDEX_PATH"

RESUME_ARG=""
CHECKPOINT_PATH="${OUTPUT_DIR}/latest.pth"

if [ "$1" == "resume" ]; then
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Resuming from checkpoint: $CHECKPOINT_PATH"
        RESUME_ARG="--resume $CHECKPOINT_PATH"
    fi
elif [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found existing checkpoint. Auto-resuming..."
    RESUME_ARG="--resume $CHECKPOINT_PATH"
fi
python3 main.py \
  --config configs/pretrain_tech.yaml \
  $RESUME_ARG \
  --opts \
    output_dir="$OUTPUT_DIR" \
    project="$PROJECT_NAME" \
    dataset.feature_path='/vePFS-0x0d/home/dht/eeg_feature_extraction_full/results/tuep/features_zscore.csv' \
    dataset.name='TUEG' \
    enable_wandb=$ENABLE_WANDB \
    dataset.batch_size=$BATCH_SIZE \
    dataset.num_workers=$NUM_WORKERS \
    dataset.persistent_workers=true \
    dataset.prefetch_factor=$PREFETCH_FACTOR \
    dataset.val_num_workers=${VAL_NUM_WORKERS:-0} \
    dataset.val_persistent_workers=${VAL_PERSISTENT_WORKERS:-false} \
    dataset.val_timeout=${VAL_TIMEOUT:-0} \
    dataset.cache_path="$SHARED_INDEX_PATH" \
    dataset.feature_cache_path="$SHARED_FEATURE_CACHE_PATH" \
    dataset.enable_feature_filter_cache=true \
    dataset.io_group_shuffle=true \
    dataset.io_group_chunk_size=$IO_GROUP_CHUNK_SIZE \
    dataset.h5_rdcc_nbytes=$H5_RDCC_NBYTES \
    dataset.h5_cache_size=$H5_CACHE_SIZE \
    optimizer.lr=${LR:-1e-3} \
    epochs=${EPOCHS:-50} \
    val_freq_split=${VAL_FREQ_SPLIT:-5} \
    model.pretrain_tasks=['reconstruction','feature_pred'] \
    loss.feature_loss_weight=1.0 \
    loss.use_dynamic_loss=false

echo "Experiment Finished."
