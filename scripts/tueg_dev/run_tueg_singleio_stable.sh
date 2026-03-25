#!/bin/bash
set -euo pipefail

source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba

# Single-process shared-IO mode: one dataloader group is built once and reused across models.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

export SHARED_CACHE_DIR=${SHARED_CACHE_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/shared_cache}
export SHARED_INDEX_PATH=${SHARED_INDEX_PATH:-$SHARED_CACHE_DIR/tueg_dataset_index.json}
export SHARED_FEATURE_CACHE_PATH=${SHARED_FEATURE_CACHE_PATH:-$SHARED_CACHE_DIR/tuep_feature_map.pkl}
mkdir -p "$SHARED_CACHE_DIR"

# ===== User Controls =====
MODELS=${MODELS:-cbramod,eegmamba,reve,tech}
VARIANTS=${VARIANTS:-recon,feat_only,joint}
RESUME=${RESUME:-never}
TINY=${TINY:-false}

# Tuned for current low memory usage + stability against val deadlocks.
BATCH_SIZE=${BATCH_SIZE:-160}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-0}
VAL_PERSISTENT_WORKERS=${VAL_PERSISTENT_WORKERS:-false}
VAL_PREFETCH_FACTOR=${VAL_PREFETCH_FACTOR:-2}
LOADER_TIMEOUT=${LOADER_TIMEOUT:-300}
VAL_TIMEOUT=${VAL_TIMEOUT:-0}
VAL_FREQ_SPLIT=${VAL_FREQ_SPLIT:-8}

OUTPUT_BASE=${OUTPUT_BASE:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_singleio_stable}
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT"
# ========================

TINY_FLAG=""
if [ "$TINY" = "true" ]; then
  TINY_FLAG="--tiny"
fi

echo "[run_tueg_singleio_stable] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_tueg_singleio_stable] MODELS=$MODELS"
echo "[run_tueg_singleio_stable] VARIANTS=$VARIANTS"
echo "[run_tueg_singleio_stable] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[run_tueg_singleio_stable] BATCH_SIZE=$BATCH_SIZE NUM_WORKERS=$NUM_WORKERS PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "[run_tueg_singleio_stable] VAL_NUM_WORKERS=$VAL_NUM_WORKERS VAL_PERSISTENT_WORKERS=$VAL_PERSISTENT_WORKERS"

python3 scripts/tueg_dev/run_multi_models.py \
  --script-dir scripts/tueg_dev \
  --models "$MODELS" \
  --variants "$VARIANTS" \
  --resume "$RESUME" \
  --output-root "$OUTPUT_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --val-num-workers "$VAL_NUM_WORKERS" \
  --val-persistent-workers "$VAL_PERSISTENT_WORKERS" \
  --val-prefetch-factor "$VAL_PREFETCH_FACTOR" \
  --loader-timeout "$LOADER_TIMEOUT" \
  --val-timeout "$VAL_TIMEOUT" \
  --val-freq-split "$VAL_FREQ_SPLIT" \
  $TINY_FLAG
