#!/bin/bash
set -euo pipefail

source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

# Keep the same shared cache path strategy as existing tueg_dev scripts.
export SHARED_CACHE_DIR=${SHARED_CACHE_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/shared_cache}
export SHARED_INDEX_PATH=${SHARED_INDEX_PATH:-$SHARED_CACHE_DIR/tueg_dataset_index.json}
export SHARED_FEATURE_CACHE_PATH=${SHARED_FEATURE_CACHE_PATH:-$SHARED_CACHE_DIR/tuep_feature_map.pkl}
mkdir -p "$SHARED_CACHE_DIR"

# ===== User Controls =====
# Models: cbramod,eegmamba,reve,tech
MODELS=${MODELS:-cbramod,eegmamba,reve,tech}

# Variants: recon,joint,scheme_a,scheme_b
VARIANTS=${VARIANTS:-joint}

# Resume policy: auto / always / never
RESUME=${RESUME:-auto}

# Debug quick run: true / false
TINY=${TINY:-false}

# Unified output directory for this batch run.
OUTPUT_BASE=${OUTPUT_BASE:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_multi}
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT"
# ========================

TINY_FLAG=""
if [ "$TINY" = "true" ]; then
  TINY_FLAG="--tiny"
fi

echo "[run_tueg_multi] MODELS=$MODELS"
echo "[run_tueg_multi] VARIANTS=$VARIANTS"
echo "[run_tueg_multi] RESUME=$RESUME"
echo "[run_tueg_multi] RUN_NAME=$RUN_NAME"
echo "[run_tueg_multi] OUTPUT_ROOT=$OUTPUT_ROOT"

echo "[run_tueg_multi] Shared index cache: $SHARED_INDEX_PATH"
echo "[run_tueg_multi] Shared feature cache: $SHARED_FEATURE_CACHE_PATH"

python3 scripts/tueg_dev/run_multi_models.py \
  --script-dir scripts/tueg_dev \
  --models "$MODELS" \
  --variants "$VARIANTS" \
  --resume "$RESUME" \
  --output-root "$OUTPUT_ROOT" \
  $TINY_FLAG
