#!/bin/bash
set -euo pipefail

source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

export SHARED_CACHE_DIR=${SHARED_CACHE_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/shared_cache}
export SHARED_INDEX_PATH=${SHARED_INDEX_PATH:-$SHARED_CACHE_DIR/tueg_dataset_index.json}
export SHARED_FEATURE_CACHE_PATH=${SHARED_FEATURE_CACHE_PATH:-$SHARED_CACHE_DIR/tuep_feature_map.pkl}
mkdir -p "$SHARED_CACHE_DIR"

# ===== User Controls =====
VARIANTS=${VARIANTS:-recon,feat_only,joint}
RESUME=${RESUME:-auto}   # auto / always / never
EPOCHS=${EPOCHS:-20}
VAL_FREQ_SPLIT=${VAL_FREQ_SPLIT:-0}  # 0 => only epoch-end validation

# One GPU per model mapping.
GPU_CBRAMOD=${GPU_CBRAMOD:-0}
GPU_EEGMAMBA=${GPU_EEGMAMBA:-1}
GPU_REVE=${GPU_REVE:-2}
GPU_TECH=${GPU_TECH:-3}

# Batch size defaults from smoke result.
BATCH_CBRAMOD=${BATCH_CBRAMOD:-1024}
BATCH_EEGMAMBA=${BATCH_EEGMAMBA:-1024}
BATCH_REVE=${BATCH_REVE:-1024}
BATCH_TECH=${BATCH_TECH:-1024}

# LR scaling rule: 512 -> 1e-3, scale linearly with batch size.
BASE_BS=${BASE_BS:-512}
BASE_LR=${BASE_LR:-1e-3}
LR_CBRAMOD=${LR_CBRAMOD:-$(awk -v b="$BATCH_CBRAMOD" -v bb="$BASE_BS" -v bl="$BASE_LR" 'BEGIN{printf "%.10g", bl*(b/bb)}')}
LR_EEGMAMBA=${LR_EEGMAMBA:-$(awk -v b="$BATCH_EEGMAMBA" -v bb="$BASE_BS" -v bl="$BASE_LR" 'BEGIN{printf "%.10g", bl*(b/bb)}')}
LR_REVE=${LR_REVE:-$(awk -v b="$BATCH_REVE" -v bb="$BASE_BS" -v bl="$BASE_LR" 'BEGIN{printf "%.10g", bl*(b/bb)}')}
LR_TECH=${LR_TECH:-$(awk -v b="$BATCH_TECH" -v bb="$BASE_BS" -v bl="$BASE_LR" 'BEGIN{printf "%.10g", bl*(b/bb)}')}

OUTPUT_BASE=${OUTPUT_BASE:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_onegpu_full}
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT"
# ========================

if [[ "$RESUME" != "auto" && "$RESUME" != "always" && "$RESUME" != "never" ]]; then
  echo "Unsupported RESUME=$RESUME (use auto|always|never)"
  exit 1
fi

echo "[run_tueg_onegpu_full] VARIANTS=$VARIANTS"
echo "[run_tueg_onegpu_full] RESUME=$RESUME EPOCHS=$EPOCHS VAL_FREQ_SPLIT=$VAL_FREQ_SPLIT"
echo "[run_tueg_onegpu_full] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[run_tueg_onegpu_full] cbramod gpu=$GPU_CBRAMOD bs=$BATCH_CBRAMOD lr=$LR_CBRAMOD"
echo "[run_tueg_onegpu_full] eegmamba gpu=$GPU_EEGMAMBA bs=$BATCH_EEGMAMBA lr=$LR_EEGMAMBA"
echo "[run_tueg_onegpu_full] reve gpu=$GPU_REVE bs=$BATCH_REVE lr=$LR_REVE"
echo "[run_tueg_onegpu_full] tech gpu=$GPU_TECH bs=$BATCH_TECH lr=$LR_TECH"

IFS=',' read -r -a VAR_ARR <<< "$VARIANTS"
status=0
LAST_PID=""

run_one() {
  local model="$1"
  local variant="$2"
  local gpu="$3"
  local bs="$4"
  local lr="$5"

  local script="scripts/tueg_dev/run_${model}_${variant}.sh"
  local name="${model}_${variant}"
  local out_dir="$OUTPUT_ROOT/$name"
  local log_file="$OUTPUT_ROOT/${name}.log"
  mkdir -p "$out_dir"

  local resume_arg=""
  if [[ "$RESUME" == "always" ]]; then
    resume_arg="resume"
  elif [[ "$RESUME" == "never" ]]; then
    rm -f "$out_dir/latest.pth" || true
    rm -f "$out_dir/best.pth" || true
  fi

  echo "[launch] $name gpu=$gpu bs=$bs lr=$lr" >&2
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export BATCH_SIZE="$bs"
    export LR="$lr"
    export EPOCHS="$EPOCHS"
    export VAL_FREQ_SPLIT="$VAL_FREQ_SPLIT"
    export OUTPUT_DIR="$out_dir"
    export PROJECT_NAME="ptft-${name}"
    export SHARED_CACHE_DIR
    export SHARED_INDEX_PATH
    export SHARED_FEATURE_CACHE_PATH
    bash "$script" $resume_arg
  ) >"$log_file" 2>&1 &

  LAST_PID="$!"
}

for vraw in "${VAR_ARR[@]}"; do
  variant="$(echo "$vraw" | xargs)"
  if [[ -z "$variant" ]]; then
    continue
  fi

  echo "[batch] variant=$variant"

  run_one "cbramod" "$variant" "$GPU_CBRAMOD" "$BATCH_CBRAMOD" "$LR_CBRAMOD"
  p1="$LAST_PID"
  run_one "eegmamba" "$variant" "$GPU_EEGMAMBA" "$BATCH_EEGMAMBA" "$LR_EEGMAMBA"
  p2="$LAST_PID"
  run_one "reve" "$variant" "$GPU_REVE" "$BATCH_REVE" "$LR_REVE"
  p3="$LAST_PID"
  run_one "tech" "$variant" "$GPU_TECH" "$BATCH_TECH" "$LR_TECH"
  p4="$LAST_PID"

  for p in "$p1" "$p2" "$p3" "$p4"; do
    if ! wait "$p"; then
      status=1
    fi
  done

  if [[ "$status" -ne 0 ]]; then
    echo "[batch] variant=$variant failed, stop further variants"
    break
  fi

  echo "[batch] variant=$variant done"
done

echo "[run_tueg_onegpu_full] finished status=$status"
exit "$status"
