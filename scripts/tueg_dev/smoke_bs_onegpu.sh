#!/bin/bash
set -euo pipefail

source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba

MODELS=${MODELS:-cbramod,eegmamba,reve,tech}
GPU_LIST=${GPU_LIST:-0,1,2,3}
CANDIDATES=${CANDIDATES:-512,640,768,896,1024}
VARIANT=${VARIANT:-joint}

BASE_LR=${BASE_LR:-1e-3}
OUTPUT_BASE=${OUTPUT_BASE:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/smoke}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_bs_smoke}
OUT_DIR="$OUTPUT_BASE/$RUN_NAME"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
IFS=',' read -r -a BS_ARR <<< "$CANDIDATES"

SUMMARY="$OUT_DIR/summary.csv"
echo "model,gpu,best_bs,best_lr,status" > "$SUMMARY"

echo "[smoke] out_dir=$OUT_DIR"
echo "[smoke] models=$MODELS variant=$VARIANT candidates=$CANDIDATES"

for i in "${!MODEL_ARR[@]}"; do
  model="$(echo "${MODEL_ARR[$i]}" | xargs)"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"

  best_bs=0
  best_lr=""
  status="failed_all"

  for bs in "${BS_ARR[@]}"; do
    bs="$(echo "$bs" | xargs)"
    lr=$(awk -v b="$bs" -v base="$BASE_LR" 'BEGIN { printf "%.10g", base * (b / 512.0) }')

    test_out="$OUT_DIR/${model}_bs${bs}"
    test_log="$LOG_DIR/${model}_bs${bs}.log"
    mkdir -p "$test_out"

    echo "[smoke] model=$model gpu=$gpu bs=$bs lr=$lr"

    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE \
    python3 scripts/tueg_dev/run_multi_models.py \
      --script-dir scripts/tueg_dev \
      --models "$model" \
      --variants "$VARIANT" \
      --resume never \
      --tiny \
      --output-root "$test_out" \
      --batch-size "$bs" \
      --num-workers 2 \
      --prefetch-factor 2 \
      --val-num-workers 0 \
      --val-timeout 0 \
      --val-freq-split 0 \
      --epochs 1 \
      > "$test_log" 2>&1
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
      best_bs="$bs"
      best_lr="$lr"
      status="ok"
    else
      if [ "$best_bs" -gt 0 ]; then
        status="ok"
      fi
      break
    fi
  done

  if [ "$best_bs" -eq 0 ]; then
    best_lr="n/a"
  fi
  echo "$model,$gpu,$best_bs,$best_lr,$status" >> "$SUMMARY"
done

echo "[smoke] done"
echo "[smoke] summary=$SUMMARY"
cat "$SUMMARY"
