#!/bin/bash
set -euo pipefail

# Resume the 4 recon models under one existing run directory for +N epochs.
# Default behavior targets the known safeio run and extends 20 -> 30 epochs.

RUN_DIR=${RUN_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio}
EXTRA_EPOCHS=${EXTRA_EPOCHS:-10}

# Keep these aligned with the original safe-IO recon run.
GPU_CBRAMOD=${GPU_CBRAMOD:-0}
GPU_EEGMAMBA=${GPU_EEGMAMBA:-1}
GPU_REVE=${GPU_REVE:-2}
GPU_TECH=${GPU_TECH:-3}

BATCH_CBRAMOD=${BATCH_CBRAMOD:-1024}
BATCH_EEGMAMBA=${BATCH_EEGMAMBA:-1024}
BATCH_REVE=${BATCH_REVE:-512}
BATCH_TECH=${BATCH_TECH:-1280}

LR_CBRAMOD=${LR_CBRAMOD:-0.002}
LR_EEGMAMBA=${LR_EEGMAMBA:-0.002}
LR_REVE=${LR_REVE:-0.001}
LR_TECH=${LR_TECH:-0.0025}

NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-0}
VAL_PERSISTENT_WORKERS=${VAL_PERSISTENT_WORKERS:-false}
VAL_TIMEOUT=${VAL_TIMEOUT:-0}
VAL_FREQ_SPLIT=${VAL_FREQ_SPLIT:-0}

if [[ ! -d "$RUN_DIR" ]]; then
  echo "RUN_DIR not found: $RUN_DIR"
  exit 1
fi

for m in cbramod eegmamba reve tech; do
  if [[ ! -f "$RUN_DIR/${m}_recon/latest.pth" ]]; then
    echo "Missing checkpoint: $RUN_DIR/${m}_recon/latest.pth"
    exit 1
  fi
  if [[ ! -f "$RUN_DIR/${m}_recon/log.csv" ]]; then
    echo "Missing log.csv: $RUN_DIR/${m}_recon/log.csv"
    exit 1
  fi
done

epoch_cbramod=$(awk -F, 'NR>1 && $1+0>max {max=$1+0} END {print max+0}' "$RUN_DIR/cbramod_recon/log.csv")
epoch_eegmamba=$(awk -F, 'NR>1 && $1+0>max {max=$1+0} END {print max+0}' "$RUN_DIR/eegmamba_recon/log.csv")
epoch_reve=$(awk -F, 'NR>1 && $1+0>max {max=$1+0} END {print max+0}' "$RUN_DIR/reve_recon/log.csv")
epoch_tech=$(awk -F, 'NR>1 && $1+0>max {max=$1+0} END {print max+0}' "$RUN_DIR/tech_recon/log.csv")

if [[ "$epoch_cbramod" != "$epoch_eegmamba" || "$epoch_cbramod" != "$epoch_reve" || "$epoch_cbramod" != "$epoch_tech" ]]; then
  echo "Epoch mismatch across models in $RUN_DIR"
  echo "cbramod=$epoch_cbramod eegmamba=$epoch_eegmamba reve=$epoch_reve tech=$epoch_tech"
  echo "Please align checkpoints/logs first, then resume together."
  exit 1
fi

cur_max_epoch=$epoch_cbramod
current_epochs=$((cur_max_epoch + 1))
target_epochs=$((current_epochs + EXTRA_EPOCHS))

OUTPUT_BASE=$(dirname "$RUN_DIR")
RUN_NAME=$(basename "$RUN_DIR")

echo "[resume] RUN_DIR=$RUN_DIR"
echo "[resume] current_epochs=$current_epochs EXTRA_EPOCHS=$EXTRA_EPOCHS target_epochs=$target_epochs"
echo "[resume] This will run recon only and append/update each model log.csv"

cd /vePFS-0x0d/home/cx/ptft

export PYTHONUNBUFFERED=1

RUN_NAME="$RUN_NAME" \
OUTPUT_BASE="$OUTPUT_BASE" \
VARIANTS=recon \
RESUME=always \
EPOCHS="$target_epochs" \
VAL_FREQ_SPLIT="$VAL_FREQ_SPLIT" \
GPU_CBRAMOD="$GPU_CBRAMOD" \
GPU_EEGMAMBA="$GPU_EEGMAMBA" \
GPU_REVE="$GPU_REVE" \
GPU_TECH="$GPU_TECH" \
BATCH_CBRAMOD="$BATCH_CBRAMOD" \
BATCH_EEGMAMBA="$BATCH_EEGMAMBA" \
BATCH_REVE="$BATCH_REVE" \
BATCH_TECH="$BATCH_TECH" \
LR_CBRAMOD="$LR_CBRAMOD" \
LR_EEGMAMBA="$LR_EEGMAMBA" \
LR_REVE="$LR_REVE" \
LR_TECH="$LR_TECH" \
NUM_WORKERS="$NUM_WORKERS" \
PREFETCH_FACTOR="$PREFETCH_FACTOR" \
VAL_NUM_WORKERS="$VAL_NUM_WORKERS" \
VAL_PERSISTENT_WORKERS="$VAL_PERSISTENT_WORKERS" \
VAL_TIMEOUT="$VAL_TIMEOUT" \
bash scripts/tueg_dev/run_tueg_onegpu_full.sh

echo "[resume] Training command finished. Verifying log.csv..."
for m in cbramod eegmamba reve tech; do
  log_csv="$RUN_DIR/${m}_recon/log.csv"
  max_epoch=$(awk -F, 'NR>1 && $1+0>max {max=$1+0} END {print max+0}' "$log_csv")
  rows=$(wc -l < "$log_csv")
  echo "[resume] ${m}_recon: rows=$rows max_epoch=$max_epoch file=$log_csv"
done

echo "[resume] Done."