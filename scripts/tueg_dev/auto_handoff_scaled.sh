#!/bin/bash
set -euo pipefail

# Wait until current run finishes recon for all 4 models, then launch a scaled
# follow-up run for feat_only + joint.

CURRENT_RUN_NAME=${CURRENT_RUN_NAME:-20260313_181700_onegpu_full_safeio}
OUTPUT_BASE=${OUTPUT_BASE:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs}
CURRENT_RUN_DIR="$OUTPUT_BASE/$CURRENT_RUN_NAME"

POLL_SECONDS=${POLL_SECONDS:-60}

FOLLOWUP_RUN_NAME=${FOLLOWUP_RUN_NAME:-${CURRENT_RUN_NAME}_scaled_next}
FOLLOWUP_LOG=${FOLLOWUP_LOG:-$OUTPUT_BASE/$FOLLOWUP_RUN_NAME.auto_handoff.log}

mkdir -p "$OUTPUT_BASE"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$FOLLOWUP_LOG"
}

all_recon_finished() {
  local models=(cbramod eegmamba reve tech)
  local m
  for m in "${models[@]}"; do
    local f="$CURRENT_RUN_DIR/${m}_recon.log"
    if [[ ! -f "$f" ]]; then
      return 1
    fi
    if ! grep -q "Experiment Finished\." "$f"; then
      return 1
    fi
  done
  return 0
}

log "auto handoff watcher started"
log "watching recon completion under: $CURRENT_RUN_DIR"
log "follow-up run name: $FOLLOWUP_RUN_NAME"

while true; do
  if all_recon_finished; then
    log "all recon logs report 'Experiment Finished.'"
    break
  fi
  sleep "$POLL_SECONDS"
done

# Stop the old orchestrator before it launches its own feat_only/joint batch.
pkill -f 'scripts/tueg_dev/run_tueg_onegpu_full.sh' || true
sleep 2

log "launching scaled follow-up for VARIANTS=feat_only,joint"
(
  cd /vePFS-0x0d/home/cx/ptft
  export PYTHONUNBUFFERED=1
  RUN_NAME="$FOLLOWUP_RUN_NAME" \
  VARIANTS=feat_only,joint \
  RESUME=auto \
  EPOCHS=20 \
  VAL_FREQ_SPLIT=0 \
  BATCH_CBRAMOD=1024 \
  LR_CBRAMOD=0.002 \
  BATCH_EEGMAMBA=1024 \
  LR_EEGMAMBA=0.002 \
  BATCH_REVE=512 \
  LR_REVE=0.001 \
  BATCH_TECH=1280 \
  LR_TECH=0.0025 \
  NUM_WORKERS=4 \
  PREFETCH_FACTOR=2 \
  VAL_NUM_WORKERS=0 \
  VAL_PERSISTENT_WORKERS=false \
  VAL_TIMEOUT=0 \
  bash scripts/tueg_dev/run_tueg_onegpu_full.sh
) >> "$FOLLOWUP_LOG" 2>&1 &

NEW_PID=$!
log "follow-up launched, pid=$NEW_PID"
