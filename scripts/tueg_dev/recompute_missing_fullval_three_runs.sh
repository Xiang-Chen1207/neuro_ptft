#!/bin/bash
set -euo pipefail

source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

REPO_DIR=/vePFS-0x0d/home/cx/ptft
EVAL_SCRIPT="$REPO_DIR/scripts/tueg_dev/eval_val_from_checkpoints.py"

RUN_DIRS=(
  /vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio
  /vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next
  /vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen
)

# Optional override: comma-separated absolute run dirs.
# Example:
# TARGET_RUN_DIRS="/path/run_a,/path/run_b" bash ...
if [[ -n "${TARGET_RUN_DIRS:-}" ]]; then
  IFS=',' read -r -a RUN_DIRS <<< "$TARGET_RUN_DIRS"
fi

# true: run cbramod/eegmamba/reve/tech concurrently per run dir (one GPU each)
# false: run models serially.
PARALLEL_MODE=${PARALLEL_MODE:-true}

log() {
  echo "[$(date '+%F %T')] $*"
}

gpu_for_model() {
  local model_dir_name="$1"
  case "$model_dir_name" in
    cbramod_*) echo 0 ;;
    eegmamba_*) echo 1 ;;
    reve_*) echo 2 ;;
    tech_*) echo 3 ;;
    *) echo 0 ;;
  esac
}

compute_missing_epochs() {
  local model_dir="$1"
  local fullcsv="$model_dir/val_metrics_recomputed_fullval.csv"

  mapfile -t ckpt_epochs < <(ls "$model_dir"/checkpoint_epoch_*.pth 2>/dev/null | sed -E 's/.*checkpoint_epoch_([0-9]+)\.pth/\1/' | sort -n -u)
  mapfile -t done_epochs < <(awk -F, 'NR>1{print $1}' "$fullcsv" 2>/dev/null | sort -n -u)

  if [[ ${#ckpt_epochs[@]} -eq 0 ]]; then
    return
  fi

  comm -23 \
    <(printf "%s\n" "${ckpt_epochs[@]}" | sed '/^$/d' | sort -n -u) \
    <(printf "%s\n" "${done_epochs[@]}" | sed '/^$/d' | sort -n -u)
}

merge_csv() {
  local base_csv="$1"
  local extra_csv="$2"

  python - "$base_csv" "$extra_csv" << 'PY'
import csv
import sys

base_csv, extra_csv = sys.argv[1], sys.argv[2]

rows_by_epoch = {}
all_keys = set(["epoch"])

for path in [base_csv, extra_csv]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            e = int(float(r["epoch"]))
            rows_by_epoch[e] = r
            all_keys.update(r.keys())

ordered = ["epoch"] + sorted(k for k in all_keys if k != "epoch")
rows = [rows_by_epoch[e] for e in sorted(rows_by_epoch.keys())]

with open(base_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=ordered)
    w.writeheader()
    for r in rows:
      out = {k: r.get(k, "") for k in ordered}
      w.writerow(out)

print(f"merged_rows={len(rows)}")
PY
}

eval_one_model() {
  local model_dir="$1"
  local model_name
  model_name=$(basename "$model_dir")
  local model_log="$model_dir/recompute_val_fullval.log"
  local fullcsv="$model_dir/val_metrics_recomputed_fullval.csv"
  local fallback_csv="$model_dir/val_metrics_recomputed.csv"

  if [[ ! -f "$fullcsv" && -f "$fallback_csv" ]]; then
    cp "$fallback_csv" "$fullcsv"
    log "seed fullval csv from fallback for $model_name"
  fi

  if [[ ! -f "$fullcsv" ]]; then
    echo "missing base csv for $model_name: $fullcsv" >&2
    return 1
  fi

  mapfile -t missing < <(compute_missing_epochs "$model_dir")
  if [[ ${#missing[@]} -eq 0 ]]; then
    log "$model_name no missing checkpoints"
    return 0
  fi

  local gpu
  gpu=$(gpu_for_model "$model_name")
  log "$model_name missing=${#missing[@]} gpu=$gpu epochs=$(printf '%s ' "${missing[@]}")"

  local tmpdir
  tmpdir=$(mktemp -d)
  local epoch
  for epoch in "${missing[@]}"; do
    ln -s "$model_dir/checkpoint_epoch_${epoch}.pth" "$tmpdir/checkpoint_epoch_${epoch}.pth"
  done

  {
    echo "[$(date '+%F %T')] start missing recompute: $model_name"
    python -u "$EVAL_SCRIPT" \
      --run-dir "$tmpdir" \
      --pattern 'checkpoint_epoch_*.pth' \
      --device "cuda:${gpu}" \
      --full-val \
      --out-csv "$tmpdir/missing_fullval.csv"

    merge_csv "$fullcsv" "$tmpdir/missing_fullval.csv"
    echo "[$(date '+%F %T')] done missing recompute: $model_name"
  } >> "$model_log" 2>&1

  rm -rf "$tmpdir"
}

run_one_folder_batch() {
  local run_dir="$1"
  local mdir
  local pids=()
  local fail=0
  local p

  log "=== run_dir: $run_dir ==="

  if [[ "$PARALLEL_MODE" == "true" ]]; then
    for mdir in "$run_dir"/cbramod_* "$run_dir"/eegmamba_* "$run_dir"/reve_* "$run_dir"/tech_*; do
      [[ -d "$mdir" ]] || continue
      eval_one_model "$mdir" &
      pids+=("$!")
    done

    for p in "${pids[@]}"; do
      if ! wait "$p"; then
        fail=1
      fi
    done

    if [[ "$fail" -ne 0 ]]; then
      log "run_dir failed: $run_dir"
      return 1
    fi
  else
    for mdir in "$run_dir"/cbramod_* "$run_dir"/eegmamba_* "$run_dir"/reve_* "$run_dir"/tech_*; do
      [[ -d "$mdir" ]] || continue
      if ! eval_one_model "$mdir"; then
        log "model failed: $(basename "$mdir")"
        return 1
      fi
    done
  fi

  log "run_dir completed: $run_dir"
}

cd "$REPO_DIR"
log "start recompute missing fullval across 3 run dirs"

for r in "${RUN_DIRS[@]}"; do
  run_one_folder_batch "$r"
done

log "all done"
