#!/bin/bash
set -euo pipefail

source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

SHARED_CACHE_DIR=${SHARED_CACHE_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/shared_cache}
SHARED_INDEX_PATH=${SHARED_INDEX_PATH:-$SHARED_CACHE_DIR/tueg_dataset_index.json}
SHARED_FEATURE_CACHE_PATH=${SHARED_FEATURE_CACHE_PATH:-$SHARED_CACHE_DIR/tuep_feature_map.pkl}
mkdir -p "$SHARED_CACHE_DIR"

# ===== User Controls =====
# Models: cbramod,eegmamba,reve,tech
MODELS=${MODELS:-cbramod,eegmamba,reve,tech}

# Variants: joint,recon,feat_only (comma-separated)
VARIANTS=${VARIANTS:-joint}

# GPU ids used for parallel jobs. Jobs are mapped in order.
GPU_LIST=${GPU_LIST:-0,1,2,3}

# Per-job dataloader knobs to avoid overloading CPU/IO in parallel runs.
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}

OUTPUT_BASE=${OUTPUT_BASE:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)_parallel}
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT"
# ========================

IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"

if [ "${#MODEL_ARR[@]}" -eq 0 ]; then
	echo "No models selected"
	exit 1
fi

if [ "${#GPU_ARR[@]}" -eq 0 ]; then
	echo "No GPUs selected"
	exit 1
fi

echo "[run_tueg_parallel] MODELS=$MODELS"
echo "[run_tueg_parallel] VARIANTS=$VARIANTS"
echo "[run_tueg_parallel] GPU_LIST=$GPU_LIST"
echo "[run_tueg_parallel] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[run_tueg_parallel] SHARED_INDEX_PATH=$SHARED_INDEX_PATH"
echo "[run_tueg_parallel] SHARED_FEATURE_CACHE_PATH=$SHARED_FEATURE_CACHE_PATH"

declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()

IFS=',' read -r -a VARIANT_ARR <<< "$VARIANTS"
declare -a JOB_MODELS=()
declare -a JOB_VARIANTS=()

for variant in "${VARIANT_ARR[@]}"; do
	v="$(echo "$variant" | xargs)"
	if [ -z "$v" ]; then
		continue
	fi
	for model in "${MODEL_ARR[@]}"; do
		m="$(echo "$model" | xargs)"
		JOB_MODELS+=("$m")
		JOB_VARIANTS+=("$v")
	done
done

job_count=${#JOB_MODELS[@]}
if [ "$job_count" -eq 0 ]; then
	echo "No valid jobs composed from MODELS/VARIANTS."
	exit 1
fi

gpu_count=${#GPU_ARR[@]}
batch_start=0
status=0

while [ "$batch_start" -lt "$job_count" ]; do
	PIDS=()
	NAMES=()
	LOGS=()

	batch_end=$((batch_start + gpu_count - 1))
	if [ "$batch_end" -ge "$job_count" ]; then
		batch_end=$((job_count - 1))
	fi

	echo "[run_tueg_parallel] Launch batch: jobs ${batch_start}-${batch_end}"

	for idx in $(seq "$batch_start" "$batch_end"); do
		model="${JOB_MODELS[$idx]}"
		variant="${JOB_VARIANTS[$idx]}"
		gpu_slot=$((idx - batch_start))
		gpu="${GPU_ARR[$gpu_slot]}"
		script="scripts/tueg_dev/run_${model}_${variant}.sh"

		if [ ! -f "$script" ]; then
			echo "Skip: script not found for model=$model variant=$variant"
			continue
		fi

		run_name="${model}_${variant}"
		model_output="$OUTPUT_ROOT/$run_name"
		model_log="$OUTPUT_ROOT/${run_name}.log"
		mkdir -p "$model_output"

		echo "[launch] $run_name on GPU $gpu"

		(
			export CUDA_VISIBLE_DEVICES="$gpu"
			export SHARED_CACHE_DIR
			export SHARED_INDEX_PATH
			export SHARED_FEATURE_CACHE_PATH
			export NUM_WORKERS
			export PREFETCH_FACTOR
			export OUTPUT_DIR="$model_output"
			export PROJECT_NAME="ptft-${run_name}"
			bash "$script"
		) >"$model_log" 2>&1 &

		PIDS+=("$!")
		NAMES+=("$run_name")
		LOGS+=("$model_log")
	done

	if [ "${#PIDS[@]}" -eq 0 ]; then
		echo "No valid jobs launched in this batch."
		status=1
		break
	fi

	echo "[run_tueg_parallel] Launched ${#PIDS[@]} jobs in current batch."
	for i in "${!PIDS[@]}"; do
		echo "  - ${NAMES[$i]} pid=${PIDS[$i]} log=${LOGS[$i]}"
	done

	batch_status=0
	for i in "${!PIDS[@]}"; do
		pid="${PIDS[$i]}"
		name="${NAMES[$i]}"
		if wait "$pid"; then
			echo "[done] $name: ok"
		else
			echo "[done] $name: failed"
			batch_status=1
			status=1
		fi
	done

	if [ "$batch_status" -ne 0 ]; then
		echo "[run_tueg_parallel] Stop due to failure in current batch."
		break
	fi

	batch_start=$((batch_end + 1))
done

echo "[run_tueg_parallel] All jobs finished. status=$status"
exit "$status"
