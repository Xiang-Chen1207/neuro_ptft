#!/bin/bash
# Extended LP benchmark: 12 checkpoints x 12 datasets
# Datasets: TUAB, BCIC2A, SEEDIV, TUEP, TUEV, TUSZ,
#           SleepEDF_full, Physionet_MI, Workload, SEED, MDD, AD65
#
# Usage:
#   bash run_latest12_extended_suite.sh [--skip_existing] [--tiny] [--datasets DS1,DS2,...]
#
# Environment overrides:
#   PYTHON_EXEC, CUDA_VISIBLE_DEVICES, BATCH_SIZE
set -euo pipefail

cd /vePFS-0x0d/home/cx/ptft || exit 1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PYTHON_EXEC="${PYTHON_EXEC:-/vePFS-0x0d/home/cx/miniconda3/envs/labram_mamba/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

BATCH_SIZE="${BATCH_SIZE:-256}"

"${PYTHON_EXEC}" experiments/latest12_lp_benchmark/run_latest12_extended_suite.py \
  --python_exec "${PYTHON_EXEC}" \
  --batch_size "${BATCH_SIZE}" \
  --device cuda \
  "$@"
