#!/bin/bash
# Extended full fine-tuning benchmark: 12 checkpoints x 12 datasets
# Datasets: TUAB, BCIC2A, SEEDIV, TUEP, TUEV, TUSZ,
#           SleepEDF_full, Physionet_MI, Workload, SEED, MDD, AD65
#
# Usage:
#   bash run_latest12_extended_full_ft_suite.sh [--skip_existing] [--stop_on_error]
#   bash run_latest12_extended_full_ft_suite.sh --datasets SleepEDF_full,MDD,AD65
#
# Environment overrides:
#   PYTHON_EXEC, CUDA_VISIBLE_DEVICES, LR, EPOCHS, BATCH_SIZE (per-dataset defaults apply)
set -euo pipefail

cd /vePFS-0x0d/home/cx/ptft || exit 1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PYTHON_EXEC="${PYTHON_EXEC:-/vePFS-0x0d/home/cx/miniconda3/envs/labram_mamba/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"

"${PYTHON_EXEC}" experiments/latest12_lp_benchmark/run_latest12_extended_full_ft_suite.py \
  --python_exec "${PYTHON_EXEC}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  "$@"
