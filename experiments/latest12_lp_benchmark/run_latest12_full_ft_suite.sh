#!/bin/bash
set -euo pipefail

cd /vePFS-0x0d/home/cx/ptft || exit 1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Default to labram_mamba env python, can be overridden by exporting PYTHON_EXEC.
PYTHON_EXEC="${PYTHON_EXEC:-/vePFS-0x0d/home/cx/miniconda3/envs/labram_mamba/bin/python}"

# In-script tuning knobs for full fine-tuning.
# You can still override from CLI by passing --lr/--batch_size.
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Multi-GPU DataParallel setting (single process).
# Example: CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../run_latest12_full_ft_suite.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

"${PYTHON_EXEC}" experiments/latest12_lp_benchmark/run_latest12_full_ft_suite.py \
  --python_exec "${PYTHON_EXEC}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --cuda_visible_devices "${CUDA_VISIBLE_DEVICES}" \
  "$@"
