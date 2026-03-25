#!/bin/bash
set -euo pipefail

cd /vePFS-0x0d/home/cx/ptft || exit 1

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Default to labram_mamba env python, can be overridden by exporting PYTHON_EXEC.
PYTHON_EXEC="${PYTHON_EXEC:-/vePFS-0x0d/home/cx/miniconda3/envs/labram_mamba/bin/python}"

"${PYTHON_EXEC}" experiments/latest12_lp_benchmark/run_latest12_suite.py \
  --python_exec "${PYTHON_EXEC}" \
  "$@"
