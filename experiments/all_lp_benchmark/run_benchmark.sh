#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
PYTHON_EXEC="/vePFS-0x0d/home/cx/miniconda3/envs/labram/bin/python"
CHECKPOINT="/vePFS-0x0d/home/cx/ptft/output/joint_pretrain/best.pth"
OUTPUT_CSV="experiments/all_lp_benchmark/results_full.csv"

echo "Starting Benchmark on ALL datasets..."
echo "Checkpoint: $CHECKPOINT"

$PYTHON_EXEC experiments/all_lp_benchmark/run_benchmark.py \
    --checkpoint "$CHECKPOINT" \
    --output_csv "$OUTPUT_CSV" \
    --batch_size 256 \
    --device cuda

echo "Benchmark Completed."
