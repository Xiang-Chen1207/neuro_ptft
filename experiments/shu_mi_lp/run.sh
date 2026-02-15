#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
PYTHON_EXEC="/vePFS-0x0d/home/cx/miniconda3/envs/labram/bin/python"
CHECKPOINT="/vePFS-0x0d/home/cx/ptft/output/joint_pretrain/best.pth"
OUTPUT_CSV="experiments/shu_mi_lp/results_shumi.csv"

echo "Starting SHU_MI Linear Probing..."
echo "Checkpoint: $CHECKPOINT"

$PYTHON_EXEC experiments/shu_mi_lp/run_shu_mi_lp.py \
    --checkpoint "$CHECKPOINT" \
    --output_csv "$OUTPUT_CSV" \
    --batch_size 256 \
    --device cuda

echo "SHU_MI LP Completed."
