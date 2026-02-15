#!/bin/bash

# Ensure we are in the project root
cd /vePFS-0x0d/home/cx/ptft || exit

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Use labram_mamba env which should support all models including Mamba
PYTHON_EXEC="/vePFS-0x0d/home/cx/miniconda3/envs/labram_mamba/bin/python"

# Define models: Name:Path:BatchSize
MODELS=(
    "joint_pretrain:/vePFS-0x0d/home/cx/ptft/output/joint_pretrain/best.pth:256"
    "st_eegformer_large:/vePFS-0x0d/home/cx/ptft/output/pretrain_st_eegformer_large/best.pth:64"
    "joint_eegmamba:/vePFS-0x0d/home/cx/ptft/output/joint_pretrain_eegmamba/best.pth:128"
)

echo "Starting Benchmark Suite..."

for entry in "${MODELS[@]}"; do
    IFS=":" read -r name path batch_size <<< "$entry"
    
    output_csv="experiments/all_lp_benchmark/results_${name}.csv"
    
    echo "================================================================"
    echo "Running Benchmark for: $name"
    echo "Checkpoint: $path"
    echo "Batch Size: $batch_size"
    echo "Output: $output_csv"
    echo "================================================================"
    
    $PYTHON_EXEC experiments/all_lp_benchmark/run_benchmark.py \
        --checkpoint "$path" \
        --output_csv "$output_csv" \
        --batch_size "$batch_size" \
        --device cuda \
        "$@"
        
    echo "Finished $name"
    echo ""
done

echo "All benchmarks completed."
