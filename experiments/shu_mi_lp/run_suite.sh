#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
# Using labram_mamba environment to support all models
PYTHON_EXEC="/vePFS-0x0d/home/cx/miniconda3/envs/labram_mamba/bin/python"

# Define models: Name:Path:BatchSize
MODELS=(
    "joint_pretrain:/vePFS-0x0d/home/cx/ptft/output/joint_pretrain/best.pth:256"
    "st_eegformer_large:/vePFS-0x0d/home/cx/ptft/output/pretrain_st_eegformer_large/best.pth:64"
    "joint_eegmamba:/vePFS-0x0d/home/cx/ptft/output/joint_pretrain_eegmamba/best.pth:128"
)

echo "Starting SHU_MI Linear Probing Benchmark Suite..."

for entry in "${MODELS[@]}"; do
    IFS=":" read -r name path batch_size <<< "$entry"
    
    output_csv="experiments/shu_mi_lp/results_shumi_${name}.csv"
    
    echo "================================================================"
    echo "Running SHU_MI LP for: $name"
    echo "Checkpoint: $path"
    echo "Batch Size: $batch_size"
    echo "Output: $output_csv"
    echo "================================================================"
    
    $PYTHON_EXEC experiments/shu_mi_lp/run_shu_mi_lp.py \
        --checkpoint "$path" \
        --output_csv "$output_csv" \
        --batch_size "$batch_size" \
        --device cuda
        
    echo "Finished $name"
    echo ""
done

echo "SHU_MI LP Benchmark Suite Completed."
