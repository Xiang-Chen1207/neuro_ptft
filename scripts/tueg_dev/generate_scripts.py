
import os

models = {
    'eegmamba': 'configs/pretrain_eegmamba.yaml',
    'cbramod': 'configs/pretrain.yaml',
    'tech': 'configs/pretrain_tech.yaml',
    'reve': 'configs/pretrain_reve.yaml'
}

feature_path = '/vePFS-0x0d/home/dht/eeg_feature_extraction_full/results/tuep/features_zscore.csv'
output_base = '/vePFS-0x0d/home/cx/ptft/scripts/tueg_dev'
feature_cache_path = '/vePFS-0x0d/home/cx/ptft/output/tueg_dev/feature_cache_tuep.pkl'

# Header
header = """#!/bin/bash
set -e
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram_mamba
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export ENABLE_WANDB=true
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-12}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-6}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-2}
VAL_PREFETCH_FACTOR=${VAL_PREFETCH_FACTOR:-2}
VAL_TIMEOUT=${VAL_TIMEOUT:-300}
IO_GROUP_CHUNK_SIZE=${IO_GROUP_CHUNK_SIZE:-2048}
H5_RDCC_NBYTES=${H5_RDCC_NBYTES:-67108864}
H5_CACHE_SIZE=${H5_CACHE_SIZE:-512}
SHARED_CACHE_DIR=${SHARED_CACHE_DIR:-/vePFS-0x0d/home/cx/ptft/output/tueg_dev/shared_cache}
SHARED_INDEX_PATH=${SHARED_INDEX_PATH:-$SHARED_CACHE_DIR/tueg_dataset_index.json}
SHARED_FEATURE_CACHE_PATH=${SHARED_FEATURE_CACHE_PATH:-$SHARED_CACHE_DIR/tuep_feature_map.pkl}
"""

# Logic block (Resume & Mkdir)
logic_block = """
# Ensure output directory exists
mkdir -p $OUTPUT_DIR
mkdir -p $SHARED_CACHE_DIR

echo "Starting Experiment: $PROJECT_NAME"
echo "Using dataset.batch_size=$BATCH_SIZE"
echo "Using shared dataset index cache: $SHARED_INDEX_PATH"

RESUME_ARG=""
CHECKPOINT_PATH="${OUTPUT_DIR}/latest.pth"

if [ "$1" == "resume" ]; then
    if [ -f "$CHECKPOINT_PATH" ]; then
        echo "Resuming from checkpoint: $CHECKPOINT_PATH"
        RESUME_ARG="--resume $CHECKPOINT_PATH"
    fi
elif [ -f "$CHECKPOINT_PATH" ]; then
    echo "Found existing checkpoint. Auto-resuming..."
    RESUME_ARG="--resume $CHECKPOINT_PATH"
fi
"""

def generate_script(model, config_path, script_type, filename):
    
    # Base opts
    opts = [
        "output_dir=\"$OUTPUT_DIR\"",
        "project=\"$PROJECT_NAME\"",
        f"dataset.feature_path='{feature_path}'",
        "dataset.name='TUEG'",
        "enable_wandb=$ENABLE_WANDB",
        "dataset.batch_size=$BATCH_SIZE",
        "dataset.num_workers=$NUM_WORKERS",
        "dataset.persistent_workers=true",
        "dataset.prefetch_factor=$PREFETCH_FACTOR",
        "dataset.val_num_workers=$VAL_NUM_WORKERS",
        "dataset.val_persistent_workers=false",
        "dataset.val_prefetch_factor=$VAL_PREFETCH_FACTOR",
        "dataset.val_pin_memory=false",
        "dataset.val_timeout=$VAL_TIMEOUT",
        "dataset.cache_path=\"$SHARED_INDEX_PATH\"",
        "dataset.feature_cache_path=\"$SHARED_FEATURE_CACHE_PATH\"",
        "dataset.enable_feature_filter_cache=true",
        "dataset.io_group_shuffle=true",
        "dataset.io_group_chunk_size=$IO_GROUP_CHUNK_SIZE",
        "dataset.h5_rdcc_nbytes=$H5_RDCC_NBYTES",
        "dataset.h5_cache_size=$H5_CACHE_SIZE",
        "optimizer.lr=1e-3",
        "epochs=50",
        "val_freq_split=5"
    ]

    project_suffix = ""
    
    if script_type == 'recon':
        project_suffix = "recon"
        opts.extend([
            "model.pretrain_tasks=['reconstruction']",
            "loss.name='mse'",
        ])
    elif script_type == 'joint':
        project_suffix = "joint"
        opts.extend([
            "model.pretrain_tasks=['reconstruction','feature_pred']",
            "loss.feature_loss_weight=1.0", 
            "loss.use_dynamic_loss=false"
        ])
    elif script_type == 'scheme_a':
        project_suffix = "feat-scheme-a"
        opts.extend([
            "model.pretrain_tasks=['reconstruction','feature_pred']",
            "model.feature_token_type='cross_attn'",
            "model.feature_token_strategy='single'",
            "loss.feature_loss_weight=1.0"
        ])
    elif script_type == 'scheme_b':
        project_suffix = "feat-scheme-b"
        opts.extend([
            "model.pretrain_tasks=['reconstruction','feature_pred']",
            "model.feature_token_type='prefix'",
            "loss.feature_loss_weight=1.0"
        ])
        if model == 'eegmamba':
            opts.append("model.name='eegmamba_prefix'")

    # Construct file content
    content = header
    if model == 'tech':
        content += "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}\n"
        content += "BATCH_SIZE=${BATCH_SIZE:-512}\n"
    content += f'\nOUTPUT_DIR="output/tueg_dev/{model}_{project_suffix}"\n'
    content += f'PROJECT_NAME="ptft-{model}-{project_suffix}"\n'
    content += f'export PROJECT_NAME\n'
    content += logic_block
    
    cmd = "python3 main.py \\\n"
    cmd += f"  --config {config_path} \\\n"
    cmd += "  $RESUME_ARG \\\n"
    cmd += "  --opts \\\n"
    
    for opt in opts:
        cmd += f"    {opt} \\\n"
    
    cmd = cmd.rstrip(" \\\n") + "\n"
    
    content += cmd
    content += '\necho "Experiment Finished."\n'
    
    with open(os.path.join(output_base, filename), 'w') as f:
        f.write(content)
    
    os.chmod(os.path.join(output_base, filename), 0o755)

# Generate
for model, config in models.items():
    generate_script(model, config, 'recon', f'run_{model}_recon.sh')
    generate_script(model, config, 'joint', f'run_{model}_joint.sh')
    generate_script(model, config, 'scheme_a', f'run_{model}_feat_scheme_a.sh')
    generate_script(model, config, 'scheme_b', f'run_{model}_feat_scheme_b.sh')

print("Generated 16 scripts in " + output_base)
