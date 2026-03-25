# CLAUDE.md - AI Assistant Guide for PTFT

## Project Overview

**PTFT (Pretrain-Finetune)** implements **Neuro-KE (Neuro-Knowledge Engine)**, a label-free knowledge integration framework for EEG foundation models. The project enhances EEG pretraining by integrating domain-specific signal priors (morphometric, spectral, non-linear features) into deep learning architectures.

### Core Concept
Neuro-KE acts as a "knowledge engine" that forces the model to encode clinically meaningful patterns by predicting z-scored EEG features (band powers, Hjorth parameters, spectral entropy, aperiodic exponents) as an auxiliary training signal during masked-autoencoding pretraining.

## Codebase Structure

```
neuro_ptft/
├── main.py                         # Unified CLI entry point (pretraining/finetuning)
├── configs/                        # 22 YAML configs (pretrain, finetune, linear probe)
├── core/
│   ├── trainer.py                  # Trainer: epoch loop, checkpointing, optimizer/scheduler
│   ├── engine.py                   # train_one_epoch(), evaluate(): multi-task losses, metrics
│   └── loss.py                     # LossFactory: masked MSE, CrossEntropy, BCEWithLogits
├── models/
│   ├── wrapper.py                  # CBraModWrapper: backbone + task-specific heads
│   ├── backbone.py                 # CBraModBackbone: spectral patch embedding + Transformer
│   ├── criss_cross_transformer.py  # Split spatial-temporal cross-attention
│   ├── eegmamba.py                 # EEG-specific Mamba architecture
│   ├── eegmamba_prefix.py          # Mamba with prefix tuning for feature prediction
│   ├── reve.py                     # ReVE (Recurrent Vision Encoder) for EEG
│   ├── st_eegformer.py             # ST-EEGFormer Transformer model
│   ├── st_eegformer_mae.py         # ST-EEGFormer with MAE pretraining
│   ├── modules/                    # Mamba config and mixer layers
│   └── tech/                       # TeChBackbone alternative architecture
├── datasets/
│   ├── builder.py                  # Factory: build_dataset() for 10+ datasets, tiny-mode
│   ├── tueg.py                     # TUEG pretraining (95/5 subject-based split)
│   ├── tuab.py                     # TUAB classification (80/10/10 split)
│   ├── tuep.py                     # TUEP event prediction
│   ├── tuev.py                     # TUEV event prediction
│   ├── tusz.py                     # TUSZ seizure detection
│   ├── seed.py                     # SEED emotion recognition
│   ├── seed_iv.py                  # SEED-IV emotion recognition
│   ├── bcic2a.py                   # BCI Competition IVa motor imagery
│   ├── joint.py                    # Multi-dataset joint training
│   ├── generic_h5.py               # Generic HDF5 loader
│   └── generic_sub_h5.py           # Generic subject-level HDF5 loader
├── utils/
│   └── util.py                     # MetricLogger, WandbLogger, mask generation
├── eeg_feature_extraction/         # Offline EEG feature computation pipeline
│   ├── feature_extractor.py        # Main feature computation with parallel processing
│   ├── selective_feature_extraction.py
│   ├── merged_segment_extraction.py
│   ├── psd_computer.py             # Power Spectral Density utilities
│   ├── data_loader.py              # EEG data I/O for feature extraction
│   ├── config.py                   # Feature extraction config and frequency bands
│   └── features/                   # Feature-type-specific modules
├── scripts/                        # 21 experiment scripts (bash + python utilities)
├── experiments/                    # 19 archived experiment runs with result summaries
├── CSBrain-main/                   # Integrated CSBrain foundation model (alternative backbone)
├── docs/                           # ICML Technical Report on Neuro-KE
├── result/                         # Aggregated linear probing results
├── eval_features.py                # Feature prediction evaluation (R², PCC per feature)
├── viz_features.py                 # Bar plot visualization for feature metrics
├── viz_compare.py                  # Compare metrics across experiments
├── verify_changes.sh               # Quick sanity test (2 samples, 1 epoch)
└── various utility scripts         # analyze_*, benchmark_*, check_*, inspect_*, count_*
```

## Quick Start Commands

```bash
# Pretraining with CBraMod (default)
python main.py --config configs/pretrain.yaml

# Pretraining with alternative backbones
python main.py --config configs/pretrain_eegmamba.yaml
python main.py --config configs/pretrain_reve.yaml
python main.py --config configs/pretrain_st_eegformer.yaml
python main.py --config configs/pretrain_tech.yaml

# Resume from checkpoint
python main.py --config configs/pretrain.yaml --resume output/flagship_cross_attn/latest.pth

# Debugging with tiny dataset (10 files, fast iteration)
python main.py --config configs/pretrain.yaml --tiny --opts epochs=1

# Finetuning on downstream tasks
python main.py --config configs/finetune_tuab.yaml
python main.py --config configs/finetune_tusz.yaml
python main.py --config configs/finetune_tuep.yaml
python main.py --config configs/finetune_tuev.yaml
python main.py --config configs/finetune_seed.yaml

# Linear probing
python main.py --config configs/linear_probe_tusz.yaml
python main.py --config configs/linear_probe_seed.yaml
python main.py --config configs/linear_probe_bcic2a.yaml

# Override config options via CLI
python main.py --config configs/pretrain.yaml --opts model.pretrain_tasks=['reconstruction'] epochs=50

# Evaluate feature prediction
python eval_features.py --config configs/pretrain.yaml --checkpoint output/best.pth --output metrics.csv

# Quick sanity check
./verify_changes.sh
```

## Configuration System

YAML configs support nested key overrides via `--opts`:

```bash
--opts key1.key2=value list_key=['a','b'] numeric=100
```

### Key Configuration Options

| Key | Values | Description |
|-----|--------|-------------|
| `task_type` | `pretraining`, `classification` | Training mode |
| `model.name` | `cbramod`, `eegmamba`, `eegmamba_prefix`, `st_eegformer_mae`, `reve`, `tech` | Backbone architecture |
| `model.pretrain_tasks` | `['reconstruction']`, `['feature_pred']`, `['reconstruction','feature_pred']` | Pretraining objectives |
| `model.feature_token_type` | `gap`, `cross_attn`, `prefix` | Feature extraction method |
| `model.feature_token_strategy` | `single`, `all`, `group` | Cross-attention token strategy |
| `model.feature_group_count` | int | Number of groups (for `group` strategy) |
| `model.d_model` | int (default 200) | Transformer hidden dimension |
| `model.n_layer` | int (default 12) | Number of Transformer layers |
| `model.nhead` | int (default 8) | Number of attention heads |
| `model.seq_len` | int (default 30) | Number of patches |
| `model.in_dim` | int (default 200) | Patch size |
| `loss.feature_loss_weight` | float | Weight for feature prediction loss |
| `val_freq_split` | int | Intra-epoch validation frequency (0 = disabled) |
| `dataset.name` | `TUEG`, `TUAB`, `TUEP`, `TUEV`, `TUSZ`, `SEED`, `SEED_IV`, `BCIC2a`, ... | Dataset name |
| `dataset.tiny` | bool | Use 10-file subset for debugging |
| `dataset.val_max_files` | int (default 50) | Limit validation files for efficiency |
| `optimizer.name` | `AdamW`, `SGD` | Optimizer |
| `scheduler.name` | `CosineAnnealingLR`, `StepLR` | Learning rate scheduler |

### Full YAML Structure

```yaml
task_type: pretraining  # or classification
output_dir: output/experiment_name
device: cuda
epochs: 100
clip_grad: 1.0
metric_key: loss  # or balanced_acc

dataset:
  name: TUEG
  dataset_dir: /path/to/data
  feature_path: /path/to/features.csv  # pretraining only
  batch_size: 128
  num_workers: 8
  seed: 42

model:
  name: cbramod
  in_dim: 200
  d_model: 200
  seq_len: 30
  n_layer: 12
  nhead: 8
  pretrain_tasks: [reconstruction, feature_pred]
  feature_dim: 62
  feature_token_type: cross_attn
  feature_token_strategy: single

loss:
  name: mse
  feature_loss_weight: 1.0

optimizer:
  name: AdamW
  lr: 1e-4
  weight_decay: 0.05

scheduler:
  name: CosineAnnealingLR
  T_max: 100
```

## Model Architectures

### Supported Backbones

| Backbone | Config Key | Description |
|----------|------------|-------------|
| **CBraMod** | `cbramod` | Spectral patch embedding + criss-cross spatial-temporal Transformer (default) |
| **EEGMamba** | `eegmamba` | EEG-specific Mamba (state-space model) architecture |
| **EEGMamba Prefix** | `eegmamba_prefix` | Mamba with prefix tokens for feature prediction |
| **ST-EEGFormer MAE** | `st_eegformer_mae` | Spatial-temporal EEGFormer with MAE decoder |
| **ReVE** | `reve` | Recurrent Vision Encoder adapted for EEG |
| **TeCh** | `tech` | TeChBackbone alternative architecture |

### CBraModBackbone (Default)
- **Patch Embedding**: Dual-domain — time-domain Conv2D + FFT spectral projection, combined via addition
- **Transformer**: 12 layers, 8 heads, d_model=200, criss-cross spatial-temporal attention
- **Input**: `(B, C, N, P)` — batch, channels, num_patches, patch_size
- **Output**: `(B, C, N, D)` where D=d_model

### Criss-Cross Transformer
- Splits attention heads between spatial (channel) and temporal (patch) dimensions
- Each dimension uses d_model//2 with independent MultiheadAttention
- Supports channel_mask and time_mask for selective attention

### Feature Prediction Strategies
1. **GAP**: Global Average Pool → MLP → features
2. **Cross-Attention (single)**: 1 learnable query → attention over all patches → all features
3. **Cross-Attention (all)**: N queries (N=feature_dim) → attention → 1 output each
4. **Cross-Attention (group)**: K queries → attention → MLP → features
5. **Prefix** (Mamba only): Prefix tokens prepended to input sequence

### Wrapper Pattern
```python
# CBraModWrapper selects backbone and attaches task heads
class CBraModWrapper(nn.Module):
    def __init__(self, config):
        self._init_backbone(model_config)           # Backbone by model.name
        self._init_task_heads(config, model_config)  # Pretraining or classification heads
```

## Datasets

| Dataset | Task | Split | Description |
|---------|------|-------|-------------|
| **TUEG** | Pretraining | 95/5 (subject) | Temple University EEG corpus, multi-task target |
| **TUAB** | Classification | 80/10/10 | Abnormal EEG detection (binary) |
| **TUEP** | Classification | subject-based | Epilepsy prediction |
| **TUEV** | Classification | subject-based | EEG event classification |
| **TUSZ** | Classification | subject-based | Seizure detection |
| **SEED** | Classification | subject-based | Emotion recognition (3 classes) |
| **SEED-IV** | Classification | subject-based | Emotion recognition (4 classes) |
| **BCIC2a** | Classification | subject-based | Motor imagery BCI |
| **Joint** | Pretraining | multi-dataset | Multi-dataset joint training |
| **Generic H5** | Flexible | configurable | Generic HDF5 file loader |

All datasets use subject-based splits to prevent data leakage. Pretraining datasets return `(x, features, mask)`, classification datasets return `(x, label)`.

## Important Patterns

### Multi-task Forward Pass (Pretraining)
```python
# Returns tuple: (reconstruction_output, mask, feature_prediction)
outputs, mask, feature_pred = model(x, mask=mask)
```

### Loss Computation
```python
# Reconstruction uses masked MSE (only masked patches contribute)
loss = criterion(outputs, x, mask=mask)  # ((pred-target)² * mask).sum() / mask.sum()

# Feature loss added with weight
if feature_pred is not None:
    loss += feature_loss_weight * F.mse_loss(feature_pred, target_features)
```

### Classification Forward Pass
```python
logits = model(x)  # returns logits or (logits,) tuple
```

### Metrics Tracking
- `SmoothedValue` maintains deque of recent values (windowed averages)
- `MetricLogger` aggregates metrics, logs every N steps
- `WandbLogger` wraps W&B for experiment tracking
- Classification metrics: balanced_accuracy, ROC-AUC, F1, Cohen's kappa (via scikit-learn)
- CSV logs written to `output_dir/log.csv`

### Checkpoint Structure
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,  # Optional
    'config': {...},
    'epoch': int,
    'best_metric': float
}
```

## EEG Feature Extraction Pipeline

The `eeg_feature_extraction/` directory contains the offline pipeline that generates z-scored features used as auxiliary training targets:

- **Band Powers**: Delta, Theta, Alpha, Beta, Gamma power (dB)
- **Ratios**: Alpha/Theta, Beta/Theta, etc.
- **Spectral**: Peak frequency, aperiodic exponent (1/f slope), spectral entropy
- **Time Domain**: Hjorth parameters (activity, mobility, complexity)
- **Non-linear**: Permutation entropy

Features are pre-computed, z-scored, and stored in CSV. Dataset classes load the CSV at init and return per-segment features.

## Testing & Validation

### Quick Sanity Test
```bash
./verify_changes.sh  # 2 samples, 1 epoch, verifies code runs end-to-end
```

### Tiny Mode
```bash
python main.py --config configs/pretrain.yaml --tiny --opts epochs=2
```
Loads only 10 H5 files for fast iteration during development.

### Intra-Epoch Validation
Set `val_freq_split: N` to validate N times per epoch (uses 50-batch mini-validation for efficiency).

## Experiment Scripts

| Script | Purpose |
|--------|---------|
| `run_flagship_cross_attn.sh` | **Primary**: Multi-task + cross-attention (single token), 4 GPUs |
| `run_ablation_gap.sh` | Ablation: Global average pooling for features |
| `run_sanity_feat_only.sh` | Sanity: Feature prediction only (verify features learnable) |
| `run_baseline_recon.sh` | Baseline: Reconstruction-only (standard MAE) |
| `run_eegmamba_pretrain.sh` | EEGMamba backbone pretraining |
| `run_st_eegformer_pretrain.sh` | ST-EEGFormer backbone pretraining |
| `run_joint_pretrain.sh` | Multi-dataset joint pretraining |

All experiment scripts support `resume` argument: `./run_flagship_cross_attn.sh resume`

### Archived Experiments (`experiments/`)
19 directories with results for linear probing (`*_lp/`), full finetuning (`*_full_ft/`), and specialized experiments. Each contains `results_final_summary.md` with balanced accuracy, ROC-AUC, F1 metrics.

## Code Conventions

### Python Style
- **Type hints**: Not consistently used; follow existing patterns
- **Docstrings**: Minimal; code is self-documenting
- **Imports**: Standard library → third-party → local modules
- **Config access**: Use `config.get('key', default)` pattern throughout

### Training Loop Pattern
- `train_one_epoch()` in `core/engine.py`: forward/backward, gradient clipping, logging, intra-epoch validation
- `evaluate()` in `core/engine.py`: metrics computation, visualization generation
- `Trainer` in `core/trainer.py`: orchestrates epochs, checkpointing, scheduler stepping

### Multi-GPU
- Uses `torch.nn.DataParallel` for data parallelism (not DistributedDataParallel)

## Data Paths (Production)

```
TUEG:          /vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset
TUAB:          /vepfs-0x0d/eeg-data/TUAB
TUEP:          /vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset
TUEV:          /vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Events
TUSZ:          /vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Seizure
Features CSV:  /vePFS-0x0d/pretrain-clip/feature_analysis/features_final_zscore.csv
```

## Dependencies

Core (in `requirements.txt`):
- `torch` (>=2.0.0), `numpy`, `scikit-learn`, `pyyaml`, `wandb`, `einops`, `matplotlib`

Additional (used in various modules):
- `h5py` — EEG data I/O
- `tqdm` — Progress bars
- `scipy` — Signal processing
- `pandas` — Feature CSV handling
- `mne` — EEG processing utilities

## WandB Integration

- **Pretraining project**: `eeg-knowledge-engine`
- **Finetuning project**: `ptft-project`
- **Entity**: `bci-foundation-model`
- **Logs**: loss, metrics, learning rate, gradient norm, reconstruction visualizations
- Disable: set `enable_wandb: false` in config

## Common Development Tasks

### Adding a New Backbone
1. Create model file in `models/` (follow `eegmamba.py` or `reve.py` patterns)
2. Register in `models/wrapper.py` — add to backbone selection in `_init_backbone()`
3. Create matching pretrain config in `configs/`
4. Create experiment script in `scripts/`

### Adding a New Pretraining Task
1. Add task name to `model.pretrain_tasks` list in config
2. Implement head in `models/wrapper.py:_init_pretraining_heads()`
3. Handle forward pass in `_forward_pretraining()`
4. Add loss computation in `core/engine.py:train_one_epoch()`

### Adding a New Dataset
1. Create dataset class in `datasets/` following `tueg.py` (pretraining) or `tuab.py` (classification) pattern
2. Register in `datasets/builder.py:build_dataset()`
3. Implement `__getitem__` returning `(x, features, mask)` for pretraining or `(x, label)` for classification
4. Create matching finetune config in `configs/`

### Debugging Training Issues
1. Enable tiny mode: `--tiny`
2. Check feature alignment: `python eval_features.py --checkpoint ... --output metrics.csv`
3. Inspect checkpoint: `python inspect_checkpoint.py path/to/checkpoint.pth`
4. Check `output/*/log.csv` for training curves
5. GPU monitoring: `python check_gpu_usage.py`

## Utility Scripts (Root Level)

| Script | Purpose |
|--------|---------|
| `eval_features.py` | Feature prediction evaluation (R², Pearson correlation, scatter plots) |
| `inspect_checkpoint.py` | Debug checkpoint contents (config, params, task type) |
| `inspect_h5.py` | Inspect HDF5 file structure |
| `check_consistency.py` | Validate H5 segment counts vs feature CSV |
| `analyze_features.py` | Statistical analysis of EEG feature distributions |
| `benchmark_*.py` | Performance profiling for feature extraction |
| `count_*_segments.py` | Dataset statistics (segment counts per label/file) |
| `check_gpu_*.py` | GPU memory and process monitoring |
| `create_pdf_report.py` | Generate PDF reports from metrics |
| `viz_features.py` | Bar plot visualization for feature metrics |
| `viz_compare.py` | Compare metrics across experiments |

## Git Workflow

- Main experiments tracked in `output/` directory (gitignored)
- Data files (*.h5, *.pth, *.csv, *.json) are gitignored
- Configuration and code changes should be committed
- Checkpoint resume support allows training interruption/restart
