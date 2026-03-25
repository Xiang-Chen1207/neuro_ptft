# latest12_lp_benchmark

Run linear probing evaluation for 12 latest checkpoints and export comparison tables.

Target run folders:

- /vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio
- /vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next
- /vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen

## Notes

- Reuses existing linear probing benchmark: experiments/joint_benchmark/run_benchmark.py
- Writes one CSV per model.
- Aggregates results to long-format, pivot comparison, and mean-BAcc ranking.

## Run

From project root:

```bash
bash experiments/latest12_lp_benchmark/run_latest12_suite.sh
```

Run full fine-tuning suite for the same 12 selected checkpoints:

```bash
bash experiments/latest12_lp_benchmark/run_latest12_full_ft_suite.sh
```

Common options:

```bash
# Tiny mode for debugging
bash experiments/latest12_lp_benchmark/run_latest12_suite.sh --tiny

# Skip models that already have per-model CSV
bash experiments/latest12_lp_benchmark/run_latest12_suite.sh --skip_existing

# Custom batch size and dataset list
bash experiments/latest12_lp_benchmark/run_latest12_suite.sh \
  --batch_size 256 \
  --datasets "TUAB,BCIC2A,SEEDIV,TUEP,TUEV,TUSZ"

# Full fine-tuning: tune lr and batch size in one command
bash experiments/latest12_lp_benchmark/run_latest12_full_ft_suite.sh \
  --selected_dataset TUEP \
  --lr 1e-4 \
  --batch_size 128 \
  --epochs 20

# Full fine-tuning with multi-GPU DataParallel
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash experiments/latest12_lp_benchmark/run_latest12_full_ft_suite.sh \
  --selected_dataset TUEP \
  --lr 5e-5 \
  --batch_size 192
```

## Outputs

Default output folder: experiments/latest12_lp_benchmark/results

- per_model/*.csv: per-checkpoint downstream result files.
- run_status.csv: run status and return code for each checkpoint.
- results_long.csv: long-format table with model metadata.
- results_compare_bacc.csv: Balanced Accuracy comparison by dataset.
- results_compare_acc.csv: Accuracy comparison by dataset.
- results_rank_mean_bacc.csv: ranking by mean Balanced Accuracy.

Full fine-tuning suite outputs:

- results_full_ft/run_status.csv: run status and return code for each model.
- results_full_ft/per_model/*.csv: one-row summary for each model run.
- results_full_ft/results_full_ft_summary.csv: aggregated full-ft best-metric table.
- results_full_ft/results_full_ft_rank.csv: ranking by best validation metric.
