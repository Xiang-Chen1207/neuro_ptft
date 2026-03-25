#!/usr/bin/env python3
"""Extended full fine-tuning benchmark: 12 checkpoints x 12 datasets.

For each (checkpoint, dataset) pair, calls:
    python main.py --config configs/finetune_<dataset>.yaml --opts ...

Datasets: TUAB, BCIC2A, SEEDIV, TUEP, TUEV, TUSZ,
          SleepEDF_full, Physionet_MI, Workload, SEED, MDD, AD65
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Best checkpoints
# ---------------------------------------------------------------------------
BEST_CHECKPOINT_PATHS: Dict[str, str] = {
    "20260313_181700_onegpu_full_safeio__cbramod_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/cbramod_recon/checkpoint_epoch_26.pth",
    "20260313_181700_onegpu_full_safeio__eegmamba_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/eegmamba_recon/checkpoint_epoch_28.pth",
    "20260313_181700_onegpu_full_safeio__reve_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/reve_recon/checkpoint_epoch_28.pth",
    "20260313_181700_onegpu_full_safeio__tech_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/tech_recon/checkpoint_epoch_27.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__cbramod_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/cbramod_feat_only/checkpoint_epoch_29.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__eegmamba_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/eegmamba_feat_only/checkpoint_epoch_28.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__reve_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/reve_feat_only/checkpoint_epoch_26.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__tech_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/tech_feat_only/checkpoint_epoch_29.pth",
    "20260316_joint_recon_feat_screen__cbramod_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/cbramod_joint/checkpoint_epoch_38.pth",
    "20260316_joint_recon_feat_screen__eegmamba_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/eegmamba_joint/checkpoint_epoch_38.pth",
    "20260316_joint_recon_feat_screen__reve_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/reve_joint/checkpoint_epoch_38.pth",
    "20260316_joint_recon_feat_screen__tech_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/tech_joint/checkpoint_epoch_38.pth",
}

# ---------------------------------------------------------------------------
# Per-dataset finetune config
# ---------------------------------------------------------------------------
DATASET_CONFIG: Dict[str, Dict] = {
    "TUAB":        {"config": "configs/finetune.yaml",            "selected_dataset": "TUAB",         "num_classes": 2,  "metric_key": "balanced_acc", "batch_size": 256},
    "TUEP":        {"config": "configs/finetune_tuep.yaml",       "selected_dataset": "TUEP",         "num_classes": 2,  "metric_key": "balanced_acc", "batch_size": 128},
    "TUEV":        {"config": "configs/finetune_tuev.yaml",       "selected_dataset": "TUEV",         "num_classes": 6,  "metric_key": "balanced_acc", "batch_size": 256},
    "TUSZ":        {"config": "configs/finetune_tusz.yaml",       "selected_dataset": "TUSZ",         "num_classes": 13, "metric_key": "balanced_acc", "batch_size": 128},
    "SEED":        {"config": "configs/finetune_seed.yaml",       "selected_dataset": "SEED",         "num_classes": 3,  "metric_key": "accuracy",     "batch_size": 32},
    "BCIC2A":      {"config": "configs/finetune_bcic2a.yaml",     "selected_dataset": "BCIC2A",       "num_classes": 4,  "metric_key": "accuracy",     "batch_size": 32},
    "SEEDIV":      {"config": "configs/finetune_seed.yaml",       "selected_dataset": "SEEDIV",       "num_classes": 4,  "metric_key": "accuracy",     "batch_size": 32},
    "SleepEDF_full": {"config": "configs/finetune_sleepedf.yaml",   "selected_dataset": "SleepEDF_full", "num_classes": 5, "metric_key": "balanced_acc", "batch_size": 128},
    "Physionet_MI":  {"config": "configs/finetune_physionet_mi.yaml", "selected_dataset": "Physionet_MI", "num_classes": 4, "metric_key": "balanced_acc", "batch_size": 128},
    "Workload":      {"config": "configs/finetune_workload.yaml",    "selected_dataset": "Workload",      "num_classes": 3, "metric_key": "balanced_acc", "batch_size": 128},
    "MDD":           {"config": "configs/finetune_mdd.yaml",         "selected_dataset": "MDD",           "num_classes": 2, "metric_key": "balanced_acc", "batch_size": 128},
    "AD65":          {"config": "configs/finetune_ad65.yaml",         "selected_dataset": "AD65",          "num_classes": 3, "metric_key": "balanced_acc", "batch_size": 128},
}

ALL_DATASETS = ",".join(DATASET_CONFIG.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended full FT benchmark: 12 ckpts x 12 datasets.")
    p.add_argument("--output_dir",        type=str, default="experiments/latest12_lp_benchmark/results_extended_full_ft")
    p.add_argument("--train_output_root", type=str, default="output/latest12_extended_full_ft")
    p.add_argument("--python_exec",       type=str, default=sys.executable)
    p.add_argument("--epochs",            type=int, default=20)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--num_workers",       type=int, default=8)
    p.add_argument("--datasets",          type=str, default=ALL_DATASETS)
    p.add_argument("--cuda_visible_devices", type=str, default="")
    p.add_argument("--save_checkpoints",  type=str, default="false")
    p.add_argument("--run_tag",           type=str, default="")
    p.add_argument("--skip_existing",     action="store_true")
    p.add_argument("--stop_on_error",     action="store_true")
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_checkpoints() -> List[Dict]:
    rows = []
    for model_tag, ckpt in sorted(BEST_CHECKPOINT_PATHS.items()):
        run_dir, model_name = model_tag.split("__", 1)
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            print(f"[WARN] Missing checkpoint: {ckpt_path}")
            continue
        rows.append({"model_tag": model_tag, "run_dir": run_dir,
                     "model_name": model_name, "model_family": model_name.split("_")[0],
                     "checkpoint": str(ckpt_path)})
    return rows


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _metric_is_loss(key: str) -> bool:
    return "loss" in key.lower()


def find_best_from_log(log_csv: Path, metric_key: str) -> Dict:
    empty = {"best_epoch": None, "best_metric": None, "best_metric_raw": None, "metric_col": None}
    if not log_csv.is_file():
        return empty
    try:
        df = pd.read_csv(log_csv)
    except Exception as exc:
        print(f"[WARN] Cannot read {log_csv}: {exc}")
        return empty
    col = f"val_{metric_key}"
    if col not in df.columns:
        fallbacks = [c for c in ("val_balanced_acc", "val_accuracy", "val_acc", "val_loss") if c in df.columns]
        if not fallbacks:
            return empty
        col = fallbacks[0]
    if "epoch" not in df.columns:
        df = df.reset_index().rename(columns={"index": "epoch"})
    series = pd.to_numeric(df[col], errors="coerce")
    valid = series.notna()
    if not valid.any():
        return empty
    vals = series[valid]
    best_idx = vals.idxmin() if _metric_is_loss(col) else vals.idxmax()
    best_raw = float(df.loc[best_idx, col])
    return {
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "best_metric": -best_raw if _metric_is_loss(col) else best_raw,
        "best_metric_raw": best_raw,
        "metric_col": col,
    }


# ---------------------------------------------------------------------------
# Run one (model, dataset) pair
# ---------------------------------------------------------------------------
def run_one(
    root: Path, python_exec: str, rec: Dict, ds_name: str, ds_cfg: Dict,
    train_output_root: Path, output_root: Path,
    epochs: int, lr: float, num_workers: int,
    save_checkpoints: str, cuda_visible_devices: str, run_tag: str,
    skip_existing: bool,
) -> int:
    suffix = f"__{run_tag}" if run_tag else ""
    safe_ds = ds_name.replace("/", "_")
    train_out = train_output_root / f"{rec['model_tag']}__{safe_ds}{suffix}"
    log_csv = train_out / "log.csv"

    rec[f"train_out__{safe_ds}"] = str(train_out)
    rec[f"log_csv__{safe_ds}"] = str(log_csv)

    if skip_existing and log_csv.is_file():
        print(f"[SKIP] {rec['model_tag']} x {ds_name}")
        return 0

    config_path = (root / ds_cfg["config"]).resolve()
    if not config_path.is_file():
        print(f"[ERROR] Config not found: {config_path}")
        return -1

    selected = ds_cfg["selected_dataset"]
    opts = [
        f"output_dir={train_out.as_posix()}",
        "enable_wandb=false",
        f"epochs={epochs}",
        f"optimizer.lr={lr}",
        f"save_checkpoints={save_checkpoints}",
        "model.use_pretrained=true",
        f"model.pretrained_path={rec['checkpoint']}",
        f"model.num_classes={ds_cfg['num_classes']}",
        f"metric_key={ds_cfg.get('metric_key', 'balanced_acc')}",
        f"selected_dataset={selected}",
        f"datasets.{selected}.batch_size={ds_cfg['batch_size']}",
        f"datasets.{selected}.num_workers={num_workers}",
    ]

    cmd = [python_exec, "main.py", "--config", str(config_path), "--opts", *opts]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("\n" + "=" * 88)
    print(f"[RUN] model={rec['model_tag']}  dataset={ds_name}")
    print(f"[RUN] ckpt={rec['checkpoint']}")
    print(f"[RUN] out={train_out}")
    print(f"[RUN] cmd={' '.join(cmd)}")
    print("=" * 88)

    ret = subprocess.run(cmd, cwd=str(root), env=env, check=False).returncode
    if ret == 0:
        best = find_best_from_log(log_csv, ds_cfg.get("metric_key", "balanced_acc"))
        # save per-(model,dataset) csv
        per_dir = output_root / "per_model_dataset"
        per_dir.mkdir(parents=True, exist_ok=True)
        per_csv = per_dir / f"{rec['model_tag'].replace('/', '__')}__{safe_ds}.csv"
        pd.DataFrame([{
            "Model_Tag": rec["model_tag"], "Dataset": ds_name,
            "Model_Family": rec["model_family"], "Checkpoint": rec["checkpoint"],
            "Train_Output_Dir": str(train_out),
            "Metric_Column": best.get("metric_col"),
            "Best_Epoch": best.get("best_epoch"),
            "Best_Metric_Raw": best.get("best_metric_raw"),
            "Best_Metric_For_Rank": best.get("best_metric"),
        }]).to_csv(per_csv, index=False)
        rec.setdefault("per_csvs", {})[ds_name] = str(per_csv)
    return ret


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def aggregate(records: List[Dict], datasets: List[str], output_root: Path) -> None:
    rows = []
    for rec in records:
        for ds in datasets:
            safe_ds = ds.replace("/", "_")
            per_csv = rec.get("per_csvs", {}).get(ds)
            ret = rec.get(f"return_code__{safe_ds}", -999)
            rows.append({"Model_Tag": rec["model_tag"], "Dataset": ds,
                         "Return_Code": ret, "Per_CSV": per_csv or ""})
    status_df = pd.DataFrame(rows)
    status_df.to_csv(output_root / "run_status.csv", index=False)
    print(f"[OK] run_status.csv saved.")

    # Collect all per-model-dataset CSVs
    long_rows = []
    per_dir = output_root / "per_model_dataset"
    if per_dir.exists():
        for csv_path in sorted(per_dir.glob("*.csv")):
            try:
                long_rows.extend(pd.read_csv(csv_path).to_dict("records"))
            except Exception:
                pass
    if not long_rows:
        print("[WARN] No per-model-dataset CSVs found.")
        return

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_root / "results_long.csv", index=False)
    print(f"[OK] results_long.csv saved.")

    pivot = long_df.pivot_table(
        index="Dataset", columns="Model_Tag",
        values="Best_Metric_Raw", aggfunc="first"
    )
    pivot.to_csv(output_root / "results_compare_best_metric.csv")
    print(f"[OK] results_compare_best_metric.csv saved.")

    summary = (
        long_df.groupby("Model_Tag", as_index=False)
        .agg(Mean_Best_Metric=("Best_Metric_For_Rank", "mean"),
             Num_Datasets=("Dataset", "nunique"))
        .sort_values("Mean_Best_Metric", ascending=False)
    )
    summary.to_csv(output_root / "results_rank.csv", index=False)
    print(f"[OK] results_rank.csv saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    root = project_root()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    output_root = (root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_output_root = (root / args.train_output_root).resolve()
    train_output_root.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints()
    print(f"[INFO] {len(checkpoints)} checkpoints, {len(datasets)} datasets -> "
          f"{len(checkpoints) * len(datasets)} total runs.")

    failed = False
    for rec in checkpoints:
        for ds_name in datasets:
            ds_cfg = DATASET_CONFIG.get(ds_name)
            if ds_cfg is None:
                print(f"[WARN] No config for dataset '{ds_name}', skipping.")
                continue

            safe_ds = ds_name.replace("/", "_")
            ret = run_one(
                root=root, python_exec=args.python_exec,
                rec=rec, ds_name=ds_name, ds_cfg=ds_cfg,
                train_output_root=train_output_root,
                output_root=output_root,
                epochs=args.epochs, lr=args.lr,
                num_workers=args.num_workers,
                save_checkpoints=args.save_checkpoints,
                cuda_visible_devices=args.cuda_visible_devices,
                run_tag=args.run_tag,
                skip_existing=args.skip_existing,
            )
            rec[f"return_code__{safe_ds}"] = ret

            if ret != 0:
                print(f"[FAIL] {rec['model_tag']} x {ds_name} (code {ret})")
                if args.stop_on_error:
                    failed = True
                    break
            else:
                print(f"[OK]   {rec['model_tag']} x {ds_name}")
        if failed:
            break

    aggregate(checkpoints, datasets, output_root)
    print("[DONE] Extended full FT benchmark finished.")


if __name__ == "__main__":
    main()
