#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


BEST_CHECKPOINT_PATHS: Dict[str, str] = {
    # recon (by min val_loss)
    "20260313_181700_onegpu_full_safeio__cbramod_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/cbramod_recon/checkpoint_epoch_26.pth",
    "20260313_181700_onegpu_full_safeio__eegmamba_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/eegmamba_recon/checkpoint_epoch_28.pth",
    "20260313_181700_onegpu_full_safeio__reve_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/reve_recon/checkpoint_epoch_28.pth",
    # NOTE: tech_recon checkpoint is included as requested, though run was marked as not converged.
    "20260313_181700_onegpu_full_safeio__tech_recon": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio/tech_recon/checkpoint_epoch_27.pth",
    # feat_only (by min val_loss_feat)
    "20260313_181700_onegpu_full_safeio_scaled_next__cbramod_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/cbramod_feat_only/checkpoint_epoch_29.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__eegmamba_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/eegmamba_feat_only/checkpoint_epoch_28.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__reve_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/reve_feat_only/checkpoint_epoch_26.pth",
    "20260313_181700_onegpu_full_safeio_scaled_next__tech_feat_only": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260313_181700_onegpu_full_safeio_scaled_next/tech_feat_only/checkpoint_epoch_29.pth",
    # joint (by min val_loss, excluding epoch-23 spike)
    "20260316_joint_recon_feat_screen__cbramod_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/cbramod_joint/checkpoint_epoch_38.pth",
    "20260316_joint_recon_feat_screen__eegmamba_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/eegmamba_joint/checkpoint_epoch_38.pth",
    "20260316_joint_recon_feat_screen__reve_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/reve_joint/checkpoint_epoch_37.pth",
    "20260316_joint_recon_feat_screen__tech_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/tech_joint/checkpoint_epoch_38.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full fine-tuning benchmark for latest12 selected checkpoints."
    )
    parser.add_argument(
        "--finetune_config",
        type=str,
        default="configs/finetune.yaml",
        help="Fine-tuning config path relative to ptft root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/latest12_lp_benchmark/results_full_ft",
        help="Output directory for this suite (relative to ptft root).",
    )
    parser.add_argument(
        "--train_output_root",
        type=str,
        default="output/latest12_full_ft",
        help="Where each model's fine-tuning run artifacts are stored (relative to ptft root).",
    )
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--save_checkpoints",
        type=str,
        default="true",
        help="Whether to save checkpoint files during full fine-tuning (true/false).",
    )
    parser.add_argument(
        "--metric_key",
        type=str,
        default="",
        help="Metric key used to rank models (defaults to config metric_key).",
    )
    parser.add_argument(
        "--selected_dataset",
        type=str,
        default="",
        help="Override selected_dataset in finetune config (e.g., TUAB/TUEP/TUEV/SEED/BCIC2A/TUSZ).",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="",
        help="Optional CUDA_VISIBLE_DEVICES override, supports multi-GPU DataParallel in one process.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional tag appended to each training output folder.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a model if its training output already has log.csv.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one model run fails.",
    )
    return parser.parse_args()


def project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_best_checkpoints() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_tag, ckpt in sorted(BEST_CHECKPOINT_PATHS.items()):
        run_dir, model_name = model_tag.split("__", 1)
        model_family = model_name.split("_")[0]
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_file():
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            continue

        rows.append(
            {
                "model_tag": model_tag,
                "run_dir": run_dir,
                "model_name": model_name,
                "model_family": model_family,
                "checkpoint": str(ckpt_path),
            }
        )
    return rows


def _metric_is_loss(metric_key: str) -> bool:
    return "loss" in metric_key.lower()


def find_best_from_log(log_csv: Path, metric_key: str) -> Dict[str, Optional[float]]:
    if not log_csv.is_file():
        return {"best_epoch": None, "best_metric": None, "best_metric_raw": None}

    try:
        df = pd.read_csv(log_csv)
    except Exception as exc:
        print(f"[WARN] Failed to read {log_csv}: {exc}")
        return {"best_epoch": None, "best_metric": None, "best_metric_raw": None}

    metric_col = f"val_{metric_key}"
    if metric_col not in df.columns:
        fallback_cols = [c for c in ("val_accuracy", "val_acc", "val_balanced_acc", "val_loss") if c in df.columns]
        if not fallback_cols:
            print(f"[WARN] No usable metric column in {log_csv}")
            return {"best_epoch": None, "best_metric": None, "best_metric_raw": None}
        metric_col = fallback_cols[0]
        print(f"[WARN] Missing {metric_col} target, fallback metric column: {metric_col}")

    if "epoch" not in df.columns:
        df = df.reset_index().rename(columns={"index": "epoch"})

    metric_series = pd.to_numeric(df[metric_col], errors="coerce")
    valid = metric_series.notna()
    if valid.sum() == 0:
        return {"best_epoch": None, "best_metric": None, "best_metric_raw": None}

    metric_values = metric_series[valid]
    best_idx = metric_values.idxmin() if _metric_is_loss(metric_col) else metric_values.idxmax()

    best_epoch = int(df.loc[best_idx, "epoch"])
    best_raw = float(df.loc[best_idx, metric_col])
    best_rank_metric = -best_raw if _metric_is_loss(metric_col) else best_raw

    return {
        "best_epoch": best_epoch,
        "best_metric": best_rank_metric,
        "best_metric_raw": best_raw,
        "metric_col": metric_col,
    }


def run_one_model(
    project_root: Path,
    python_exec: str,
    finetune_config: Path,
    output_root: Path,
    train_output_root: Path,
    rec: Dict[str, str],
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    save_checkpoints: str,
    selected_dataset: str,
    cuda_visible_devices: str,
    run_tag: str,
    metric_key: str,
    skip_existing: bool,
) -> int:
    suffix = f"__{run_tag}" if run_tag else ""
    train_output_dir = train_output_root / f"{rec['model_tag']}{suffix}"
    log_csv = train_output_dir / "log.csv"

    rec["train_output_dir"] = str(train_output_dir)
    rec["train_log_csv"] = str(log_csv)

    if skip_existing and log_csv.is_file():
        print(f"[SKIP] Existing full-ft log found: {log_csv}")
        return 0

    opts = [
        f"output_dir={train_output_dir.as_posix()}",
        "enable_wandb=false",
        f"epochs={epochs}",
        f"optimizer.lr={lr}",
        f"save_checkpoints={save_checkpoints}",
        "model.use_pretrained=true",
        f"model.pretrained_path={rec['checkpoint']}",
    ]
    if selected_dataset:
        opts.append(f"selected_dataset={selected_dataset}")
        # main.py flattens datasets[selected_dataset] into dataset after merging opts,
        # so overrides must target datasets.<name>.* to remain effective.
        opts.extend(
            [
                f"datasets.{selected_dataset}.batch_size={batch_size}",
                f"datasets.{selected_dataset}.num_workers={num_workers}",
            ]
        )
    else:
        opts.extend(
            [
                f"dataset.batch_size={batch_size}",
                f"dataset.num_workers={num_workers}",
            ]
        )

    cmd = [
        python_exec,
        "main.py",
        "--config",
        str(finetune_config),
        "--opts",
        *opts,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    print("\n" + "=" * 88)
    print(f"[RUN] model_tag={rec['model_tag']}")
    print(f"[RUN] checkpoint={rec['checkpoint']}")
    print(f"[RUN] output_dir={train_output_dir}")
    print(f"[RUN] cmd={' '.join(cmd)}")
    print("=" * 88)

    result = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)

    if result.returncode == 0:
        best_info = find_best_from_log(log_csv, metric_key)
        rec["best_epoch"] = best_info.get("best_epoch")
        rec["best_metric"] = best_info.get("best_metric")
        rec["best_metric_raw"] = best_info.get("best_metric_raw")
        rec["metric_col"] = best_info.get("metric_col")

        per_model_dir = output_root / "per_model"
        per_model_dir.mkdir(parents=True, exist_ok=True)
        per_model_csv = per_model_dir / f"{rec['model_tag'].replace('/', '__')}.csv"
        pd.DataFrame(
            [
                {
                    "Model_Tag": rec["model_tag"],
                    "Run_Dir": rec["run_dir"],
                    "Model_Name": rec["model_name"],
                    "Model_Family": rec["model_family"],
                    "Checkpoint": rec["checkpoint"],
                    "Train_Output_Dir": str(train_output_dir),
                    "Metric_Column": rec.get("metric_col"),
                    "Best_Epoch": rec.get("best_epoch"),
                    "Best_Metric_Raw": rec.get("best_metric_raw"),
                    "Best_Metric_For_Rank": rec.get("best_metric"),
                }
            ]
        ).to_csv(per_model_csv, index=False)
        rec["per_model_csv"] = str(per_model_csv)

    return result.returncode


def aggregate_results(records: List[Dict[str, str]], output_root: Path) -> None:
    status_rows = []
    summary_rows = []

    for rec in records:
        status_rows.append(
            {
                "Model_Tag": rec["model_tag"],
                "Run_Dir": rec["run_dir"],
                "Model_Name": rec["model_name"],
                "Model_Family": rec["model_family"],
                "Checkpoint": rec["checkpoint"],
                "Train_Output_Dir": rec.get("train_output_dir"),
                "Train_Log_CSV": rec.get("train_log_csv"),
                "Return_Code": rec.get("return_code", -999),
                "Per_Model_CSV": rec.get("per_model_csv"),
            }
        )

        if rec.get("return_code", -1) != 0:
            continue

        summary_rows.append(
            {
                "Model_Tag": rec["model_tag"],
                "Run_Dir": rec["run_dir"],
                "Model_Name": rec["model_name"],
                "Model_Family": rec["model_family"],
                "Checkpoint": rec["checkpoint"],
                "Metric_Column": rec.get("metric_col"),
                "Best_Epoch": rec.get("best_epoch"),
                "Best_Metric_Raw": rec.get("best_metric_raw"),
                "Best_Metric_For_Rank": rec.get("best_metric"),
                "Train_Output_Dir": rec.get("train_output_dir"),
            }
        )

    status_df = pd.DataFrame(status_rows)
    status_path = output_root / "run_status.csv"
    status_df.to_csv(status_path, index=False)
    print(f"[OK] Saved run status: {status_path}")

    if not summary_rows:
        print("[WARN] No successful runs to summarize.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_root / "results_full_ft_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[OK] Saved full-ft summary: {summary_path}")

    rank_df = summary_df.sort_values(by="Best_Metric_For_Rank", ascending=False)
    rank_path = output_root / "results_full_ft_rank.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"[OK] Saved full-ft ranking: {rank_path}")


def infer_metric_key(config_path: Path, cli_metric_key: str) -> str:
    if cli_metric_key:
        return cli_metric_key

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    metric_key = str(cfg.get("metric_key", "balanced_acc"))
    print(f"[INFO] metric_key from config: {metric_key}")
    return metric_key


def infer_selected_dataset(config_path: Path, cli_selected_dataset: str) -> str:
    if cli_selected_dataset:
        print(f"[INFO] selected_dataset from CLI: {cli_selected_dataset}")
        return cli_selected_dataset

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    selected_dataset = str(cfg.get("selected_dataset", "")).strip()
    if selected_dataset:
        print(f"[INFO] selected_dataset from config: {selected_dataset}")
    else:
        print("[WARN] selected_dataset is empty; falling back to dataset.* override keys.")
    return selected_dataset


def main() -> None:
    args = parse_args()

    project_root = project_root_from_this_file()
    finetune_config = (project_root / args.finetune_config).resolve()
    if not finetune_config.is_file():
        raise FileNotFoundError(f"finetune config not found: {finetune_config}")

    output_root = (project_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_output_root = (project_root / args.train_output_root).resolve()
    train_output_root.mkdir(parents=True, exist_ok=True)

    metric_key = infer_metric_key(finetune_config, args.metric_key)
    selected_dataset = infer_selected_dataset(finetune_config, args.selected_dataset)

    checkpoints = discover_best_checkpoints()
    print(f"[INFO] Discovered {len(checkpoints)} configured best checkpoints.")
    for rec in checkpoints:
        print(f"[INFO] {rec['model_tag']} -> {rec['checkpoint']}")

    if not checkpoints:
        raise RuntimeError("No valid checkpoints discovered.")

    for rec in checkpoints:
        ret = run_one_model(
            project_root=project_root,
            python_exec=args.python_exec,
            finetune_config=finetune_config,
            output_root=output_root,
            train_output_root=train_output_root,
            rec=rec,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_checkpoints=args.save_checkpoints,
            selected_dataset=selected_dataset,
            cuda_visible_devices=args.cuda_visible_devices,
            run_tag=args.run_tag,
            metric_key=metric_key,
            skip_existing=args.skip_existing,
        )
        rec["return_code"] = ret

        if ret != 0:
            print(f"[FAIL] Model failed: {rec['model_tag']} (return code {ret})")
            if args.stop_on_error:
                break
        else:
            print(f"[OK] Model completed: {rec['model_tag']}")

    aggregate_results(checkpoints, output_root)
    print("[DONE] Full fine-tuning suite finished.")


if __name__ == "__main__":
    main()
