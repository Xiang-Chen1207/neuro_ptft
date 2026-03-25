#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


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
    "20260316_joint_recon_feat_screen__reve_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/reve_joint/checkpoint_epoch_38.pth",
    "20260316_joint_recon_feat_screen__tech_joint": "/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs/20260316_joint_recon_feat_screen/tech_joint/checkpoint_epoch_38.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run linear probing benchmark for 12 latest.pth checkpoints and compare results."
    )
    parser.add_argument(
        "--runs_root",
        type=str,
        default="/vePFS-0x0d/home/cx/ptft/output/tueg_dev/runs",
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--run_dirs",
        type=str,
        default=(
            "20260313_181700_onegpu_full_safeio,"
            "20260313_181700_onegpu_full_safeio_scaled_next,"
            "20260316_joint_recon_feat_screen"
        ),
        help="Comma-separated run directory names under runs_root.",
    )
    parser.add_argument(
        "--benchmark_script",
        type=str,
        default="experiments/joint_benchmark/run_benchmark.py",
        help="Benchmark script path relative to ptft project root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/latest12_lp_benchmark/results",
        help="Output directory relative to ptft project root.",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable used to run benchmark_script.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--datasets",
        type=str,
        default="TUAB,BCIC2A,SEEDIV,TUEP,TUEV,TUSZ",
        help="Comma-separated downstream datasets.",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Pass --tiny to benchmark script for quick debugging.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip model if per-model CSV already exists.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one model run fails.",
    )
    return parser.parse_args()


def project_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_latest_checkpoints(runs_root: Path, run_dirs: List[str]) -> List[Dict[str, str]]:
    checkpoints: List[Dict[str, str]] = []
    for run_dir in run_dirs:
        run_path = runs_root / run_dir
        if not run_path.is_dir():
            print(f"[WARN] Run directory not found: {run_path}")
            continue

        for model_dir in sorted(p for p in run_path.iterdir() if p.is_dir()):
            model_name = model_dir.name
            model_family = model_name.split("_")[0]
            model_tag = f"{run_dir}__{model_name}"

            best_ckpt = BEST_CHECKPOINT_PATHS.get(model_tag)
            if best_ckpt is not None:
                ckpt_path = Path(best_ckpt)
                if not ckpt_path.is_file():
                    print(f"[WARN] Missing configured best checkpoint {ckpt_path}, fallback to latest.pth")
                    ckpt_path = model_dir / "latest.pth"
            else:
                ckpt_path = model_dir / "latest.pth"

            if not ckpt_path.is_file():
                print(f"[WARN] Missing checkpoint: {ckpt_path}")
                continue

            checkpoints.append(
                {
                    "run_dir": run_dir,
                    "model_name": model_name,
                    "model_family": model_family,
                    "model_tag": model_tag,
                    "checkpoint": str(ckpt_path),
                }
            )

    checkpoints.sort(key=lambda x: x["model_tag"])
    return checkpoints


def run_one_model(
    project_root: Path,
    benchmark_script: Path,
    python_exec: str,
    output_csv: Path,
    checkpoint: str,
    batch_size: int,
    device: str,
    datasets: str,
    tiny: bool,
) -> int:
    cmd = [
        python_exec,
        str(benchmark_script),
        "--checkpoint",
        checkpoint,
        "--output_csv",
        str(output_csv),
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--datasets",
        datasets,
    ]
    if tiny:
        cmd.append("--tiny")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}".rstrip(":")

    print("\n" + "=" * 88)
    print(f"[RUN] checkpoint={checkpoint}")
    print(f"[RUN] output_csv={output_csv}")
    print(f"[RUN] cmd={' '.join(cmd)}")
    print("=" * 88)

    result = subprocess.run(cmd, cwd=str(project_root), env=env, check=False)
    return result.returncode


def aggregate_results(records: List[Dict[str, str]], output_root: Path) -> None:
    long_rows = []
    status_rows = []

    for rec in records:
        per_csv = Path(rec["per_model_csv"])
        status_rows.append(
            {
                "Model_Tag": rec["model_tag"],
                "Run_Dir": rec["run_dir"],
                "Model_Name": rec["model_name"],
                "Model_Family": rec["model_family"],
                "Checkpoint": rec["checkpoint"],
                "Per_Model_CSV": str(per_csv),
                "Return_Code": rec.get("return_code", -999),
                "CSV_Exists": per_csv.is_file(),
            }
        )

        if not per_csv.is_file():
            continue

        try:
            df = pd.read_csv(per_csv)
        except Exception as exc:
            print(f"[WARN] Failed to read CSV {per_csv}: {exc}")
            continue

        required = {"Dataset", "Accuracy", "Balanced_Accuracy"}
        if not required.issubset(set(df.columns)):
            print(f"[WARN] Missing required columns in {per_csv}: {required - set(df.columns)}")
            continue

        for _, row in df.iterrows():
            long_rows.append(
                {
                    "Model_Tag": rec["model_tag"],
                    "Run_Dir": rec["run_dir"],
                    "Model_Name": rec["model_name"],
                    "Model_Family": rec["model_family"],
                    "Checkpoint": rec["checkpoint"],
                    "Dataset": row.get("Dataset"),
                    "Accuracy": row.get("Accuracy"),
                    "Balanced_Accuracy": row.get("Balanced_Accuracy"),
                    "Train_Size": row.get("Train_Size"),
                    "Test_Size": row.get("Test_Size"),
                    "Error": row.get("Error") if "Error" in df.columns else None,
                }
            )

    status_df = pd.DataFrame(status_rows)
    status_path = output_root / "run_status.csv"
    status_df.to_csv(status_path, index=False)
    print(f"[OK] Saved run status: {status_path}")

    if not long_rows:
        print("[WARN] No valid per-model results found for aggregation.")
        return

    long_df = pd.DataFrame(long_rows)
    long_path = output_root / "results_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"[OK] Saved long results: {long_path}")

    bacc_pivot = long_df.pivot_table(
        index="Dataset", columns="Model_Tag", values="Balanced_Accuracy", aggfunc="first"
    )
    bacc_pivot_path = output_root / "results_compare_bacc.csv"
    bacc_pivot.to_csv(bacc_pivot_path)
    print(f"[OK] Saved BAcc comparison: {bacc_pivot_path}")

    acc_pivot = long_df.pivot_table(
        index="Dataset", columns="Model_Tag", values="Accuracy", aggfunc="first"
    )
    acc_pivot_path = output_root / "results_compare_acc.csv"
    acc_pivot.to_csv(acc_pivot_path)
    print(f"[OK] Saved Acc comparison: {acc_pivot_path}")

    summary = (
        long_df.groupby("Model_Tag", as_index=False)
        .agg(
            Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
            Mean_Accuracy=("Accuracy", "mean"),
            Num_Datasets=("Dataset", "nunique"),
        )
        .sort_values(by="Mean_Balanced_Accuracy", ascending=False)
    )

    meta_df = long_df[["Model_Tag", "Run_Dir", "Model_Name", "Model_Family", "Checkpoint"]].drop_duplicates()
    summary = summary.merge(meta_df, on="Model_Tag", how="left")

    summary_path = output_root / "results_rank_mean_bacc.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved ranking summary: {summary_path}")


def main() -> None:
    args = parse_args()

    project_root = project_root_from_this_file()
    runs_root = Path(args.runs_root)
    run_dirs = [x.strip() for x in args.run_dirs.split(",") if x.strip()]

    benchmark_script = (project_root / args.benchmark_script).resolve()
    if not benchmark_script.is_file():
        raise FileNotFoundError(f"benchmark script not found: {benchmark_script}")

    output_root = (project_root / args.output_dir).resolve()
    per_model_dir = output_root / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_latest_checkpoints(runs_root, run_dirs)
    print(f"[INFO] Discovered {len(checkpoints)} checkpoints.")
    for rec in checkpoints:
        print(f"[INFO] {rec['model_tag']} -> {rec['checkpoint']}")

    if not checkpoints:
        raise RuntimeError("No checkpoints discovered. Please check --runs_root and --run_dirs.")

    for rec in checkpoints:
        safe_csv_name = f"{rec['model_tag'].replace('/', '__')}.csv"
        per_model_csv = per_model_dir / safe_csv_name

        rec["per_model_csv"] = str(per_model_csv)

        if args.skip_existing and per_model_csv.is_file():
            print(f"[SKIP] Existing result CSV found: {per_model_csv}")
            rec["return_code"] = 0
            continue

        ret = run_one_model(
            project_root=project_root,
            benchmark_script=benchmark_script,
            python_exec=args.python_exec,
            output_csv=per_model_csv,
            checkpoint=rec["checkpoint"],
            batch_size=args.batch_size,
            device=args.device,
            datasets=args.datasets,
            tiny=args.tiny,
        )
        rec["return_code"] = ret

        if ret != 0:
            print(f"[FAIL] Model failed: {rec['model_tag']} (return code {ret})")
            if args.stop_on_error:
                break
        else:
            print(f"[OK] Model completed: {rec['model_tag']}")

    aggregate_results(checkpoints, output_root)
    print("[DONE] Batch benchmark and comparison finished.")


if __name__ == "__main__":
    main()
