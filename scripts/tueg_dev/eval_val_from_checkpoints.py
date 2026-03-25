#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Tuple

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.engine import evaluate
from core.loss import LossFactory
from datasets.builder import build_dataloader
from models.wrapper import CBraModWrapper


def _extract_epoch(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"checkpoint_epoch_(\d+)\.pth$", name)
    return int(m.group(1)) if m else -1


def _load_config_from_checkpoint(path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Checkpoint has no valid config dict: {path}")
    return cfg


def _sync_feature_dim_from_dataset(config: Dict, val_loader) -> None:
    dataset_obj = val_loader.dataset
    if hasattr(dataset_obj, "dataset"):
        dataset_obj = dataset_obj.dataset
    if hasattr(dataset_obj, "feature_dim") and getattr(dataset_obj, "feature_dim", 0) > 0:
        detected_dim = int(dataset_obj.feature_dim)
        config_dim = int(config.get("model", {}).get("feature_dim", 0))
        if detected_dim != config_dim:
            print(
                f"[Info] model.feature_dim mismatch ({config_dim} -> {detected_dim}), auto-sync for eval"
            )
            config.setdefault("model", {})["feature_dim"] = detected_dim


def _prepare_model_and_val_loader(
    config: Dict,
    device: torch.device,
    batch_size_override: int = -1,
    use_full_val: bool = False,
):
    ds = config.setdefault("dataset", {})
    if batch_size_override > 0:
        ds["batch_size"] = int(batch_size_override)

    # Eval-only loader settings: conservative and stable.
    ds["val_num_workers"] = int(ds.get("val_num_workers", 0))
    ds["val_persistent_workers"] = bool(ds.get("val_persistent_workers", False))
    ds["val_timeout"] = int(ds.get("val_timeout", 0))
    if use_full_val:
        ds["val_max_files"] = 0

    val_loader = build_dataloader(ds["name"], ds, mode="val")

    config["model"]["task_type"] = config.get("task_type", "classification")
    _sync_feature_dim_from_dataset(config, val_loader)

    model = CBraModWrapper(config).to(device)
    loss_cfg = config.get("loss", {})
    criterion = LossFactory.get(loss_cfg.get("name", "mse"), **loss_cfg.get("params", {})).to(device)
    return model, criterion, val_loader


def _evaluate_checkpoints(
    checkpoints: List[str],
    model,
    criterion,
    val_loader,
    device: torch.device,
    task_type: str,
    limit_batches: int,
) -> List[Dict]:
    rows: List[Dict] = []
    for idx, ckpt_path in enumerate(checkpoints, start=1):
        epoch = _extract_epoch(ckpt_path)
        print(f"[{idx}/{len(checkpoints)}] Evaluating epoch {epoch}: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict")
        if state_dict is None:
            print(f"[Warn] Skip (missing model_state_dict): {ckpt_path}")
            continue

        model.load_state_dict(state_dict, strict=True)

        metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            task_type=task_type,
            log_writer=None,
            epoch=epoch,
            header=f"Val@epoch{epoch}:",
            log_prefix="val",
            limit_batches=(limit_batches if limit_batches > 0 else None),
            verbose=False,
        )

        row = {"epoch": epoch}
        for k, v in metrics.items():
            if isinstance(v, (int, float, bool, str)):
                row[f"val_{k}"] = v

        # Friendly aliases for reconstruction-only asks
        if "val_recon_pcc" in row and "val_pcc" not in row:
            row["val_pcc"] = row["val_recon_pcc"]
        if "val_recon_r2" in row and "val_r2" not in row:
            row["val_r2"] = row["val_recon_r2"]

        rows.append(row)
    return rows


def _write_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        raise RuntimeError("No evaluation rows produced.")
    keys = set()
    for r in rows:
        keys.update(r.keys())

    # Keep epoch first, then stable alphabetical order.
    ordered = ["epoch"] + sorted(k for k in keys if k != "epoch")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: int(x.get("epoch", -1))))
    print(f"[Done] Wrote {len(rows)} rows -> {out_csv}")


def _resolve_checkpoints(run_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(run_dir, pattern))
    paths = [p for p in paths if _extract_epoch(p) >= 0]
    return sorted(paths, key=_extract_epoch)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Recompute validation metrics for each epoch checkpoint.")
    ap.add_argument("--run-dir", required=True, help="Directory containing checkpoint_epoch_*.pth")
    ap.add_argument("--pattern", default="checkpoint_epoch_*.pth", help="Checkpoint glob pattern")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", default="", help="Output CSV path (default: <run-dir>/val_metrics_recomputed.csv)")
    ap.add_argument("--batch-size", type=int, default=-1, help="Optional override for eval batch size")
    ap.add_argument("--limit-batches", type=int, default=0, help="Only evaluate first N val batches (0=all)")
    ap.add_argument("--max-epochs", type=int, default=0, help="Evaluate only first N checkpoints after sorting (0=all)")
    ap.add_argument("--full-val", action="store_true", help="Use full validation split (disable val file cap)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_csv = args.out_csv or os.path.join(run_dir, "val_metrics_recomputed.csv")

    checkpoints = _resolve_checkpoints(run_dir, args.pattern)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir} with pattern {args.pattern}")

    if args.max_epochs > 0:
        checkpoints = checkpoints[: args.max_epochs]

    print(f"Found {len(checkpoints)} checkpoints in {run_dir}")
    print(f"Using device: {args.device}")

    config = _load_config_from_checkpoint(checkpoints[0])
    config["enable_wandb"] = False

    device = torch.device(args.device)
    model, criterion, val_loader = _prepare_model_and_val_loader(
        config=config,
        device=device,
        batch_size_override=args.batch_size,
        use_full_val=args.full_val,
    )

    rows = _evaluate_checkpoints(
        checkpoints=checkpoints,
        model=model,
        criterion=criterion,
        val_loader=val_loader,
        device=device,
        task_type=config.get("task_type", "classification"),
        limit_batches=args.limit_batches,
    )
    _write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
