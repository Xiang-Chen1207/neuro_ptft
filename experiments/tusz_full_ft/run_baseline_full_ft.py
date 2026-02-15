import os
import sys
import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.append(os.getcwd())

from experiments.tusz_full_ft.baseline_model import BaselineConv1D
from datasets.tusz import TUSZDataset, get_tusz_file_list


def setup_logger(output_dir, log_filename="baseline_full_ft.log"):
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_filename)),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch, device):
    if batch is None:
        return None
    if isinstance(batch, dict):
        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).long()
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(device, non_blocking=True)
        dropped = int(batch.get("dropped", 0) or 0)
        return x, y, mask, dropped
    if len(batch) == 3:
        x, y, mask = batch
        return x.to(device).float(), y.to(device).long(), mask.to(device), 0
    x, y = batch
    return x.to(device).float(), y.to(device).long(), None, 0


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp=True):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    skipped_batches = 0
    dropped_items = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        unpacked = unpack_batch(batch, device)
        if unpacked is None:
            skipped_batches += 1
            continue
        x, y, mask, dropped = unpacked
        dropped_items += dropped

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x, mask=mask)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().cpu().item())
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    if len(all_targets) == 0:
        return {"loss": float("nan"), "bacc": float("nan"), "f1": float("nan"), "skipped_batches": skipped_batches, "dropped_items": dropped_items}

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"loss": total_loss / max(1, (len(loader) - skipped_batches)), "bacc": bacc, "f1": f1, "skipped_batches": skipped_batches, "dropped_items": dropped_items}


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    skipped_batches = 0
    dropped_items = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        unpacked = unpack_batch(batch, device)
        if unpacked is None:
            skipped_batches += 1
            continue
        x, y, mask, dropped = unpacked
        dropped_items += dropped

        with autocast(enabled=use_amp):
            logits = model(x, mask=mask)
            loss = criterion(logits, y)

        total_loss += float(loss.detach().cpu().item())
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    if len(all_targets) == 0:
        return {"loss": float("nan"), "bacc": float("nan"), "f1": float("nan"), "skipped_batches": skipped_batches, "dropped_items": dropped_items}

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"loss": total_loss / max(1, (len(loader) - skipped_batches)), "bacc": bacc, "f1": f1, "skipped_batches": skipped_batches, "dropped_items": dropped_items}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/tusz_full_ft/baseline_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_drop_bad_samples", action="store_false", dest="drop_bad_samples")
    parser.set_defaults(drop_bad_samples=True)
    parser.add_argument("--fixed_length", action="store_false", dest="dynamic_length")
    parser.set_defaults(dynamic_length=True)
    args = parser.parse_args()

    setup_logger(args.output_dir)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    logging.info("Loading TUSZ file lists...")
    train_files = get_tusz_file_list(args.dataset_dir, "train", seed=args.seed)
    val_files = get_tusz_file_list(args.dataset_dir, "val", seed=args.seed)
    test_files = get_tusz_file_list(args.dataset_dir, "test", seed=args.seed)

    train_ds = TUSZDataset(train_files, mode="train", dynamic_length=args.dynamic_length, drop_bad_samples=args.drop_bad_samples)
    val_ds = TUSZDataset(val_files, mode="val", dynamic_length=args.dynamic_length, drop_bad_samples=args.drop_bad_samples)
    test_ds = TUSZDataset(test_files, mode="test", dynamic_length=args.dynamic_length, drop_bad_samples=args.drop_bad_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=TUSZDataset.collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=TUSZDataset.collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=TUSZDataset.collate, pin_memory=True)

    model = BaselineConv1D(num_channels=19, num_classes=13, dropout=0.2).to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)

    best_val_bacc = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp=use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_bacc={train_metrics['bacc']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_bacc={val_metrics['bacc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"skipped_batches(train/val)={train_metrics['skipped_batches']}/{val_metrics['skipped_batches']} "
            f"dropped_items(train/val)={train_metrics['dropped_items']}/{val_metrics['dropped_items']}"
        )

        if val_metrics["bacc"] == val_metrics["bacc"] and val_metrics["bacc"] > best_val_bacc:
            best_val_bacc = val_metrics["bacc"]
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, best_path)
            logging.info(f"New best model saved: val_bacc={best_val_bacc:.4f}")

    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(state, strict=True)
        test_metrics = evaluate(model, test_loader, criterion, device, use_amp=use_amp)
        logging.info(f"TEST loss={test_metrics['loss']:.4f} bacc={test_metrics['bacc']:.4f} f1={test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
