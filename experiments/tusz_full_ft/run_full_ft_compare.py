import os
import sys
import argparse
import logging
import warnings

# Filter harmless warnings from sklearn about missing classes in target
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import yaml

sys.path.append(os.getcwd())

from experiments.tusz_full_ft.model_dynamic import CBraModWrapperDynamic
from datasets.tusz import TUSZDataset


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(output_dir, log_filename):
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


def list_tusz_h5_files(dataset_dir):
    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            if fn.endswith(".h5") and fn.startswith("sub_"):
                all_files.append(os.path.join(root, fn))
    all_files.sort()
    return all_files


def group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        base = os.path.basename(f)
        sid = None
        if base.startswith("sub_"):
            parts = base.split("_", 2)
            if len(parts) >= 2:
                sid = parts[1].split(".")[0]
        if not sid:
            sid = base.split(".")[0]
        subject_files.setdefault(sid, []).append(f)
    return subject_files


def split_subjects(subject_ids, seed, ratios):
    ratios = [float(r) for r in ratios]
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Expected ratios like 0.8 0.1 0.1 (sum=1), got {ratios}")

    rng = np.random.RandomState(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * ratios[0]))
    n_val = int(n * ratios[1])
    if n >= 3 and n_val == 0:
        n_val = 1
    n_test = n - n_train - n_val
    if n_test <= 0 and n - n_train - 1 >= 1:
        n_val = max(1, n_val - 1)
        n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def build_loaders(dataset_dir, seed, ratios, batch_size, num_workers, dynamic_length, drop_bad_samples):
    all_h5 = list_tusz_h5_files(dataset_dir)
    if len(all_h5) == 0:
        raise ValueError(f"No sub_*.h5 found under {dataset_dir}")

    subject_files = group_files_by_subject(all_h5)
    train_subs, val_subs, test_subs = split_subjects(list(subject_files.keys()), seed, ratios)

    def gather(subs):
        out = []
        for s in subs:
            out.extend(subject_files.get(s, []))
        return out

    train_files = gather(train_subs)
    val_files = gather(val_subs)
    test_files = gather(test_subs)

    train_ds = TUSZDataset(train_files, mode="train", dynamic_length=dynamic_length, drop_bad_samples=drop_bad_samples)
    val_ds = TUSZDataset(val_files, mode="val", dynamic_length=dynamic_length, drop_bad_samples=drop_bad_samples) if val_files else None
    test_ds = TUSZDataset(test_files, mode="test", dynamic_length=dynamic_length, drop_bad_samples=drop_bad_samples) if test_files else None

    nw = int(num_workers)
    persistent_workers = (nw > 0)

    if nw > 0:
        pf = 4  # Prefetch factor
    else:
        pf = None # default

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=pf,
        collate_fn=TUSZDataset.collate,
    )
    val_loader = None
    test_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=pf,
            collate_fn=TUSZDataset.collate,
        )
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=persistent_workers,
            prefetch_factor=pf,
            collate_fn=TUSZDataset.collate,
        )

    split_info = {
        "n_subjects": len(subject_files),
        "train_subjects": len(train_subs),
        "val_subjects": len(val_subs),
        "test_subjects": len(test_subs),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
    }
    return train_loader, val_loader, test_loader, split_info


def load_pretrained_model(config, weights_path, device):
    model = CBraModWrapperDynamic(config)

    if weights_path and os.path.exists(weights_path):
        logging.info(f"Loading pretrained weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            # Skip head weights mismatch
            if k.startswith("head."):
                continue
            new_state[k] = v
        msg = model.load_state_dict(new_state, strict=False)
        logging.info(f"Weights loaded. missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    else:
        logging.info("No pretrained weights found; using random init.")

    model.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    return model


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


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, use_amp):
    if loader is None:
        return None
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    skipped_batches = 0
    dropped_items = 0

    it = tqdm(loader, desc=("Training" if train else "Evaluating"), leave=False)
    for batch in it:
        unpacked = unpack_batch(batch, device)
        if unpacked is None:
            skipped_batches += 1
            continue
        x, y, mask, dropped = unpacked
        dropped_items += dropped

        if train:
            optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(x, mask=mask)
            if isinstance(logits, dict):
                logits = logits.get("logits", logits.get("cls_pred"))
            loss = criterion(logits, y)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach().cpu().item())
        
        # Probabilities
        probs = torch.softmax(logits.detach(), dim=1)
        all_probs.append(probs.cpu())
        
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    if len(all_targets) == 0:
        return {"loss": float("nan"), "bacc": float("nan"), "f1": float("nan"), "auroc": float("nan"), "auc_pr": float("nan"), "skipped_batches": skipped_batches, "dropped_items": dropped_items}

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()
    
    bacc = balanced_accuracy_score(y_true, y_pred)
    # TUSZ has 13 classes (default in code) or dynamic
    # We should probably detect num_classes from y_probs shape or config, but 13 is standard for TUSZ
    # Actually, y_probs.shape[1] is safer
    n_classes = y_probs.shape[1]
    
    f1 = f1_score(y_true, y_pred, average="weighted", labels=list(range(n_classes)), zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except:
        auroc = float('nan')
        
    try:
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        auc_pr = average_precision_score(y_true_bin, y_probs, average='macro')
    except:
        auc_pr = float('nan')

    denom = max(1, (len(loader) - skipped_batches))
    return {"loss": total_loss / denom, "bacc": bacc, "f1": f1, "auroc": auroc, "auc_pr": auc_pr, "skipped_batches": skipped_batches, "dropped_items": dropped_items}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/finetune_tusz_dynamic.yaml")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/tusz_full_ft/results_compare")
    parser.add_argument("--baseline_path", type=str, default=None)
    parser.add_argument("--flagship_path", type=str, default=None)
    parser.add_argument("--featonly_path", type=str, default=None)
    parser.add_argument("--run_models", type=str, nargs="+", default=["Baseline", "Flagship", "FeatOnly"], help="Which models to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_drop_bad_samples", action="store_false", dest="drop_bad_samples")
    parser.set_defaults(drop_bad_samples=True)
    parser.add_argument("--fixed_length", action="store_false", dest="dynamic_length")
    parser.set_defaults(dynamic_length=True)
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    set_seed(args.seed)

    base_config = load_config(args.config)
    base_config["task_type"] = "classification"
    base_config.setdefault("model", {})
    base_config["model"]["num_classes"] = int(base_config["model"].get("num_classes", 13) or 13)
    base_config["model"]["head_type"] = base_config["model"].get("head_type", "pooling")

    train_loader, val_loader, test_loader, split_info = build_loaders(
        dataset_dir=args.dataset_dir,
        seed=args.seed,
        ratios=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dynamic_length=args.dynamic_length,
        drop_bad_samples=args.drop_bad_samples,
    )

    available_models = {
        "Baseline": args.baseline_path,
        "Flagship": args.flagship_path,
        "FeatOnly": args.featonly_path,
    }

    models_to_run = []
    for m_name in args.run_models:
        if m_name not in available_models:
            print(f"Warning: Model {m_name} not known. Skipping.")
            continue
        p = available_models[m_name]
        if p is None:
            raise ValueError(f"Path for model '{m_name}' was not provided.")
        models_to_run.append((m_name, p))

    results = []
    for name, wpath in models_to_run:
        setup_logger(args.output_dir, f"training_{name.lower()}.log")
        logging.info(f"=== TUSZ full FT compare: {name} ===")
        logging.info(f"Split: {args.split} info={split_info}")

        # Configure Head Type
        current_config = base_config.copy()
        current_config["model"] = base_config["model"].copy()
        
        if name == 'FeatOnly':
            current_config['model']['head_type'] = 'feat_cross_attn'
        elif name == 'Flagship':
            current_config['model']['head_type'] = 'flagship_concat'
        else:
            current_config['model']['head_type'] = 'pooling' # Default Baseline

        model = load_pretrained_model(current_config, wpath, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=use_amp)

        best_val_bacc = -1.0
        final_test_results = None

        epoch_results = []

        for epoch in range(args.epochs):
            train_m = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True, use_amp=use_amp)
            val_m = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False, use_amp=use_amp) if val_loader else None
            test_m = run_epoch(model, test_loader, criterion, optimizer, device, scaler, train=False, use_amp=use_amp) if test_loader else None

            msg = f"Epoch {epoch+1}/{args.epochs} train_loss={train_m['loss']:.4f} train_bacc={train_m['bacc']:.4f}"
            if val_m:
                msg += f" | Val BAcc: {val_m['bacc']:.4f} F1: {val_m['f1']:.4f} AUROC: {val_m['auroc']:.4f}"
                if val_m["bacc"] > best_val_bacc:
                    best_val_bacc = val_m["bacc"]
            if test_m:
                msg += f" | Test BAcc: {test_m['bacc']:.4f} F1: {test_m['f1']:.4f} AUROC: {test_m['auroc']:.4f}"
                final_test_results = test_m # Store last one or logic to store best val's test?
                # Usually we want the test result corresponding to best val, but here we just log all
            
            logging.info(msg)
            print(msg)
            
            epoch_data = {
                "epoch": epoch + 1,
                "train": train_m,
                "val": val_m,
                "test": test_m
            }
            epoch_results.append(epoch_data)

        results.append({
            "model": name,
            "best_val_bacc": best_val_bacc,
            "epoch_history": epoch_results
        })

    report_path = os.path.join(args.output_dir, "results_full_ft_compare.md")
    with open(report_path, "w") as f:
        f.write("# TUSZ Full Fine-tuning Comparative Results\n\n")
        f.write(f"- Split (subject): train/val/test = {args.split}\n")
        f.write(f"- Split info: {split_info}\n\n")
        
        for r in results:
            f.write(f"## Model: {r['model']}\n")
            f.write(f"**Best Val BAcc:** {r['best_val_bacc']:.4f}\n\n")
            f.write("| Epoch | Train Loss | Train BAcc | Val Loss | Val BAcc | Val AUROC | Val AUC-PR | Val F1 | Test Loss | Test BAcc | Test AUROC | Test AUC-PR | Test F1 |\n")
            f.write("|------:|-----------:|-----------:|---------:|---------:|----------:|------------:|-------:|----------:|----------:|-----------:|------------:|--------:|\n")
            for ep in r["epoch_history"]:
                e_num = ep["epoch"]
                tr = ep["train"]
                vl = ep["val"] or {"loss": float("nan"), "bacc": float("nan"), "auroc": float("nan"), "auc_pr": float("nan"), "f1": float("nan")}
                ts = ep["test"] or {"loss": float("nan"), "bacc": float("nan"), "auroc": float("nan"), "auc_pr": float("nan"), "f1": float("nan")}
                f.write(f"| {e_num} | {tr['loss']:.4f} | {tr['bacc']:.4f} | {vl['loss']:.4f} | {vl['bacc']:.4f} | {vl['auroc']:.4f} | {vl['auc_pr']:.4f} | {vl['f1']:.4f} | {ts['loss']:.4f} | {ts['bacc']:.4f} | {ts['auroc']:.4f} | {ts['auc_pr']:.4f} | {ts['f1']:.4f} |\n")
            f.write("\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
