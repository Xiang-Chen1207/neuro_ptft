import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import logging

# Add project root to path
import sys
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
# CHANGED: Import TUEV instead of TUAB
from datasets.tuev import TUEVDataset, get_tuev_file_list

# Define load_config directly to resolve import error
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Define setup_logger directly to resolve import error
def setup_logger(output_dir, log_filename='training.log'):
    # Remove existing handlers to avoid duplicate logs when switching models
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_filename)),
            logging.StreamHandler()
        ]
    )

def load_pretrained_model(config, weights_path, device):
    # Initialize model wrapper
    # Note: Full fine-tuning uses the classification head
    # The config should specify num_classes, etc.
    model = CBraModWrapper(config)
    
    # Load backbone weights
    print(f"Loading pretrained weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        # Filter state dict to load only backbone or matching keys
        # For full FT, we generally load the backbone. 
        # If the keys match, we load them.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        # Load strictly=False because the classification head might be new or different
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Proceeding might be dangerous if weights are crucial, but let's allow it for debugging if needed
        # sys.exit(1)
        
    model.to(device)
    
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(dataloader, desc="Training", leave=False):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        with torch.amp.autocast('cuda'):
            output = model(data)
            
            # Check output type
            if isinstance(output, dict):
                logits = output.get('logits', output.get('cls_pred', None))
            else:
                logits = output
                
            if logits is None:
                raise ValueError("Model did not return logits!")
                
            loss = criterion(logits, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            if isinstance(output, dict):
                logits = output.get('logits', output.get('cls_pred', None))
            else:
                logits = output
                
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            # Probabilities for AUC/AUROC
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    # Explicitly provide labels to ensure consistent behavior, though 'weighted' ignores missing classes (weight=0)
    # TUEV has 6 classes (0-5)
    f1 = f1_score(all_targets, all_preds, average='weighted', labels=list(range(6)), zero_division=0)
    
    # AUROC and AUC-PR
    try:
        all_targets_np = np.array(all_targets)
        all_probs_np = np.array(all_probs)
        n_classes = all_probs_np.shape[1]
        
        if n_classes == 2:
            # Binary case
            auroc = roc_auc_score(all_targets_np, all_probs_np[:, 1])
            auc_pr = average_precision_score(all_targets_np, all_probs_np[:, 1])
        else:
            # Multiclass case
            # Binarize labels for AUC-PR
            # Handle missing classes in y_true by explicit classes list if known, 
            # but here we rely on what's present or try/except.
            # label_binarize needs to know all possible classes to make columns match probs
            # We assume classes are 0..n_classes-1
            y_true_bin = label_binarize(all_targets_np, classes=range(n_classes))
            
            # multi_class='ovr' handles missing classes in y_true gracefully usually,
            # but 'weighted' average might warn if a class is totally missing.
            auroc = roc_auc_score(all_targets_np, all_probs_np, multi_class='ovr', average='macro')
            auc_pr = average_precision_score(y_true_bin, all_probs_np, average='macro')
            
    except Exception as e:
        # print(f"Metric Error: {e}")
        auroc = float('nan')
        auc_pr = float('nan')
        
    return {
        "loss": total_loss / len(dataloader), 
        "acc": acc,
        "bacc": bacc, 
        "f1": f1,
        "auroc": auroc,
        "auc_pr": auc_pr
    }

def main():
    parser = argparse.ArgumentParser()
    # CHANGED: default config to tuev
    parser.add_argument('--config', type=str, default='configs/finetune_tuev.yaml')
    parser.add_argument('--baseline_path', type=str, default=None)
    parser.add_argument('--flagship_path', type=str, default=None)
    parser.add_argument('--featonly_path', type=str, default=None)
    parser.add_argument('--run_models', type=str, nargs='+', default=['Baseline', 'Flagship', 'FeatOnly'], help='Models to run')
    # CHANGED: default dataset dir to TUEV path
    parser.add_argument('--dataset_dir', type=str, default='/vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Events')
    # CHANGED: default output dir
    parser.add_argument('--output_dir', type=str, default='experiments/tuev_full_ft/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs='+', 
                        default=[1.0])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=16)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load Dataset Index
    print("Loading Dataset Index...")
    # CHANGED: Use get_tuev_file_list
    train_files_orig = get_tuev_file_list(args.dataset_dir, 'train', seed=args.seed)
    test_files_orig = get_tuev_file_list(args.dataset_dir, 'test', seed=args.seed)
    
    all_files = train_files_orig + test_files_orig
    print(f"Total files loaded: {len(all_files)}")
    
    # CHANGED: TUEV Subject ID extraction logic
    def get_subject_from_path(path):
        basename = os.path.basename(path)
        # TUEV: sub_aaaaaaar.h5 -> aaaaaaar
        try:
            # sub_aaaaaaar.h5 -> split('_')[1] -> aaaaaaar.h5 -> split('.')[0] -> aaaaaaar
            return basename.split('_')[1].split('.')[0]
        except:
            return basename.split('.')[0]

    print("Grouping all files by subject...")
    all_file_subjects = [get_subject_from_path(f) for f in all_files]
    unique_subjects = np.unique(all_file_subjects)
    total_subjects = len(unique_subjects)
    print(f"Total Unique Subjects (Train + Test): {total_subjects}")
    
    # Global Shuffle
    rng = np.random.RandomState(args.seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)
    
    # Load Config template
    base_config = load_config(args.config)
    # Ensure classification task
    base_config['task_type'] = 'classification'
    # CHANGED: 6 Classes for TUEV
    base_config['model']['num_classes'] = 6
    # Ensure sequence length is 1 for TUEV
    base_config['model']['seq_len'] = 1
    
    # Results storage
    results = []
    
    # Iterate Ratios
    for ratio in sorted(args.ratios):
        if ratio >= 1.0:
            n_subs = total_subjects
            current_subjects = shuffled_subjects
            ratio_display = "100%"
        else:
            n_subs = max(1, int(total_subjects * ratio))
            current_subjects = shuffled_subjects[:n_subs]
            ratio_display = f"{ratio*100:.1f}%"
            
        print(f"\n=== Ratio: {ratio_display} ({n_subs} Subjects) ===")
        
        # Split 80/10/10 for TUEV (Full FT Standard)
        n_train = int(0.8 * n_subs)
        n_val = int(0.1 * n_subs)
        
        if n_train == 0: n_train = 1
        if n_val == 0 and n_subs > 1: n_val = 1
        
        # Adjust indices
        train_subs = current_subjects[:n_train]
        val_subs = current_subjects[n_train:n_train+n_val]
        test_subs = current_subjects[n_train+n_val:]
        
        print(f"Split Sizes (Subjects): Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
        
        # Create Subsets of files
        train_subs_set = set(train_subs)
        val_subs_set = set(val_subs)
        test_subs_set = set(test_subs)
        
        # Helper to filter files
        train_files_subset = [f for f, s in zip(all_files, all_file_subjects) if s in train_subs_set]
        val_files_subset = [f for f, s in zip(all_files, all_file_subjects) if s in val_subs_set]
        test_files_subset = [f for f, s in zip(all_files, all_file_subjects) if s in test_subs_set]
        
        if len(train_files_subset) == 0:
            print("Warning: No train files selected!")
            continue
            
        # CHANGED: Use TUEVDataset
        train_dataset = TUEVDataset(train_files_subset)
        val_dataset = TUEVDataset(val_files_subset) if val_files_subset else None
        test_dataset = TUEVDataset(test_files_subset) if test_files_subset else None
        
        # Prepare DataLoaders
        pf = 4 if args.num_workers > 0 else None
        persistent = (args.num_workers > 0)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=persistent, prefetch_factor=pf, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=persistent, prefetch_factor=pf, pin_memory=True) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=persistent, prefetch_factor=pf, pin_memory=True) if test_dataset else None
        
        # Define Models to Run
        available_models = {
            'Baseline': args.baseline_path,
            'Flagship': args.flagship_path,
            'FeatOnly': args.featonly_path
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
        
        for model_name, weights_path in models_to_run:
            print(f"--- Training {model_name} ---")
            
            # Setup logger for this model
            log_filename = f"training_{model_name.lower()}_{ratio_display.replace('%','')}.log"
            setup_logger(args.output_dir, log_filename)
            logging.info(f"Starting training for model: {model_name}, Ratio: {ratio_display}")
            
            # Configure Head Type
            current_config = base_config.copy()
            current_config['model'] = base_config['model'].copy()
            
            if model_name == 'FeatOnly':
                current_config['model']['head_type'] = 'feat_cross_attn'
            elif model_name == 'Flagship':
                current_config['model']['head_type'] = 'flagship_concat'
            else:
                current_config['model']['head_type'] = 'pooling' # Default Baseline
            
            # Reset Model for each run
            model = load_pretrained_model(current_config, weights_path, device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
            criterion = nn.CrossEntropyLoss()
            
            best_val_bacc = 0.0
            
            epoch_results = []
            
            # Training Loop
            scaler = torch.amp.GradScaler('cuda')
            
            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
                
                # Validate
                val_stats = {"loss": float("nan"), "acc": float("nan"), "bacc": float("nan"), "f1": float("nan"), "auroc": float("nan"), "auc_pr": float("nan")}
                if val_loader:
                    val_stats = evaluate(model, val_loader, criterion, device)
                    
                    if val_stats['bacc'] > best_val_bacc:
                        best_val_bacc = val_stats['bacc']
                        
                # Test every epoch
                test_stats = {"loss": float("nan"), "acc": float("nan"), "bacc": float("nan"), "f1": float("nan"), "auroc": float("nan"), "auc_pr": float("nan")}
                if test_loader:
                    test_stats = evaluate(model, test_loader, criterion, device)

                log_msg = (f"Epoch {epoch+1}/{args.epochs} | "
                           f"Train Loss: {train_loss:.4f} | "
                           f"Val BAcc: {val_stats['bacc']:.4f} F1: {val_stats['f1']:.4f} AUROC: {val_stats['auroc']:.4f} | "
                           f"Test BAcc: {test_stats['bacc']:.4f} F1: {test_stats['f1']:.4f} AUROC: {test_stats['auroc']:.4f}")
                print(log_msg)
                logging.info(log_msg)
                
                epoch_results.append({
                    "epoch": epoch + 1,
                    "train": {"loss": train_loss, "acc": train_acc},
                    "val": val_stats,
                    "test": test_stats
                })
            
            results.append({
                'ratio': ratio_display,
                'n_subs': n_subs,
                'model': model_name,
                'best_val_bacc': best_val_bacc,
                'epoch_history': epoch_results
            })

    # Generate Report
    print("\n=== Final Results Summary ===")
    
    report_path = os.path.join(args.output_dir, 'results_final_ft.md')
    with open(report_path, 'w') as f:
        f.write("# TUEV Full Fine-tuning Comparative Results (80/10/10 Split)\n\n")
        f.write(f"| Ratio | NumSub | Model | Best Val BAcc (%) |\n")
        f.write(f"|-------|--------|-------|-------------------|\n")
        
        for r in results:
            f.write(f"| {r['ratio']} | {r['n_subs']} | {r['model']} | {r['best_val_bacc']*100:.2f} |\n")
        
        f.write("\n\n# Detailed Epoch History\n")
        for r in results:
            f.write(f"\n## {r['model']} (Ratio: {r['ratio']})\n")
            f.write("| Epoch | Train Loss | Val BAcc | Val F1 | Val AUROC | Val AUC-PR | Test BAcc | Test F1 | Test AUROC | Test AUC-PR |\n")
            f.write("|-------|------------|----------|--------|-----------|------------|-----------|---------|------------|-------------|\n")
            for ep in r['epoch_history']:
                f.write(f"| {ep['epoch']} | {ep['train']['loss']:.4f} | "
                        f"{ep['val']['bacc']:.4f} | {ep['val']['f1']:.4f} | {ep['val']['auroc']:.4f} | {ep['val']['auc_pr']:.4f} | "
                        f"{ep['test']['bacc']:.4f} | {ep['test']['f1']:.4f} | {ep['test']['auroc']:.4f} | {ep['test']['auc_pr']:.4f} |\n")
            
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
