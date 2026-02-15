import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import logging
import warnings

# Filter harmless warnings from sklearn about missing classes in target
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# Add project root to path
import sys
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
from datasets.tuep import TUEPDataset, get_tuep_file_list

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(output_dir, log_filename='training.log'):
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
    model = CBraModWrapper(config)
    print(f"Loading pretrained weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        
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
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None: continue
        
        data = batch['x'].to(device)
        target = batch['y'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output = model(data)
            
            if isinstance(output, dict):
                logits = output.get('logits', output.get('cls_pred', None))
            else:
                logits = output
                
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
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None: continue
            
            data = batch['x'].to(device)
            target = batch['y'].to(device)
            
            output = model(data)
            if isinstance(output, dict):
                logits = output.get('logits', output.get('cls_pred', None))
            else:
                logits = output
                
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Prob for class 1
            
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    # TUEP has 2 classes
    f1 = f1_score(all_targets, all_preds, average='weighted', labels=list(range(2)), zero_division=0)
    try:
        auroc = roc_auc_score(all_targets, all_probs)
        auc_pr = average_precision_score(all_targets, all_probs)
    except Exception as e:
        print(f"Warning: Metric calculation failed: {e}")
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
    parser.add_argument('--config', type=str, default='configs/finetune_tuep.yaml')
    parser.add_argument('--baseline_path', type=str, default=None)
    parser.add_argument('--flagship_path', type=str, default=None)
    parser.add_argument('--featonly_path', type=str, default=None)
    parser.add_argument('--run_models', type=str, nargs='+', default=['Baseline', 'Flagship', 'FeatOnly'], help='Models to run')
    parser.add_argument('--dataset_dir', type=str, default='/vePFS-0x0d/home/cx/ptft/tuep')
    parser.add_argument('--output_dir', type=str, default='experiments/tuep_full_ft/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load Data Lists
    print("Loading File Lists...")
    train_files = get_tuep_file_list(args.dataset_dir, mode='train', seed=args.seed)
    val_files = get_tuep_file_list(args.dataset_dir, mode='val', seed=args.seed)
    test_files = get_tuep_file_list(args.dataset_dir, mode='test', seed=args.seed)
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Initialize Datasets
    # Use a shared cache path to avoid re-indexing multiple times if possible, or distinct ones?
    # TUEPDataset takes cache_path. Let's use one.
    cache_path = os.path.join(args.output_dir, 'dataset_index.json')
    
    train_dataset = TUEPDataset(train_files, input_size=12000, cache_path=cache_path)
    val_dataset = TUEPDataset(val_files, input_size=12000, cache_path=cache_path)
    test_dataset = TUEPDataset(test_files, input_size=12000, cache_path=cache_path)
    
    # Load Config
    base_config = load_config(args.config)
    base_config['task_type'] = 'classification'
    base_config['model']['num_classes'] = 2
    base_config['model']['seq_len'] = 60
    
    # Prepare DataLoaders
    pf = 4 if args.num_workers > 0 else None
    persistent = (args.num_workers > 0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, persistent_workers=persistent, 
                              prefetch_factor=pf, pin_memory=True, collate_fn=TUEPDataset.collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, persistent_workers=persistent, 
                            prefetch_factor=pf, pin_memory=True, collate_fn=TUEPDataset.collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, persistent_workers=persistent, 
                             prefetch_factor=pf, pin_memory=True, collate_fn=TUEPDataset.collate)
    
    available_models = {
        'Baseline': args.baseline_path,
        'Flagship': args.flagship_path,
        'FeatOnly': args.featonly_path
    }
    
    results = []
    
    for m_name in args.run_models:
        if m_name not in available_models: continue
        weights_path = available_models[m_name]
        if weights_path is None: continue
        
        print(f"--- Training {m_name} ---")
        setup_logger(args.output_dir, f"training_{m_name.lower()}.log")
        
        # Configure Head Type
        current_config = base_config.copy()
        current_config['model'] = base_config['model'].copy()
        
        if m_name == 'FeatOnly':
            current_config['model']['head_type'] = 'feat_cross_attn'
        elif m_name == 'Flagship':
            current_config['model']['head_type'] = 'flagship_concat'
        else:
            current_config['model']['head_type'] = 'pooling' # Default Baseline
            
        model = load_pretrained_model(current_config, weights_path, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        
        scaler = torch.amp.GradScaler('cuda')
        
        best_val_bacc = 0.0
        epoch_results = []
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            
            val_stats = evaluate(model, val_loader, criterion, device)
            
            if val_stats['bacc'] > best_val_bacc:
                best_val_bacc = val_stats['bacc']
            
            test_stats = evaluate(model, test_loader, criterion, device)
            
            log_msg = (f"Epoch {epoch+1}/{args.epochs} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Val BAcc: {val_stats['bacc']:.4f} F1: {val_stats['f1']:.4f} AUROC: {val_stats['auroc']:.4f} | "
                       f"Test BAcc: {test_stats['bacc']:.4f} F1: {test_stats['f1']:.4f} AUROC: {test_stats['auroc']:.4f}")
            print(log_msg)
            logging.info(log_msg)
            
            epoch_results.append({
                "epoch": epoch+1,
                "train": {"loss": train_loss, "acc": train_acc},
                "val": val_stats,
                "test": test_stats
            })
            
        results.append({
            'model': m_name,
            'best_val_bacc': best_val_bacc,
            'epoch_history': epoch_results
        })
        
    # Report
    report_path = os.path.join(args.output_dir, 'results_final_ft.md')
    with open(report_path, 'w') as f:
        f.write("# TUEP Full Fine-tuning Results\n\n")
        f.write("| Model | Best Val BAcc (%) |\n")
        f.write("|-------|-------------------|\n")
        for r in results:
            f.write(f"| {r['model']} | {r['best_val_bacc']*100:.2f} |\n")
            
        f.write("\n\n# Detailed History\n")
        for r in results:
            f.write(f"\n## {r['model']}\n")
            f.write("| Epoch | Train Loss | Val BAcc | Val F1 | Val AUROC | Val AUC-PR | Test BAcc | Test F1 | Test AUROC | Test AUC-PR |\n")
            f.write("|-------|------------|----------|--------|-----------|------------|-----------|---------|------------|-------------|\n")
            for ep in r['epoch_history']:
                f.write(f"| {ep['epoch']} | {ep['train']['loss']:.4f} | "
                        f"{ep['val']['bacc']:.4f} | {ep['val']['f1']:.4f} | {ep['val']['auroc']:.4f} | {ep['val']['auc_pr']:.4f} | "
                        f"{ep['test']['bacc']:.4f} | {ep['test']['f1']:.4f} | {ep['test']['auroc']:.4f} | {ep['test']['auc_pr']:.4f} |\n")
                
    print(f"Results saved to {report_path}")

if __name__ == '__main__':
    main()
