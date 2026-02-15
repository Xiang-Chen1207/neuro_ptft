import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
import logging
import sys

# Add project root to path
sys.path.append(os.getcwd())

from experiments.tusz_full_ft.model_dynamic import CBraModWrapperDynamic
from datasets.tusz import TUSZDataset, get_tusz_file_list

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
    model = CBraModWrapperDynamic(config)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pretrained weights from {weights_path}...")
        try:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'): k = k[7:]
                new_state_dict[k] = v
                
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("No pretrained weights found or path not provided. initializing random weights.")
        
    model.to(device)
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
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
        return x, y, mask
    if len(batch) == 3:
        data, target, mask = batch
        return data.to(device).float(), target.to(device).long(), mask.to(device)
    data, target = batch
    return data.to(device).float(), target.to(device).long(), None

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    skipped_batches = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        unpacked = unpack_batch(batch, device)
        if unpacked is None:
            skipped_batches += 1
            continue
        data, target, mask = unpacked
        
        optimizer.zero_grad()
        
        with autocast():
            # Pass mask to model
            output = model(data, mask=mask)
            
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
        
    denom = max(1, (len(dataloader) - skipped_batches))
    return total_loss / denom, correct / max(1, total)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        skipped_batches = 0
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            unpacked = unpack_batch(batch, device)
            if unpacked is None:
                skipped_batches += 1
                continue
            data, target, mask = unpacked
            
            output = model(data, mask=mask)
            if isinstance(output, dict):
                logits = output.get('logits', output.get('cls_pred', None))
            else:
                logits = output
                
            loss = criterion(logits, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    bacc = balanced_accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    denom = max(1, (len(dataloader) - skipped_batches))
    return total_loss / denom, bacc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/finetune_tusz.yaml')
    parser.add_argument('--weights_path', type=str, default=None)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments/tusz_full_ft/results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    logging.info("Loading TUSZ Dataset...")
    train_files = get_tusz_file_list(args.dataset_dir, 'train', seed=args.seed)
    val_files = get_tusz_file_list(args.dataset_dir, 'val', seed=args.seed)
    test_files = get_tusz_file_list(args.dataset_dir, 'test', seed=args.seed)
    
    # Use dynamic_length=True
    train_dataset = TUSZDataset(train_files, mode='train', dynamic_length=True)
    val_dataset = TUSZDataset(val_files, mode='val', dynamic_length=True)
    test_dataset = TUSZDataset(test_files, mode='test', dynamic_length=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=TUSZDataset.collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=TUSZDataset.collate, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, collate_fn=TUSZDataset.collate, pin_memory=True)
    
    # Config
    base_config = load_config(args.config)
    base_config['task_type'] = 'classification'
    
    logging.info(f"Initializing model with weights: {args.weights_path}")
    model = load_pretrained_model(base_config, args.weights_path, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    best_val_bacc = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_bacc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val BAcc: {val_bacc:.4f} F1: {val_f1:.4f}")
        
        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            torch.save(model.state_dict(), best_model_path)
            logging.info("New best model saved.")
            
    # Test
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_bacc, test_f1 = evaluate(model, test_loader, criterion, device)
        logging.info(f"TEST RESULTS | BAcc: {test_bacc:.4f} F1: {test_f1:.4f}")
    
if __name__ == '__main__':
    main()
