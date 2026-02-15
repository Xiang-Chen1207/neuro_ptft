import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.wrapper import CBraModWrapper
from datasets.builder import build_dataset

# Dataset Paths (Hardcoded based on exploration)
DATASET_PATHS = {
    'TUAB': '/vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5/downstream/TUAB',
    'TUEP': '/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset',
    'TUEV': '/vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Events',
    'TUSZ': '/vePFS-0x0d/pretrain-clip/benchmark_dataloader/hdf5_output/TUH_Seizure',
    'SEED': '/vePFS-0x0d/home/downstream_data/SEED',
    'BCIC2A': '/vePFS-0x0d/home/downstream_data/BCIC2A',
    'SEEDIV': '/vePFS-0x0d/eeg-data/SEEDIV'
}

# Known Class Counts for Robustness
NUM_CLASSES = {
    'TUAB': 2,
    'TUEP': 2,
    'TUEV': 6,
    'TUSZ': 13, # 0-12 usually, but might be less in subset. Standard is 13? Or is it 3? Check TUSZ logic. TUSZ has 12 classes? Let's check config.
    # finetune_tusz.yaml says 13.
    'SEED': 3,
    'BCIC2A': 4,
    'SEEDIV': 4
}

# Linear Probe Model (GPU accelerated)
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def run_logistic_regression(X_train, y_train, X_test, y_test, num_classes=None, device='cuda', seed=42):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Check classes
    unique_train = np.unique(y_train)
    unique_test = np.unique(y_test)
    print(f"  Train Classes: {unique_train} (Count: {len(unique_train)})")
    print(f"  Test Classes: {unique_test} (Count: {len(unique_test)})")
    
    input_dim = X_train.shape[1]
    
    if num_classes is None:
        # Robust class count inference
        max_class = max(y_train.max(), y_test.max())
        num_classes = int(max_class + 1)
        # Ensure at least 2 classes
        num_classes = max(num_classes, 2)
    
    print(f"  Training LR: Input Dim={input_dim}, Classes={num_classes}, Train Samples={len(y_train)}")
    
    # Normalize Features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Prepare Data
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    model = LogisticRegressionTorch(input_dim, num_classes).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    def closure():
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
    # Eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs.data, 1)
        
    correct = (predicted == y_test_t).sum().item()
    acc = correct / len(y_test_t)
    
    bacc = balanced_accuracy_score(y_test, predicted.cpu().numpy())
    
    return acc, bacc

def extract_features(model, dataset, device, batch_size=64, num_workers=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, collate_fn=getattr(dataset, 'collate', None))
    
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting", leave=False):
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    data, target = batch[0], batch[1]
                else:
                    continue # Skip weird batches
            elif isinstance(batch, dict):
                data = batch.get('x')
                if data is None:
                    data = batch.get('eeg')
                if data is None:
                    data = batch.get('input')
                    
                target = batch.get('y')
                if target is None:
                    target = batch.get('label')
                if target is None:
                    target = batch.get('target')
                
            if data is None: continue
            
            data = data.to(device).float()
            
            # Forward Backbone
            # Output: (B, C, N, D)
            feats = model.backbone(data)
            
            # GAP: Mean over Channel (1) and Patch (2) dimensions
            # Verify shape first
            if feats.ndim == 4:
                # (B, C, N, D) -> (B, D)
                pooled = feats.mean(dim=[1, 2])
            elif feats.ndim == 3:
                # (B, N, D) -> (B, D)
                pooled = feats.mean(dim=1)
            else:
                pooled = feats
                
            features_list.append(pooled.cpu().numpy())
            labels_list.append(target.cpu().numpy())
            
    if not features_list:
        raise ValueError("No samples extracted from dataset")

    return np.concatenate(features_list), np.concatenate(labels_list)

def warmup_mamba(model, device, config):
    model_cfg = config.get('model', {})
    if model_cfg.get('name') != 'eegmamba':
        return

    print(f"DEBUG: Warming up eegmamba kernels on {device}...")
    
    # Infer shapes
    seq_len = int(model_cfg.get('seq_len', 60))
    patch_size = int(model_cfg.get('in_dim', 200))
    batch_size = 2
    ch_num = 21 # Dummy channel count
    
    x = torch.zeros(batch_size, ch_num, seq_len, patch_size, device=device, dtype=torch.float32)
    mask = torch.zeros(batch_size, ch_num, seq_len, device=device, dtype=torch.float32)
    
    try:
        model_to_call = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
             # We only need backbone forward really, but wrapper is safer
             # Wrapper forward: (x, mask=None, ...)
             _ = model_to_call(x, mask=mask)
             
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("DEBUG: Warmup successful.")
    except Exception as e:
        print(f"WARNING: Warmup failed with error: {e}. Proceeding anyway...")

def main():
    parser = argparse.ArgumentParser(description="Run Linear Probing Benchmark on All Datasets")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='experiments/all_lp_benchmark/results.csv')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tiny', action='store_true', help="Use tiny subset for debugging")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        # Clean state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        # Ensure config has minimal required fields for wrapper
        if 'model' not in config: config['model'] = {}
        # Force task type to avoid errors in wrapper initialization
        config['task_type'] = 'pretraining' 
        
        model = CBraModWrapper(config)
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        
        if torch.cuda.device_count() > 1 and device.type == 'cuda':
            model.backbone = nn.DataParallel(model.backbone)
            
        # Warmup for Mamba if needed
        warmup_mamba(model, device, config)
            
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    results = []
    
    # 2. Iterate Datasets
    # datasets_to_run = ['TUAB', 'TUEP', 'TUEV', 'TUSZ', 'SEED', 'BCIC2A', 'SEEDIV']
    # Re-ordering for logical flow
    datasets_to_run = ['TUAB', 'BCIC2A', 'SEEDIV', 'TUEP', 'TUEV', 'TUSZ']
    
    for ds_name in datasets_to_run:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*40}")
        
        ds_path = DATASET_PATHS.get(ds_name)
        if not ds_path:
            print(f"Path not found for {ds_name}, skipping.")
            continue
            
        ds_config = {
            'dataset_dir': ds_path,
            'seed': 42,
            'tiny': args.tiny,
            # Some datasets might need specific params
            'batch_size': args.batch_size
        }
        
        # TUSZ needs 'input_size' sometimes, but let's hope defaults work or are inferred
        if ds_name == 'TUSZ':
             ds_config['input_size'] = 2000 # 10s * 200Hz
        
        # TUEP defaults to 60s, but we likely want 10s for consistency with pretraining
        if ds_name == 'TUEP':
             ds_config['input_size'] = 2000 # 10s * 200Hz
        
        try:
            # Build Train
            print("Building Train Dataset...")
            train_ds = build_dataset(ds_name, ds_config, mode='train')
            
            if len(train_ds) == 0:
                 print("Train dataset is empty, skipping.")
                 results.append({'Dataset': ds_name, 'Error': 'Empty Train Set'})
                 continue

            # Build Test
            print("Building Test Dataset...")
            test_ds = build_dataset(ds_name, ds_config, mode='test')
            
            if len(test_ds) == 0:
                 print("Test dataset is empty, skipping.")
                 results.append({'Dataset': ds_name, 'Error': 'Empty Test Set'})
                 continue
            
            # Extract Features
            print("Extracting Train Features...")
            X_train, y_train = extract_features(model, train_ds, device, args.batch_size)
            
            print("Extracting Test Features...")
            X_test, y_test = extract_features(model, test_ds, device, args.batch_size)
            
            # Run Linear Probe
            print("Running Linear Probe...")
            # Use known num_classes if available, else robust infer
            known_classes = NUM_CLASSES.get(ds_name)
            acc, bacc = run_logistic_regression(X_train, y_train, X_test, y_test, num_classes=known_classes, device=device)
            
            print(f"RESULT [{ds_name}]: Acc={acc:.4f}, BAcc={bacc:.4f}")
            
            results.append({
                'Dataset': ds_name,
                'Accuracy': acc,
                'Balanced_Accuracy': bacc,
                'Train_Size': len(y_train),
                'Test_Size': len(y_test)
            })
            
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'Dataset': ds_name,
                'Accuracy': -1,
                'Balanced_Accuracy': -1,
                'Error': str(e)
            })
            
    # 3. Save Results
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print(df)

if __name__ == '__main__':
    main()
