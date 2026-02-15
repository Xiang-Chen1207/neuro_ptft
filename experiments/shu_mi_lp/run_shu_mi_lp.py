import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import glob
import json
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import OrderedDict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.wrapper import CBraModWrapper

# SHU_MI Path
SHU_MI_PATH = '/vePFS-0x0d/pretrain-clip/chr/test_datasets/datasets/SHU_MI'

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

def _index_worker_shumi(h5_path):
    samples = []
    errors = []
    try:
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                if not isinstance(trial_group, h5py.Group):
                    continue
                
                # Check for segment0 (or segments)
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                
                # If no explicit segment sub-groups, maybe data is directly under trial
                # But SHU_MI seems to have segment0
                if not segment_keys:
                    # Fallback or check direct
                    target_group = trial_group
                    seg_key = None
                else:
                    # Usually segment0
                    seg_key = segment_keys[0]
                    target_group = trial_group[seg_key]
                
                # Check for data
                if 'eeg' in target_group:
                    shape = target_group['eeg'].shape
                elif 'data' in target_group:
                    shape = target_group['data'].shape
                else:
                    continue # No data found
                
                # Try to read label from attributes
                label = -1
                if 'label' in target_group.attrs:
                    label = int(target_group.attrs['label'])
                elif 'label' in trial_group.attrs:
                    label = int(trial_group.attrs['label'])
                
                samples.append({
                    'file_path': h5_path,
                    'trial_key': trial_key,
                    'segment_key': seg_key,
                    'shape': shape,
                    'label': label
                })
                
    except Exception as e:
        errors.append({
            'file_path': h5_path,
            'error': str(e)
        })
    return {'samples': samples, 'errors': errors}

class SHUMIDataset(Dataset):
    def __init__(self, file_list, input_size=12000, transform=None, cache_path=None, **kwargs):
        super().__init__()
        self.input_size = input_size 
        self.transform = transform
        self.drop_bad_samples = kwargs.get('drop_bad_samples', True)
        
        # Cache handling
        if cache_path is None:
            import hashlib
            h = hashlib.md5(str(sorted(file_list)).encode()).hexdigest()
            cache_path = f"output/shumi_index_{h[:8]}.json"
            
        self.samples = self._load_or_generate_index(file_list, cache_path)
        
        # Filter out samples with invalid labels if necessary
        self.samples = [s for s in self.samples if s['label'] != -1]
        
        self.file_cache = OrderedDict()
        self.cache_size = 64

    def _load_or_generate_index(self, file_list, cache_path):
        full_index = []
        input_files_set = set(file_list)
        reindex = True
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    full_index = data.get('samples', [])
                    # Simple check if cache is relevant (not robust but okay for demo)
                    if len(full_index) > 0:
                        reindex = False
            except:
                pass
        
        if reindex:
            print(f"Indexing {len(file_list)} files for SHUMIDataset...")
            max_workers = min(16, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_index_worker_shumi, file_list), total=len(file_list)))
            
            full_index = [s for res in results for s in res.get('samples', [])]
            
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({'samples': full_index}, f)
                
        return [s for s in full_index if s['file_path'] in input_files_set]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path = info['file_path']
        trial_key = info['trial_key']
        seg_key = info['segment_key']
        label = info['label']
        
        try:
            if h5_path in self.file_cache:
                f = self.file_cache[h5_path]
                self.file_cache.move_to_end(h5_path)
            else:
                if len(self.file_cache) >= self.cache_size:
                    self.file_cache.popitem(last=False)[1].close()
                f = h5py.File(h5_path, 'r', rdcc_nbytes=1024*1024, libver='latest')
                self.file_cache[h5_path] = f
            
            group = f[trial_key][seg_key] if seg_key else f[trial_key]
            
            if 'eeg' in group:
                dset = group['eeg']
            elif 'data' in group:
                dset = group['data']
            else:
                raise KeyError("No data found")
                
            # Read data
            # Shape is usually (32, 800) -> (Channels, Time)
            raw = dset[:]
            
            # Normalize
            tensor = torch.from_numpy(raw).float()
            
            # Simple channel-wise normalization
            mean = tensor.mean(dim=1, keepdim=True)
            std = tensor.std(dim=1, keepdim=True)
            tensor = (tensor - mean) / (std + 1e-6)
            
            # Patchify logic (Same as generic_h5)
            # We need to reshape to (C, N, P) for the model wrapper
            # Assuming patch_size=200
            patch_size = 200
            C, T = tensor.shape
            
            # If T < patch_size, pad
            if T < patch_size:
                pad = patch_size - T
                tensor = torch.nn.functional.pad(tensor, (0, pad))
                T = patch_size
                
            remainder = T % patch_size
            if remainder != 0:
                pad = patch_size - remainder
                tensor = torch.nn.functional.pad(tensor, (0, pad))
            
            num_patches = tensor.shape[1] // patch_size
            tensor = tensor.view(C, num_patches, patch_size)
            
            if self.transform:
                tensor = self.transform(tensor)
                
            return tensor, label
            
        except Exception as e:
            if self.drop_bad_samples:
                return None
            raise e
    
    def collate(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
            
        # batch is list of (tensor, label)
        data = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        
        # Stack
        # data: list of (C, N, P). If N varies, we need padding.
        # For SHU_MI, T is fixed (800), so N is fixed (4).
        try:
            data = torch.stack(data)
            targets = torch.tensor(targets, dtype=torch.long)
        except:
            # Handle variable length if necessary (not implemented for simplicity)
            print("Warning: Variable length batch? Padding not implemented.")
            return None
            
        return data, targets

# --- Linear Probe Utils ---

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def run_logistic_regression(X_train, y_train, X_test, y_test, num_classes=None, device='cuda', seed=42):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    unique_train = np.unique(y_train)
    print(f"  Train Classes: {unique_train} (Count: {len(unique_train)})")
    
    input_dim = X_train.shape[1]
    if num_classes is None:
        num_classes = int(max(y_train.max(), y_test.max()) + 1)
    
    print(f"  Training LR: Input Dim={input_dim}, Classes={num_classes}, Train Samples={len(y_train)}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
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
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs.data, 1)
        
    correct = (predicted == y_test_t).sum().item()
    acc = correct / len(y_test_t)
    bacc = balanced_accuracy_score(y_test, predicted.cpu().numpy())
    
    return acc, bacc

def extract_features(model, dataset, device, batch_size=64):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, collate_fn=dataset.collate)
    
    features_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting", leave=False):
            if batch is None: continue
            data, target = batch
            data = data.to(device).float()
            
            # Forward Backbone
            # (B, C, N, D)
            feats = model.backbone(data)
            
            # GAP
            if feats.ndim == 4:
                pooled = feats.mean(dim=[1, 2])
            elif feats.ndim == 3:
                pooled = feats.mean(dim=1)
            else:
                pooled = feats
                
            features_list.append(pooled.cpu().numpy())
            labels_list.append(target.cpu().numpy())
            
    if not features_list:
        raise ValueError("No samples extracted")

    return np.concatenate(features_list), np.concatenate(labels_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='experiments/shu_mi_lp/results_shumi.csv')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint.get('config', {})
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        if 'model' not in config: config['model'] = {}
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
    
    # 2. Prepare Data
    print("Preparing SHU_MI Dataset...")
    all_files = glob.glob(os.path.join(SHU_MI_PATH, "*.h5"))
    all_files.sort()
    
    if not all_files:
        print("No SHU_MI files found!")
        sys.exit(1)
        
    # Split: Last 5 subjects as test (approx 20%)
    # Assuming file names are sub_1.h5 ... sub_25.h5
    # Sorting should put sub_1, sub_10, ...
    # Let's sort numerically to be safe
    try:
        all_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    except:
        pass # Fallback to string sort
        
    test_count = 5
    train_files = all_files[:-test_count]
    test_files = all_files[-test_count:]
    
    print(f"Train Files: {len(train_files)}")
    print(f"Test Files: {len(test_files)}")
    
    train_ds = SHUMIDataset(train_files, cache_path="experiments/shu_mi_lp/cache/train_index.json")
    test_ds = SHUMIDataset(test_files, cache_path="experiments/shu_mi_lp/cache/test_index.json")
    
    # 3. Run Pipeline
    print("Extracting Train Features...")
    X_train, y_train = extract_features(model, train_ds, device, args.batch_size)
    
    print("Extracting Test Features...")
    X_test, y_test = extract_features(model, test_ds, device, args.batch_size)
    
    print("Running Linear Probe...")
    acc, bacc = run_logistic_regression(X_train, y_train, X_test, y_test, device=device, seed=args.seed)
    
    print(f"SHU_MI Result: Acc={acc:.4f}, BAcc={bacc:.4f}")
    
    # Save
    df = pd.DataFrame([{
        'Dataset': 'SHU_MI',
        'Accuracy': acc,
        'Balanced_Accuracy': bacc,
        'Train_Size': len(y_train),
        'Test_Size': len(y_test)
    }])
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved to {args.output_csv}")

if __name__ == '__main__':
    main()
