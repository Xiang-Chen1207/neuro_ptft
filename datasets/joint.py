
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from .generic_h5 import GenericH5Dataset

def get_joint_file_list(data_csv_path, mode='train', seed=42, val_split=0.1):
    """
    Reads dataset paths from csv, finds files, and splits (1-val_split)/val_split per dataset.
    """
    if not os.path.exists(data_csv_path):
        # Try relative to project root
        if os.path.exists(os.path.join('datasets', data_csv_path)):
             data_csv_path = os.path.join('datasets', data_csv_path)
        else:
             print(f"Warning: data.csv not found at {data_csv_path}")
             return []

    with open(data_csv_path, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
        
    final_file_list = []
    
    for dpath in paths:
        # Handle relative paths
        if not os.path.isabs(dpath):
            # Assume relative to project root or some base
            # User example: pretrain-clip/...
            # Try prepending /vePFS-0x0d/ or check if exists
            if not os.path.exists(dpath):
                alt_path = os.path.join('/vePFS-0x0d', dpath)
                if os.path.exists(alt_path):
                    dpath = alt_path
                elif os.path.exists(os.path.join('/vePFS-0x0d/home/cx/ptft', dpath)):
                    dpath = os.path.join('/vePFS-0x0d/home/cx/ptft', dpath)
        
        if not os.path.exists(dpath):
            print(f"Warning: Dataset path {dpath} does not exist. Skipping.")
            continue
            
        # Glob files
        # Check standard patterns
        files = sorted(glob.glob(os.path.join(dpath, '**', '*.h5'), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(dpath, '*.h5')))
            
        # Filter for sub_ if likely subject based (heuristic from user request)
        # But generic should be generic.
        
        if not files:
            print(f"Warning: No .h5 files found in {dpath}")
            continue
            
        print(f"Dataset {os.path.basename(dpath)}: Found {len(files)} files.")
        
        # Split (1-val_split)/val_split
        rng = np.random.RandomState(seed)
        rng.shuffle(files)
        
        n_val = max(1, int(len(files) * val_split))
        n_train = len(files) - n_val
        
        if mode == 'train':
            dataset_files = files[:n_train]
        else:
            dataset_files = files[n_train:]
            
        final_file_list.extend(dataset_files)
        
    print(f"[{mode}] Total files for Joint Dataset: {len(final_file_list)}")
    return final_file_list

class JointDataset(Dataset):
    def __init__(self, file_list, **kwargs):
        # We delegate to GenericH5Dataset
        # We can pass the full file list to one GenericH5Dataset instance
        # since it handles list of files.
        self.dataset = GenericH5Dataset(file_list, **kwargs)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def collate(batch):
        return collate_joint(batch)

def collate_joint(batch):
    """
    Collate function for variable length channels and time.
    batch: list of (tensor, label) where tensor is (C, N, P)
    """
    kept = [b for b in batch if b is not None]
    dropped = len(batch) - len(kept)
    if len(kept) == 0:
        return None
        
    tensors, labels = zip(*kept)
    
    # tensor shape: (C, N, P)
    # C varies, N varies. P is fixed (200).
    
    max_c = max(t.shape[0] for t in tensors)
    max_n = max(t.shape[1] for t in tensors)
    patch_size = tensors[0].shape[2]
    
    batch_size = len(tensors)
    
    # Output buffer: (B, MaxC, MaxN, P)
    out_x = torch.zeros(batch_size, max_c, max_n, patch_size, dtype=torch.float32)
    
    # PyTorch MultiheadAttention key_padding_mask: True for values to be IGNORED (padding).
    channel_mask = torch.ones(batch_size, max_c, dtype=torch.bool) # Init to True (all padded)
    time_mask = torch.ones(batch_size, max_n, dtype=torch.bool)    # Init to True (all padded)
    
    for i, t in enumerate(tensors):
        c, n, p = t.shape
        out_x[i, :c, :n, :] = t
        
        # Unmask valid regions (set to False)
        channel_mask[i, :c] = False
        time_mask[i, :n] = False
        
    # Stack labels
    if isinstance(labels[0], torch.Tensor):
        y = torch.stack(labels, 0)
    else:
        y = torch.as_tensor(labels)
        
    return {
        'x': out_x,
        'y': y,
        'channel_mask': channel_mask, # (B, C)
        'time_mask': time_mask,       # (B, N)
        'dropped': dropped
    }
