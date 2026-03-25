
import os
import glob
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from .generic_h5 import GenericH5Dataset
from tqdm import tqdm

DATASET_FEATURE_MAP = {
    'merged_final_dataset': ('tueg', 'features_final_zscore.csv'),
    'SEED': ('seed', 'features_zscore.csv'),
    'Workload_MATB': ('workload', 'features_zscore.csv'),
    'SleepEDF': ('sleepedf', 'features_zscore.csv'),
    'SHU_MI': ('shumi', 'features_zscore.csv')
}

def _load_joint_features(file_list, feature_root_dir):
    """
    Loads features for multiple datasets and maps them to H5 file paths.
    Using a union of all feature columns (padding with zeros + mask).
    """
    print(f"Loading features from {feature_root_dir}...")
    
    # 1. Index file paths by basename
    basename_to_paths = {}
    for f in file_list:
        bn = os.path.basename(f)
        if bn not in basename_to_paths:
            basename_to_paths[bn] = []
        basename_to_paths[bn].append(f)
    
    meta_cols = ['aug_type', 'segment_id', 'window_idx', 'start_time', 'subject_id', 'source_file', 'trial_id', 'session_id', 'label', 'end_time', 'Unnamed: 0']
    
    # 2. Collect all unique feature columns
    all_feature_cols = set()
    valid_csvs = {} # key -> path
    
    for key, (subdir, csv_name) in DATASET_FEATURE_MAP.items():
        csv_path = os.path.join(feature_root_dir, subdir, csv_name)
        if not os.path.exists(csv_path):
             alt_path = os.path.join(feature_root_dir, csv_name)
             if os.path.exists(alt_path):
                 csv_path = alt_path
             else:
                 continue
        valid_csvs[key] = csv_path
        
        try:
            # Read header only
            df_head = pd.read_csv(csv_path, nrows=0) 
            cols = [c for c in df_head.columns if c not in meta_cols]
            all_feature_cols.update(cols)
        except Exception as e:
            print(f"Warning: Failed to read header for {key}: {e}")
            
    master_columns = sorted(list(all_feature_cols))
    feature_dim = len(master_columns)
    col_to_idx = {c: i for i, c in enumerate(master_columns)}
    
    print(f"Total unique features across datasets: {feature_dim}")
    
    feature_map = {}
    
    for key, csv_path in valid_csvs.items():
        print(f"Processing features for {key} from {csv_path}...")
        try:
            # Optimize: Precompute basename -> full_path for this dataset key
            # Use more precise matching to avoid false positives (e.g., SEED matching SEEDIV)
            key_basename_map = {}
            
            # Define dataset-specific path patterns for more accurate matching
            key_patterns = {
                'merged_final_dataset': ['merged_final_dataset', 'tuh'],
                'SEED': ['SEED'],  # Avoid matching SEEDIV
                'SEEDIV': ['SEEDIV'],
                'Workload_MATB': ['Workload_MATB', 'workload', 'MATB'],
                'SleepEDF': ['SleepEDF', 'sleepedf', 'sleep'],
                'SHU_MI': ['SHU_MI', 'shumi']
            }
            
            # Get patterns for this key, fallback to key itself if not in patterns
            patterns = key_patterns.get(key, [key])
            
            for bn, paths in basename_to_paths.items():
                matched = False
                for p in paths:
                    # Check if any pattern matches in the path
                    for pattern in patterns:
                        # Use word boundary matching: pattern should appear as a directory name or filename component
                        # Check if pattern appears as a separate component in the path
                        path_parts = p.replace('/', ' ').replace('\\', ' ').split()
                        if any(pattern.lower() == part.lower() or pattern.lower() in part.lower() for part in path_parts):
                            # Additional check: for SEED, ensure it's not SEEDIV
                            if key == 'SEED' and 'seediv' in p.lower():
                                continue
                            key_basename_map[bn] = p
                            matched = True
                            break
                    if matched:
                        break
            
            if not key_basename_map:
                print(f"Warning: No matching files found for dataset key {key} in file list.")
                print(f"  Searched for patterns: {patterns}")
                print(f"  Sample paths: {list(basename_to_paths.values())[:3] if basename_to_paths else 'None'}")
                continue

            df = pd.read_csv(csv_path)
            # Filter window_idx == 0
            if 'window_idx' in df.columns:
                df = df[df['window_idx'] == 0]
            
            # Identify columns present in this CSV
            curr_cols = [c for c in df.columns if c not in meta_cols]
            
            # Map current columns to master indices
            # Using numpy for indexing later
            curr_indices = np.array([col_to_idx[c] for c in curr_cols], dtype=np.int64)
            
            # Convert to numpy
            vals = df[curr_cols].values.astype(np.float32)
            source_files = df['source_file'].values
            seg_ids = df['segment_id'].values.astype(int)
            
            count = 0
            for i in tqdm(range(len(vals)), desc=f"Mapping {key}"):
                fname = os.path.basename(str(source_files[i]))
                sid = seg_ids[i]
                
                matched_path = key_basename_map.get(fname)
                if matched_path:
                    # Store tuple (raw_values, indices) to save memory
                    # We will expand it to full vector in Dataset.__getitem__
                    feature_map[(matched_path, sid)] = (vals[i], curr_indices)
                    count += 1
            
            print(f"Mapped {count} feature entries for {key}.")
            
        except Exception as e:
            print(f"Error loading features for {key}: {e}")
            
    print(f"Total feature entries mapped: {len(feature_map)}")
    return feature_map, feature_dim

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
    def __init__(self, file_list, feature_path=None, **kwargs):
        # We delegate to GenericH5Dataset
        # We can pass the full file list to one GenericH5Dataset instance
        # since it handles list of files.
        
        feature_map = None
        feature_dim = 0
        if feature_path:
             feature_map, feature_dim = _load_joint_features(file_list, feature_path)
             kwargs['feature_map'] = feature_map
             kwargs['feature_dim'] = feature_dim
             
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
    batch: list of (tensor, features, feature_mask) OR (tensor, label)
    """
    kept = [b for b in batch if b is not None]
    dropped = len(batch) - len(kept)
    if len(kept) == 0:
        return None
        
    # Check if we have features (3 elements) or just label (2 elements)
    elem = kept[0]
    has_features = (len(elem) == 3)
    
    if has_features:
        tensors, features, masks = zip(*kept)
        # Stack features and masks
        y = torch.stack(features, 0)
        feature_mask = torch.stack(masks, 0)
    else:
        # Check if we have 3 elements (tensor, features, mask) but maybe first elem was None and skipped?
        # If elem has 3 items, it is (tensor, feature, mask)
        if len(elem) == 3:
             tensors, features, masks = zip(*kept)
             y = torch.stack(features, 0)
             feature_mask = torch.stack(masks, 0)
        else:
             tensors, labels = zip(*kept)
             if isinstance(labels[0], torch.Tensor):
                 y = torch.stack(labels, 0)
             else:
                 y = torch.as_tensor(labels)
             feature_mask = None
    
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
        
    return {
        'x': out_x,
        'y': y,
        'channel_mask': channel_mask, # (B, C)
        'time_mask': time_mask,       # (B, N)
        'feature_mask': feature_mask,
        'dropped': dropped
    }
