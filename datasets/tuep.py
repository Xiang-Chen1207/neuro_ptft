import os
import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import OrderedDict

# Define paths for labels
NO_EP_CSV = '/vePFS-0x0d/home/cx/ptft/tuep/no_ep.csv'
WITH_EP_CSV = '/vePFS-0x0d/home/cx/ptft/tuep/with_ep.csv'

def get_subjects_from_csv(csv_path):
    subjects = []
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if parts:
                    subjects.append(parts[0])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return subjects

def get_tuep_file_list(dataset_dir, mode, seed=42):
    # Load subjects
    no_ep_subs = get_subjects_from_csv(NO_EP_CSV)
    with_ep_subs = get_subjects_from_csv(WITH_EP_CSV)
    
    # Exclude problematic subjects
    excluded_subs = {'aaaaajqo'}
    
    # Create file list with labels
    # Label 0: No Epilepsy
    # Label 1: With Epilepsy
    
    data_infos = []
    
    # Debug: Check if dataset_dir exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist!")
        return []
    
    # Try finding files recursively first as fallback
    print(f"Searching for files in {dataset_dir}...")
    all_files = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.h5') and 'sub_' in f:
                # Store full path mapped by filename or subject id
                # Key: sub_aaaaaaar.h5 -> aaaaaaar
                try:
                    sid = f.split('sub_')[1].split('.')[0]
                    all_files[sid] = os.path.join(root, f)
                except:
                    pass

    print(f"Found {len(all_files)} total H5 files in directory.")

    for sub in no_ep_subs:
        if sub in excluded_subs: continue
        
        # Check direct path
        fname = f"sub_{sub}.h5"
        fpath = os.path.join(dataset_dir, fname)
        
        if os.path.exists(fpath):
            data_infos.append({'path': fpath, 'label': 0, 'subject_id': sub})
        elif sub in all_files:
            data_infos.append({'path': all_files[sub], 'label': 0, 'subject_id': sub})
        else:
            # print(f"Warning: Subject {sub} file not found.")
            pass
            
    for sub in with_ep_subs:
        if sub in excluded_subs: continue
        
        fname = f"sub_{sub}.h5"
        fpath = os.path.join(dataset_dir, fname)
        
        if os.path.exists(fpath):
            data_infos.append({'path': fpath, 'label': 1, 'subject_id': sub})
        elif sub in all_files:
            data_infos.append({'path': all_files[sub], 'label': 1, 'subject_id': sub})
        else:
            # print(f"Warning: Subject {sub} file not found.")
            pass
            
    # Split by subject (stratified)
    class_0 = [x for x in data_infos if x['label'] == 0]
    class_1 = [x for x in data_infos if x['label'] == 1]
    
    rng = np.random.RandomState(seed)
    rng.shuffle(class_0)
    rng.shuffle(class_1)
    
    def split_list(lst, ratios=(0.8, 0.1)):
        n = len(lst)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # The rest is test
        
        return {
            'train': lst[:n_train],
            'val': lst[n_train:n_train+n_val],
            'test': lst[n_train+n_val:]
        }
        
    splits_0 = split_list(class_0)
    splits_1 = split_list(class_1)
    
    # Map 'eval' to 'val' if needed
    if mode == 'eval':
        mode = 'val'
    
    if mode not in splits_0:
        # Fallback or error
        return []

    final_split = splits_0[mode] + splits_1[mode]
    
    # Shuffle the combined list
    rng.shuffle(final_split)
    
    print(f"[{mode}] Found {len(final_split)} subjects for TUEP (Class 0: {len(splits_0[mode])}, Class 1: {len(splits_1[mode])})")
    
    return final_split

def _index_worker(info):
    # Info is a dict: {'path': ..., 'label': ...}
    h5_path = info['path']
    label = info['label']
    samples = []
    errors = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
             for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                # Check for segments
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                if segment_keys:
                    for seg_key in segment_keys:
                        samples.append({
                            'file_path': h5_path,
                            'trial_key': trial_key,
                            'segment_key': seg_key,
                            'label': label
                        })
                else:
                    samples.append({
                        'file_path': h5_path,
                        'trial_key': trial_key,
                        'segment_key': None,
                        'label': label
                    })
    except Exception as e:
        errors.append({
            'file_path': h5_path,
            'trial_key': None,
            'segment_key': None,
            'stage': 'index',
            'error': f"{type(e).__name__}: {e}",
        })
    return {'samples': samples, 'errors': errors}

def _collate_drop_none(batch):
    kept = [b for b in batch if b is not None]
    dropped = len(batch) - len(kept)
    if len(kept) == 0:
        return None
    xs, ys = zip(*kept)
    x = torch.stack(xs, 0)
    y0 = ys[0]
    if isinstance(y0, torch.Tensor):
        y = torch.stack(ys, 0)
    else:
        y = torch.as_tensor(ys)
    return {'x': x, 'y': y, 'mask': None, 'dropped': dropped}

class TUEPDataset(Dataset):
    def __init__(self, file_list, input_size=12000, transform=None, cache_path='dataset_index_tuep.json', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.transform = transform
        self.drop_bad_samples = bool(kwargs.get('drop_bad_samples', True))
        
        # Standard 19 channels
        self.target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        self.source_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2']
        
        self.channel_indices = []
        for target in self.target_channels:
            try:
                idx = self.source_channels.index(target)
                self.channel_indices.append(idx)
            except ValueError:
                print(f"Warning: Channel {target} not found in source map.")
        
        self.samples = self._load_or_generate_index(file_list, cache_path)
        
        # File handle cache
        self.file_cache = OrderedDict()
        self.cache_size = 64

    @staticmethod
    def collate(batch):
        return _collate_drop_none(batch)

    def _load_or_generate_index(self, file_list, cache_path):
        full_index = []
        error_index = []
        # file_list contains dicts, so we use path as key
        input_files_set = set(item['path'] for item in file_list)
        reindex = True
        
        # 1. Try Loading Cache
        if os.path.exists(cache_path):
            print(f"Loading index from {cache_path}...")
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        full_index = data
                    else:
                        full_index = data.get('samples', [])
                        error_index = data.get('errors', [])
                
                # Check coverage
                cached_files = set(s['file_path'] for s in full_index)
                if input_files_set.issubset(cached_files):
                     reindex = False
                else:
                     print("Cache incomplete. Re-indexing...")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # 2. Re-index if needed
        if reindex:
            print(f"Indexing {len(file_list)} files...")
            max_workers = min(32, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_index_worker, file_list), total=len(file_list)))
            new_samples = [s for res in results for s in res.get('samples', [])]
            new_errors = [e for res in results for e in res.get('errors', [])]
            
            existing_samples_map = { (s['file_path'], s['trial_key'], s['segment_key']): s for s in full_index }
            for s in new_samples:
                key = (s['file_path'], s['trial_key'], s['segment_key'])
                existing_samples_map[key] = s
            
            full_index = list(existing_samples_map.values())
            error_index.extend(new_errors)
            
            try:
                with open(cache_path, 'w') as f:
                    bad_files = sorted(list({e.get('file_path') for e in error_index if e.get('file_path')}))
                    payload = {
                        'samples': full_index,
                        'errors': error_index,
                        'stats': {
                            'n_files': len(file_list),
                            'n_samples': len(full_index),
                            'n_errors': len(error_index),
                            'n_bad_files': len(bad_files),
                        }
                    }
                    json.dump(payload, f)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
            
            if len(error_index) > 0:
                bad_files = {e.get('file_path') for e in error_index if e.get('file_path')}
                print(f"Index summary: samples={len(full_index)} errors={len(error_index)} bad_files={len(bad_files)}")
                
        # 3. Filter for current split
        filtered_samples = [s for s in full_index if s['file_path'] in input_files_set]
        print(f"Dataset initialized: {len(filtered_samples)} samples (from {len(file_list)} files).")
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path, trial_key, seg_key, label = info['file_path'], info['trial_key'], info['segment_key'], info['label']
        
        data = np.zeros((len(self.channel_indices), self.input_size), dtype=np.float32)
        try:
            # Manage File Cache
            if h5_path in self.file_cache:
                f = self.file_cache[h5_path]
                self.file_cache.move_to_end(h5_path)
            else:
                if len(self.file_cache) >= self.cache_size:
                    old_path, old_f = self.file_cache.popitem(last=False)
                    try: old_f.close()
                    except: pass
                
                f = h5py.File(h5_path, 'r', rdcc_nbytes=4*1024*1024, libver='latest')
                self.file_cache[h5_path] = f

            # Read Data
            group = f[trial_key][seg_key] if seg_key else f[trial_key]
            if 'eeg' in group:
                raw = group['eeg'][:]
            elif 'data' in group:
                raw = group['data'][:]
            else:
                raise KeyError(f"No data found in {h5_path} key {trial_key}/{seg_key}")

            # Handle Transpose if needed (Standard is [Channels, Time])
            # Some files might be [Time, Channels]
            # Usually we expect 19-21 channels.
            if raw.shape[0] > raw.shape[1] and raw.shape[1] <= 30: # Likely (Time, Channel)
                raw = raw.T
            
            # Select channels
            # Note: channel_indices assumes raw is [Channels, Time] and channel order matches source_channels
            # If raw has more channels than source_channels, we assume standard order.
            # But if raw has FEWER channels or DIFFERENT order, we might crash.
            # Let's add a safety check for bounds
            
            try:
                raw = raw[self.channel_indices, :]
            except IndexError:
                # Fallback: If indices are out of bounds, maybe the file has fewer channels?
                # Or maybe the channel mapping is wrong for this specific file.
                # For now, let's pad with zeros if channel is missing? Or skip?
                # Better: Create a zero buffer and fill available channels.
                
                # Assume raw is [Available_Channels, Time]
                # We need to map available channels to target channels.
                # But we don't have channel names here easily without reading attributes.
                # Given the error "index 17 is out of bounds for axis 0 with size 17",
                # it means raw has 17 channels, but we asked for index 17 (which is the 18th channel).
                # The file likely has fewer channels than the standard 21.
                raise ValueError(
                    f"Channel mismatch in {h5_path}. Expected max index {max(self.channel_indices)}, got {raw.shape[0]} channels."
                )

            if raw.shape[1] < self.input_size:
                pad = self.input_size - raw.shape[1]
                data = np.pad(raw, ((0,0), (0, pad)), 'constant')
            else:
                data = raw[:, :self.input_size]
        except Exception as e:
            if h5_path in self.file_cache:
                try: self.file_cache.pop(h5_path).close()
                except: pass
            if self.drop_bad_samples:
                return None
            raise RuntimeError(f"Bad sample: {h5_path} {trial_key}/{seg_key}: {type(e).__name__}: {e}") from e

        tensor = torch.from_numpy(data).float()
        # Normalize
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)
        
        # Patchify
        patch_size = 200
        if self.input_size % patch_size == 0:
            num_patches = self.input_size // patch_size
            tensor = tensor.view(tensor.shape[0], num_patches, patch_size)

        if self.transform is not None:
            tensor = self.transform(tensor)
            
        return tensor, label
