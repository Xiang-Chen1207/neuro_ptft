import os
import glob
import h5py
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import OrderedDict

# --- Splitting Logic from cbramod_tuab/run_finetuning.py ---

def _group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        basename = os.path.basename(f)
        # sub_00000001.h5 -> 00000001
        try:
            subject_id = basename.split('_')[1].split('.')[0]
        except IndexError:
            # Fallback if filename format differs
            subject_id = basename.split('.')[0]
            
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)
    return subject_files

def _split_subjects(subject_files, seed, ratios=(0.8, 0.1)):
    # Default is already 80/10/10 (0.8, 0.1 implies 0.1 test)
    # User request: 80% train, 10% val, 10% test for full ft.
    # Linear probing: 80% train, 20% test.
    # We will handle linear probing logic by passing ratios=(0.8, 0.0) from the caller if needed.
    unique_subjects = sorted(list(subject_files.keys()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_subjects)
    
    n_subjects = len(unique_subjects)
    n_train = int(n_subjects * ratios[0])
    n_val = int(n_subjects * ratios[1])
    
    splits = {
        'train': unique_subjects[:n_train],
        'val': unique_subjects[n_train:n_train+n_val],
        'test': unique_subjects[n_train+n_val:]
    }
    
    files_split = {}
    for key in ['train', 'val', 'test']:
        files = []
        for s in splits[key]:
            files.extend(subject_files[s])
        files_split[key] = files
        
    return files_split

def get_tuab_file_list(dataset_dir, mode, seed=42):
    # Get all H5 files in root
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, 'sub_*.h5')))
    if not all_h5_files:
        # Fallback to recursive if not found in root
        all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', 'sub_*.h5'), recursive=True))
        
    print(f"Found {len(all_h5_files)} H5 files in {dataset_dir}")
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_h5_files)
    
    subject_files = _group_files_by_subject(all_h5_files)
    splits = _split_subjects(subject_files, seed)
    
    # Map 'eval' to 'val' if needed
    if mode == 'eval':
        mode = 'val'
        
    return splits.get(mode, [])

# --- Indexing Logic from cbramod_tuab/util/datasets.py ---

def _index_worker(h5_path):
    samples = []
    errors = []
    label_map = {'normal': 0, 'abnormal': 1}
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                task_name = trial_group.attrs.get('task_name')
                if isinstance(task_name, bytes):
                    task_name = task_name.decode('utf-8')
                
                label = label_map.get(task_name, -1)
                if label == -1:
                    continue

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

class TUABDataset(Dataset):
    def __init__(self, file_list, input_size=12000, transform=None, cache_path='dataset_index.json', **kwargs):
        super().__init__()
        self.input_size = input_size
        self.transform = transform
        self.drop_bad_samples = bool(kwargs.get('drop_bad_samples', True))
        
        # Standard 19 channels from cbramod_tuab/run_finetuning.py
        self.target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        # Note: 'T7'/'T8' might be used in some systems instead of 'T3'/'T4'. TUAB often uses T3/T4.
        # The source mapping below assumes the file has these names.
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
        self.cache_size = 128 # Keep 128 files open per worker

    @staticmethod
    def collate(batch):
        return _collate_drop_none(batch)

    def _load_or_generate_index(self, file_list, cache_path):
        full_index = []
        error_index = []
        input_files_set = set(file_list)
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
                
                # Check coverage: Ensure all input files are in the cache
                cached_files = set(s['file_path'] for s in full_index)
                if input_files_set.issubset(cached_files):
                     reindex = False
                else:
                     print("Cache incomplete. Re-indexing...")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # 2. Re-index if needed
        if reindex:
            # Important: To be safe and mimic reference, we should index EVERYTHING if possible,
            # but here we only have access to 'file_list'. 
            # Ideally, we should index 'input_files_set' and MERGE it with existing cache, or overwrite.
            # To be robust: We will index the provided file_list.
            print(f"Indexing {len(file_list)} files...")
            max_workers = min(32, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_index_worker, file_list), total=len(file_list)))
            new_samples = [s for res in results for s in res.get('samples', [])]
            new_errors = [e for res in results for e in res.get('errors', [])]
            
            # Merge with existing index if available to avoid losing other splits
            # (Though simple overwrite is safer if we assume we run this for all splits eventually)
            # Let's simple append new unique samples
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
                    try:
                        old_f.close()
                    except:
                        pass
                
                # Optimize read with 4MB chunk cache, use latest libver for speed
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

            if raw.shape[0] != 21 and raw.shape[1] == 21:
                raw = raw.T
            
            raw = raw[self.channel_indices, :]
            
            if raw.shape[1] < self.input_size:
                pad = self.input_size - raw.shape[1]
                data = np.pad(raw, ((0,0), (0, pad)), 'constant')
            else:
                data = raw[:, :self.input_size]
        except Exception as e:
            # If cache error, try to reopen
            if h5_path in self.file_cache:
                try:
                    self.file_cache.pop(h5_path).close()
                except:
                    pass
            if self.drop_bad_samples:
                return None
            raise RuntimeError(f"Bad sample: {h5_path} {trial_key}/{seg_key}: {type(e).__name__}: {e}") from e

        tensor = torch.from_numpy(data).float()
        # Normalize
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)
        
        # Patchify: (C, T) -> (C, N, P)
        patch_size = 200
        if self.input_size % patch_size == 0:
            num_patches = self.input_size // patch_size
            tensor = tensor.view(tensor.shape[0], num_patches, patch_size)

        if self.transform is not None:
            tensor = self.transform(tensor)
            
        return tensor, label
