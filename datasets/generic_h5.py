
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

def _index_worker(h5_path):
    samples = []
    errors = []
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                if not isinstance(trial_group, h5py.Group):
                    continue
                
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                if segment_keys:
                    for seg_key in segment_keys:
                        # Try to find data to get shape
                        shape = None
                        if 'eeg' in trial_group[seg_key]:
                            shape = trial_group[seg_key]['eeg'].shape
                        elif 'data' in trial_group[seg_key]:
                            shape = trial_group[seg_key]['data'].shape
                            
                        samples.append({
                            'file_path': h5_path,
                            'trial_key': trial_key,
                            'segment_key': seg_key,
                            'shape': shape
                        })
                else:
                    # Maybe data is directly in trial?
                    shape = None
                    if 'eeg' in trial_group:
                        shape = trial_group['eeg'].shape
                    elif 'data' in trial_group:
                        shape = trial_group['data'].shape
                        
                    samples.append({
                        'file_path': h5_path,
                        'trial_key': trial_key,
                        'segment_key': None,
                        'shape': shape
                    })
    except Exception as e:
        errors.append({
            'file_path': h5_path,
            'error': str(e)
        })
    return {'samples': samples, 'errors': errors}

class GenericH5Dataset(Dataset):
    def __init__(self, file_list, input_size=12000, transform=None, cache_path=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.transform = transform
        self.drop_bad_samples = kwargs.get('drop_bad_samples', True)
        
        # Cache handling
        if cache_path is None:
            # Generate a hash of file list or use a temp name
            import hashlib
            h = hashlib.md5(str(sorted(file_list)).encode()).hexdigest()
            cache_path = f"output/dataset_index_{h[:8]}.json"
            
        self.samples = self._load_or_generate_index(file_list, cache_path)
        
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
                    cached_files = set(s['file_path'] for s in full_index)
                    if input_files_set.issubset(cached_files):
                        reindex = False
            except:
                pass
        
        if reindex:
            print(f"Indexing {len(file_list)} files for GenericH5Dataset...")
            max_workers = min(32, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_index_worker, file_list), total=len(file_list)))
            
            full_index = [s for res in results for s in res.get('samples', [])]
            
            # Save cache
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({'samples': full_index}, f)
                
        # Filter
        return [s for s in full_index if s['file_path'] in input_files_set]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path = info['file_path']
        trial_key = info['trial_key']
        seg_key = info['segment_key']
        
        try:
            # File Cache
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
                
            # Slicing Optimization
            # First get shape without reading data
            shape = dset.shape
            
            # Determine orientation: (21, T) or (T, 21)
            is_transposed = False
            if len(shape) == 2 and shape[0] > shape[1] and shape[1] < 200:
                is_transposed = True # (T, 21)
                time_dim = 0
            else:
                time_dim = 1 # (21, T) or other

            current_len = shape[time_dim]
            
            if current_len > self.input_size:
                 # Slice Read
                 start = np.random.randint(0, current_len - self.input_size + 1)
                 
                 if is_transposed:
                     # (T, 21) -> read (start:end, :)
                     raw = dset[start:start+self.input_size, :]
                     raw = raw.T
                 else:
                     # (21, T) -> read (:, start:end)
                     raw = dset[:, start:start+self.input_size]
            else:
                 # Read all
                 raw = dset[:]
                 if is_transposed:
                     raw = raw.T

            # Normalize
            tensor = torch.from_numpy(raw).float()
            mean = tensor.mean(dim=1, keepdim=True)
            std = tensor.std(dim=1, keepdim=True)
            tensor = (tensor - mean) / (std + 1e-6)
            
            # Patchify is NOT done here because length is variable. 
            # We return (C, T) tensor. 
            # Wait, TUEGDataset patchifies in __getitem__.
            # If we patchify here, we return (C, N, P).
            # It's better to patchify here so collate handles patches.
            
            patch_size = 200
            # Pad to multiple of patch_size
            C, T = tensor.shape
            remainder = T % patch_size
            if remainder != 0:
                pad = patch_size - remainder
                tensor = torch.nn.functional.pad(tensor, (0, pad))
            
            num_patches = tensor.shape[1] // patch_size
            tensor = tensor.view(C, num_patches, patch_size)
            
            if self.transform:
                tensor = self.transform(tensor)
                
            return tensor, 0 # Dummy label
            
        except Exception as e:
            if self.drop_bad_samples:
                return None
            raise e
