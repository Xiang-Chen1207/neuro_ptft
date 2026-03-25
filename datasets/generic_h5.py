
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


def _decode_attr_value(raw):
    if raw is None:
        return None

    if isinstance(raw, (bytes, np.bytes_)):
        try:
            return raw.decode('utf-8').strip()
        except Exception:
            return str(raw).strip()

    arr = np.asarray(raw)
    if arr.size == 0:
        return None

    if arr.dtype.kind in {'S', 'U', 'O'}:
        val = arr.reshape(-1)[0]
        if isinstance(val, (bytes, np.bytes_)):
            try:
                return val.decode('utf-8').strip()
            except Exception:
                return str(val).strip()
        return str(val).strip()

    if arr.ndim == 0 or arr.size == 1:
        return int(arr.reshape(-1)[0])

    return int(np.argmax(arr))


def _decode_category_list(file_attrs):
    if 'category_list' not in file_attrs:
        return []
    cats = []
    for item in np.asarray(file_attrs['category_list']).reshape(-1):
        if isinstance(item, (bytes, np.bytes_)):
            try:
                cats.append(item.decode('utf-8').strip())
            except Exception:
                cats.append(str(item).strip())
        else:
            cats.append(str(item).strip())
    return cats


def _extract_label_from_attrs(dset, group, trial_group, file_handle):
    categories = _decode_category_list(file_handle.attrs)

    dataset_name = ''
    if 'dataset_name' in file_handle.attrs:
        dataset_name = str(_decode_attr_value(file_handle.attrs['dataset_name']) or '').lower()

    # FACED: prefer task_name/category_list multi-class mapping.
    if 'faced' in dataset_name and trial_group is not None and categories and 'task_name' in trial_group.attrs:
        task_name = _decode_attr_value(trial_group.attrs['task_name'])
        if isinstance(task_name, str) and task_name in categories:
            return int(categories.index(task_name))

    candidates = []
    for obj in (dset, group, trial_group):
        if obj is None:
            continue
        for key in ('label', 'task_label'):
            if key in obj.attrs:
                candidates.append(obj.attrs[key])

    if trial_group is not None and 'task_name' in trial_group.attrs:
        candidates.append(trial_group.attrs['task_name'])

    for raw in candidates:
        parsed = _decode_attr_value(raw)
        if parsed is None:
            continue

        if isinstance(parsed, str):
            if not parsed:
                continue
            if categories and parsed in categories:
                return int(categories.index(parsed))
            # FACED-like fallback for valence binary probe.
            low = parsed.lower()
            if low.startswith('pos'):
                return 1
            if low.startswith('neg') or low.startswith('neu'):
                return 0
            continue

        return int(parsed)

    return None


def _pick_sleepfm_channel(x_group):
    preferred = ['EEG_F4-Cz', 'EEG_C4-M1', 'EEG_O2-Cz', 'EEG']
    for key in preferred:
        if key in x_group:
            return key
    keys = list(x_group.keys())
    return keys[0] if keys else None

def _index_worker(h5_path):
    samples = []
    errors = []
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            trial_keys = [k for k in f.keys() if k.startswith('trial')]
            for trial_key in trial_keys:
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

            # SleepEDF_full / SleepFM-style layout: top-level x(group) + y(dataset)
            if not trial_keys and 'x' in f and 'y' in f:
                x_obj = f['x']
                y_obj = f['y']
                if isinstance(y_obj, h5py.Dataset):
                    n_epochs = int(y_obj.shape[0]) if len(y_obj.shape) > 0 else 0
                    if isinstance(x_obj, h5py.Group):
                        channel_key = _pick_sleepfm_channel(x_obj)
                        if channel_key is not None and channel_key in x_obj:
                            dset = x_obj[channel_key]
                            for i in range(min(n_epochs, int(dset.shape[0]))):
                                samples.append({
                                    'file_path': h5_path,
                                    'trial_key': None,
                                    'segment_key': None,
                                    'sample_kind': 'xy',
                                    'sample_index': i,
                                    'channel_key': channel_key,
                                    'shape': tuple(dset.shape[1:]) if len(dset.shape) > 1 else tuple(dset.shape),
                                })
                    elif isinstance(x_obj, h5py.Dataset):
                        for i in range(min(n_epochs, int(x_obj.shape[0]))):
                            samples.append({
                                'file_path': h5_path,
                                'trial_key': None,
                                'segment_key': None,
                                'sample_kind': 'xy',
                                'sample_index': i,
                                'channel_key': None,
                                'shape': tuple(x_obj.shape[1:]) if len(x_obj.shape) > 1 else tuple(x_obj.shape),
                            })
    except Exception as e:
        errors.append({
            'file_path': h5_path,
            'error': str(e)
        })
    return {'samples': samples, 'errors': errors}

class GenericH5Dataset(Dataset):
    def __init__(self, file_list, input_size=12000, transform=None, cache_path=None, feature_map=None, feature_dim=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.transform = transform
        self.drop_bad_samples = kwargs.get('drop_bad_samples', True)
        
        self.feature_map = feature_map
        self.feature_dim = feature_dim

        # Cache handling
        if cache_path is None:
            # Generate a hash of file list or use a temp name
            import hashlib
            h = hashlib.md5(str(sorted(file_list)).encode()).hexdigest()
            cache_path = f"output/dataset_index_{h[:8]}.json"
            
        self.samples = self._load_or_generate_index(file_list, cache_path)
        
        self.file_cache = OrderedDict()
        self.cache_size = kwargs.get('cache_size', 256) # Increased cache size for better hit rate

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
        sample_kind = info.get('sample_kind', 'trial_segment')
        sample_index = int(info.get('sample_index', -1))
        channel_key = info.get('channel_key', None)
        
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
            
            group = None
            trial_group = None
            dset = None
            label = None

            if sample_kind == 'xy':
                if sample_index < 0:
                    raise ValueError('Invalid sample_index for xy sample')

                x_obj = f['x']
                y_obj = f['y']
                if isinstance(x_obj, h5py.Group):
                    key = channel_key if channel_key in x_obj else _pick_sleepfm_channel(x_obj)
                    if key is None:
                        raise KeyError('No channel available under x group')
                    raw = x_obj[key][sample_index]
                else:
                    raw = x_obj[sample_index]

                raw = np.asarray(raw)
                if raw.ndim == 1:
                    raw = raw[np.newaxis, :]

                label = int(np.asarray(y_obj[sample_index]).reshape(-1)[0])
            else:
                group = f[trial_key][seg_key] if seg_key else f[trial_key]
                trial_group = f[trial_key]

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
            
            # Retrieve features if map exists
            features = None
            if self.feature_map is not None:
                basename = os.path.basename(h5_path)
                # Parse segment id from key (e.g. 'segment20' -> 20)
                seg_id = 0
                if seg_key and seg_key.startswith('segment'):
                    try:
                        seg_id = int(seg_key.replace('segment', ''))
                    except ValueError:
                        seg_id = 0
                
                # Try full path first (for JointDataset disambiguation)
                features_data = self.feature_map.get((h5_path, seg_id))
                
                # Fallback to basename
                if features_data is None:
                    features_data = self.feature_map.get((basename, seg_id))

                if features_data is not None:
                    # features_data is (raw_vals_np, indices_np) from JointDataset
                    # or just raw_vals_np from Single Dataset (if not using Joint loader)
                    
                    if isinstance(features_data, tuple) and len(features_data) == 2:
                        raw_vals, indices = features_data
                        
                        full_feat = torch.zeros(self.feature_dim, dtype=torch.float32)
                        full_mask = torch.zeros(self.feature_dim, dtype=torch.float32)
                        
                        # Robust Scaling
                        raw_vals = np.clip(raw_vals, -6.0, 6.0)
                        
                        # Fill
                        full_feat[indices] = torch.from_numpy(raw_vals)
                        full_mask[indices] = 1.0
                        
                        features = full_feat
                        feature_mask = full_mask
                    else:
                        # Legacy/Single Dataset mode (assuming full vector)
                        features_np = features_data
                        features_np = np.clip(features_np, -6.0, 6.0)
                        features = torch.from_numpy(features_np)
                        feature_mask = torch.ones(self.feature_dim, dtype=torch.float32)
                else:
                    # Missing features for this segment
                    features = torch.zeros(self.feature_dim, dtype=torch.float32)
                    feature_mask = torch.zeros(self.feature_dim, dtype=torch.float32)
            
            if features is not None:
                return tensor, features, feature_mask
            else:
                if label is None:
                    # Try to read downstream label from attributes.
                    label = _extract_label_from_attrs(
                        dset=dset,
                        group=group,
                        trial_group=trial_group,
                        file_handle=f,
                    )

                # Keep backward-compatible fallback for unlabeled/pretraining data.
                if label is None:
                    label = 0
                return tensor, label
            
        except Exception as e:
            if self.drop_bad_samples:
                return None
            raise e
