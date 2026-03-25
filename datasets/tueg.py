import os
import glob
import h5py
import json
import pickle
import hashlib
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import OrderedDict

# --- Splitting Logic (adapted from TUAB) ---

def _group_files_by_subject(all_h5_files):
    subject_files = {}
    for f in all_h5_files:
        basename = os.path.basename(f)
        try:
            subject_id = basename.split('_')[1].split('.')[0]
        except IndexError:
            subject_id = basename.split('.')[0]
            
        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)
    return subject_files

def get_tueg_file_list(dataset_dir, mode='train', seed=42, subset_fraction=None):
    dataset_dir = os.path.abspath(dataset_dir)
    cache_root = os.path.join(dataset_dir, '.tueg_cache')
    cache_key = hashlib.md5(dataset_dir.encode('utf-8')).hexdigest()[:12]
    cache_path = os.path.join(cache_root, f'filelist_{cache_key}.pkl')
    all_h5_files = None
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            cached_dir = cached.get('dataset_dir')
            cached_files = cached.get('all_h5_files')
            sample_files = cached_files[: min(64, len(cached_files))] if isinstance(cached_files, list) else []
            if cached_dir == dataset_dir and isinstance(cached_files, list) and all(os.path.exists(p) for p in sample_files):
                all_h5_files = cached_files
                print(f"Loaded cached file list: {len(all_h5_files)} files")
    except Exception:
        all_h5_files = None
    if all_h5_files is None:
        all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
        all_h5_files = [f for f in all_h5_files if 'sub_' in os.path.basename(f)]
        try:
            os.makedirs(cache_root, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({'dataset_dir': dataset_dir, 'all_h5_files': all_h5_files}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        print(f"Found {len(all_h5_files)} H5 files in {dataset_dir}")
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_h5_files)
    
    subject_files = _group_files_by_subject(all_h5_files)
    unique_subjects = sorted(list(subject_files.keys()))
    rng.shuffle(unique_subjects)
    
    n_subjects = len(unique_subjects)
    n_train = int(n_subjects * 0.95)
    
    splits = {
        'train': unique_subjects[:n_train],
        'val': unique_subjects[n_train:],
        'test': []
    }
    
    if mode == 'train' and subset_fraction is not None:
        try:
            subset_fraction = float(subset_fraction)
            if 0.0 < subset_fraction < 1.0:
                original_train = splits['train']
                n_original = len(original_train)
                n_keep = int(n_original * subset_fraction)
                n_keep = max(1, n_keep)
                
                print(f"[{mode}] Subsetting active: Using {subset_fraction:.2%} of subjects.")
                print(f"Reducing training subjects from {n_original} to {n_keep}.")
                
                splits['train'] = original_train[:n_keep]
        except ValueError:
            print(f"Warning: Invalid subset_fraction {subset_fraction}, ignoring.")

    target_subjects = splits.get(mode, [])
    file_list = []
    for s in target_subjects:
        file_list.extend(subject_files[s])
        
    return file_list

def _index_worker(h5_path):
    samples = []
    errors = []
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            for trial_key in [k for k in f.keys() if k.startswith('trial')]:
                trial_group = f[trial_key]
                label = -1 
                segment_keys = [k for k in trial_group.keys() if k.startswith('segment')]
                if segment_keys:
                    for seg_key in segment_keys:
                        seg_group = trial_group[seg_key]
                        if 'eeg' in seg_group:
                            dset_path = f"{trial_key}/{seg_key}/eeg"
                        elif 'data' in seg_group:
                            dset_path = f"{trial_key}/{seg_key}/data"
                        else:
                            continue
                        samples.append({
                            'file_path': h5_path,
                            'trial_key': trial_key,
                            'segment_key': seg_key,
                            'dset_path': dset_path,
                            'label': label
                        })
                else:
                    if 'eeg' in trial_group:
                        dset_path = f"{trial_key}/eeg"
                    elif 'data' in trial_group:
                        dset_path = f"{trial_key}/data"
                    else:
                        continue
                    samples.append({
                        'file_path': h5_path,
                        'trial_key': trial_key,
                        'segment_key': None,
                        'dset_path': dset_path,
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


def _parse_segment_id(seg_key):
    if not seg_key:
        return 0
    if isinstance(seg_key, str) and seg_key.startswith('segment'):
        try:
            return int(seg_key.replace('segment', ''))
        except ValueError:
            return 0
    return 0


class TUEGFileGroupedSampler(Sampler):
    def __init__(self, dataset, chunk_size=256, seed=42):
        self.dataset = dataset
        self.chunk_size = int(max(1, chunk_size))
        self.seed = int(seed)
        self.epoch = 0
        self._iter_count = 0
        file_to_indices = {}
        for i, s in enumerate(dataset.samples):
            fp = s['file_path']
            if fp not in file_to_indices:
                file_to_indices[fp] = []
            file_to_indices[fp].append(i)
        self.file_to_indices = file_to_indices
        self.file_keys = list(file_to_indices.keys())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch + self._iter_count)
        self._iter_count += 1
        per_file = {}
        for k, idxs in self.file_to_indices.items():
            arr = np.asarray(idxs, dtype=np.int64)
            rng.shuffle(arr)
            per_file[k] = arr
        file_order = list(self.file_keys)
        rng.shuffle(file_order)
        cursors = {k: 0 for k in file_order}
        remaining = int(sum(len(per_file[k]) for k in file_order))
        while remaining > 0:
            for k in file_order:
                c = cursors[k]
                arr = per_file[k]
                if c >= len(arr):
                    continue
                end = min(c + self.chunk_size, len(arr))
                chunk = arr[c:end]
                for idx in chunk:
                    yield int(idx)
                consumed = end - c
                cursors[k] = end
                remaining -= consumed
                if remaining <= 0:
                    break

    def __len__(self):
        return len(self.dataset)

def _collate_drop_none(batch):
    kept = [b for b in batch if b is not None]
    dropped = len(batch) - len(kept)
    if len(kept) == 0:
        return None
    
    elem = kept[0]
    if len(elem) == 3:
        xs, ys, masks = zip(*kept)
        x = torch.stack(xs, 0)
        y = torch.stack(ys, 0)
        feature_mask = torch.stack(masks, 0)
        return {'x': x, 'y': y, 'feature_mask': feature_mask, 'dropped': dropped}
    else:
        xs, ys = zip(*kept)
        x = torch.stack(xs, 0)
        y0 = ys[0]
        if isinstance(y0, torch.Tensor):
            y = torch.stack(ys, 0)
        else:
            y = torch.as_tensor(ys)
        return {'x': x, 'y': y, 'mask': None, 'dropped': dropped}

class TUEGDataset(Dataset):
    def __init__(self, file_list, input_size=2000, transform=None, cache_path='dataset_index.json', feature_path=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.transform = transform
        self.drop_bad_samples = bool(kwargs.get('drop_bad_samples', True))
        self.feature_clip_min = float(kwargs.get('feature_clip_min', -6.0))
        self.feature_clip_max = float(kwargs.get('feature_clip_max', 6.0))
        self.h5_rdcc_nbytes = int(kwargs.get('h5_rdcc_nbytes', 8 * 1024 * 1024))
        self.cache_size = int(kwargs.get('h5_cache_size', 128))
        self.enable_feature_filter_cache = bool(kwargs.get('enable_feature_filter_cache', True))
        self.feature_filter_cache_path = kwargs.get('feature_filter_cache_path')
        self.feature_source_signature = None
        
        self.file_handles = OrderedDict()
        self.feature_map = None
        self.feature_dim = 0
        self.feature_names = []
        self._zero_features = torch.zeros(0, dtype=torch.float32)
        self._one_mask = torch.zeros(0, dtype=torch.float32)
        self._zero_mask = torch.zeros(0, dtype=torch.float32)
        if feature_path and os.path.exists(feature_path):
            print(f"Loading features from {feature_path}...")
            try:
                import pandas as pd
                feature_cache_path = kwargs.get('feature_cache_path', f"{feature_path}.pkl")
                feature_cache_hit = False
                cached = None
                if feature_cache_path and os.path.exists(feature_cache_path):
                    try:
                        with open(feature_cache_path, 'rb') as f:
                            cached = pickle.load(f)
                        src_mtime = os.path.getmtime(feature_path)
                        cache_mtime = cached.get('feature_source_mtime')
                        if cache_mtime == src_mtime:
                            self.feature_map = cached.get('feature_map')
                            self.feature_names = cached.get('feature_names', [])
                            self.feature_dim = int(cached.get('feature_dim', len(self.feature_names)))
                            feature_cache_hit = self.feature_map is not None
                    except Exception:
                        cached = None
                if not feature_cache_hit:
                    header = pd.read_csv(feature_path, nrows=0)
                    cols = list(header.columns)
                    if 'source_file' not in cols or 'segment_id' not in cols:
                        raise ValueError("feature csv缺少source_file或segment_id")
                    usecols = [c for c in ['source_file', 'segment_id', 'window_idx'] if c in cols]
                    dtype_map = {'segment_id': np.int32, 'window_idx': np.int16}
                    meta_cols = ['aug_type', 'segment_id', 'window_idx', 'start_time', 'subject_id', 'source_file']
                    feature_cols = [c for c in cols if c not in meta_cols]
                    for c in feature_cols:
                        dtype_map[c] = np.float32
                    usecols.extend(feature_cols)
                    df = pd.read_csv(feature_path, usecols=usecols, dtype=dtype_map, memory_map=True)
                    if 'window_idx' in df.columns:
                        df = df[df['window_idx'] == 0]
                    basename = df['source_file'].astype(str).str.rsplit('/', n=1).str[-1]
                    seg_ids = df['segment_id'].to_numpy(np.int32, copy=False)
                    vals = df[feature_cols].to_numpy(np.float32, copy=False)
                    np.clip(vals, self.feature_clip_min, self.feature_clip_max, out=vals)
                    self.feature_map = {(str(basename.iloc[i]), int(seg_ids[i])): vals[i] for i in range(len(df))}
                    self.feature_names = feature_cols
                    self.feature_dim = len(feature_cols)
                    if feature_cache_path:
                        cache_dir = os.path.dirname(feature_cache_path)
                        if cache_dir:
                            os.makedirs(cache_dir, exist_ok=True)
                        with open(feature_cache_path, 'wb') as f:
                            pickle.dump({
                                'feature_source_mtime': os.path.getmtime(feature_path),
                                'feature_map': self.feature_map,
                                'feature_names': self.feature_names,
                                'feature_dim': self.feature_dim,
                            }, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    print(f"Loaded feature cache from {feature_cache_path}")
                meta_cols = ['aug_type', 'segment_id', 'window_idx', 'start_time', 'subject_id', 'source_file']
                if not self.feature_names and self.feature_map is not None:
                    example = next(iter(self.feature_map.values())) if len(self.feature_map) > 0 else None
                    self.feature_dim = int(example.shape[0]) if example is not None else 0
                    self.feature_names = [f'f{i}' for i in range(self.feature_dim)]
                print(f"Found {self.feature_dim} features.")
                print(f"Indexed {len(self.feature_map)} feature entries.")
                try:
                    src_mtime = os.path.getmtime(feature_path)
                    self.feature_source_signature = f"{os.path.abspath(feature_path)}:{src_mtime:.6f}:{self.feature_dim}"
                except Exception:
                    self.feature_source_signature = f"{os.path.abspath(feature_path)}:{self.feature_dim}"
                    
            except Exception as e:
                print(f"Error loading features: {e}")
                self.feature_map = None
        
        self.target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
        self.source_channels = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2']
        
        self.channel_indices = []
        for target in self.target_channels:
            try:
                idx = self.source_channels.index(target)
                self.channel_indices.append(idx)
            except ValueError:
                print(f"Warning: Channel {target} not found in source map.")
        self.channel_indices_np = np.asarray(self.channel_indices, dtype=np.int64)
        self.channel_sorted_order = np.argsort(self.channel_indices_np)
        self.channel_read_indices_np = self.channel_indices_np[self.channel_sorted_order]
        self.channel_restore_order = np.argsort(self.channel_sorted_order)
        
        self.samples = self._load_or_generate_index(file_list, cache_path)
        self._zero_features = torch.zeros(self.feature_dim, dtype=torch.float32)
        self._one_mask = torch.ones(self.feature_dim, dtype=torch.float32)
        self._zero_mask = torch.zeros(self.feature_dim, dtype=torch.float32)

    @staticmethod
    def collate(batch):
        return _collate_drop_none(batch)

    def build_grouped_sampler(self, batch_size, seed=42, chunk_size=None):
        if chunk_size is None:
            chunk_size = max(int(batch_size) * 2, 128)
        return TUEGFileGroupedSampler(self, chunk_size=chunk_size, seed=seed)

    def _load_or_generate_index(self, file_list, cache_path):
        full_index = []
        error_index = []
        input_files_set = set(file_list)
        reindex = True
        
        cache_ext = os.path.splitext(cache_path)[1].lower()
        use_pickle = cache_ext in ['.pkl', '.pickle']
        if os.path.exists(cache_path):
            print(f"Loading index from {cache_path}...")
            try:
                if use_pickle:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(cache_path, 'r') as f:
                        data = json.load(f)
                if isinstance(data, list):
                    full_index = data
                else:
                    full_index = data.get('samples', [])
                    error_index = data.get('errors', [])
                
                cached_files = set(s['file_path'] for s in full_index)
                if input_files_set.issubset(cached_files):
                     reindex = False
                else:
                     print("Cache incomplete. Re-indexing...")
            except Exception as e:
                print(f"Error loading cache: {e}")
        
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
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
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
                tmp_path = f"{cache_path}.tmp"
                if use_pickle:
                    with open(tmp_path, 'wb') as f:
                        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(tmp_path, 'w') as f:
                        json.dump(payload, f)
                os.replace(tmp_path, cache_path)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
            
            if len(error_index) > 0:
                bad_files = {e.get('file_path') for e in error_index if e.get('file_path')}
                print(f"Index summary: samples={len(full_index)} errors={len(error_index)} bad_files={len(bad_files)}")
                
        filtered_samples = [s for s in full_index if s['file_path'] in input_files_set]
        file_sig = hashlib.md5('\n'.join(sorted(input_files_set)).encode('utf-8')).hexdigest()
        
        if self.feature_map is not None:
            feature_filter_cache_path = self.feature_filter_cache_path or f"{cache_path}.feature_filter.pkl"
            if self.enable_feature_filter_cache and os.path.exists(feature_filter_cache_path):
                try:
                    with open(feature_filter_cache_path, 'rb') as f:
                        cached_filter = pickle.load(f)
                    if (
                        cached_filter.get('file_sig') == file_sig and
                        cached_filter.get('feature_sig') == self.feature_source_signature and
                        isinstance(cached_filter.get('samples'), list)
                    ):
                        filtered_samples = cached_filter['samples']
                        print(f"Loaded filtered sample cache: {len(filtered_samples)}")
                except Exception:
                    pass
            if len(filtered_samples) > 0 and 'seg_id' in filtered_samples[0] and 'basename' in filtered_samples[0]:
                pass
            else:
                print(f"Filtering samples without features (Total before: {len(filtered_samples)})...")
                valid_samples = []
                missing_count = 0
                for s in filtered_samples:
                    basename = os.path.basename(s['file_path'])
                    seg_key = s.get('segment_key')
                    seg_id = _parse_segment_id(seg_key)
                    s['basename'] = basename
                    s['seg_id'] = seg_id
                    if (basename, seg_id) in self.feature_map:
                        valid_samples.append(s)
                    else:
                        missing_count += 1
                
                print(f"Removed {missing_count} samples without features.")
                filtered_samples = valid_samples
                if self.enable_feature_filter_cache:
                    try:
                        ff_cache_dir = os.path.dirname(feature_filter_cache_path)
                        if ff_cache_dir:
                            os.makedirs(ff_cache_dir, exist_ok=True)
                        with open(feature_filter_cache_path, 'wb') as f:
                            pickle.dump(
                                {
                                    'file_sig': file_sig,
                                    'feature_sig': self.feature_source_signature,
                                    'samples': filtered_samples,
                                },
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL
                            )
                    except Exception:
                        pass

        unique_trials = set((s['file_path'], s['trial_key']) for s in filtered_samples)
        print(f"Dataset initialized: {len(filtered_samples)} samples (from {len(file_list)} files).")
        print(f"Unique trials found: {len(unique_trials)}")
        
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path = info['file_path']
        trial_key = info.get('trial_key')
        seg_key = info.get('segment_key')
        
        data = np.zeros((len(self.channel_indices), self.input_size), dtype=np.float32)
        
        # --- IO Optimization: Use Persistent File Handles (LRU Cache) ---
        try:
            # Check if file is already open
            if h5_path in self.file_handles:
                f = self.file_handles[h5_path]
                self.file_handles.move_to_end(h5_path)
            else:
                try:
                    f = h5py.File(
                        h5_path,
                        'r',
                        rdcc_nbytes=self.h5_rdcc_nbytes,
                        libver='latest',
                        swmr=True,
                    )
                except Exception:
                    f = h5py.File(h5_path, 'r', rdcc_nbytes=self.h5_rdcc_nbytes, libver='latest')
                if len(self.file_handles) >= self.cache_size:
                    old_path, old_f = self.file_handles.popitem(last=False)
                    try:
                        old_f.close()
                    except Exception:
                        pass
                
                self.file_handles[h5_path] = f

            dset_path = info.get('dset_path')
            if dset_path is not None:
                dset = f[dset_path]
            else:
                # Fallback for old cache
                group = f[trial_key][seg_key] if seg_key else f[trial_key]
                if 'eeg' in group:
                    dset = group['eeg']
                elif 'data' in group:
                    dset = group['data']
                else:
                    raise KeyError("No data")

            shape = dset.shape
            
            is_transposed = False
            if len(shape) == 2 and shape[0] != 21 and shape[1] == 21:
                is_transposed = True
                time_dim = 0
            else:
                time_dim = 1

            current_len = shape[time_dim]
            
            if current_len >= self.input_size:
                if is_transposed:
                    raw = dset[:self.input_size, :]
                    raw = np.asarray(raw, dtype=np.float32)
                    raw = raw[:, self.channel_read_indices_np]
                    raw = raw[:, self.channel_restore_order]
                    data = raw.T
                else:
                    raw = dset[:, :self.input_size]
                    raw = np.asarray(raw, dtype=np.float32)
                    raw = raw[self.channel_read_indices_np, :]
                    data = raw[self.channel_restore_order, :]
            else:
                if is_transposed:
                    raw = dset[:, :]
                    raw = np.asarray(raw, dtype=np.float32)
                    raw = raw[:, self.channel_read_indices_np]
                    raw = raw[:, self.channel_restore_order]
                    data = raw.T
                else:
                    raw = dset[:, :]
                    raw = np.asarray(raw, dtype=np.float32)
                    raw = raw[self.channel_read_indices_np, :]
                    data = raw[self.channel_restore_order, :]
                pad = self.input_size - data.shape[1]
                data = np.pad(data, ((0,0), (0, pad)), 'constant')
                    
        except Exception as e:
            if h5_path in self.file_handles:
                try:
                    self.file_handles.pop(h5_path).close()
                except:
                    pass
            
            if self.drop_bad_samples:
                return None
            raise RuntimeError(f"Bad sample: {h5_path} {trial_key}/{seg_key}: {type(e).__name__}: {e}") from e

        # Normalize using numpy (faster for small arrays)
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, ddof=1, keepdims=True)
        data = (data - mean) / (std + 1e-6)
        
        tensor = torch.from_numpy(np.ascontiguousarray(data))
        
        patch_size = 200
        if self.input_size % patch_size == 0:
            num_patches = self.input_size // patch_size
            tensor = tensor.view(tensor.shape[0], num_patches, patch_size)

        if self.transform is not None:
            tensor = self.transform(tensor)
            
        features = None
        if self.feature_map is not None:
            basename = info.get('basename')
            if basename is None:
                basename = os.path.basename(h5_path)
            seg_id = info.get('seg_id')
            if seg_id is None:
                seg_id = _parse_segment_id(seg_key)
            features_np = self.feature_map.get((basename, seg_id))
            if features_np is not None:
                features = torch.from_numpy(features_np)
            else:
                features = self._zero_features
        
        if features is not None:
            return tensor, features, self._one_mask
        else:
            return tensor, self._zero_features, self._zero_mask

    def close(self):
        for _, f in list(self.file_handles.items()):
            try:
                f.close()
            except Exception:
                pass
        self.file_handles.clear()

    def __del__(self):
        self.close()
