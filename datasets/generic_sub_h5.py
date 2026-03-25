"""Generic dataset for downstream EEG datasets stored as sub_*.h5 files.

Expected HDF5 structure (same as SEED/SEEDIV):
  <file>.h5
    trial0/
      segment0/
        eeg  (dataset, shape: [C, T], attrs: label=<int or array>)
      segment1/
      ...
    trial1/
      ...

Supported datasets using this loader:
  SleepEDF_full, Physionet_MI, Workload, MDD, AD65
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def get_sub_h5_file_list(
    dataset_dir: str,
    mode: str = "train",
    seed: int = 42,
    test_ratio: float = 0.2,
) -> List[str]:
    """Split sub_*.h5 files into train/val/test by subject.

    Uses stratified splitting by subject-level label (read from the first
    segment of each file) so that all classes appear in both train and test.
    Falls back to sorted numeric split if label reading fails.
    """
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    all_files = sorted(
        glob.glob(os.path.join(dataset_dir, "sub_*.h5")),
        key=lambda p: _extract_sub_id(p),
    )
    if not all_files:
        raise ValueError(f"No sub_*.h5 files found in {dataset_dir}")

    print(f"[{mode}] Found {len(all_files)} subject files in {dataset_dir}")

    # Try to read per-subject label for stratified split
    subject_labels = _read_subject_labels(all_files)

    rng = np.random.RandomState(seed)

    if subject_labels is not None:
        # Stratified split: group files by label, split each group
        from collections import defaultdict
        label_to_files: dict = defaultdict(list)
        for f, lbl in zip(all_files, subject_labels):
            label_to_files[lbl].append(f)

        train_files: List[str] = []
        test_files: List[str] = []
        for lbl in sorted(label_to_files.keys()):
            grp = label_to_files[lbl]
            rng_grp = np.random.RandomState(seed + lbl)  # deterministic per class
            idx = np.arange(len(grp))
            rng_grp.shuffle(idx)
            n_test = max(1, int(np.ceil(len(grp) * test_ratio)))
            test_files.extend([grp[i] for i in idx[:n_test]])
            train_files.extend([grp[i] for i in idx[n_test:]])

        # Sort for reproducibility
        train_files = sorted(train_files, key=_extract_sub_id)
        test_files = sorted(test_files, key=_extract_sub_id)
        print(f"[{mode}] Stratified split: {len(train_files)} train / {len(test_files)} test")
    else:
        # Fallback: simple numeric order split
        n_test = max(1, int(np.ceil(len(all_files) * test_ratio)))
        train_files = all_files[:-n_test]
        test_files = all_files[-n_test:]
        print(f"[{mode}] Numeric split: {len(train_files)} train / {len(test_files)} test")

    splits: dict = {
        "train": train_files,
        "val": test_files,
        "test": test_files,
    }
    file_list = splits.get(mode, all_files)
    print(f"[{mode}] Using {len(file_list)} files")
    return file_list


def _read_subject_labels(files: List[str]) -> Optional[List[int]]:
    """Read the label of the first segment from each subject file.

    Returns a list of ints (one per file), or None if reading fails.
    """
    labels: List[int] = []
    try:
        for path in files:
            with h5py.File(path, "r") as f:
                trial_keys = [k for k in f.keys() if k.startswith("trial")]
                if not trial_keys:
                    return None
                tk = trial_keys[0]
                seg_keys = [k for k in f[tk].keys() if k.startswith("segment")]
                if not seg_keys:
                    return None
                sk = seg_keys[0]
                if "eeg" not in f[tk][sk]:
                    return None
                raw = np.asarray(f[tk][sk]["eeg"].attrs.get("label", -1))
                labels.append(int(raw.flat[0]))
    except Exception as exc:
        print(f"[split] Warning: could not read subject labels ({exc}), using numeric split.")
        return None
    return labels


def _extract_sub_id(path: str) -> int:
    try:
        return int(os.path.basename(path).split("_")[1].split(".")[0])
    except Exception:
        return 999999


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _index_worker(h5_path: str) -> dict:
    samples = []
    errors = []
    try:
        with h5py.File(h5_path, "r") as f:
            for trial_key in sorted(f.keys()):
                if not trial_key.startswith("trial"):
                    continue
                trial_group = f[trial_key]
                if not isinstance(trial_group, h5py.Group):
                    continue
                for seg_key in sorted(trial_group.keys()):
                    if not seg_key.startswith("segment"):
                        continue
                    seg_group = trial_group[seg_key]
                    if not isinstance(seg_group, h5py.Group):
                        continue
                    if "eeg" not in seg_group:
                        continue
                    dset = seg_group["eeg"]
                    label = -1
                    if "label" in dset.attrs:
                        raw = dset.attrs["label"]
                        raw = np.asarray(raw)
                        if raw.ndim == 0:
                            label = int(raw)
                        elif raw.size == 1:
                            label = int(raw.flat[0])
                        else:
                            label = int(np.argmax(raw))
                    samples.append(
                        {
                            "file_path": h5_path,
                            "trial_key": trial_key,
                            "segment_key": seg_key,
                            "label": label,
                        }
                    )
    except Exception as exc:
        errors.append(
            {
                "file_path": h5_path,
                "trial_key": None,
                "segment_key": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return {"samples": samples, "errors": errors}


def _collate_drop_none(batch):
    kept = [b for b in batch if b is not None]
    if not kept:
        return None
    xs, ys = zip(*kept)
    x = torch.stack(xs, 0)
    y0 = ys[0]
    y = torch.stack(ys, 0) if isinstance(y0, torch.Tensor) else torch.as_tensor(ys)
    return {"x": x, "y": y, "mask": None, "dropped": len(batch) - len(kept)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GenericSubH5Dataset(Dataset):
    """Reads sub_*.h5 files whose internal structure mirrors SEEDDataset.

    Parameters
    ----------
    file_list : list of str
        Paths to subject h5 files.
    input_size : int
        Number of time-steps per sample (default 200 = 1 s @ 200 Hz).
    patch_size : int
        Patch width for patchification. Set to 0 to disable patchification.
    num_channels : int or None
        Expected channel count. None = infer from data.
    cache_path : str
        Path to JSON index cache.
    mode : str
        'train' | 'val' | 'test'
    drop_bad_samples : bool
        Return None instead of raising on bad samples (collate skips them).
    label_offset : int
        Value added to every raw label after loading (default 0).
        Use -1 for datasets with 1-based labels (e.g. MDD: 1/2 -> 0/1,
        AD65: 1/2/3 -> 0/1/2).
    """

    def __init__(
        self,
        file_list: List[str],
        input_size: int = 200,
        patch_size: int = 200,
        num_channels: Optional[int] = None,
        cache_path: str = "dataset_index_generic.json",
        mode: str = "train",
        drop_bad_samples: bool = True,
        label_offset: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.mode = mode
        self.drop_bad_samples = drop_bad_samples
        self.label_offset = label_offset
        self._num_channels = num_channels  # None = infer

        self.samples = self._build_index(file_list, cache_path)
        self._file_set = set(file_list)

        # LRU file handle cache
        self._file_cache: OrderedDict = OrderedDict()
        self._cache_size = 64

    # ------------------------------------------------------------------
    @staticmethod
    def collate(batch):
        return _collate_drop_none(batch)

    # ------------------------------------------------------------------
    def _build_index(self, file_list: List[str], cache_path: str) -> List[dict]:
        input_set = set(file_list)
        full_index: List[dict] = []

        # Try loading cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                full_index = data.get("samples", data) if isinstance(data, dict) else data
                cached_files = {s["file_path"] for s in full_index}
                if input_set.issubset(cached_files):
                    filtered = [s for s in full_index if s["file_path"] in input_set]
                    print(f"[index] Loaded {len(filtered)} samples from cache.")
                    return filtered
                print("[index] Cache incomplete, re-indexing...")
            except Exception as exc:
                print(f"[index] Cache read failed ({exc}), re-indexing...")

        # Build fresh index
        print(f"[index] Indexing {len(file_list)} files...")
        workers = min(8, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as exe:
            results = list(
                tqdm(exe.map(_index_worker, file_list), total=len(file_list), desc="indexing")
            )

        new_samples = [s for r in results for s in r["samples"]]
        new_errors = [e for r in results for e in r["errors"]]

        if new_errors:
            print(f"[index] {len(new_errors)} errors during indexing.")

        # Merge with any existing cache entries for other files
        existing_map = {(s["file_path"], s["trial_key"], s["segment_key"]): s for s in full_index}
        for s in new_samples:
            existing_map[(s["file_path"], s["trial_key"], s["segment_key"])] = s
        merged = list(existing_map.values())

        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"samples": merged, "errors": new_errors}, f)
        except Exception as exc:
            print(f"[index] Could not save cache: {exc}")

        filtered = [s for s in merged if s["file_path"] in input_set]
        print(f"[index] {len(filtered)} samples ready.")
        return filtered

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        h5_path = info["file_path"]
        trial_key = info["trial_key"]
        seg_key = info["segment_key"]
        label = info["label"]

        try:
            f = self._get_file_handle(h5_path)
            dset = f[trial_key][seg_key]["eeg"]
            raw: np.ndarray = dset[:]  # (C, T) or (T, C)
        except Exception as exc:
            self._evict_file_handle(h5_path)
            if self.drop_bad_samples:
                return None
            raise RuntimeError(
                f"Bad sample {h5_path} {trial_key}/{seg_key}: {exc}"
            ) from exc

        # Apply label offset (e.g. -1 for 1-based datasets like MDD, AD65)
        label = label + self.label_offset

        # Ensure (C, T)
        if raw.ndim == 2 and raw.shape[1] < raw.shape[0]:
            raw = raw.T

        num_ch = raw.shape[0]

        # Infer / validate channel count
        if self._num_channels is None:
            self._num_channels = num_ch
        expected_ch = self._num_channels

        # Time crop / pad
        T = raw.shape[1] if raw.ndim == 2 else 0
        data = np.zeros((expected_ch, self.input_size), dtype=np.float32)
        ch_use = min(num_ch, expected_ch)
        if T >= self.input_size:
            start = (
                np.random.randint(0, T - self.input_size + 1)
                if self.mode == "train"
                else (T - self.input_size) // 2
            )
            data[:ch_use] = raw[:ch_use, start : start + self.input_size].astype(np.float32)
        else:
            data[:ch_use, :T] = raw[:ch_use].astype(np.float32)

        tensor = torch.from_numpy(data)  # (C, T)

        # Per-channel z-score
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-6)

        # Patchify: (C, T) -> (C, n_patches, patch_size)
        if self.patch_size > 0 and self.input_size % self.patch_size == 0:
            n_patches = self.input_size // self.patch_size
            tensor = tensor.view(expected_ch, n_patches, self.patch_size)

        return tensor, label

    # ------------------------------------------------------------------
    def _get_file_handle(self, path: str) -> h5py.File:
        if path in self._file_cache:
            self._file_cache.move_to_end(path)
            return self._file_cache[path]
        if len(self._file_cache) >= self._cache_size:
            _, old_f = self._file_cache.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass
        fh = h5py.File(path, "r", rdcc_nbytes=4 * 1024 * 1024, libver="latest")
        self._file_cache[path] = fh
        return fh

    def _evict_file_handle(self, path: str) -> None:
        if path in self._file_cache:
            try:
                self._file_cache.pop(path).close()
            except Exception:
                pass
