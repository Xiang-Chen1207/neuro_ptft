import os
from torch.utils.data import DataLoader


def _worker_init_disable_h5_locking(_worker_id):
    # Avoid network filesystem lock contention in HDF5 workers.
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

def build_dataset(name, config, mode='train'):
    # config contains all keys from the 'dataset' section of yaml, including 'dataset_dir'
    if name == 'TUEG':
        from .tueg import TUEGDataset, get_tueg_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files using TUEG logic (95/5 for pretraining)
        file_list = get_tueg_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
        elif mode == 'val' and config.get('limit_val', True):
            # For validation efficiency, default to a small subset (Mini-Val).
            # Set dataset.val_max_files<=0 to disable this cap and use full val split.
            val_max_files = int(config.get('val_max_files', 50))
            if val_max_files > 0:
                print(f"[{mode}] Limiting validation set to {val_max_files} files for efficiency.")
                file_list = file_list[:val_max_files]
            else:
                print(f"[{mode}] Using full validation split (no file cap).")
        
        # 2. Initialize Dataset
        return TUEGDataset(file_list=file_list, **config)
    elif name == 'TUAB':
        from .tuab import TUABDataset, get_tuab_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files using reference logic
        file_list = get_tuab_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
        
        # 2. Initialize Dataset
        return TUABDataset(file_list=file_list, **config)
    elif name == 'TUEP':
        from .tuep import TUEPDataset, get_tuep_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files
        file_list = get_tuep_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
            
        # 2. Initialize Dataset
        return TUEPDataset(file_list=file_list, **config)
    elif name == 'TUEV':
        from .tuev import TUEVDataset, get_tuev_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files
        file_list = get_tuev_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
            
        # 2. Initialize Dataset
        return TUEVDataset(file_list=file_list, **config)
    elif name == 'TUSZ':
        from .tusz import TUSZDataset, get_tusz_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files
        file_list = get_tusz_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 10 files.")
            file_list = file_list[:10]
            
        # 2. Initialize Dataset
        return TUSZDataset(file_list=file_list, mode=mode, **config)
    elif name == 'SEED':
        from .seed import SEEDDataset, get_seed_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files
        file_list = get_seed_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 5 files.")
            file_list = file_list[:5]
            
        # 2. Initialize Dataset
        return SEEDDataset(file_list=file_list, mode=mode, **config)
    elif name == 'SEEDIV':
        from .seed_iv import SEEDIVDataset, get_seed_iv_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        file_list = get_seed_iv_file_list(dataset_dir, mode, seed=seed)
        
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 5 files.")
            file_list = file_list[:5]
            
        return SEEDIVDataset(file_list=file_list, mode=mode, **config)
    elif name == 'BCIC2A':
        from .bcic2a import BCIC2ADataset, get_bcic2a_file_list
        dataset_dir = config.get('dataset_dir')
        seed = config.get('seed', 42)
        
        # 1. Get split files
        file_list = get_bcic2a_file_list(dataset_dir, mode, seed=seed)
        
        # Handle Tiny Mode
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 5 files.")
            file_list = file_list[:5]
            
        # 2. Initialize Dataset
        return BCIC2ADataset(file_list=file_list, mode=mode, **config)
    elif name == 'JOINT':
        from .joint import JointDataset, get_joint_file_list
        # For JOINT, dataset_dir can point to the data.csv or be ignored if data_csv is provided separately
        # Support passing a raw list of paths via dataset_dir (hack for sequential training)
        # Or support data_csv as a list?
        
        data_csv = config.get('data_csv', 'datasets/data.csv')
        
        # New Feature: If data_csv is NOT a file but looks like a directory path, 
        # treat it as a single-dataset JOINT mode (Sequential Training)
        # This allows passing a direct path like "/path/to/SEED" instead of creating a temp csv.
        if data_csv and not os.path.isfile(data_csv) and os.path.isdir(data_csv):
             print(f"[JOINT] Detected directory path in data_csv. Switching to Single-Dir Joint Mode: {data_csv}")
             
             dpath = data_csv
             import glob
             import numpy as np
             files = sorted(glob.glob(os.path.join(dpath, '**', '*.h5'), recursive=True))
             if not files:
                 files = sorted(glob.glob(os.path.join(dpath, '*.h5')))
                 
             print(f"Dataset {os.path.basename(dpath)}: Found {len(files)} files.")
             
             seed = config.get('seed', 42)
             val_split = config.get('val_split', 0.1)
             
             rng = np.random.RandomState(seed)
             rng.shuffle(files)
             n_val = max(1, int(len(files) * val_split))
             n_train = len(files) - n_val
             
             if mode == 'train':
                 file_list = files[:n_train]
             elif mode == 'val':
                 file_list = files[n_train:]
             else:
                 file_list = files # Test or all
                 
        else:
            # Standard CSV mode
            seed = config.get('seed', 42)
            val_split = config.get('val_split', 0.1)
            file_list = get_joint_file_list(data_csv, mode, seed=seed, val_split=val_split)
        
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 20 files.")
            file_list = file_list[:20]
            
        return JointDataset(file_list=file_list, **config)
    elif name in ('SleepEDF_full', 'Physionet_MI', 'Workload', 'MDD', 'AD65', 'SEED_downstream'):
        from .generic_sub_h5 import GenericSubH5Dataset, get_sub_h5_file_list
        dataset_dir = config.get('dataset_dir')
        seed_val = config.get('seed', 42)

        # Per-dataset defaults
        _DATASET_DEFAULTS = {
            'SleepEDF_full': {'input_size': 200, 'patch_size': 200, 'num_channels': None, 'test_ratio': 0.2, 'label_offset': 0},
            'Physionet_MI':  {'input_size': 200, 'patch_size': 200, 'num_channels': None, 'test_ratio': 0.2, 'label_offset': 0},
            'Workload':      {'input_size': 200, 'patch_size': 200, 'num_channels': None, 'test_ratio': 0.2, 'label_offset': 0},
            'MDD':           {'input_size': 200, 'patch_size': 200, 'num_channels': None, 'test_ratio': 0.2, 'label_offset': -1},
            'AD65':          {'input_size': 200, 'patch_size': 200, 'num_channels': None, 'test_ratio': 0.2, 'label_offset': -1},
            'SEED_downstream': {'input_size': 400, 'patch_size': 200, 'num_channels': 62, 'test_ratio': 0.2, 'label_offset': 0},
        }
        defaults = _DATASET_DEFAULTS.get(name, {})
        input_size  = int(config.get('input_size',  defaults.get('input_size', 200)))
        patch_size  = int(config.get('patch_size',  defaults.get('patch_size', 200)))
        num_channels = config.get('num_channels', defaults.get('num_channels', None))
        test_ratio   = float(config.get('test_ratio', defaults.get('test_ratio', 0.2)))
        label_offset = int(config.get('label_offset', defaults.get('label_offset', 0)))

        # Use dataset-specific cache to avoid collisions
        cache_path = config.get(
            'index_cache_path',
            f'/tmp/dataset_index_{name.lower()}.json'
        )

        file_list = get_sub_h5_file_list(
            dataset_dir, mode=mode, seed=seed_val, test_ratio=test_ratio
        )

        if config.get('tiny', False):
            print(f'[{mode}] Tiny mode: limiting to 5 files.')
            file_list = file_list[:5]

        return GenericSubH5Dataset(
            file_list=file_list,
            input_size=input_size,
            patch_size=patch_size,
            num_channels=num_channels,
            cache_path=cache_path,
            mode=mode,
            drop_bad_samples=bool(config.get('drop_bad_samples', True)),
            label_offset=label_offset,
        )
    else:
        raise ValueError(f"Dataset {name} not supported")

def build_dataloader(name, config, mode='train'):
    dataset = build_dataset(name, config, mode)
    
    collate_fn = getattr(dataset, 'collate', None)
    
    # Shuffle for train AND val (as per cbramod_tuab reference for intra-epoch val)
    # Test usually remains unshuffle, but val should be shuffled for random sampling during intra-validation
    shuffle = (mode == 'train' or mode == 'val')
    
    base_num_workers = int(config.get('num_workers', 4))
    if mode == 'train':
        num_workers = int(config.get('train_num_workers', base_num_workers))
    elif mode == 'val':
        num_workers = int(config.get('val_num_workers', min(4, base_num_workers)))
    else:
        num_workers = int(config.get('test_num_workers', min(2, base_num_workers)))

    if mode == 'train':
        persistent_workers = bool(config.get('persistent_workers', True)) and num_workers > 0
    elif mode == 'val':
        persistent_workers = bool(config.get('val_persistent_workers', False)) and num_workers > 0
    else:
        persistent_workers = bool(config.get('test_persistent_workers', False)) and num_workers > 0

    if num_workers > 0:
        if mode == 'train':
            prefetch_factor = int(config.get('prefetch_factor', 4))
        elif mode == 'val':
            prefetch_factor = int(config.get('val_prefetch_factor', 2))
        else:
            prefetch_factor = int(config.get('test_prefetch_factor', 2))
    else:
        prefetch_factor = None

    if mode == 'train':
        pin_memory = bool(config.get('pin_memory', True))
    elif mode == 'val':
        pin_memory = bool(config.get('val_pin_memory', False))
    else:
        pin_memory = bool(config.get('test_pin_memory', False))

    timeout_key = f'{mode}_timeout'
    timeout = int(config.get(timeout_key, config.get('loader_timeout', 0)))

    sampler = None
    if (
        name == 'TUEG' and
        (mode == 'train' or mode == 'val') and
        bool(config.get('io_group_shuffle', True)) and
        hasattr(dataset, 'build_grouped_sampler')
    ):
        sampler = dataset.build_grouped_sampler(
            batch_size=config.get('batch_size', 32),
            seed=config.get('seed', 42),
            chunk_size=config.get('io_group_chunk_size', None)
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        timeout=timeout,
        worker_init_fn=_worker_init_disable_h5_locking if name == 'TUEG' else None,
    )
