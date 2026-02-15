import os
from torch.utils.data import DataLoader

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
            # For validation efficiency, limit to a small subset (Mini-Val)
            # 50 files is enough for metric stability (~few hundred samples)
            print(f"[{mode}] Limiting validation set to 50 files for efficiency.")
            file_list = file_list[:50]
        
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
        data_csv = config.get('data_csv', 'datasets/data.csv')
        seed = config.get('seed', 42)
        val_split = config.get('val_split', 0.1)
        
        file_list = get_joint_file_list(data_csv, mode, seed=seed, val_split=val_split)
        
        if config.get('tiny', False):
            print(f"[{mode}] Tiny mode active: Limiting file list to 20 files.")
            file_list = file_list[:20]
            
        return JointDataset(file_list=file_list, **config)
    else:
        raise ValueError(f"Dataset {name} not supported")

def build_dataloader(name, config, mode='train'):
    dataset = build_dataset(name, config, mode)
    
    collate_fn = getattr(dataset, 'collate', None)
    
    # Shuffle for train AND val (as per cbramod_tuab reference for intra-epoch val)
    # Test usually remains unshuffle, but val should be shuffled for random sampling during intra-validation
    shuffle = (mode == 'train' or mode == 'val')
    
    return DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=shuffle,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', True),
        persistent_workers=config.get('persistent_workers', True),
        prefetch_factor=config.get('prefetch_factor', 4) if config.get('num_workers', 4) > 0 else None
    )
