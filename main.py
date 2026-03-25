import argparse
import yaml
import sys
from core.trainer import Trainer
from models.wrapper import CBraModWrapper
from datasets.builder import build_dataloader

def merge_cfg_from_list(cfg, opts):
    """Merge config from list of keys and values."""
    for opt in opts:
        key, value = opt.split('=', 1)
        keys = key.split('.')
        
        # Parse value
        # Handle lists: [a,b]
        if value.startswith('[') and value.endswith(']'):
            value = value[1:-1].split(',')
            value = [v.strip() for v in value]
            # Try to convert elements to int/float if possible? 
            # For tasks, they are strings. For simplicity, keep strings if quote present?
            # Or just strip quotes.
            value = [v.replace("'", "").replace('"', "") for v in value]
        else:
            # Try int, float, bool
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass # Keep as string
        
        # Traverse dict
        sub_cfg = cfg
        for k in keys[:-1]:
            sub_cfg = sub_cfg.setdefault(k, {})
        sub_cfg[keys[-1]] = value
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Unified Entry Point for PT/FT")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line, e.g. model.pretrain_tasks=['reconstruction']")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny subset of data for debugging")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pth) to resume training from")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    if args.opts:
        print(f"Overriding config with: {args.opts}")
        config = merge_cfg_from_list(config, args.opts)
        
    print(f"Loading config from {args.config}")
    
    # Logic to flatten datasets -> dataset
    if 'datasets' in config:
        selected = config.get('selected_dataset')
        if not selected:
             # Fallback or error
             raise ValueError("Config has 'datasets' but no 'selected_dataset' specified.")
        
        if selected not in config['datasets']:
             raise ValueError(f"Selected dataset '{selected}' not found in 'datasets'.")
             
        print(f"Active Dataset: {selected}")
        config['dataset'] = config['datasets'][selected]

    # Tiny Mode Logic
    if args.tiny:
        print("!!! TINY MODE ACTIVATED: Using small data subset for debugging !!!")
        # Pass a flag to dataset config to limit files
        # We need to modify dataset config. 
        # Since build_dataloader calls Dataset class, we can inject a 'limit_files' param if supported,
        # or we can hack it here if we knew how dataset loads.
        # TUEGDataset takes file_list. builder.py likely globs files.
        # Let's check builder.py. Assuming we can pass 'tiny': True to config['dataset']
        config['dataset']['tiny'] = True

    if 'dataset' not in config:
        raise ValueError("Config must contain 'dataset' section.")
    if 'batch_size' not in config['dataset']:
        raise ValueError("Config dataset must contain 'batch_size'.")
    expected_batch_size = int(config['dataset']['batch_size'])
    print(f"Configured dataset.batch_size={expected_batch_size}")
    
    # 1. Build Datasets
    print("Building datasets...")
    train_loader = build_dataloader(config['dataset']['name'], config['dataset'], mode='train')
    train_batch_size = int(getattr(train_loader, 'batch_size', 0))
    print(f"Train DataLoader batch_size={train_batch_size}")
    if train_batch_size != expected_batch_size:
        raise ValueError(
            f"Batch size mismatch: configured {expected_batch_size}, train loader {train_batch_size}"
        )
    
    # Auto-sync feature_dim from dataset to model config
    # This ensures that if the feature CSV changes (e.g. reduced features), 
    # the model architecture adapts automatically.
    dataset_obj = train_loader.dataset
    if hasattr(dataset_obj, 'dataset'): # Handle Subset/Wrapper if present
        dataset_obj = dataset_obj.dataset
        
    if hasattr(dataset_obj, 'feature_dim') and dataset_obj.feature_dim > 0:
        detected_dim = dataset_obj.feature_dim
        config_dim = config['model'].get('feature_dim', 0)
        
        if detected_dim != config_dim:
            print(f"Warning: Config feature_dim ({config_dim}) does not match dataset feature_dim ({detected_dim}).")
            print(f"Automatically updating config.model.feature_dim to {detected_dim}")
            config['model']['feature_dim'] = detected_dim

    val_loader = None
    # Enable validation for pretraining if requested (val_freq_split > 0)
    # We always build it if not explicitly disabled, as engine handles frequency
    # But to save memory if user doesn't want it, check val_freq_split
    if config.get('val_freq_split', 0) > 0:
        print("Building validation dataset (mini-split for monitoring)...")
        val_loader = build_dataloader(config['dataset']['name'], config['dataset'], mode='val')
        val_batch_size = int(getattr(val_loader, 'batch_size', 0))
        print(f"Val DataLoader batch_size={val_batch_size}")
        if val_batch_size != expected_batch_size:
            raise ValueError(
                f"Batch size mismatch: configured {expected_batch_size}, val loader {val_batch_size}"
            )
    
    # 2. Build Model
    print("Building model...")
    # Inject task type into model config part for wrapper logic
    config['model']['task_type'] = config['task_type']
    model = CBraModWrapper(config)
    
    # 3. Setup Trainer
    print("Setting up trainer...")
    trainer = Trainer(config, model, train_loader, val_loader)
    
    # 4. Run
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
    else:
        print("Starting training from scratch...")
    trainer.train(resume_path=args.resume)

if __name__ == "__main__":
    main()
