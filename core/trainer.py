import torch
import os
import csv
import yaml
from pathlib import Path
from .loss import LossFactory
from .engine import train_one_epoch, evaluate
from utils.util import WandbLogger

class Trainer:
    def __init__(self, config, model, train_loader, val_loader=None, test_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        if torch.cuda.device_count() > 1 and self.device.type == 'cuda':
             if config.get('task_type') == 'pretraining' and config.get('model', {}).get('name') == 'eegmamba':
                 self._warmup_eegmamba_kernels()
             print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
             self.model = torch.nn.DataParallel(self.model)
        
        self._setup_optimizer()
        self._setup_loss()
        
        self.output_dir = Path(config.get('output_dir', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB
        self.log_writer = WandbLogger(config)

    def _warmup_eegmamba_kernels(self):
        model_cfg = self.config.get('model', {})
        dataset_cfg = self.config.get('dataset', {})
        seq_len = int(model_cfg.get('seq_len', 60))
        patch_size = int(model_cfg.get('in_dim', 200))
        batch_size = int(min(2, dataset_cfg.get('batch_size', 2)))
        ch_num = int(model_cfg.get('warmup_channels', 21))
        
        # Warmup only on the primary device. 
        # Iterating over all devices in the same process can cause Triton context issues.
        device = self.device
        print(f"DEBUG: Warming up eegmamba kernels on {device}...")
        
        # Ensure model is on device (should be already)
        self.model.to(device)
        
        x = torch.zeros(batch_size, ch_num, seq_len, patch_size, device=device, dtype=torch.float32)
        mask = torch.zeros(batch_size, ch_num, seq_len, device=device, dtype=torch.float32)
        
        try:
            out = self.model(x, mask=mask)
            if isinstance(out, tuple):
                out = out[0] if out[0] is not None else out[-1]
            loss = out.float().mean()
            loss.backward()
            if hasattr(self.model, "zero_grad"):
                self.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            print("DEBUG: Warmup successful.")
        except Exception as e:
            print(f"WARNING: Warmup failed with error: {e}. Proceeding anyway...")
            
        # Restore model to device (redundant if we didn't move it)
        self.model.to(self.device)
        
    def _setup_optimizer(self):
        # Use copy() to avoid modifying original config (important for checkpoint saving)
        opt_config = self.config.get('optimizer', {'name': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.05}).copy()
        name = opt_config.pop('name')
        if name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_config)
        elif name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), **opt_config)
        else:
            raise ValueError(f"Optimizer {name} not supported")

        # Scheduler setup - use copy() to preserve original config
        sched_config = self.config.get('scheduler', {'name': 'CosineAnnealingLR', 'T_max': 100}).copy()
        sched_name = sched_config.pop('name', 'CosineAnnealingLR')
        
        if sched_name == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **sched_config)
        elif sched_name == 'StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **sched_config)
        else:
            print(f"Warning: Scheduler {sched_name} not supported/implemented. No scheduler used.")
            self.scheduler = None
        
    def _setup_loss(self):
        loss_name = self.config.get('loss', {}).get('name', 'mse')
        loss_params = self.config.get('loss', {}).get('params', {})
        self.criterion = LossFactory.get(loss_name, **loss_params).to(self.device)
        
    def _truncate_csv_log(self, start_epoch):
        """Truncate CSV log to remove entries >= start_epoch when resuming training."""
        csv_file = self.output_dir / 'log.csv'
        if not csv_file.exists():
            return

        # Read existing data
        with open(csv_file, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = [row for row in reader if int(row.get('epoch', 0)) < start_epoch]

        # Rewrite file with truncated data
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Truncated log.csv to epoch < {start_epoch} ({len(rows)} rows kept)")

    def train(self, resume_path=None):
        epochs = self.config.get('epochs', 10)
        best_metric = -float('inf')
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_path is not None:
            start_epoch, best_metric = self.load_checkpoint(resume_path)
            print(f"Resuming training from epoch {start_epoch}")
            # Truncate CSV log to avoid duplicate/conflicting epoch entries
            self._truncate_csv_log(start_epoch)

        # Calculate val_freq dynamically
        num_steps = len(self.train_loader)
        # Default split is 5 if not specified. If 0, disable intra-epoch val.
        # Use config value if present, otherwise default to 5.
        val_freq_split = self.config.get('val_freq_split')
        if val_freq_split is None:
            val_freq_split = 0

        if val_freq_split > 0:
            val_freq = max(1, int(num_steps / val_freq_split))
            print(f"Validation Frequency set to {val_freq} steps (total steps per epoch: {num_steps}, split: {val_freq_split})")
        else:
            val_freq = -1 # Disabled
            print("Intra-epoch validation disabled (val_freq_split <= 0)")

        feature_loss_weight = self.config.get('loss', {}).get('feature_loss_weight', 0.0)
        use_dynamic_loss = self.config.get('loss', {}).get('use_dynamic_loss', False)

        for epoch in range(start_epoch, epochs):
            train_stats = train_one_epoch(
                self.model, 
                self.train_loader, 
                self.optimizer, 
                self.criterion, 
                self.device, 
                epoch,
                max_norm=self.config.get('clip_grad', 1.0),
                log_writer=self.log_writer,
                val_loader=self.val_loader,
                val_freq=val_freq,
                task_type=self.config.get('task_type', 'classification'),
                feature_loss_weight=feature_loss_weight,
                use_dynamic_loss=use_dynamic_loss
            )
            
            # Use 'loss' from returned stats if available, else None
            print(f"Epoch {epoch}: Train Loss: {train_stats.get('loss', 'N/A')}")
            
            if self.val_loader:
                val_metrics = evaluate(
                    self.model, 
                    self.val_loader, 
                    self.criterion, 
                    self.device, 
                    task_type=self.config.get('task_type', 'classification'),
                    log_writer=self.log_writer,
                    epoch=epoch,
                    header='Val:',
                    verbose=True
                )
                print(f"Epoch {epoch}: Val Metrics: {val_metrics}")
                
                # Log validation metrics to WandB
                if self.log_writer:
                    # Add 'val/' prefix if not present (evaluate might add it? No, evaluate returns raw metric names)
                    # But wait, evaluate logic for 'val_intra_loss' adds prefix?
                    # Let's just add 'val/' prefix here to be safe and consistent
                    wandb_metrics = {}
                    for k, v in val_metrics.items():
                        # Avoid double prefixing if keys already have 'val/' (unlikely from evaluate return)
                        key = f"val/{k}" if not k.startswith('val/') else k
                        wandb_metrics[key] = v
                    self.log_writer.log(wandb_metrics, step=epoch)

                # Checkpoint logic
                metric_key = self.config.get('metric_key', 'acc')
                # Try 'accuracy' if 'acc' is missing (common difference)
                if metric_key not in val_metrics and 'accuracy' in val_metrics:
                    metric_key = 'accuracy'
                
                # Determine metric value (handling minimization for loss)
                if metric_key in val_metrics:
                    val_value = val_metrics[metric_key]
                    # If metric is loss, we want to minimize it. 
                    # Since the logic below maximizes best_metric, we negate loss.
                    if 'loss' in metric_key.lower():
                        current_metric = -val_value
                    else:
                        current_metric = val_value
                else:
                    # Fallback: use negative training loss (minimize train loss)
                    current_metric = -train_stats.get('loss', 0)
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    self.save_checkpoint('best.pth', epoch=epoch, best_metric=best_metric)
            
            # CSV Logging
            log_stats = {'epoch': epoch}
            for k, v in train_stats.items():
                if isinstance(v, (int, float, str, bool)):
                    log_stats[f'train_{k}'] = v
            
            if self.val_loader and 'val_metrics' in locals():
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float, str, bool)):
                        log_stats[f'val_{k}'] = v
            
            csv_file = self.output_dir / 'log.csv'
            file_exists = csv_file.exists()
            
            # Determine fieldnames. If file exists, read the header.
            # If not, use current keys. 
            # Note: This assumes keys don't change. If they might, we need to handle it.
            # For simplicity, we assume consistent keys after epoch 0.
            
            current_keys = sorted(log_stats.keys())
            
            if not file_exists:
                 with open(csv_file, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=current_keys)
                    writer.writeheader()
                    writer.writerow(log_stats)
            else:
                # Read existing data to check for header updates
                with open(csv_file, mode='r', newline='') as f:
                    reader = csv.DictReader(f)
                    # Handle empty file case
                    if reader.fieldnames:
                        header = list(reader.fieldnames)
                        existing_data = list(reader)
                    else:
                        header = []
                        existing_data = []
                
                # Check if we have new keys that aren't in the header
                new_keys = [k for k in log_stats.keys() if k not in header]
                
                if new_keys:
                    print(f"New metrics detected in log: {new_keys}. Updating log.csv structure.")
                    header.extend(new_keys)
                    # Rewrite the entire file with new header
                    with open(csv_file, mode='w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header)
                        writer.writeheader()
                        writer.writerows(existing_data)
                        writer.writerow(log_stats)
                else:
                    # Append to existing file
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header)
                        writer.writerow(log_stats)

            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch=epoch, best_metric=best_metric)
            self.save_checkpoint('latest.pth', epoch=epoch, best_metric=best_metric)

            # Step scheduler at epoch level (not batch level)
            if self.scheduler is not None:
                self.scheduler.step()

        self.log_writer.finish()
            
    def save_checkpoint(self, filename, epoch=None, best_metric=None):
        if not self.config.get('save_checkpoints', True):
            print(f"Skipping checkpoint save (save_checkpoints=false): {filename}")
            return
        path = self.output_dir / filename
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'best_metric': best_metric
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state restored successfully")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}. Continuing with fresh optimizer state.")
        else:
            print("Warning: No optimizer state found in checkpoint. Using fresh optimizer state.")

        # Load scheduler state if available
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state restored successfully")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}. Continuing with fresh scheduler state.")
        elif self.scheduler is not None:
            print("Warning: No scheduler state found in checkpoint. Using fresh scheduler state.")

        # Get epoch and best_metric
        start_epoch = checkpoint.get('epoch', -1) + 1  # Resume from next epoch
        best_metric = checkpoint.get('best_metric', -float('inf'))

        print(f"Resumed from epoch {checkpoint.get('epoch', 'N/A')}, will start at epoch {start_epoch}")
        print(f"Previous best metric: {best_metric}")

        return start_epoch, best_metric
