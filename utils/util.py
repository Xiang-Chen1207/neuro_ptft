import os
import random
import signal
import time
import datetime
import json
from collections import defaultdict, deque
import wandb

import numpy as np
import torch
import torch.distributed as dist

def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask

def to_tensor(array):
    return torch.from_numpy(array).float()

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f}"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
        ]
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

class WandbLogger:
    def __init__(self, config):
        self.active = False
        if not is_main_process():
            return

        enable_wandb = config.get('enable_wandb', False)
        if isinstance(enable_wandb, str):
            enable_wandb = enable_wandb.strip().lower() in {"1", "true", "yes", "y", "on"}
        if not enable_wandb:
            return

        # Prioritize environment variable for security
        # If api_key is in config, use it but warn about security risk
        if config.get('api_key'):
            if os.environ.get('WANDB_API_KEY'):
                print("Warning: WANDB_API_KEY found in both environment variable and config. Using environment variable.")
            else:
                print("Warning: api_key found in config file. For security, consider using WANDB_API_KEY environment variable instead.")
                os.environ['WANDB_API_KEY'] = config['api_key']

        mode = config.get('wandb_mode') or os.environ.get('WANDB_MODE')
        if isinstance(mode, str):
            mode = mode.strip().lower()

        netrc_path = os.path.expanduser("~/.netrc")
        has_key = bool(os.environ.get("WANDB_API_KEY")) or bool(config.get("api_key"))
        if not has_key and not os.path.exists(netrc_path):
            mode = "disabled"

        if mode in {"disabled"}:
            os.environ["WANDB_MODE"] = "disabled"
            return

        try:
            wandb.init(
                project=config.get('project', 'ptft-project'),
                config=config,
                entity=config.get('entity'),
                mode=mode,
            )
            self.active = True
        except Exception as e:
            print(f"W&B disabled due to init failure: {e}")
            self.active = False
    
    def log(self, data, step=None):
        if self.active:
            wandb.log(data, step=step)
    
    def log_image(self, images, step=None):
        if self.active:
            wandb.log(images, step=step)
            
    def finish(self):
        if self.active:
            wandb.finish()

def calc_regression_metrics(preds, target, chunk_size=256):
    """
    Memory-efficient PyTorch implementation of PCC and R2 using chunking.
    Inputs: preds (B, D), target (B, D)
    Returns: {'pcc': float, 'r2': float, 'r2_per_channel': tensor}
    """
    # Ensure inputs are float
    preds = preds.float()
    target = target.float()
    
    B = preds.shape[0]
    D = preds.shape[1]
    
    # Accumulators for Global R2
    ss_res_accum = torch.zeros(D, device=preds.device)
    ss_tot_accum = torch.zeros(D, device=preds.device)
    
    # Accumulators for Mean PCC
    pcc_sum = 0.0
    
    # Process in chunks to save memory
    num_chunks = (B + chunk_size - 1) // chunk_size
    
    # Global mean for R2 (needs to be computed over full batch first, or estimated?
    # Strictly speaking, R2 SS_tot requires global mean. 
    # Computing global mean is cheap (B*D sum), let's do it first.
    # But wait, large reduction might also be heavy? 
    # No, sum reduction is much cheaper than pairwise operations.
    # But to be super safe, let's chunk mean calculation too if needed.
    # For now, let's assume B*D sum fits in memory (it's much smaller than intermediate tensors).
    target_mean = torch.mean(target, dim=0) # (D,)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, B)
        
        p_chunk = preds[start_idx:end_idx]
        t_chunk = target[start_idx:end_idx]
        
        # --- PCC Calculation ---
        # PCC(x, y) = Cov(x, y) / (Std(x) * Std(y))
        # Row-wise mean
        vx = p_chunk - torch.mean(p_chunk, dim=1, keepdim=True)
        vy = t_chunk - torch.mean(t_chunk, dim=1, keepdim=True)
        
        cost = torch.sum(vx * vy, dim=1)
        
        std_x = torch.sqrt(torch.sum(vx ** 2, dim=1)) + 1e-8
        std_y = torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-8
        
        chunk_pcc = cost / (std_x * std_y)
        pcc_sum += torch.sum(chunk_pcc).item()
        
        # --- R2 Calculation ---
        # R2 = 1 - SS_res / SS_tot
        ss_res_accum += torch.sum((t_chunk - p_chunk) ** 2, dim=0)
        ss_tot_accum += torch.sum((t_chunk - target_mean) ** 2, dim=0)
        
        # Free memory
        del vx, vy, cost, std_x, std_y, chunk_pcc, p_chunk, t_chunk
    
    mean_pcc = pcc_sum / B
    
    # Final R2
    # Robust R2: Mask out channels with near-zero variance to avoid explosion
    # Aggressive Threshold: 1e-2 (Std < 0.1) to ignore flat channels
    valid_mask = ss_tot_accum > 1e-2
    r2_per_channel = torch.zeros(D, device=preds.device)
    
    if valid_mask.any():
        r2_vals = 1 - ss_res_accum[valid_mask] / ss_tot_accum[valid_mask]
        # Robustness: Clamp R2 to avoid negative explosion affecting global average
        # R2 < -100 is effectively "garbage", no need to distinguish from -10000
        r2_vals = torch.clamp(r2_vals, min=-100.0, max=1.0)
        
        r2_per_channel[valid_mask] = r2_vals
        # Only average over valid channels
        mean_r2 = r2_per_channel[valid_mask].mean().item()
    else:
        mean_r2 = 0.0
    
    return {'pcc': mean_pcc, 'r2': mean_r2, 'r2_per_channel': r2_per_channel.detach()}

def visualize_eeg_batch(x, x_hat, mask, sample_idx=0):
    """
    Create a matplotlib figure comparing original and reconstructed EEG.
    x, x_hat: (B, C, N, P) or (B, C, S)
    mask: (B, C, N) - 1 if masked
    """
    import matplotlib.pyplot as plt
    
    # Handle data shapes: Flatten patch dim if necessary
    # (B, C, N, P) -> (B, C, N*P)
    if x.ndim == 4:
        B, C, N, P = x.shape
        x_flat = x.view(B, C, -1).detach().cpu().numpy()
        x_hat_flat = x_hat.view(B, C, -1).detach().cpu().numpy()
        
        # Handle Mask: (B, C, N) -> (B, C, N*P)
        if mask is not None:
            if mask.ndim == 3: # (B, C, N)
                # Check if N matches
                if mask.shape[2] == N:
                    # mask (B, C, N) -> unsqueeze -> (B, C, N, 1)
                    # repeat(1, 1, 1, P) -> (B, C, N, P)
                    mask_flat = mask.unsqueeze(-1).repeat(1, 1, 1, P).view(B, C, -1).detach().cpu().numpy()
                else:
                    # Mismatch, maybe mask is already flat or different shape?
                    # Just resize to match x_flat for visualization purposes (nearest)
                    # For now, let's just print warning and ignore mask viz
                    print(f"Viz Warning: Mask shape {mask.shape} does not match N={N}")
                    mask_flat = np.zeros_like(x_flat)
            elif mask.ndim == 2: # (B, N) or (B, S)
                 # Assume (B, N) broadcast to C
                 mask_expanded = mask.unsqueeze(1).repeat(1, C, 1) # (B, C, N)
                 # mask_expanded (B, C, N) -> unsqueeze -> (B, C, N, 1)
                 # repeat(1, 1, 1, P) -> (B, C, N, P)
                 mask_flat = mask_expanded.unsqueeze(-1).repeat(1, 1, 1, P).view(B, C, -1).detach().cpu().numpy()
            else:
                mask_flat = np.zeros_like(x_flat)
        else:
            mask_flat = np.zeros_like(x_flat)
            
    else:
        x_flat = x.detach().cpu().numpy()
        x_hat_flat = x_hat.detach().cpu().numpy()
        if mask is not None:
             mask_flat = mask.detach().cpu().numpy()
        else:
             mask_flat = np.zeros_like(x_flat)
        
    # Select sample
    orig = x_flat[sample_idx]
    recon = x_hat_flat[sample_idx]
    m = mask_flat[sample_idx]
    
    # Plot first 3 channels
    num_channels = min(3, orig.shape[0])
    fig, axes = plt.subplots(num_channels, 1, figsize=(10, 2*num_channels), sharex=True)
    if num_channels == 1: axes = [axes]
    
    for i in range(num_channels):
        ax = axes[i]
        # Plot Original (Blue)
        ax.plot(orig[i], label='Original', color='tab:blue', alpha=0.7, linewidth=1)
        # Plot Reconstructed (Orange)
        ax.plot(recon[i], label='Recon', color='tab:orange', alpha=0.7, linewidth=1)
        
        # Highlight Masked Regions
        # Find segments where m[i] == 1
        masked_indices = np.where(m[i] == 1)[0]
        if len(masked_indices) > 0:
            # Simple shading for masked regions
            # To do this cleanly, we shade ranges.
            # Lazy way: scatter plot or fill_between where mask is 1
            ax.fill_between(np.arange(len(m[i])), -5, 5, where=(m[i]==1), 
                            color='gray', alpha=0.2, label='Masked')
        
        ax.set_ylabel(f'Ch {i}')
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')
            
    plt.tight_layout()
    plt.close(fig) # Prevent display
    return fig
