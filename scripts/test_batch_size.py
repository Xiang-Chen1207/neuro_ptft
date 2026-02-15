
import os
import torch
import torch.nn as nn
import argparse
import sys
import yaml
from torch.utils.data import DataLoader, Dataset
import time

# Mock classes to avoid full dependency loading if possible, 
# but best to import actual model to get real memory usage.
sys.path.append(os.getcwd())

try:
    from models.wrapper import CBraModWrapper
except ImportError:
    print("Error: Could not import model. Run this script from project root.")
    sys.exit(1)

def get_max_memory():
    return torch.cuda.max_memory_allocated() / 1024**3

def test_batch_size(batch_size, config, device):
    print(f"\nTesting Batch Size: {batch_size}")
    
    # 1. Create Model
    try:
        model = CBraModWrapper(config).to(device)
        model.train()
    except Exception as e:
        print(f"Failed to create model: {e}")
        return False

    # 2. Create Dummy Batch
    # Simulate JOINT dataset output: (B, C, N, P)
    # Using max typical values to stress test
    # C=62 (common max in your data), N=60 (seq_len), P=200
    C = 62
    N = config['model']['seq_len']
    P = config['model']['in_dim']
    
    # Note: batch_size here is TOTAL batch size. 
    # If using DataParallel, we should divide by num_gpus, but here we test on Single GPU to find per-GPU limit.
    # The user asks "how much can batch_size be", usually referring to the total across all GPUs or per GPU?
    # Usually we want to know max per GPU.
    
    try:
        x = torch.randn(batch_size, C, N, P, device=device)
        mask = torch.zeros(batch_size, C, N, device=device) # No masking (all valid) to maximize computation
        
        # 3. Forward Pass
        torch.cuda.reset_peak_memory_stats()
        start_mem = get_max_memory()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Forward
        outputs = model(x, mask=mask)
        if isinstance(outputs, tuple):
            pred = outputs[0]
        else:
            pred = outputs
            
        # Loss
        if pred is not None:
            loss = pred.mean()
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        end_mem = get_max_memory()
        print(f"Success! Peak Memory: {end_mem:.2f} GB")
        
        del model, x, mask, outputs, optimizer, loss
        torch.cuda.empty_cache()
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM triggered at batch size {batch_size}")
            return False
        else:
            print(f"Runtime Error: {e}")
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain_st_eegformer.yaml")
    parser.add_argument("--start_batch", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_batch", type=int, default=128)
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override with Small config if not already (to match user's latest change)
    # Actually we should use the file as is to test WHAT IS THERE.
    # But just in case, let's print the config key params
    print(f"Model Config: d_model={config['model'].get('d_model')}, layers={config['model'].get('n_layer')}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {torch.cuda.get_device_name(0)}")
    
    # Binary Search or Linear Search?
    # Linear is safer to find exact point.
    
    current_batch = args.start_batch
    max_safe_batch = 0
    
    # Strategy: Doubling until fail, then binary search? 
    # Or just stepping. Let's do doubling first.
    
    print("Phase 1: Doubling Batch Size...")
    while current_batch <= args.max_batch:
        success = test_batch_size(current_batch, config, device)
        if success:
            max_safe_batch = current_batch
            current_batch *= 2
        else:
            break
            
    print(f"\nPhase 2: Fine-tuning between {max_safe_batch} and {current_batch}...")
    low = max_safe_batch
    high = min(current_batch, args.max_batch)
    
    if high > low:
        # Binary search
        while low < high:
            mid = (low + high + 1) // 2
            if mid == low: break # Converged
            
            success = test_batch_size(mid, config, device)
            if success:
                low = mid
            else:
                high = mid - 1
                
    final_max = low
    print(f"\n\n============== RESULT ==============")
    print(f"Max safe batch size PER GPU: {final_max}")
    print(f"If using 4 GPUs, total batch size can be: {final_max * 4}")
    print(f"Recommended safe total batch size (90% capacity): {int(final_max * 4 * 0.9)}")
    print(f"====================================")

if __name__ == "__main__":
    main()
