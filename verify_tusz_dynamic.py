import sys
import os
import torch
import numpy as np
sys.path.append(os.getcwd())

from datasets.tusz import TUSZDataset, get_tusz_file_list
from experiments.tusz_full_ft.model_dynamic import CBraModWrapperDynamic

def test_dataset():
    print("Testing TUSZDataset with dynamic_length=True...")
    # Using the path found in previous configs
    dataset_dir = '/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure'
    
    if not os.path.exists(dataset_dir):
        # Try finding where dataset might be from the user's environment or files I saw
        # The user's prompt mentioned /vePFS-0x0d/home/cx/ptft/datasets/tusz.py
        # But real data path might be elsewhere.
        # I'll check if I can find any h5 files to test.
        # For now, just print warning.
        print(f"Warning: Dataset dir {dataset_dir} not found. Skipping dataset integration test.")
        return

    try:
        file_list = get_tusz_file_list(dataset_dir, mode='train', seed=42)
    except Exception as e:
        print(f"Error getting file list: {e}")
        return

    if not file_list:
        print("No files found.")
        return
        
    # Test with a small subset
    dataset = TUSZDataset(file_list[:5], dynamic_length=True)
    print(f"Dataset length: {len(dataset)}")
    
    for i in range(min(3, len(dataset))):
        try:
            data, label = dataset[i]
            print(f"Sample {i}: Shape {data.shape}, Label {label}")
            # Expected shape: (19, N, 200)
            assert data.dim() == 3
            assert data.shape[0] == 19 # Channel count
            assert data.shape[2] == 200 # Patch size
            print("  -> Valid shape")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")

def test_model_dynamic():
    print("\nTesting CBraModWrapperDynamic...")
    config = {
        'model': {
            'in_dim': 200,
            'd_model': 200,
            'dim_feedforward': 800,
            'seq_len': 10,
            'n_layer': 2, # Small for speed
            'nhead': 4,
            'dropout': 0.1,
            'num_classes': 13,
            'head_type': 'pooling'
        },
        'task_type': 'classification'
    }
    
    try:
        model = CBraModWrapperDynamic(config)
        model.eval()
        
        # Create fake batch with padding
        # Batch size 2. 
        # Sample 1: 5 patches
        # Sample 2: 3 patches (padded to 5)
        
        B, C, N, P = 2, 19, 5, 200
        x = torch.randn(B, C, N, P)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[1, 3:] = True # Pad last 2 patches of 2nd sample
        
        print(f"Input: {x.shape}, Mask: {mask.shape}")
        
        with torch.no_grad():
            out = model(x, mask=mask)
        
        print(f"Output: {out.shape}")
        assert out.shape == (B, 13)
        print("  -> Forward pass successful")
    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_model_dynamic()
    test_dataset()
