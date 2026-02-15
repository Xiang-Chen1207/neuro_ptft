import os
import glob
import h5py
import multiprocessing as mp
from functools import partial
import time

def count_segments_in_file(h5_path):
    """Count segments in a single H5 file."""
    try:
        count = 0
        with h5py.File(h5_path, 'r') as f:
            # Structure is typically: group -> segment_name -> eeg_data
            # Or group -> segment
            # Let's inspect the structure. Usually it's:
            # root -> trial_X -> segment_Y
            
            for trial_name in f.keys():
                trial_group = f[trial_name]
                if isinstance(trial_group, h5py.Group):
                    # Count segments inside trial
                    # Segments usually start with 'segment_' or similar, 
                    # but we can just count groups that have 'eeg_data' dataset or just count keys?
                    # The data_loader iterates: for segment_name in trial_group
                    count += len(trial_group.keys())
        return count
    except Exception as e:
        # print(f"Error reading {h5_path}: {e}")
        return 0

def count_total_segments(dataset_dir):
    print(f"Scanning for H5 files in {dataset_dir}...")
    h5_files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    print(f"Found {len(h5_files)} H5 files.")
    
    start_time = time.time()
    
    # Parallel counting
    n_workers = min(mp.cpu_count(), 20) # IO bound, can use more workers
    print(f"Counting segments using {n_workers} workers...")
    
    total_segments = 0
    with mp.Pool(n_workers) as pool:
        counts = pool.map(count_segments_in_file, h5_files)
        total_segments = sum(counts)
        
    elapsed = time.time() - start_time
    
    print(f"\nTotal Segments: {total_segments}")
    print(f"Time taken: {elapsed:.2f} seconds")
    
    # Calculate total processing time estimate
    # Based on our benchmark: ~1.2 seg/s (slowest feature)
    throughput = 1.2
    est_seconds = total_segments / throughput
    est_hours = est_seconds / 3600
    est_days = est_hours / 24
    
    print(f"\nEstimated processing time (Connectivity @ 1.2 seg/s):")
    print(f"{est_hours:.2f} hours")
    print(f"{est_days:.2f} days")

if __name__ == "__main__":
    DATASET_DIR = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset"
    count_total_segments(DATASET_DIR)
