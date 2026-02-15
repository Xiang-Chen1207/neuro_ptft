import os
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
import concurrent.futures
import warnings
import h5py

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.data_loader import EEGDataLoader
from eeg_feature_extraction.feature_extractor import FeatureExtractor
from eeg_feature_extraction.config import Config, FrequencyBands, ChannelGroups

# Filter warnings
warnings.filterwarnings("ignore")

def find_first_h5_file(dataset_dir):
    """Find the first H5 file in the dataset directory."""
    print(f"Searching for H5 files in {dataset_dir}...")
    files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    if not files:
        raise FileNotFoundError(f"No H5 files found in {dataset_dir}")
    print(f"Found {len(files)} files. Using first one: {files[0]}")
    return files[0]

def process_segment_full_wrapper(args):
    """
    Wrapper for full feature extraction (all groups).
    args: (segment_data, config_dict, feature_groups, kwargs)
    """
    segment_data, config_dict, feature_groups, extra_kwargs = args
    
    # Rebuild config
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
            
    # Restore nested dataclasses
    if isinstance(config.freq_bands, dict):
        try:
            config.freq_bands = FrequencyBands(**config.freq_bands)
        except Exception:
            config.freq_bands = FrequencyBands()

    if isinstance(config.channel_groups, dict):
        try:
            config.channel_groups = ChannelGroups(**config.channel_groups)
        except Exception:
            config.channel_groups = ChannelGroups()
            
    # Initialize extractor
    extractor = FeatureExtractor(config=config, n_jobs=1)
    
    try:
        start_time = time.time()
        
        # We assume microstate analyzer is NOT needed because we are skipping microstate
        # Or if we include it, we need to pass it.
        # But user said "去除microstate".
        
        # Extract features
        # Note: FeatureExtractor.extract_features handles multiple groups
        # We need to pass kwargs to each group computation.
        # FeatureExtractor.extract_features passes **kwargs to all compute methods.
        
        try:
            _ = extractor.extract_features(
                segment_data, 
                feature_groups=feature_groups,
                **extra_kwargs
            )
        except Exception as e:
            # Check for CuPy OOM by string or type if imported
            if "OutOfMemoryError" in str(e) or "CUDA_ERROR_OUT_OF_MEMORY" in str(e):
                # Fallback to CPU if possible (re-init extractor?)
                # Or just skip this segment
                print(f"GPU OOM in worker: {e}. Consider reducing n_workers.")
                return None
            raise e
        
        elapsed = time.time() - start_time
        return elapsed
    except Exception as e:
        print(f"Error inside worker: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_benchmark_scenario(h5_path, n_segments, segment_duration_sec, segment_points):
    print(f"\n--- Scenario: {segment_duration_sec}s segments ({segment_points} points) ---")
    print(f"Target: {n_segments} segments")
    
    # Load Data
    loader = EEGDataLoader(h5_path)
    subject_info = loader.get_subject_info()
    
    # Load raw data
    raw_segments = []
    print("Loading raw segments...")
    # Iterate until we have enough
    # We might need to cycle if file is small
    collected = 0
    
    # To get 1000 segments, we might need multiple files or loop
    # Let's just loop over the generator
    all_available = []
    for _, _, seg in loader.iter_segments():
        all_available.append(seg.eeg_data)
        if len(all_available) >= 50: # Limit memory usage, we will recycle these
            break
            
    if not all_available:
        print("No data found.")
        return

    # Prepare tasks
    # We will reuse the available segments to reach n_segments
    tasks_data = []
    for i in range(n_segments):
        # Pick a segment (round robin)
        base_seg = all_available[i % len(all_available)]
        
        # Resize/Crop/Tile to target length
        current_len = base_seg.shape[1]
        target_len = segment_points
        
        if current_len >= target_len:
            seg_final = base_seg[:, :target_len]
        else:
            # Tile
            repeats = int(np.ceil(target_len / current_len))
            seg_long = np.tile(base_seg, (1, repeats))
            seg_final = seg_long[:, :target_len]
            
        tasks_data.append(seg_final)

    # Config
    config = Config()
    config.sampling_rate = subject_info.sampling_rate
    config.use_gpu = True
    if subject_info.channel_names:
        config.update_from_electrode_names(subject_info.channel_names)
        
    extractor_main = FeatureExtractor(config)
    config_dict = extractor_main._get_config_dict()
    config_dict['use_gpu'] = True
    
    # Feature Groups to run
    # User said: "测试全部特征，去除microstate"
    # So we list all, remove microstate.
    all_groups = [
        'time_domain', 
        'frequency_domain', 
        'complexity', 
        'connectivity', 
        'network', 
        'de_features',
        # 'microstate', # Excluded
        'composite'
    ]
    
    # Kwargs to exclude specific features
    extra_kwargs = {
        'skip_slow_entropy': True,      # Exclude sample/approx entropy
        'skip_correlation': True        # Exclude mean_interchannel_correlation
    }
    
    # Add a fallback for multiprocessing logic to debug failure
    # Why did it fail? "All failed" means result was None.
    # process_segment_full_wrapper returns None on exception.
    # We should print the exception inside the wrapper to debug.
    
    print(f"Feature Groups: {all_groups}")
    print(f"Exclusions: {extra_kwargs}")
    
    # Parallel Execution
    try:
        # Try to get the actual number of usable CPUs (respecting container quotas/affinity)
        n_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback to total CPU count
        n_workers = mp.cpu_count()
    
    # LIMIT n_workers to prevent GPU OOM
    # If we are using GPU, we must be conservative.
    # 26 workers * 100MB+ per context = 2.6GB+, but context overhead can be higher.
    # Also CUDA context creation is expensive.
    # We cap at 12 to be safe for GPU.
    MAX_GPU_WORKERS = 20
    if n_workers > MAX_GPU_WORKERS:
        print(f"Limiting workers to {MAX_GPU_WORKERS} to prevent GPU OOM (Available: {n_workers})")
        n_workers = MAX_GPU_WORKERS
    else:
        if n_workers > 2:
             n_workers -= 1
        
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    print(f"Starting processing with {n_workers} workers (Safe Throttle)...")
    
    # Prepare arguments
    mp_args = [(seg, config_dict, all_groups, extra_kwargs) for seg in tasks_data]
    
    ts_start = time.time()
    completed_times = []
    
    # Increase chunksize to reduce overhead?
    # Or just use map
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_segment_full_wrapper, t) for t in mp_args]
        
        # Progress bar
        done_count = 0
        try:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    completed_times.append(res)
                done_count += 1
                if done_count % 100 == 0:
                    print(f"Processed {done_count}/{n_segments}...")
        except Exception as e:
            print(f"Executor loop error: {e}")
                
    total_wall_time = time.time() - ts_start
    
    # Analysis
    if not completed_times:
        print("All failed.")
        return
        
    avg_time = np.mean(completed_times)
    throughput = len(completed_times) / total_wall_time
    
    print(f"\nResults for {segment_duration_sec}s segments:")
    print(f"  Total Wall Time: {total_wall_time:.2f} s")
    print(f"  Avg Time per Segment (Internal): {avg_time*1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} seg/s")
    print(f"  Est. 1M Segments: {(1_000_000/throughput)/3600:.2f} hours")
    
    # Also write to a summary file
    with open("Comprehensive_Benchmark_Results.txt", "a") as f:
        f.write(f"\n--- {segment_duration_sec}s Segment Benchmark ---\n")
        f.write(f"Total Wall Time: {total_wall_time:.2f} s\n")
        f.write(f"Throughput: {throughput:.2f} seg/s\n")
        f.write(f"Est. 1M Segments: {(1_000_000/throughput)/3600:.2f} hours\n")

def main():
    DATASET_DIR = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset"
    try:
        h5_file = find_first_h5_file(DATASET_DIR)
        
        # 1. Test 60s segments (12000 points)
        # fs = 200 (assumed from previous runs)
        # We can get fs from file but let's assume 200 for calculation logic
        # If actual fs is different, code handles it via config.
        # We pass points = duration * 200
        
        # First verify FS
        loader = EEGDataLoader(h5_file)
        fs = loader.get_subject_info().sampling_rate
        print(f"Sampling Rate: {fs} Hz")
        
        # 60s
        run_benchmark_scenario(h5_file, n_segments=1000, segment_duration_sec=60, segment_points=int(60*fs))
        
        # 10s
        run_benchmark_scenario(h5_file, n_segments=1000, segment_duration_sec=10, segment_points=int(10*fs))
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
