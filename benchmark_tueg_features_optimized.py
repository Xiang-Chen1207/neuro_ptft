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

def process_segment_wrapper(args):
    """
    Wrapper for multiprocessing.
    args: (segment_data, feature_group, config_dict, microstate_analyzer)
    """
    segment_data, feature_group, config_dict, microstate_centroids = args
    
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
            
    # Initialize extractor (use_gpu is in config)
    # n_jobs=1 because we are already in a worker process
    extractor = FeatureExtractor(config=config, n_jobs=1)
    
    # Rebuild microstate analyzer if needed
    ms_analyzer = None
    if microstate_centroids is not None:
        from eeg_feature_extraction.features.microstate import MicrostateAnalyzer
        ms_analyzer = MicrostateAnalyzer(n_states=4, use_gpu=config.use_gpu)
        ms_analyzer.centroids = microstate_centroids

    try:
        start_time = time.time()
        
        # Ensure 60s length (repeat if necessary)
        # 60s * 200Hz = 12000 samples
        target_len = int(60 * config.sampling_rate)
        current_len = segment_data.shape[1]
        
        if current_len < target_len:
            # Repeat
            repeats = int(np.ceil(target_len / current_len))
            data_long = np.tile(segment_data, (1, repeats))
            data_final = data_long[:, :target_len]
        else:
            data_final = segment_data[:, :target_len]
            
        kwargs = {}
        if feature_group == 'microstate' and ms_analyzer:
            kwargs['microstate_analyzer'] = ms_analyzer
        
        # 显式跳过慢速熵特征
        if feature_group == 'complexity':
            kwargs['skip_slow_entropy'] = True

        _ = extractor.extract_features(
            data_final, 
            feature_groups=[feature_group],
            **kwargs
        )
        
        elapsed = time.time() - start_time
        return elapsed
    except Exception as e:
        # print(f"Error: {e}")
        return None

def benchmark_features_optimized(h5_path, max_segments=100, timeout_per_seg=30.0):
    print(f"\nBenchmarking feature extraction (Optimized: GPU + MP)")
    print(f"Target: {max_segments} segments, 60s each")
    print(f"Timeout: {timeout_per_seg}s per segment")
    
    # 1. Load Data
    loader = EEGDataLoader(h5_path)
    subject_info = loader.get_subject_info()
    print(f"Subject: {subject_info.subject_id}, Sampling Rate: {subject_info.sampling_rate} Hz, Channels: {subject_info.n_channels}")
    
    # Get segments
    segments_data = []
    print("Loading raw segments...")
    for i, (trial_name, segment_name, segment) in enumerate(loader.iter_segments()):
        segments_data.append(segment.eeg_data)
        if len(segments_data) >= max_segments:
            break
            
    if not segments_data:
        print("No data found.")
        return

    # 2. Setup Configuration (Enable GPU)
    config = Config()
    config.sampling_rate = subject_info.sampling_rate
    config.use_gpu = True # Enable GPU
    # Sync channels
    if subject_info.channel_names:
        config.update_from_electrode_names(subject_info.channel_names)
    
    # Prepare config dict for workers
    extractor_main = FeatureExtractor(config) # Just to get helper methods
    config_dict = extractor_main._get_config_dict()
    config_dict['use_gpu'] = True # Ensure GPU is ON

    # 3. Pre-compute Microstate Template (Main Process, GPU)
    # We use the first few segments to generate template
    print("\nPre-computing microstate template (GPU)...")
    microstate_centroids = None
    try:
        # Use MicrostateAnalyzer manually
        from eeg_feature_extraction.features.microstate import MicrostateAnalyzer
        
        analyzer = MicrostateAnalyzer(n_states=4, use_gpu=True)
        # Use first 10 segments
        template_segments = segments_data[:10]
        analyzer.fit_from_segments(template_segments)
        microstate_centroids = analyzer.centroids
        print("Microstate template generated.")
    except Exception as e:
        print(f"Microstate template generation failed: {e}")

    # 4. Benchmark Loop
    feature_groups = [
        'complexity', 
        'connectivity', 
        'network', 
        'de_features',
        'microstate'
    ]
    
    results = {}
    
    # Use ProcessPoolExecutor
    n_workers = min(10, mp.cpu_count()) 
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"\nUsing {n_workers} workers.")
    
    for group in feature_groups:
        print(f"\nTesting group: {group} ...")
        
        if group == 'microstate' and microstate_centroids is None:
            print("Skipping microstate (no template).")
            continue
            
        # Prepare tasks
        tasks = []
        for seg in segments_data:
            tasks.append((seg, group, config_dict, microstate_centroids))
            
        # Run
        ts_start = time.time()
        completed_times = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_segment_wrapper, t) for t in tasks]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    completed_times.append(res)
                
        total_wall_time = time.time() - ts_start
        
        if not completed_times:
            print(f"  Failed (0 success)")
            results[group] = None
            continue
            
        avg_time = np.mean(completed_times)
        max_time = np.max(completed_times)
        
        print(f"  Avg Time per segment (internal): {avg_time*1000:.2f} ms")
        print(f"  Max Time per segment (internal): {max_time*1000:.2f} ms")
        print(f"  Total Wall Time (100 segs, {n_workers} workers): {total_wall_time:.2f} s")
        print(f"  Effective Throughput: {len(completed_times)/total_wall_time:.2f} seg/s")
        
        if avg_time > timeout_per_seg:
            print(f"  WARNING: Exceeded {timeout_per_seg}s limit.")
            
        results[group] = {
            'avg_ms': avg_time * 1000,
            'max_ms': max_time * 1000,
            'throughput': len(completed_times)/total_wall_time
        }

    # Final Summary
    print(f"\n\n=== Final Optimized Benchmark Results ({len(segments_data)} segments, 60s each) ===")
    print(f"{'Feature Group':<20} | {'Avg (ms)':<10} | {'Throughput (seg/s)':<20} | {'Est. 1M Segs (hours)':<20}")
    print("-" * 80)
    
    for group in feature_groups:
        res = results.get(group)
        if res:
            est_hours = (1_000_000 / res['throughput']) / 3600
            print(f"{group:<20} | {res['avg_ms']:<10.2f} | {res['throughput']:<20.2f} | {est_hours:<20.2f}")
        else:
            print(f"{group:<20} | Failed")

if __name__ == "__main__":
    DATASET_DIR = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset"
    try:
        h5_file = find_first_h5_file(DATASET_DIR)
        benchmark_features_optimized(h5_file, max_segments=100)
    except Exception as e:
        print(f"Benchmark failed: {e}")
