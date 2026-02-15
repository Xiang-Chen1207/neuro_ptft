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
import traceback

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.data_loader import EEGDataLoader
from eeg_feature_extraction.feature_extractor import FeatureExtractor
from eeg_feature_extraction.config import Config, FrequencyBands, ChannelGroups
from eeg_feature_extraction.psd_computer import PSDComputer

# Import specific feature classes
from eeg_feature_extraction.features.connectivity import ConnectivityFeatures
from eeg_feature_extraction.features.network import NetworkFeatures

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

def profile_connectivity_components(args):
    """
    Profile individual components of Connectivity and Network features.
    args: (segment_data, config_dict)
    """
    segment_data, config_dict = args
    
    # Rebuild config
    config = Config()
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    # Initialize classes
    psd_computer = PSDComputer(config)
    conn_features = ConnectivityFeatures(config)
    net_features = NetworkFeatures(config)
    
    times = {}
    
    try:
        # 1. Coherence Calculation (Matrix)
        t0 = time.time()
        # Compute coherence for alpha band (used by mean_alpha_coherence and network)
        coherence_matrix = psd_computer.compute_coherence(segment_data, band=(8.0, 13.0))
        times['calc_coherence_matrix'] = (time.time() - t0) * 1000
        
        # 2. PLV Calculation (Matrix) - Theta band
        t0 = time.time()
        # Need to call the standalone function or internal method
        # connectivity.py: compute_plv_matrix
        from eeg_feature_extraction.features.connectivity import compute_plv_matrix
        plv_matrix = compute_plv_matrix(segment_data, config.sampling_rate, (4.0, 8.0))
        times['calc_plv_matrix_theta'] = (time.time() - t0) * 1000
        
        # 3. Graph Metrics (Network)
        # Using the coherence matrix from step 1
        t0 = time.time()
        adj_matrix = net_features._threshold_matrix(coherence_matrix)
        times['net_thresholding'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        _ = net_features._compute_clustering_coefficient(adj_matrix)
        times['net_clustering'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        _ = net_features._compute_characteristic_path_length(adj_matrix)
        times['net_path_length'] = (time.time() - t0) * 1000
        
        t0 = time.time()
        _ = net_features._compute_global_efficiency(adj_matrix)
        times['net_efficiency'] = (time.time() - t0) * 1000
        
        # 4. Correlation (Connectivity)
        t0 = time.time()
        _ = conn_features._compute_mean_correlation(segment_data)
        times['conn_correlation'] = (time.time() - t0) * 1000
        
        return times
        
    except Exception as e:
        # print(f"Error: {e}")
        # traceback.print_exc()
        return None

def run_profiling(h5_path, n_segments=10):
    print(f"\nProfiling Connectivity & Network Components")
    print(f"Segments: {n_segments}")
    
    # Load Data
    loader = EEGDataLoader(h5_path)
    subject_info = loader.get_subject_info()
    
    segments_data = []
    for i, (trial_name, segment_name, segment) in enumerate(loader.iter_segments()):
        segments_data.append(segment.eeg_data)
        if len(segments_data) >= n_segments:
            break
            
    # Config
    config = Config()
    config.sampling_rate = subject_info.sampling_rate
    config.use_gpu = True
    if subject_info.channel_names:
        config.update_from_electrode_names(subject_info.channel_names)
        
    extractor_main = FeatureExtractor(config)
    config_dict = extractor_main._get_config_dict()
    config_dict['use_gpu'] = True
    
    # Run profiling
    n_workers = min(5, mp.cpu_count()) # Smaller pool for profiling
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    tasks = [(seg, config_dict) for seg in segments_data]
    
    all_results = defaultdict(list)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(profile_connectivity_components, t) for t in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                for k, v in res.items():
                    all_results[k].append(v)
                    
    # Report
    print(f"\n\n=== Component Profiling Results (Avg ms per segment) ===")
    print(f"{'Component':<30} | {'Avg Time (ms)':<15} | {'Min':<10} | {'Max':<10}")
    print("-" * 75)
    
    # Sort by avg time
    sorted_keys = sorted(all_results.keys(), key=lambda k: np.mean(all_results[k]), reverse=True)
    
    for k in sorted_keys:
        vals = all_results[k]
        print(f"{k:<30} | {np.mean(vals):<15.2f} | {np.min(vals):<10.2f} | {np.max(vals):<10.2f}")

if __name__ == "__main__":
    DATASET_DIR = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset"
    try:
        h5_file = find_first_h5_file(DATASET_DIR)
        run_profiling(h5_file, n_segments=10)
    except Exception as e:
        print(f"Profiling failed: {e}")
