import os
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Add current directory to path to ensure we can import eeg_feature_extraction
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.data_loader import EEGDataLoader
from eeg_feature_extraction.feature_extractor import FeatureExtractor
from eeg_feature_extraction.config import Config

def find_first_h5_file(dataset_dir):
    """Find the first H5 file in the dataset directory."""
    print(f"Searching for H5 files in {dataset_dir}...")
    files = glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True)
    if not files:
        raise FileNotFoundError(f"No H5 files found in {dataset_dir}")
    print(f"Found {len(files)} files. Using first one: {files[0]}")
    return files[0]

def benchmark_features(h5_path, max_segments=100):
    print(f"\nBenchmarking feature extraction on: {h5_path}")
    print(f"Limit: {max_segments} segments")
    
    # Initialize Loader
    loader = EEGDataLoader(h5_path)
    subject_info = loader.get_subject_info()
    print(f"Subject: {subject_info.subject_id}, Sampling Rate: {subject_info.sampling_rate} Hz, Channels: {subject_info.n_channels}")
    
    # Initialize Extractor
    extractor = FeatureExtractor(config=None, n_jobs=1)
    extractor._sync_config_with_subject_info(subject_info)
    extractor._rebuild_computers()
    
    # Get segments
    all_segments = []
    print("Loading segments...")
    for i, (trial_name, segment_name, segment) in enumerate(loader.iter_segments()):
        all_segments.append(segment)
        if len(all_segments) >= max_segments:
            break
    
    print(f"Loaded {len(all_segments)} segments for benchmarking.")
    if len(all_segments) == 0:
        print("No segments found in this file.")
        return

    # Define feature groups containing NEW features (based on New_Features_Analysis.md)
    feature_groups = [
        'frequency_domain', 
        'complexity', 
        'connectivity', 
        'network', 
        'composite',
        'de_features',
        'microstate'
    ]
    
    print("\nBenchmarking the following groups (containing new features):")
    print(", ".join(feature_groups))
    
    # Pre-compute microstate template if needed
    microstate_analyzer = None
    if 'microstate' in feature_groups:
        print("\nPre-computing microstate template (using first 10 segments)...")
        try:
            t0 = time.time()
            microstate_analyzer = extractor._compute_microstate_template(
                loader, verbose=False, segments_per_trial=10, use_gpu=False
            )
            print(f"Microstate template generation took: {time.time() - t0:.4f}s")
        except Exception as e:
            print(f"Microstate template generation failed: {e}")
            # If generation fails, we might skip microstate benchmarking or let it fail in loop
            # But better to just remove it to avoid noise
            if 'microstate' in feature_groups:
                feature_groups.remove('microstate')

    # Benchmark Loop
    stats = defaultdict(list)
    
    print(f"\nStarting benchmark...")
    
    # Warmup
    if len(all_segments) > 0:
        try:
            _ = extractor.extract_features(all_segments[0].eeg_data, feature_groups=['frequency_domain'])
        except:
            pass
    
    for i, segment in enumerate(all_segments):
        # Test each feature group individually
        for group in feature_groups:
            try:
                start_time = time.time()
                
                kwargs = {}
                if group == 'microstate' and microstate_analyzer:
                    kwargs['microstate_analyzer'] = microstate_analyzer
                
                _ = extractor.extract_features(
                    segment.eeg_data, 
                    feature_groups=[group],
                    **kwargs
                )
                
                elapsed = (time.time() - start_time) * 1000 # ms
                stats[group].append(elapsed)
                
            except Exception as e:
                # print(f"Error computing {group}: {e}")
                pass
        
        # Simple progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(all_segments)} segments...")

    # Final Report
    print(f"\n\n=== Final Benchmark Results ({len(all_segments)} segments) ===")
    print(f"{'Feature Group':<20} | {'Avg Time (ms)':<15} | {'Std Dev':<10} | {'Min':<10} | {'Max':<10}")
    print("-" * 75)
    
    total_avg_time = 0
    for group in feature_groups:
        times = stats[group]
        if times:
            avg_t = np.mean(times)
            std_t = np.std(times)
            min_t = np.min(times)
            max_t = np.max(times)
            print(f"{group:<20} | {avg_t:<15.2f} | {std_t:<10.2f} | {min_t:<10.2f} | {max_t:<10.2f}")
            total_avg_time += avg_t
        else:
            print(f"{group:<20} | Failed")
            
    print("-" * 75)
    print(f"Total time per segment (sum of averages): {total_avg_time:.2f} ms")

if __name__ == "__main__":
    # Path from pretrain.yaml
    DATASET_DIR = "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset"
    
    try:
        h5_file = find_first_h5_file(DATASET_DIR)
        benchmark_features(h5_file, max_segments=100)
    except Exception as e:
        print(f"Benchmark failed: {e}")
