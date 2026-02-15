import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

def get_h5_files(dataset_path):
    # Support both directory with .h5 files and single .h5 file
    if os.path.isfile(dataset_path):
        return [dataset_path]
    elif os.path.isdir(dataset_path):
        files = sorted(glob(os.path.join(dataset_path, "*.h5")))
        if not files:
            # Try recursive search if no h5 in root
             files = sorted(glob(os.path.join(dataset_path, "**/*.h5"), recursive=True))
        return files
    else:
        return []

def calculate_duration(h5_path, dataset_name):
    total_samples = 0
    fs = None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get sampling rate from attributes
            if 'rsFreq' in f.attrs:
                fs = float(f.attrs['rsFreq'])
            
            # Iterate through trials and segments
            # Structure usually: trialX -> segmentY -> eeg (dataset) or data (dataset)
            # Or sometimes: trialX -> data
            # Based on previous exploration:
            # TUEG: trial0/segment0/eeg (21, 12000), rsFreq=200
            # SEED: trial0/segment0/eeg (62, 400), rsFreq=200
            # MATB: trial0/segment0/eeg (62, 400), rsFreq=200
            # SleepEDF: trial0/segment0/eeg (2, 6000), rsFreq=200
            # SHU_MI: trial0/segment0/eeg (32, 800), rsFreq=200
            
            for trial_key in f.keys():
                trial_grp = f[trial_key]
                if not isinstance(trial_grp, h5py.Group):
                    continue
                    
                for segment_key in trial_grp.keys():
                    segment_node = trial_grp[segment_key]
                    
                    data_node = None
                    if isinstance(segment_node, h5py.Dataset):
                        # Should not happen based on exploration, but just in case
                        data_node = segment_node
                    elif isinstance(segment_node, h5py.Group):
                        if 'eeg' in segment_node:
                            data_node = segment_node['eeg']
                        elif 'data' in segment_node:
                            data_node = segment_node['data']
                    
                    if data_node is not None:
                        # Shape is usually (channels, time)
                        total_samples += data_node.shape[-1]
                        
    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return 0, 0

    if fs is None or fs <= 0:
        return 0, 0
        
    return total_samples, fs

def process_dataset(name, path):
    print(f"Processing {name} from {path}...")
    files = get_h5_files(path)
    if not files:
        print(f"  No .h5 files found for {name}")
        return
    
    total_seconds = 0
    total_files = len(files)
    
    # Process a subset if too many files to save time, or all files for accuracy
    # For now, let's process all files but show progress
    for file_path in tqdm(files, desc=f"Scanning {name}"):
        samples, fs = calculate_duration(file_path, name)
        if fs > 0:
            total_seconds += samples / fs
            
    hours = total_seconds / 3600
    print(f"  {name}: {total_files} files, Total Duration: {total_seconds:.2f} sec ({hours:.2f} hours)")
    return total_seconds

def main():
    datasets = {
        "TUEG": "/vePFS-0x0d/pretrain-clip/output_tuh_full_pipeline/merged_final_dataset",
        "SEED": "/vePFS-0x0d/home/downstream_data/SEED/",
        "MATB": "/pretrain-clip/hdf5_datasets/Workload_MATB",
        "SleepEDF": "/vePFS-0x0d/eeg-data/SleepEDF",
        "SHU_MI": "/vePFS-0x0d/pretrain-clip/chr/test_datasets/datasets/SHU_MI"
    }
    
    grand_total_seconds = 0
    
    for name, path in datasets.items():
        duration = process_dataset(name, path)
        if duration:
            grand_total_seconds += duration
            
    grand_hours = grand_total_seconds / 3600
    print(f"\nGrand Total Duration: {grand_total_seconds:.2f} sec ({grand_hours:.2f} hours)")

if __name__ == "__main__":
    main()
