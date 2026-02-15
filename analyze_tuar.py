import os
import glob
import h5py
import numpy as np
import collections

def analyze_tuar_files():
    dataset_dir = '/vePFS-0x0d/pretrain-clip/chr/label/tuh_eeg_artifact/hdf5_output/TUH_Artifact'
    
    print(f"Searching in {dataset_dir}...")
    all_h5_files = sorted(glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
    
    if not all_h5_files:
        print("No H5 files found!")
        return

    print(f"Found {len(all_h5_files)} H5 files. Analyzing first 20...")
    
    segment_lengths = []
    num_channels = set()
    label_shapes = set()
    unique_labels = set()
    
    for h5_path in all_h5_files[:20]:
        try:
            with h5py.File(h5_path, 'r') as f:
                for trial_key in f.keys():
                    if not trial_key.startswith('trial'): continue
                    trial_group = f[trial_key]
                    
                    for seg_key in trial_group.keys():
                        if not seg_key.startswith('segment'): continue
                        seg_group = trial_group[seg_key]
                        
                        if 'eeg' in seg_group:
                            dset = seg_group['eeg']
                            
                            # Shape: (Channels, Time) or (Time, Channels)
                            shape = dset.shape
                            
                            # Heuristic: Channels usually ~20, Time >> 20
                            if shape[0] < shape[1]:
                                chans, time = shape[0], shape[1]
                            else:
                                chans, time = shape[1], shape[0]
                                
                            num_channels.add(chans)
                            segment_lengths.append(time)
                            
                            if 'label' in dset.attrs:
                                label_vec = dset.attrs['label']
                                if hasattr(label_vec, '__len__'):
                                    label_shapes.add(len(label_vec))
                                    unique_labels.add(np.argmax(label_vec))
                                else:
                                    label_shapes.add(1)
                                    unique_labels.add(int(label_vec))
                            
                            # Check Channel Names
                            if 'ch_names' in dset.attrs:
                                ch_names = dset.attrs['ch_names']
                                # Handle bytes vs str
                                ch_names = [c.decode('utf-8') if isinstance(c, bytes) else c for c in ch_names]
                                
                                # Check against standard 19
                                target_channels = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
                                # Alternatives
                                alt_map = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
                                
                                present = []
                                missing = []
                                
                                # Normalize ch_names to upper
                                ch_names_upper = [c.upper() for c in ch_names]
                                
                                for tgt in target_channels:
                                    if tgt in ch_names_upper:
                                        present.append(tgt)
                                    elif tgt in alt_map and alt_map[tgt] in ch_names_upper:
                                        present.append(tgt) # Found via alt
                                    # Reverse check (if target is T3 but file has T7)
                                    elif tgt in ['T3', 'T4', 'T5', 'T6']:
                                         # inverse mapping check not easy without full map, 
                                         # but usually we check if standard set is covered.
                                         # Let's just check typical variants
                                         if tgt == 'T3' and 'T7' in ch_names_upper: present.append(tgt)
                                         elif tgt == 'T4' and 'T8' in ch_names_upper: present.append(tgt)
                                         elif tgt == 'T5' and 'P7' in ch_names_upper: present.append(tgt)
                                         elif tgt == 'T6' and 'P8' in ch_names_upper: present.append(tgt)
                                         else:
                                             missing.append(tgt)
                                    else:
                                        missing.append(tgt)
                                
                                if len(missing) == 0:
                                    # print(f"{h5_path}: All 19 channels present.")
                                    pass
                                else:
                                    print(f"{os.path.basename(h5_path)}: Missing {len(missing)} channels: {missing}. Total file channels: {len(ch_names)}")
                                    # print(f"  Available: {ch_names}")

        except Exception as e:
            # print(f"Error reading {h5_path}: {e}")
            pass

    print("\n=== Analysis Results ===")
    print(f"Number of Channels found: {num_channels}")
    
    if segment_lengths:
        print(f"Segment Lengths (samples): Min={min(segment_lengths)}, Max={max(segment_lengths)}, Mean={np.mean(segment_lengths):.2f}")
        # Assuming 200Hz? Need to check if sampling rate is available
        # But usually TUH H5s are 200Hz
        print(f"Segment Lengths (seconds @ 200Hz): Min={min(segment_lengths)/200:.2f}s, Max={max(segment_lengths)/200:.2f}s")
    
    print(f"Label Vector Lengths (Num Classes): {label_shapes}")
    print(f"Unique Labels Found (Indices): {sorted(list(unique_labels))}")

if __name__ == '__main__':
    analyze_tuar_files()
