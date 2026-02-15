import sys
import os
import glob
import h5py
import numpy as np

def check_tusz_classes():
    dataset_dir = '/vePFS-0x0d/home/hanrui/ptft_qwen/tuh_seizure_output/TUH_Seizure'
    
    # Find a few h5 files
    search_dir = os.path.join(dataset_dir, 'TUH_Seizure')
    if not os.path.exists(search_dir):
        search_dir = dataset_dir
        
    all_h5_files = sorted(glob.glob(os.path.join(search_dir, '**', '*.h5'), recursive=True))
    all_h5_files = [f for f in all_h5_files if 'sub_' in os.path.basename(f)]
    
    print(f"Found {len(all_h5_files)} files. Checking first 20 for label info...")
    
    labels_found = set()
    label_vec_len = None
    
    for h5_path in all_h5_files[:50]:
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
                            if 'label' in dset.attrs:
                                label_vec = dset.attrs['label']
                                
                                if hasattr(label_vec, '__len__'):
                                    if label_vec_len is None:
                                        label_vec_len = len(label_vec)
                                        print(f"Label vector length detected: {label_vec_len}")
                                    
                                    idx = np.argmax(label_vec)
                                    labels_found.add(idx)
                                else:
                                    labels_found.add(int(label_vec))
        except Exception as e:
            pass
            
    print(f"Unique labels found in subset: {sorted(list(labels_found))}")
    if label_vec_len is not None:
        print(f"Vector length (Num Classes): {label_vec_len}")
    else:
        print(f"Max label found: {max(labels_found) if labels_found else 'None'}")

if __name__ == '__main__':
    check_tusz_classes()
