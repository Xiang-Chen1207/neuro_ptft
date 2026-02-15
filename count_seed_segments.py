
import os
import glob
import h5py
from tqdm import tqdm

def count_segments():
    dataset_dir = '/vePFS-0x0d/home/downstream_data/SEED'
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.h5')))
    
    total_segments = 0
    file_stats = {}
    
    print(f"Found {len(files)} files.")
    
    for file_path in tqdm(files):
        basename = os.path.basename(file_path)
        file_segment_count = 0
        try:
            with h5py.File(file_path, 'r') as f:
                # Structure: trialX / segmentY
                for trial_key in f.keys():
                    if not trial_key.startswith('trial'):
                        continue
                    trial_group = f[trial_key]
                    if isinstance(trial_group, h5py.Group):
                         segments = [k for k in trial_group.keys() if k.startswith('segment')]
                         file_segment_count += len(segments)
        except Exception as e:
            print(f"Error reading {basename}: {e}")
            
        file_stats[basename] = file_segment_count
        total_segments += file_segment_count
        
    print("-" * 30)
    print(f"Total Segments in SEED: {total_segments}")
    print("-" * 30)
    # print("Breakdown per file:")
    # for name, count in file_stats.items():
    #     print(f"  {name}: {count}")

if __name__ == "__main__":
    count_segments()
