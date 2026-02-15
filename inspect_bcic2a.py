
import h5py
import numpy as np

file_path = '/vePFS-0x0d/home/downstream_data/BCIC2A/sub_1.h5'

def inspect_file():
    labels = set()
    shapes = set()
    lengths = set()
    
    def collect_stats(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith('eeg'):
            shapes.add(obj.shape)
            if 'label' in obj.attrs:
                # Handle label if it's an array or scalar
                lbl = obj.attrs['label']
                if hasattr(lbl, '__iter__'):
                    if len(lbl) == 1:
                         labels.add(lbl[0])
                    else:
                         labels.add(tuple(lbl))
                else:
                    labels.add(lbl)
            if 'time_length' in obj.attrs:
                lengths.add(obj.attrs['time_length'])

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting file: {file_path}")
            print("-" * 30)
            f.visititems(collect_stats)
            
            print("Summary Statistics:")
            print(f"  Unique EEG Shapes: {shapes}")
            print(f"  Unique Labels: {sorted(list(labels))}")
            print(f"  Unique Time Lengths (s): {sorted(list(lengths))}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_file()
