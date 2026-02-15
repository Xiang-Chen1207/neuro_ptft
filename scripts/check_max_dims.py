
import json
import os

cache_path = "/vePFS-0x0d/home/cx/ptft/output/joint_pretrain/dataset_index_joint.json"

if not os.path.exists(cache_path):
    print(f"Error: {cache_path} not found.")
    exit(1)

print(f"Loading {cache_path}...")
with open(cache_path, 'r') as f:
    data = json.load(f)
    
samples = data.get('samples', [])
print(f"Total samples: {len(samples)}")

max_channels = 0
max_time = 0

for s in samples:
    shape = s['shape']
    if not shape: continue
    
    # Heuristic to distinguish (C, T) vs (T, C)
    # Usually C < 200, T > 1000
    dim0, dim1 = shape
    
    if dim0 < 200 and dim1 > dim0:
        c, t = dim0, dim1
    elif dim1 < 200 and dim0 > dim1:
        c, t = dim1, dim0
    else:
        # Ambiguous or square-ish? Assume smaller is C
        c = min(dim0, dim1)
        t = max(dim0, dim1)
        
    if c > max_channels:
        max_channels = c
    if t > max_time:
        max_time = t
        
print(f"Max Channels found: {max_channels}")
print(f"Max Time points found: {max_time}")
print(f"Note: Time points will be truncated to 12000 by dataset loader.")
