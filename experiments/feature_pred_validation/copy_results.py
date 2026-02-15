
import os
import shutil

src_dir = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final_viz"
dst_dir = "/vePFS-0x0d/home/cx/ptft/result/feature"

print(f"Copying from {src_dir} to {dst_dir}...")

try:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"Created {dst_dir}")

    # Copy contents
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
            print(f"Copied dir: {item}")
        else:
            shutil.copy2(s, d)
            print(f"Copied file: {item}")
            
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
