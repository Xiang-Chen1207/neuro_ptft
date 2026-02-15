
import shutil
import os

src = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final_viz/combined_analysis_half_col.pdf"
dst = "/vePFS-0x0d/home/cx/ptft/result/feature/combined_analysis_half_col.pdf"

try:
    shutil.copy2(src, dst)
    print(f"Copied to {dst}")
except Exception as e:
    print(f"Error: {e}")
