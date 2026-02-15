import pandas as pd
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.feature_extractor import FeatureExtractor
from eeg_feature_extraction.config import Config
from eeg_feature_extraction.features.base import FeatureRegistry

def analyze_new_features():
    # 1. Load existing features from CSV
    csv_path = '/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/60sfeature_metrics_eval_feat_only.csv'
    try:
        df = pd.read_csv(csv_path)
        existing_features = set(df['Feature Name'].values)
        print(f"Loaded {len(existing_features)} existing features from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Get all available features from code
    # We need to instantiate classes or use registry to get names
    # FeatureRegistry.get_all_feature_classes() returns {group_name: class}
    # We can inspect .feature_names of each class
    
    # Force registration by importing feature modules
    # (Already done in feature_extractor.py, but let's be safe)
    from eeg_feature_extraction import feature_extractor
    
    all_groups = FeatureRegistry.get_all_feature_classes()
    
    new_features_by_group = {}
    
    print("\nAnalyzing feature groups...")
    for group_name, feature_cls in all_groups.items():
        # feature_cls.feature_names is a list of strings
        cls_features = feature_cls.feature_names
        
        new_features = []
        for feat in cls_features:
            # Check if feat is in existing_features
            # Note: CSV might have 'feature_std' but code produces 'feature'.
            # If 'feature' is in CSV, it's existing.
            # If 'feature' is NOT in CSV, it's new.
            # (We ignore whether 'feature_std' is in CSV for now, assuming base feature is what matters)
            
            if feat not in existing_features:
                new_features.append(feat)
        
        if new_features:
            new_features_by_group[group_name] = new_features
            print(f"Group '{group_name}': {len(new_features)} new features found.")
        else:
            print(f"Group '{group_name}': All features already exist.")

    # 3. Generate Markdown Report
    md_content = "# New Features Analysis\n\n"
    md_content += "This report lists features available in the codebase that are **not** present in `60sfeature_metrics_eval_feat_only.csv`.\n\n"
    
    total_new = sum(len(v) for v in new_features_by_group.values())
    md_content += f"**Total New Features:** {total_new}\n\n"
    
    md_content += "## Breakdown by Group\n\n"
    
    for group, features in new_features_by_group.items():
        md_content += f"### {group.replace('_', ' ').title()}\n"
        md_content += f"- **Count:** {len(features)}\n"
        md_content += "- **Features:**\n"
        for f in features:
            md_content += f"  - `{f}`\n"
        md_content += "\n"
        
    output_md_path = "New_Features_Analysis.md"
    with open(output_md_path, 'w') as f:
        f.write(md_content)
    
    print(f"\nMarkdown report generated: {output_md_path}")
    
    # 4. Return list of groups to benchmark
    return list(new_features_by_group.keys())

if __name__ == "__main__":
    analyze_new_features()
