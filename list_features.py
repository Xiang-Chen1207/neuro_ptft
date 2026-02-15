import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg_feature_extraction.config import Config
from eeg_feature_extraction.features.time_domain import TimeDomainFeatures
from eeg_feature_extraction.features.frequency_domain import FrequencyDomainFeatures
from eeg_feature_extraction.features.complexity import ComplexityFeatures
from eeg_feature_extraction.features.connectivity import ConnectivityFeatures
from eeg_feature_extraction.features.network import NetworkFeatures
from eeg_feature_extraction.features.de_features import DEFeatures
from eeg_feature_extraction.features.microstate import MicrostateFeatures
from eeg_feature_extraction.features.composite import CompositeFeatures

def list_all_features():
    config = Config()
    
    # Instantiate all feature classes
    feature_classes = [
        ('Time Domain', TimeDomainFeatures(config)),
        ('Frequency Domain', FrequencyDomainFeatures(config)),
        ('Complexity', ComplexityFeatures(config)),
        ('Connectivity', ConnectivityFeatures(config)),
        ('Network', NetworkFeatures(config)),
        ('DE Features', DEFeatures(config)),
        ('Microstate', MicrostateFeatures(config)),
        ('Composite', CompositeFeatures(config))
    ]
    
    all_names = []
    
    print("\n--- All Features List ---\n")
    
    global_idx = 1
    
    for group_name, feature_obj in feature_classes:
        # print(f"### {group_name}")
        for name in feature_obj.feature_names:
            # Check if it's already in list (shouldn't be, but good to check)
            # if name not in all_names:
            all_names.append(name)
            # print(f"{global_idx}. {name}")
            global_idx += 1
            
    return all_names

if __name__ == "__main__":
    names = list_all_features()
    
    # Generate Markdown list
    md_content = "\n\n## 9. Full Feature List\n\nTotal count: {}\n\n".format(len(names))
    for i, name in enumerate(names, 1):
        md_content += f"{i}. {name}\n"
        
    print(md_content)
    
    # Write to file
    with open("feature_list_dump.txt", "w") as f:
        f.write(md_content)
