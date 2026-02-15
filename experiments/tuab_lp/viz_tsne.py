import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import csv
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.manifold import TSNE
import random
from collections import defaultdict

# Add project root to path
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
from datasets.tuab import TUABDataset, get_tuab_file_list

# Reuse config logic from extract_features.py
def get_config(model_type):
    config = {
        'model': {
            'in_dim': 200,
            'd_model': 200,
            'dim_feedforward': 800,
            'seq_len': 60,
            'n_layer': 12,
            'nhead': 8,
            'dropout': 0.1,
            'num_classes': 1,
            'pretrain_tasks': ['reconstruction'],
            'feature_token_type': 'gap',
            'feature_token_strategy': 'single',
            'feature_dim': 200,
        },
        'task_type': 'pretraining'
    }

    if model_type == 'neuro_ke':
        config['model']['pretrain_tasks'] = ['reconstruction', 'feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'single'
        config['model']['feature_dim'] = 62
    elif model_type == 'feat_only':
        config['model']['pretrain_tasks'] = ['feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'single'
        config['model']['feature_dim'] = 62

    return config

def load_model(model_type, weights_path, device):
    config = get_config(model_type)
    model = CBraModWrapper(config)
    
    print(f"Loading weights for {model_type} from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    return model

def get_subject_id(file_path):
    basename = os.path.basename(file_path)
    try:
        return basename.split('_')[1].split('.')[0]
    except:
        return basename.split('.')[0]

def extract_embeddings(model, dataloader, device, model_type):
    embeddings = []
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc=f"Extracting {model_type}"):
            data = data.to(device)
            
            # Backbone
            feats = model.backbone(data) # (B, C, N, D)
            
            # Always use GAP for EEG embedding (200-dim) as requested
            emb = feats.mean(dim=[1, 2]) # (B, D)
            
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def main():
    # Configuration
    dataset_dir = '/vepfs-0x0d/eeg-data/TUAB'
    output_csv = 'experiments/tuab_lp/tsne_embeddings.csv'
    
    models_config = [
        {
            'type': 'recon',
            'path': '/vePFS-0x0d/home/chen/related_projects/CBraMod/pretrained_weights/pretrained_weights.pth',
            'name': 'Baseline (Recon)'
        },
        {
            'type': 'neuro_ke',
            'path': '/vePFS-0x0d/home/cx/ptft/output_old/flagship_cross_attn/checkpoint_epoch_6.pth',
            'name': 'Neuro-KE'
        },
        {
            'type': 'feat_only',
            'path': '/vepfs-0x0d/home/cx/ptft/output/sanity_feat_only_all_60s/checkpoint_epoch_25.pth',
            'name': 'FeatOnly'
        }
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set a new seed for "re-calculation" with different points
    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Prepare Dataset & Sampling
    print("Initializing dataset...")
    # Get all files to ensure we have enough subjects
    # Using 'train' + 'test' + 'val' essentially means scanning everything if we implement custom list
    # But TUABDataset needs a list. Let's use get_tuab_file_list for all splits or manually scan.
    # To be safe and reuse existing logic, let's just get all files via glob as in count_tuab.py
    # actually get_tuab_file_list has a split logic.
    # Let's combine train/val/test splits to get FULL dataset.
    
    all_files = []
    for split in ['train', 'val', 'test']:
        all_files.extend(get_tuab_file_list(dataset_dir, split, seed=42))
    
    # Deduplicate files just in case
    all_files = sorted(list(set(all_files)))
    
    dataset = TUABDataset(
        file_list=all_files,
        input_size=12000,
        cache_path='tuab_full_index.json' # Try to reuse the one we created or create new
    )
    
    # Group by Subject
    print("Grouping by subject...")
    subject_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        subj_id = get_subject_id(sample['file_path'])
        subject_indices[subj_id].append(idx)
        
    all_subjects = list(subject_indices.keys())
    print(f"Found {len(all_subjects)} subjects.")
    
    # Sample fewer subjects for faster execution during interactive session
    n_subjects = min(300, len(all_subjects))
    selected_subjects = random.sample(all_subjects, n_subjects)
    print(f"Selected {n_subjects} subjects.")
    
    # Sample 1 segment per subject
    selected_indices = []
    labels = []
    subj_ids = []
    
    for subj in selected_subjects:
        # Pick one random index for this subject
        idx = random.choice(subject_indices[subj])
        selected_indices.append(idx)
        labels.append(dataset.samples[idx]['label'])
        subj_ids.append(subj)
        
    subset = Subset(dataset, selected_indices)
    dataloader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)
    
    # Container for all results
    all_results = []
    
    # 2. Extract Features for each model
    for model_cfg in models_config:
        print(f"\nProcessing {model_cfg['name']}...")
        model = load_model(model_cfg['type'], model_cfg['path'], device)
        
        embs = extract_embeddings(model, dataloader, device, model_cfg['type'])
        
        # Store for t-SNE
        for i, emb in enumerate(embs):
            all_results.append({
                'model': model_cfg['name'],
                'label': labels[i],
                'subject': subj_ids[i],
                'embedding': emb
            })
            
    # 3. t-SNE
    print("\nRunning t-SNE...")
    # Prepare matrix
    X = np.array([r['embedding'] for r in all_results])
    
    # Use TSNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)
    
    # 4. Save to CSV
    print(f"Saving results to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'model', 'label', 'subject'])
        
        for i, r in enumerate(all_results):
            writer.writerow([
                X_embedded[i, 0],
                X_embedded[i, 1],
                r['model'],
                r['label'],
                r['subject']
            ])
            
    print("Done!")

if __name__ == "__main__":
    main()
