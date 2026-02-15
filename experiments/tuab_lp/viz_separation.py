import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import random
from collections import defaultdict

# Add project root to path
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
from datasets.tuab import TUABDataset, get_tuab_file_list

# --- Reuse extraction logic ---
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

def extract_embeddings(model, dataloader, device):
    embeddings = []
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Extracting"):
            data = data.to(device)
            feats = model.backbone(data)
            # Always GAP for consistency in comparison
            emb = feats.mean(dim=[1, 2]) 
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def main():
    # Setup
    dataset_dir = '/vepfs-0x0d/eeg-data/TUAB'
    output_dir = 'experiments/tuab_lp/plots_separation'
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 1. Dataset & Sampling
    print("Preparing dataset...")
    # Using a subset for speed and clarity
    all_files = []
    for split in ['train', 'val', 'test']:
        all_files.extend(get_tuab_file_list(dataset_dir, split, seed=42))
    all_files = sorted(list(set(all_files)))
    
    dataset = TUABDataset(all_files, input_size=12000, cache_path='tuab_full_index.json')
    
    # Sample 1000 subjects
    subject_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        subj_id = get_subject_id(sample['file_path'])
        subject_indices[subj_id].append(idx)
    
    all_subjects = list(subject_indices.keys())
    # Fixed seed for reproducibility
    random.seed(42) 
    selected_subjects = random.sample(all_subjects, min(1000, len(all_subjects)))
    
    selected_indices = []
    labels = []
    for subj in selected_subjects:
        idx = random.choice(subject_indices[subj])
        selected_indices.append(idx)
        labels.append(dataset.samples[idx]['label'])
        
    subset = Subset(dataset, selected_indices)
    dataloader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)
    labels = np.array(labels)
    
    metrics_data = []
    
    # 2. Process each model
    for model_cfg in models_config:
        print(f"\nAnalyzing {model_cfg['name']}...")
        model = load_model(model_cfg['type'], model_cfg['path'], device)
        
        # Extract High-Dim Embeddings (200-d)
        embeddings = extract_embeddings(model, dataloader, device)
        
        # Calculate Metrics (on 200-d data)
        sil_score = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        print(f"  Silhouette Score: {sil_score:.4f} (Higher is better)")
        print(f"  Calinski-Harabasz: {ch_score:.4f} (Higher is better)")
        
        metrics_data.append({
            'Model': model_cfg['name'],
            'Silhouette Score': sil_score,
            'CH Score': ch_score
        })
        
        # t-SNE for Visualization
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(embeddings)
        
        # DataFrame for Plotting
        df_plot = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y': X_embedded[:, 1],
            'Label': ['Abnormal' if l == 1 else 'Normal' for l in labels]
        })
        
        # 3. Enhanced Visualization (KDE + Scatter)
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")
        
        # KDE Contour (Density)
        # Levels=5 to show core density
        sns.kdeplot(
            data=df_plot, x='x', y='y', hue='Label', 
            fill=True, alpha=0.2, levels=5, 
            palette={'Normal': 'blue', 'Abnormal': 'red'},
            thresh=0.05
        )
        
        # Scatter (Points)
        sns.scatterplot(
            data=df_plot, x='x', y='y', hue='Label', 
            palette={'Normal': 'blue', 'Abnormal': 'red'},
            s=30, alpha=0.6, edgecolor='w', linewidth=0.5
        )
        
        plt.title(f"{model_cfg['name']}\nSilhouette: {sil_score:.3f}", fontsize=15)
        
        # Save
        filename = f"separation_{model_cfg['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot to {save_path}")

    # Save metrics table
    metrics_df = pd.DataFrame(metrics_data)
    print("\n=== Separation Metrics Comparison ===")
    print(metrics_df.to_string(index=False))
    metrics_path = os.path.join(output_dir, 'separation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
