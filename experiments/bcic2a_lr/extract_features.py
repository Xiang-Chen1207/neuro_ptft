import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score
import logging

# Add project root to path
import sys
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
# BCIC2A Imports
from datasets.bcic2a import BCIC2ADataset, get_bcic2a_file_list

# Define load_config directly to resolve import error
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config(model_type):
    config = {
        'model': {
            'in_dim': 200,
            'd_model': 200,
            'dim_feedforward': 800,
            # BCIC2A: 4s (1000 samples) -> 5 patches of 200
            'seq_len': 5,
            'n_layer': 12,
            'nhead': 8,
            'dropout': 0.1,
            # BCIC2A: 4 classes
            'num_classes': 4,
            'pretrain_tasks': ['reconstruction'],
            'feature_token_type': 'gap',
            'feature_token_strategy': 'single',
            'feature_dim': 200,
        },
        'task_type': 'pretraining'
    }

    if model_type == 'neuro_ke':
        # Multi-task: reconstruction + feature prediction
        config['model']['pretrain_tasks'] = ['reconstruction', 'feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'single' # Single query (200 dim)
        config['model']['feature_dim'] = 62
    elif model_type == 'neuro_ke_pooled':
        # Same as neuro_ke but we will pool features during extraction
        config['model']['pretrain_tasks'] = ['reconstruction', 'feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'all' # Load as 'all' to match weights
        config['model']['feature_dim'] = 62
    elif model_type == 'feat_only':
        # Single-task: feature prediction only (sanity_feat_only)
        config['model']['pretrain_tasks'] = ['feature_pred']
        config['model']['feature_token_type'] = 'cross_attn'
        config['model']['feature_token_strategy'] = 'single'
        config['model']['feature_dim'] = 62

    return config

def load_model(model_type, weights_path, device):
    config = get_config(model_type)
    model = CBraModWrapper(config)
    
    print(f"Loading weights from {weights_path}...")
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
        # sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs for feature extraction.")
        model.backbone = nn.DataParallel(model.backbone)
        # Note: We only parallelize the backbone as it's the heaviest part.
        # The heads (GAP, CrossAttn) will run on the main GPU, which is efficient enough.
        
    return model

def get_subject_id(file_path):
    basename = os.path.basename(file_path)
    # BCIC2A logic: sub_X.h5
    try:
        return basename.split('_')[1].split('.')[0]
    except:
        return basename.split('.')[0]

def extract_features_dataset(model, dataset, device, batch_size, num_workers, model_type):
    # Optimize num_workers based on CPU count
    if num_workers is None:
        num_workers = min(16, os.cpu_count() or 1)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
    
    eeg_feats_list = []
    feat_tokens_list = []
    pred_features_list = []
    labels_list = []
    
    subject_ids = [get_subject_id(s['file_path']) for s in dataset.samples]
    
    print(f"Extracting features for {len(dataset)} samples...")
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            
            # Backbone (Parallelized)
            feats = model.backbone(data) # (B, C, N, D)
            
            # 1. EEG Token (GAP)
            # feats is on device (likely gpu:0 if using DataParallel)
            eeg_feat = feats.mean(dim=[1, 2])
            eeg_feats_list.append(eeg_feat.cpu().numpy())
            
            # 2. Feat Token & Predicted Features (Cross Attn) - For Neuro-KE and Feat-Only
            if model_type in ['neuro_ke', 'feat_only', 'neuro_ke_pooled']:
                B, C, N, D = feats.shape
                feats_flat = feats.view(B, C * N, D)
                query = model.feat_query.expand(B, -1, -1)
                attn_output, _ = model.feat_attn(query, feats_flat, feats_flat)
                
                # Check strategy to determine shape
                if model.feature_token_strategy == 'single':
                    feat_token = attn_output.squeeze(1)
                elif model.feature_token_strategy == 'all':
                     # For 'all', attn_output is (B, 62, D)
                     
                     if model_type == 'neuro_ke_pooled':
                         # Pool to (B, D) e.g. mean pooling over 62 features
                         feat_token = attn_output.mean(dim=1) # (B, D)
                     else:
                         # Flatten to (B, 62*D)
                         feat_token = attn_output.reshape(B, -1)
                else:
                    feat_token = attn_output.reshape(B, -1)
                
                feat_tokens_list.append(feat_token.cpu().numpy())
                
                if model_type != 'neuro_ke_pooled':
                     pred_feat = model.feature_head(feat_token if model.feature_token_strategy != 'all' else attn_output)
                     if model.feature_token_strategy == 'all':
                          # head expects (B, 62, D) -> (B, 62, 1) -> (B, 62)
                          pred_feat = model.feature_head(attn_output).squeeze(-1)
                     else:
                          pred_feat = model.feature_head(feat_token)
                          
                     pred_features_list.append(pred_feat.detach().cpu().numpy())
            
            labels_list.append(target.numpy())
            
    eeg_feats = np.concatenate(eeg_feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    subject_ids = np.array(subject_ids)
    
    feat_tokens = None
    if feat_tokens_list:
        feat_tokens = np.concatenate(feat_tokens_list, axis=0)
    pred_features = None
    if pred_features_list:
        pred_features = np.concatenate(pred_features_list, axis=0)
        
    return {
        'eeg': eeg_feats,
        'feat': feat_tokens,
        'pred': pred_features,
        'labels': labels,
        'subjects': subject_ids
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['recon', 'neuro_ke', 'feat_only', 'neuro_ke_pooled'], required=True)
    parser.add_argument('--dataset_dir', type=str, default='/vePFS-0x0d/home/downstream_data/BCIC2A')
    parser.add_argument('--weights_path', type=str, default='/vepfs-0x0d/home/cx/ptft/output/flagship_fixed/checkpoint_epoch_16.pth')
    parser.add_argument('--output_dir', type=str, default='experiments/bcic2a_lr/features')
    parser.add_argument('--output_name', type=str, default=None, help='Custom output filename')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading Dataset Index...")
    train_files = get_bcic2a_file_list(args.dataset_dir, 'train', seed=42)
    val_files = get_bcic2a_file_list(args.dataset_dir, 'val', seed=42)
    
    train_dataset = BCIC2ADataset(train_files, input_size=1000)
    test_dataset = BCIC2ADataset(val_files, input_size=1000)
    
    # 2. Load Model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_type, args.weights_path, device)
    
    # 3. Extract
    print("Extracting Train Features...")
    train_data = extract_features_dataset(model, train_dataset, device, args.batch_size, args.num_workers, args.model_type)
    
    print("Extracting Test Features...")
    test_data = extract_features_dataset(model, test_dataset, device, args.batch_size, args.num_workers, args.model_type)
    
    # 4. Save
    output_path = os.path.join(args.output_dir, f"{args.model_type}_features.npz")
    
    save_dict = {
        'train_eeg': train_data['eeg'],
        'train_labels': train_data['labels'],
        'train_subjects': train_data['subjects'],
        'test_eeg': test_data['eeg'],
        'test_labels': test_data['labels'],
        'test_subjects': test_data['subjects']
    }
    
    if args.model_type in ['neuro_ke', 'feat_only', 'neuro_ke_pooled']:
        save_dict['train_feat'] = train_data['feat']
        save_dict['test_feat'] = test_data['feat']
        if train_data['pred'] is not None and test_data['pred'] is not None:
            save_dict['train_pred'] = train_data['pred']
            save_dict['test_pred'] = test_data['pred']
        
    # Support custom output name
    output_filename = getattr(args, 'output_name', None)
    if output_filename is None:
        output_filename = f"{args.model_type}_features.npz"
        
    output_path = os.path.join(args.output_dir, output_filename)
    print(f"Saving to {output_path}...")
    
    np.savez_compressed(output_path, **save_dict)
    print("Done.")

if __name__ == '__main__':
    main()
