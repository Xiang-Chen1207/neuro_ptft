import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(os.getcwd())

from models.wrapper import CBraModWrapper
from datasets.tusz import TUSZDataset, get_tusz_file_list

def get_config(model_type):
    # Base config for feature extraction
    config = {
        'model': {
            'in_dim': 200,
            'd_model': 200,
            'dim_feedforward': 800,
            'seq_len': 10, # Nominal
            'n_layer': 12,
            'nhead': 8,
            'dropout': 0.1,
            'num_classes': 13, 
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
        
    model.to(device)
    model.eval()
    return model

def extract_features_dataset(model, dataset, device, num_workers, model_type):
    # Force batch_size=1 for variable length handling without padding
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    eeg_feats_list = []
    feat_tokens_list = []
    labels_list = []
    
    print(f"Extracting features for {len(dataset)} samples...")
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            # data shape: (1, C, N, P)
            
            # Backbone
            feats = model.backbone(data) # (1, C, N, D)
            
            # 1. EEG Token (GAP)
            # Pool over Channel(1) and Time(2) -> (1, D)
            eeg_feat = feats.mean(dim=[1, 2])
            eeg_feats_list.append(eeg_feat.cpu().numpy())
            
            # 2. Feat Token (Cross Attn)
            if model_type in ['neuro_ke', 'feat_only']:
                B, C, N, D = feats.shape
                feats_flat = feats.view(B, C * N, D)
                query = model.feat_query.expand(B, -1, -1)
                attn_output, _ = model.feat_attn(query, feats_flat, feats_flat)
                
                if model.feature_token_strategy == 'single':
                    feat_token = attn_output.squeeze(1) # (B, D)
                else:
                    feat_token = attn_output.reshape(B, -1)
                    
                feat_tokens_list.append(feat_token.cpu().numpy())
            
            labels_list.append(target.numpy())
            
    eeg_feats = np.concatenate(eeg_feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    feat_tokens = None
    if feat_tokens_list:
        feat_tokens = np.concatenate(feat_tokens_list, axis=0)
        
    return {
        'eeg': eeg_feats,
        'feat': feat_tokens,
        'labels': labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['recon', 'neuro_ke', 'feat_only'], required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='experiments/tusz_lp/features')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print("Loading Dataset Index...")
    train_files = get_tusz_file_list(args.dataset_dir, 'train', seed=args.seed)
    val_files = get_tusz_file_list(args.dataset_dir, 'val', seed=args.seed)
    test_files = get_tusz_file_list(args.dataset_dir, 'test', seed=args.seed)
    
    # Merge Train and Val for LP
    train_val_files = train_files + val_files
    
    # Use dynamic_length=True
    train_dataset = TUSZDataset(train_val_files, dynamic_length=True)
    test_dataset = TUSZDataset(test_files, dynamic_length=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_type, args.weights_path, device)
    
    print("Extracting Train+Val Features...")
    train_data = extract_features_dataset(model, train_dataset, device, args.num_workers, args.model_type)
    
    print("Extracting Test Features...")
    test_data = extract_features_dataset(model, test_dataset, device, args.num_workers, args.model_type)
    
    output_path = os.path.join(args.output_dir, f"{args.model_type}_features.npz")
    print(f"Saving to {output_path}...")
    
    save_dict = {
        'train_eeg': train_data['eeg'],
        'train_labels': train_data['labels'],
        'test_eeg': test_data['eeg'],
        'test_labels': test_data['labels']
    }
    
    if args.model_type in ['neuro_ke', 'feat_only']:
        save_dict['train_feat'] = train_data['feat']
        save_dict['test_feat'] = test_data['feat']
        
    np.savez_compressed(output_path, **save_dict)
    print("Done.")

if __name__ == '__main__':
    main()
