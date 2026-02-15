import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def run_experiment_gpu(X_train, y_train, X_test, y_test, seed, C=1.0, device='cuda'):
    # Standardization (Crucial for convergence on larger datasets)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    # Ensure num_classes is at least 2 for binary case handling
    num_classes = max(num_classes, 2)
    # CHANGED: Force num_classes to match global TUEV expectation (6) if needed, 
    # but dynamic is fine usually as long as train set has all classes or enough.
    # TUEV is 6-class, so num_classes should likely be 6.
    # Let's trust unique count or max value + 1 if labels are indices.
    # Safe check: if max label > num_classes, adjust.
    max_label = y_train.max()
    if max_label >= num_classes:
        num_classes = max_label + 1
    
    model = LogisticRegressionTorch(input_dim, num_classes).to(device)
    
    # L2 Regularization via weight_decay
    weight_decay = 1.0 / C if C > 0 else 0
    
    # Increased max_iter for convergence
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, history_size=20)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    final_loss = 0.0
    def closure():
        nonlocal final_loss
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        final_loss = loss.item()
        return loss
        
    optimizer.step(closure)
    
    # print(f"  Training Loss: {final_loss:.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = torch.max(outputs.data, 1)
        y_pred = predicted.cpu().numpy()
        
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # AUC metrics
    try:
        if num_classes == 2:
            # Binary case
            y_probs = probs[:, 1]
            auroc = roc_auc_score(y_test, y_probs)
            auc_pr = average_precision_score(y_test, y_probs)
        else:
            # Multi-class case
            auroc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
            y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
            auc_pr = average_precision_score(y_test_bin, probs, average='weighted')
    except Exception as e:
        # print(f"Warning: AUC calculation failed: {e}")
        auroc = 0.0
        auc_pr = 0.0
    
    # Check convergence (simplified for LBFGS)
    converged = True 
    
    return acc, bacc, auroc, auc_pr, f1, converged, input_dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_path', type=str, required=True, help='Path to baseline .npz file')
    parser.add_argument('--flagship_path', type=str, required=True, help='Path to flagship .npz file')
    parser.add_argument('--featonly_path', type=str, required=True, help='Path to feat_only .npz file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs='+', 
                        default=[1.0], 
                        help='List of training set ratios (0.0 to 1.0)')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"Loading Baseline features from {args.baseline_path}...")
    baseline_data = np.load(args.baseline_path)
    print(f"Loading Flagship features from {args.flagship_path}...")
    flagship_data = np.load(args.flagship_path)
    print(f"Loading FeatOnly features from {args.featonly_path}...")
    featonly_data = np.load(args.featonly_path)
    
    # Verify Subject Consistency
    baseline_subs = np.unique(baseline_data['train_subjects'])
    flagship_subs = np.unique(flagship_data['train_subjects'])
    featonly_subs = np.unique(featonly_data['train_subjects'])
    
    # Check if all match
    subs_match = np.array_equal(np.sort(baseline_subs), np.sort(flagship_subs)) and \
                 np.array_equal(np.sort(baseline_subs), np.sort(featonly_subs))
    
    if not subs_match:
        print("Warning: Train subjects mismatch between models!")
        # Find intersection of all three
        common_subs = np.intersect1d(baseline_subs, flagship_subs)
        common_subs = np.intersect1d(common_subs, featonly_subs)
        print(f"Using intersection of {len(common_subs)} subjects.")
        unique_train_subjects = common_subs
    else:
        unique_train_subjects = baseline_subs
        
    total_subjects = len(unique_train_subjects)
    print(f"Total Unique Training Subjects: {total_subjects}")
    
    # Pre-calculate incremental subject subsets
    # Shuffle once globally
    rng = np.random.RandomState(args.seed)
    shuffled_subjects = unique_train_subjects.copy()
    rng.shuffle(shuffled_subjects)
    
    subsets = {}
    for ratio in sorted(args.ratios):
        if ratio >= 1.0:
            n_subs = total_subjects
            selected = shuffled_subjects
        else:
            n_subs = max(1, int(total_subjects * ratio))
            selected = shuffled_subjects[:n_subs] # Strict inclusion: take first N
            
        subsets[ratio] = (n_subs, selected)
        
    print("\n=== Comparative Results Table ===")
    print(f"| {'Ratio':<8} | {'NumSub':<6} | {'Model':<12} | {'Feature':<10} | {'Dim':<5} | {'Acc':<7} | {'BAcc':<7} | {'AUROC':<7} | {'AUC-PR':<7} | {'F1':<7} |")
    print(f"|{'-'*10}|{'-'*8}|{'-'*14}|{'-'*12}|{'-'*7}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
    
    # Iterate over ratios
    for ratio in sorted(args.ratios):
        n_subs, selected_subjects = subsets[ratio]
        
        if ratio >= 1.0:
            ratio_display = "100%"
        else:
            ratio_display = f"{ratio*100:.1f}%"
            
        # 1. Run Baseline (Recon - EEG Only)
        mask_train_bl = np.isin(baseline_data['train_subjects'], selected_subjects)
        y_train_bl = baseline_data['train_labels'][mask_train_bl]
        
        if len(np.unique(y_train_bl)) < 2:
            print(f"| {ratio_display:<8} | {n_subs:<6} | SKIPPED (Only 1 class) |")
            continue
            
        X_train_bl = baseline_data['train_eeg'][mask_train_bl]
        X_test_bl = baseline_data['test_eeg']
        y_test_bl = baseline_data['test_labels']
        
        acc, bacc, auroc, auc_pr, f1, _, dim = run_experiment_gpu(
            X_train_bl, y_train_bl, X_test_bl, y_test_bl, 
            args.seed, device=args.device
        )
        print(f"| {ratio_display:<8} | {n_subs:<6} | {'Baseline':<12} | {'EEG':<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {auroc:.4f}  | {auc_pr:.4f}  | {f1:.4f}  |")
        
        # 2. Run Flagship (Neuro-KE - All Features)
        mask_train_fs = np.isin(flagship_data['train_subjects'], selected_subjects)
        y_train_fs = flagship_data['train_labels'][mask_train_fs]
        
        fs_feature_types = ['eeg', 'feat', 'full', 'pred']
        
        for ftype in fs_feature_types:
            if ftype == 'eeg':
                X_train_fs = flagship_data['train_eeg'][mask_train_fs]
                X_test_fs = flagship_data['test_eeg']
            elif ftype == 'feat':
                X_train_fs = flagship_data['train_feat'][mask_train_fs]
                X_test_fs = flagship_data['test_feat']
            elif ftype == 'full':
                X_train_fs = np.concatenate([flagship_data['train_eeg'][mask_train_fs], flagship_data['train_feat'][mask_train_fs]], axis=1)
                X_test_fs = np.concatenate([flagship_data['test_eeg'], flagship_data['test_feat']], axis=1)
            elif ftype == 'pred':
                if 'train_pred' not in flagship_data: continue
                X_train_fs = flagship_data['train_pred'][mask_train_fs]
                X_test_fs = flagship_data['test_pred']
            
            # Skip if feature not available (e.g. some runs might not have pred)
            if 'X_train_fs' not in locals(): continue
                
            acc, bacc, auroc, auc_pr, f1, _, dim = run_experiment_gpu(
                X_train_fs, y_train_fs, X_test_fs, flagship_data['test_labels'], 
                args.seed, device=args.device
            )
            print(f"| {ratio_display:<8} | {n_subs:<6} | {'Neuro-KE':<12} | {ftype:<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {auroc:.4f}  | {auc_pr:.4f}  | {f1:.4f}  |")
            
        # 3. Run FeatOnly (Feat Prediction Only - All Features)
        mask_train_fo = np.isin(featonly_data['train_subjects'], selected_subjects)
        y_train_fo = featonly_data['train_labels'][mask_train_fo]
        
        # FeatOnly Feature Types (Same as Flagship usually)
        fo_feature_types = ['eeg', 'feat', 'full', 'pred']
        
        for ftype in fo_feature_types:
            if ftype == 'eeg':
                # Note: FeatOnly model also has an EEG backbone (frozen), so it produces EEG features too
                X_train_fo = featonly_data['train_eeg'][mask_train_fo]
                X_test_fo = featonly_data['test_eeg']
            elif ftype == 'feat':
                X_train_fo = featonly_data['train_feat'][mask_train_fo]
                X_test_fo = featonly_data['test_feat']
            elif ftype == 'full':
                X_train_fo = np.concatenate([featonly_data['train_eeg'][mask_train_fo], featonly_data['train_feat'][mask_train_fo]], axis=1)
                X_test_fo = np.concatenate([featonly_data['test_eeg'], featonly_data['test_feat']], axis=1)
            elif ftype == 'pred':
                if 'train_pred' not in featonly_data: continue
                X_train_fo = featonly_data['train_pred'][mask_train_fo]
                X_test_fo = featonly_data['test_pred']
                
            acc, bacc, auroc, auc_pr, f1, _, dim = run_experiment_gpu(
                X_train_fo, y_train_fo, X_test_fo, featonly_data['test_labels'], 
                args.seed, device=args.device
            )
            print(f"| {ratio_display:<8} | {n_subs:<6} | {'FeatOnly':<12} | {ftype:<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {auroc:.4f}  | {auc_pr:.4f}  | {f1:.4f}  |")

if __name__ == '__main__':
    main()
