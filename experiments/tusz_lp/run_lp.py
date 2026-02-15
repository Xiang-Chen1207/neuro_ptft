import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, label_binarize
import warnings

warnings.filterwarnings("ignore")

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def run_experiment_gpu(X_train, y_train, X_test, y_test, seed, C=1.0, device='cuda'):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    input_dim = X_train.shape[1]
    # Determine num_classes
    # For TUSZ, we might have missing classes in small subsets, but we should target 13 or max(label)+1
    # Let's rely on max label found in training or passed explicitly.
    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    
    model = LogisticRegressionTorch(input_dim, num_classes).to(device)
    
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, history_size=20)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    def closure():
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        return loss
        
    optimizer.step(closure)
    
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
            auroc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted', labels=np.arange(num_classes))
            y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
            auc_pr = average_precision_score(y_test_bin, probs, average='weighted')
    except Exception as e:
        print(f"Warning: AUC calculation failed: {e}")
        auroc = 0.0
        auc_pr = 0.0
    
    return acc, bacc, f1, auroc, auc_pr, input_dim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_path', type=str, required=True)
    parser.add_argument('--flagship_path', type=str, required=True)
    parser.add_argument('--featonly_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs='+', default=[1.0])
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"Loading features...")
    baseline_data = np.load(args.baseline_path)
    flagship_data = np.load(args.flagship_path)
    featonly_data = np.load(args.featonly_path)
    
    # We assume train/test splits are consistent across feature files if generated from same dataset split
    
    n_train = len(baseline_data['train_labels'])
    indices = np.arange(n_train)
    
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    
    print("\n=== TUSZ Linear Probe Results ===")
    print(f"| {'Ratio':<8} | {'Model':<12} | {'Feature':<10} | {'Dim':<5} | {'Acc':<7} | {'BAcc':<7} | {'AUROC':<7} | {'AUC-PR':<7} | {'F1':<7} |")
    print(f"|{'-'*10}|{'-'*14}|{'-'*12}|{'-'*7}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|")
    
    for ratio in sorted(args.ratios):
        ratio_display = f"{ratio*100:.1f}%"
        n_sub = int(n_train * ratio)
        if n_sub < 1: n_sub = 1
        
        curr_indices = indices[:n_sub]
        
        # 1. Baseline
        X_train = baseline_data['train_eeg'][curr_indices]
        y_train = baseline_data['train_labels'][curr_indices]
        X_test = baseline_data['test_eeg']
        y_test = baseline_data['test_labels']
        
        acc, bacc, f1, auroc, auc_pr, dim = run_experiment_gpu(X_train, y_train, X_test, y_test, args.seed, device=args.device)
        print(f"| {ratio_display:<8} | {'Baseline':<12} | {'EEG':<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {auroc:.4f}  | {auc_pr:.4f}  | {f1*100:.2f}% |")
        
        # 2. Flagship
        # Try different feature types
        for ftype in ['eeg', 'feat', 'full']:
            if ftype == 'eeg':
                X_train = flagship_data['train_eeg'][curr_indices]
                X_test = flagship_data['test_eeg']
            elif ftype == 'feat' and 'train_feat' in flagship_data:
                X_train = flagship_data['train_feat'][curr_indices]
                X_test = flagship_data['test_feat']
            elif ftype == 'full' and 'train_feat' in flagship_data:
                X_train = np.concatenate([flagship_data['train_eeg'][curr_indices], flagship_data['train_feat'][curr_indices]], axis=1)
                X_test = np.concatenate([flagship_data['test_eeg'], flagship_data['test_feat']], axis=1)
            else:
                continue
                
            y_train = flagship_data['train_labels'][curr_indices]
            
            acc, bacc, f1, auroc, auc_pr, dim = run_experiment_gpu(X_train, y_train, X_test, y_test, args.seed, device=args.device)
            print(f"| {ratio_display:<8} | {'Neuro-KE':<12} | {ftype:<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {auroc:.4f}  | {auc_pr:.4f}  | {f1*100:.2f}% |")

        # 3. FeatOnly
        for ftype in ['eeg', 'feat', 'full']:
            if ftype == 'eeg':
                X_train = featonly_data['train_eeg'][curr_indices]
                X_test = featonly_data['test_eeg']
            elif ftype == 'feat' and 'train_feat' in featonly_data:
                X_train = featonly_data['train_feat'][curr_indices]
                X_test = featonly_data['test_feat']
            elif ftype == 'full' and 'train_feat' in featonly_data:
                X_train = np.concatenate([featonly_data['train_eeg'][curr_indices], featonly_data['train_feat'][curr_indices]], axis=1)
                X_test = np.concatenate([featonly_data['test_eeg'], featonly_data['test_feat']], axis=1)
            else:
                continue
                
            y_train = featonly_data['train_labels'][curr_indices]
            
            acc, bacc, f1, auroc, auc_pr, dim = run_experiment_gpu(X_train, y_train, X_test, y_test, args.seed, device=args.device)
            print(f"| {ratio_display:<8} | {'FeatOnly':<12} | {ftype:<10} | {dim:<5} | {acc*100:.2f}% | {bacc*100:.2f}% | {auroc:.4f}  | {auc_pr:.4f}  | {f1*100:.2f}% |")

if __name__ == '__main__':
    main()
