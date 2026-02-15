import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import yaml
import os
from tqdm import tqdm
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.wrapper import CBraModWrapper
from datasets.builder import build_dataloader
import utils.util as utils

def get_feature_category(feature_name):
    name = feature_name.lower()
    if 'ratio' in name: return 'Ratios'
    if any(k in name for k in ['aperiodic', 'spectral', 'peak_frequency', 'individual_alpha']): return 'Structure'
    if any(k in name for k in ['power', 'alpha', 'beta', 'delta', 'theta', 'gamma']): return 'Power'
    return 'Time'

def plot_density_scatter(ax, y_true, y_pred, title, r2, pcc, cmap='mako'):
    # Subsample for KDE speed if needed
    if len(y_true) > 5000:
        idx = np.random.choice(len(y_true), 5000, replace=False)
        x_kde = y_true[idx]
        y_kde = y_pred[idx]
    else:
        x_kde, y_kde = y_true, y_pred
        
    xy = np.vstack([x_kde, y_kde])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x_kde[idx], y_kde[idx], z[idx]
    
    # Use mako as requested (Blue-Green-Black)
    # ADJUSTED: Reduced alpha and size to prevent "black blob" effect on dense real data
    # rasterized=True is CRITICAL for PDF export with many points (keeps file size small and fast)
    ax.scatter(x, y, c=z, s=8, cmap='mako', edgecolor='none', alpha=0.6, rasterized=True)
    
    # Diagonal
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, color='#333333', linestyle='--', lw=1, alpha=0.6)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Stats box
    stats_text = f"$R^2={r2:.2f}$\n$r={pcc:.2f}$"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9, fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.9))

def generate_combined_figure(df, all_preds, all_targets, feature_names, output_path, figsize=(12, 8)):
    # ICML Style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8])
    
    # --- LEFT: 2x2 Best Features ---
    df = df.copy() # Avoid modifying original
    df['Category'] = df['Feature Name'].apply(get_feature_category)
    
    # Define Explicit Order: Time -> Power -> Structure -> Ratios
    cat_order_list = ['Time', 'Power', 'Structure', 'Ratios']
    
    gs_left = gs[:, 0].subgridspec(2, 2, wspace=0.3, hspace=0.3)
    
    # Find best features and Plot Scatters in fixed order
    for i, cat in enumerate(cat_order_list):
        cat_df = df[df['Category'] == cat]
        if cat_df.empty: continue
        
        best_row = cat_df.loc[cat_df['R2'].idxmax()]
        feat_idx = best_row['Feature Index']
        feat_name = best_row['Feature Name']
        
        ax = fig.add_subplot(gs_left[i//2, i%2])
        
        y_true = all_targets[:, feat_idx]
        y_pred = all_preds[:, feat_idx]
        
        plot_density_scatter(ax, y_true, y_pred, f"{cat}: {feat_name}", best_row['R2'], best_row['PCC'])
        
        if i%2 == 0: ax.set_ylabel("Predicted")
        if i//2 == 1: ax.set_xlabel("True")
        
    # --- RIGHT: Bar Chart (Style 2 Cool Categorical) ---
    ax_right = fig.add_subplot(gs[:, 1])
    
    # Sort Logic for Bar Chart:
    # 1. Category Order: Ratios -> Structure -> Power -> Time (Reverse for barh so Time is Top)
    # 2. Within Category: R2 Ascending (Small -> Large) so Largest R2 is Top
    
    cat_dtype = pd.CategoricalDtype(categories=cat_order_list[::-1], ordered=True)
    df['Category'] = df['Category'].astype(cat_dtype)
    
    df_sorted = df.sort_values(by=['Category', 'R2'], ascending=[True, True])
    
    pal = {
        'Time': '#4c78a8', # Steel Blue
        'Power': '#72b7b2', # Teal
        'Structure': '#b279a2', # Muted Purple
        'Ratios': '#54a24b'  # Muted Green
    }
    
    colors = [pal.get(c, '#888888') for c in df_sorted['Category']]
    
    bars = ax_right.barh(np.arange(len(df)), df_sorted['R2'], color=colors, height=0.8, alpha=0.9)
    
    # Separators
    df_reset = df_sorted.reset_index(drop=True)
    current_cat = None
    for i in range(len(df_reset)):
        cat = df_reset.loc[i, 'Category']
        if current_cat is not None and cat != current_cat:
            ax_right.axhline(i - 0.5, color='white', linestyle='-', linewidth=2, alpha=1.0)
        current_cat = cat
        
    ax_right.set_yticks([])
    ax_right.set_xlabel("R2 Score", fontsize=11, fontweight='bold')
    
    # NEW: Set X-axis to 1.0 and add vertical line
    ax_right.set_xlim(0, 1.05)
    ax_right.axvline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    
    # Legend on Top
    # Order: Time -> Power -> Structure -> Ratios
    handles = [plt.Rectangle((0,0),1,1, color=pal[c]) for c in cat_order_list if c in pal]
    
    # Align legend with left title
    ax_right.legend(handles, [c for c in cat_order_list if c in pal], 
              loc='lower center', bbox_to_anchor=(0.5, 1.02), 
              ncol=4, frameon=False, fontsize=9)
    
    sns.despine(ax=ax_right, left=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_top_features_data(df, all_preds, all_targets, output_path):
    """Save data for top features to NPZ for quick replotting."""
    categories = ['Time', 'Power', 'Structure', 'Ratios']
    data_dict = {}
    
    df['Category'] = df['Feature Name'].apply(get_feature_category)
    
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        if cat_df.empty: continue
        best_row = cat_df.loc[cat_df['R2'].idxmax()]
        feat_idx = best_row['Feature Index']
        feat_name = best_row['Feature Name']
        
        data_dict[f"{cat}_name"] = feat_name
        data_dict[f"{cat}_r2"] = best_row['R2']
        data_dict[f"{cat}_pcc"] = best_row['PCC']
        data_dict[f"{cat}_true"] = all_targets[:, feat_idx]
        data_dict[f"{cat}_pred"] = all_preds[:, feat_idx]
        
    np.savez(output_path, **data_dict)
    print(f"Top features data saved to {output_path}")

def generate_full_grid_figure(df, all_preds, all_targets, feature_names, output_path):
    """
    Generate a single 8x8 grid figure directly using matplotlib.
    Minimalist style: Only Feature Name (Large), no ticks, no stats.
    """
    print("Generating 8x8 Full Grid Figure (Direct)...")
    
    # Sort features by Category then R2
    df_grid = df.copy()
    df_grid['Category'] = df_grid['Feature Name'].apply(get_feature_category)
    cat_order = {'Time': 0, 'Power': 1, 'Structure': 2, 'Ratios': 3}
    df_grid['CatOrder'] = df_grid['Category'].map(cat_order)
    df_grid = df_grid.sort_values(by=['CatOrder', 'R2'], ascending=[True, False])
    
    ordered_indices = df_grid['Feature Index'].values
    ordered_names = df_grid['Feature Name'].values
    
    # Layout: 8x8
    ROWS = 8
    COLS = 8
    
    # Large Figure Size
    # A4 is ~8x11. We want 8x8 squares.
    # Let's make it 16x16 inches for high detail source
    fig, axes = plt.subplots(ROWS, COLS, figsize=(20, 24))
    axes = axes.flatten()
    
    # Adjust spacing: Close packing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.1, hspace=0.3)
    
    for i in tqdm(range(len(axes)), desc="Plotting Grid"):
        ax = axes[i]
        
        if i < len(ordered_indices):
            feat_idx = ordered_indices[i]
            fname = ordered_names[i]
            
            y_true = all_targets[:, feat_idx]
            y_pred = all_preds[:, feat_idx]
            
            # Subsample for KDE speed if needed (keep 5000 for quality)
            if len(y_true) > 5000:
                idx_sub = np.random.choice(len(y_true), 5000, replace=False)
                x_kde, y_kde = y_true[idx_sub], y_pred[idx_sub]
            else:
                x_kde, y_kde = y_true, y_pred
            
            # Calculate Density
            try:
                xy = np.vstack([x_kde, y_kde])
                z = gaussian_kde(xy)(xy)
                idx_sort = z.argsort()
                x_plot, y_plot, z_plot = x_kde[idx_sort], y_kde[idx_sort], z[idx_sort]
            except:
                # Fallback if KDE fails (e.g. constant value)
                x_plot, y_plot = x_kde, y_kde
                z_plot = 'blue'
            
            # Scatter (Rasterized)
            # Increased s slightly for visibility in grid
            ax.scatter(x_plot, y_plot, c=z_plot, s=10, cmap='mako', edgecolor='none', alpha=0.6, rasterized=True)
            
            # Diagonal
            lims = [-3, 3]
            ax.plot(lims, lims, color='#333333', linestyle='--', lw=1, alpha=0.5)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            
            # Clean up
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('on') # Keep box border
            
            # Title: Feature Name only
            # Split long names
            short_name = fname
            if len(short_name) > 20: # Aggressive splitting
                 parts = short_name.split('_')
                 # Try to split into 2 or 3 lines if very long
                 if len(parts) >= 4:
                     # 3 lines max
                     chunk = len(parts)//2
                     short_name = "_".join(parts[:chunk]) + "\n" + "_".join(parts[chunk:])
                 else:
                     mid = len(parts)//2
                     short_name = "_".join(parts[:mid]) + "\n" + "_".join(parts[mid:])
            
            # Reduce font size to prevent overlap
            # User requested larger font now that wrapping is better
            ax.set_title(short_name, fontsize=12, fontweight='bold', pad=3)
            
        else:
            ax.axis('off')
            
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Feature Prediction Metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--output", type=str, default="feature_metrics.csv", help="Output CSV path")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name (e.g. TUEG, TUAB)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on (train/val/test)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config['dataset']['name'] = args.dataset
        print(f"Overriding dataset to {args.dataset}")

    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size

    # Ensure task type is pretraining
    config['task_type'] = 'pretraining'
    config['model']['task_type'] = 'pretraining'
    
    # Load Model
    print("Building model...")
    model = CBraModWrapper(config)
    model.to(args.device)
    model.eval()

    # Load Checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Handle state dict keys if wrapped in 'model' or DDP
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)

    # Build Validation Loader
    print(f"Building dataloader for split: {args.split}...")
    # Disable validation limit for full evaluation
    config['dataset']['limit_val'] = False
    val_loader = build_dataloader(config['dataset']['name'], config['dataset'], mode=args.split)
    
    # Feature Names
    feature_names = None
    if hasattr(val_loader.dataset, 'feature_names'):
        feature_names = val_loader.dataset.feature_names
    elif hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'feature_names'):
         feature_names = val_loader.dataset.dataset.feature_names
    
    if feature_names is None:
        print("Warning: Could not find feature names in dataset. Using indices.")

    # Prepare for Scatter Plots (Raw Data Collection)
    # To avoid OOM with 28k samples, we can subsample or just store all (28k * 66 * 4 bytes ~ 7MB, very small)
    all_preds = []
    all_targets = []

    # Evaluation Loop
    print("Starting evaluation...")
    
    # Accumulators
    mse_accum = None
    var_accum = None
    count = 0
    
    # For R2: 1 - SSE/SST
    # SSE = sum((pred - target)^2)
    # SST = sum((target - mean)^2) -> This requires global mean.
    # Alternative R2 calculation: 
    # Use online Welford algorithm or just sum(target) and sum(target^2) to get variance?
    # Simpler: Accumulate sum_squared_error and sum_squared_total (using batch mean approximation or 2-pass?)
    # 2-pass is expensive. 
    # Let's use the implementation in utils.calc_regression_metrics which does batch-wise averaging?
    # No, batch-wise averaging of R2 is not mathematically correct for global R2.
    # Correct way: Accumulate SSE and SST globally.
    # SST = sum((y - y_global_mean)^2). 
    # We need y_global_mean first. 
    # Approximation: Use batch mean as proxy? No, that's bad.
    # Better: Accumulate sum(y) and sum(y^2) to compute global variance later?
    # Var(y) = E[y^2] - (E[y])^2
    # SST = N * Var(y)
    
    sum_y = None
    sum_y_sq = None
    sum_sq_err = None
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if isinstance(batch, dict):
                # Handle dictionary batch
                if 'eeg' in batch:
                    x = batch['eeg']
                elif 'input' in batch:
                    x = batch['input']
                elif 'x' in batch:
                    x = batch['x']
                else:
                    # Try to find the input tensor
                    for k, v in batch.items():
                         if isinstance(v, torch.Tensor) and v.ndim >= 3:
                             x = v
                             break
                
                if 'features' in batch:
                    y = batch['features']
                elif 'target' in batch:
                    y = batch['target']
                elif 'label' in batch:
                    y = batch['label']
                elif 'y' in batch:
                    y = batch['y']
                else:
                    # Fallback: assume second item in items list if it's a tensor
                    # Or check for 'feature_label'
                    if 'feature_label' in batch:
                         y = batch['feature_label']
                    else:
                         print(f"Warning: Could not find target features in batch keys: {batch.keys()}")
                         continue
                    
                mask = batch.get('mask', None)
                
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    mask = None
                elif len(batch) == 3:
                    x, y, mask = batch
                else:
                    # Handle cases with more items (e.g., filename, index, etc.)
                    # Assume first 3 are x, y, mask
                    x = batch[0]
                    y = batch[1]
                    mask = batch[2]
            else:
                print(f"Unknown batch type: {type(batch)}")
                continue
            
            x = x.to(args.device).float()
            
            # y contains features in pretraining
            if isinstance(y, torch.Tensor):
                target_features = y.to(args.device).float()
            else:
                # Should not happen in this codebase for pretraining
                target_features = y.to(args.device).float()

            if mask is not None:
                mask = mask.to(args.device)

            # Forward
            # Explicitly create a zero-mask (no masking) for full signal inference
            # Mask shape should be (B, C, N) with all zeros
            # We need patch count N. 
            # x shape: (B, C, N, P) or similar? 
            # Let's check wrapper.py: _generate_mask uses x.shape -> B, C, N, P
            if x.ndim == 4:
                B, C, N, P = x.shape
                zero_mask = torch.zeros((B, C, N), device=x.device, dtype=torch.long)
            elif x.ndim == 3: # Maybe (B, N, D) or something?
                # wrapper says: B, C, N, P = x.shape. So assume 4D.
                B, C, N = x.shape
                zero_mask = torch.zeros((B, C, N), device=x.device, dtype=torch.long)
            else:
                 # Fallback if shape is weird, let model generate it? No, we want zero mask.
                 # Just pass None and hope? No, wrapper generates 50% mask if None.
                 # We MUST generate zero mask.
                 # Let's trust wrapper's expectation of 4D input for now.
                 if x.ndim == 4:
                     B, C, N, P = x.shape
                     zero_mask = torch.zeros((B, C, N), device=x.device, dtype=torch.long)
                 else:
                     # Attempt to construct valid mask
                     # Try to match shape of mask if it was passed by loader
                     if mask is not None:
                         zero_mask = torch.zeros_like(mask)
                     else:
                         # Risky fallback
                         print("Warning: Input x is not 4D and no mask provided. Cannot infer N.")
                         zero_mask = None # This will trigger random mask in model

            # model(x, mask) -> out, mask, feature_pred
            outputs = model(x, mask=zero_mask)
            
            feature_pred = None
            if isinstance(outputs, tuple):
                 if len(outputs) == 3:
                     _, _, feature_pred = outputs
                 elif len(outputs) == 2:
                     _, _ = outputs # No feature pred?
            
            if feature_pred is None:
                print("Error: Model did not return feature predictions. Check config 'pretrain_tasks'.")
                return

            # Initialize accumulators
            B, D = feature_pred.shape
            if sum_sq_err is None:
                sum_sq_err = torch.zeros(D, device=args.device)
                sum_y = torch.zeros(D, device=args.device)
                sum_y_sq = torch.zeros(D, device=args.device)
            
            # Accumulate stats
            # SSE
            diff = feature_pred - target_features
            sum_sq_err += torch.sum(diff ** 2, dim=0)
            
            # Stats for SST
            sum_y += torch.sum(target_features, dim=0)
            sum_y_sq += torch.sum(target_features ** 2, dim=0)
            
            # Stats for PCC: Need E[xy], E[x], E[y], E[x^2], E[y^2]
            # We already have E[y] (sum_y) and E[y^2] (sum_y_sq)
            # Need sum_x, sum_x_sq, sum_xy
            if 'sum_x' not in locals():
                sum_x = torch.zeros(D, device=args.device)
                sum_x_sq = torch.zeros(D, device=args.device)
                sum_xy = torch.zeros(D, device=args.device)
            
            sum_x += torch.sum(feature_pred, dim=0)
            sum_x_sq += torch.sum(feature_pred ** 2, dim=0)
            sum_xy += torch.sum(feature_pred * target_features, dim=0)
            
            count += B
            
            # Store raw data for scatter plots (CPU to save GPU mem)
            all_preds.append(feature_pred.cpu())
            all_targets.append(target_features.cpu())
            
    # Compute Metrics
    # MSE = SSE / N
    mse_per_channel = sum_sq_err / count
    rmse_per_channel = torch.sqrt(mse_per_channel)
    
    # Global Mean
    mean_y = sum_y / count
    mean_x = sum_x / count
    
    # SST = sum(y^2) - 2*mean*sum(y) + N*mean^2
    #     = sum_y_sq - count*mean_y^2 (Simplified)
    sst_per_channel = sum_y_sq - count * (mean_y ** 2)
    
    # R2 = 1 - SSE / SST
    # Handle division by zero (constant features)
    valid_mask = sst_per_channel > 1e-6
    r2_per_channel = torch.zeros_like(sst_per_channel)
    r2_per_channel[valid_mask] = 1 - (sum_sq_err[valid_mask] / sst_per_channel[valid_mask])
    
    # PCC Calculation
    # PCC = (E[xy] - E[x]E[y]) / (std_x * std_y)
    # std_x = sqrt(E[x^2] - (E[x])^2)
    
    # Numerator: N * Cov(x,y) = sum_xy - N * mean_x * mean_y
    numerator = sum_xy - count * mean_x * mean_y
    
    # Denominator
    var_x_n = sum_x_sq - count * (mean_x ** 2)
    var_y_n = sum_y_sq - count * (mean_y ** 2) # This is SST
    
    denominator = torch.sqrt(var_x_n * var_y_n)
    
    pcc_per_channel = torch.zeros_like(r2_per_channel)
    # Avoid div by zero
    valid_pcc = denominator > 1e-8
    pcc_per_channel[valid_pcc] = numerator[valid_pcc] / denominator[valid_pcc]

    # Prepare DataFrame
    metrics_data = {
        'Feature Index': range(len(r2_per_channel)),
        'R2': r2_per_channel.cpu().numpy(),
        'PCC': pcc_per_channel.cpu().numpy(),
        'RMSE': rmse_per_channel.cpu().numpy(),
        'MSE': mse_per_channel.cpu().numpy()
    }
    
    if feature_names:
        # Ensure length matches
        if len(feature_names) == len(r2_per_channel):
            metrics_data['Feature Name'] = feature_names
        else:
            print(f"Warning: Feature names count ({len(feature_names)}) != output dim ({len(r2_per_channel)})")
            
    df = pd.DataFrame(metrics_data)
    
    # Reorder columns if Name exists
    if 'Feature Name' in df.columns:
        cols = ['Feature Index', 'Feature Name', 'R2', 'PCC', 'RMSE', 'MSE']
        df = df[cols]
        
    # Sort by R2 descending
    df = df.sort_values('R2', ascending=False)
    
    print(f"Saving metrics to {args.output}")
    df.to_csv(args.output, index=False)
    
    print("Top 10 Features by R2:")
    print(df.head(10))

    # --- Generate Visualizations ---
    print("Generating Visualizations...")
    
    output_dir = os.path.dirname(args.output)
    if not output_dir:
        output_dir = "."
    
    output_stem = os.path.splitext(os.path.basename(args.output))[0]
    viz_base_dir = os.path.join(output_dir, f"{output_stem}_viz")
    scatter_dir = os.path.join(viz_base_dir, "scatter_plots")
    os.makedirs(scatter_dir, exist_ok=True)
    
    # Concatenate all data
    all_preds_tensor = torch.cat(all_preds, dim=0) # (N, D)
    all_targets_tensor = torch.cat(all_targets, dim=0) # (N, D)
    
    all_preds_np = all_preds_tensor.cpu().numpy()
    all_targets_np = all_targets_tensor.cpu().numpy()
    
    # 1. Generate 62 Individual Scatter Plots (Density Style)
    print(f"Generating {all_preds_np.shape[1]} individual scatter plots...")
    num_features = all_preds_np.shape[1]
    
    for i in tqdm(range(num_features), desc="Plotting Scatters"):
        fname = feature_names[i] if feature_names else f"Feature {i}"
        safe_fname = fname.replace('/', '_').replace(' ', '_')
        
        r2_val = r2_per_channel[i].item()
        pcc_val = pcc_per_channel[i].item()
        
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        
        y_true = all_targets_np[:, i]
        y_pred = all_preds_np[:, i]
        
        plot_density_scatter(ax, y_true, y_pred, fname, r2_val, pcc_val)
        
        ax.set_xlabel("True Value (Z-scored)")
        ax.set_ylabel("Predicted Value")
        plt.tight_layout()
        
        # Save as PDF for high quality, but with rasterized scatter points
        plt.savefig(os.path.join(scatter_dir, f"{safe_fname}.pdf"), dpi=300)
        plt.close()
        
    print(f"Individual scatter plots saved to {scatter_dir}/")
    
    # 2. Generate Combined Summary Figure (3 Sizes)
    print("Generating Combined Summary Figures (Small, Medium, Large)...")
    
    # Save Top Features Data for quick replotting
    top_feats_path = os.path.join(viz_base_dir, "top_features_data.npz")
    save_top_features_data(df, all_preds_np, all_targets_np, top_feats_path)
    
    # Large (Original)
    combined_path_L = os.path.join(viz_base_dir, "combined_analysis_style2_L.png")
    generate_combined_figure(df, all_preds_np, all_targets_np, feature_names, combined_path_L, figsize=(12, 8))
    
    # Medium (Standard Paper Width) -> Half Column optimized
    # Half column width is typically 3.25 to 3.5 inches.
    # If we make the figure smaller but keep font sizes, everything looks bigger.
    # Previous M was (10, 6). Let's try (8, 5) or even smaller for half-column density.
    # But user wants "canvas smaller so text is bigger".
    # If we use figsize=(8, 5) and keep font sizes, it simulates a larger relative font.
    combined_path_M = os.path.join(viz_base_dir, "combined_analysis_style2_M.png")
    generate_combined_figure(df, all_preds_np, all_targets_np, feature_names, combined_path_M, figsize=(8, 5))
    
    # Small (Compact)
    combined_path_S = os.path.join(viz_base_dir, "combined_analysis_style2_S.png")
    generate_combined_figure(df, all_preds_np, all_targets_np, feature_names, combined_path_S, figsize=(8, 5))
    
    # 3. Generate 8x8 Grid PDF (Vector/Raster Hybrid)
    print("Generating 8x8 Grid PDF (Appendix)...")
    appendix_dir = os.path.join(viz_base_dir, "appendix")
    os.makedirs(appendix_dir, exist_ok=True)
    grid_pdf_path = os.path.join(appendix_dir, "appendix_scatter_grid_8x8.pdf")
    
    # Use the new direct generation function
    generate_full_grid_figure(df, all_preds_np, all_targets_np, feature_names, grid_pdf_path)
    
    print(f"8x8 Grid PDF saved to {grid_pdf_path}")
    print(f"Combined figures saved to {viz_base_dir}")

if __name__ == "__main__":
    main()
