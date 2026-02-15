
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import os

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

def generate_figure_from_saved_data(df, top_features_data, output_path, figsize=(8, 5)):
    # ICML Style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8])
    
    # --- LEFT: 2x2 Best Features ---
    # Define Explicit Order: Time -> Power -> Structure -> Ratios
    cat_order_list = ['Time', 'Power', 'Structure', 'Ratios']
    
    gs_left = gs[:, 0].subgridspec(2, 2, wspace=0.3, hspace=0.3)
    
    for i, cat in enumerate(cat_order_list):
        ax = fig.add_subplot(gs_left[i//2, i%2])
        
        # Load data from NPZ
        # Keys are like "Time_name", "Time_true", etc.
        if f"{cat}_name" not in top_features_data:
            continue
            
        feat_name = str(top_features_data[f"{cat}_name"])
        y_true = top_features_data[f"{cat}_true"]
        y_pred = top_features_data[f"{cat}_pred"]
        r2 = top_features_data[f"{cat}_r2"]
        pcc = top_features_data[f"{cat}_pcc"]
        
        plot_density_scatter(ax, y_true, y_pred, f"{cat}: {feat_name}", r2, pcc)
        
        if i%2 == 0: ax.set_ylabel("Predicted")
        if i//2 == 1: ax.set_xlabel("True")
        
    # --- RIGHT: Bar Chart (Style 2 Cool Categorical) ---
    ax_right = fig.add_subplot(gs[:, 1])
    
    # Process DF for Bar Chart
    df = df.copy()
    if 'Category' not in df.columns:
        df['Category'] = df['Feature Name'].apply(get_feature_category)
    
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
    
    # Set X-axis to 1.0 and add vertical line
    ax_right.set_xlim(0, 1.05)
    ax_right.axvline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    
    # Legend on Top
    handles = [plt.Rectangle((0,0),1,1, color=pal[c]) for c in cat_order_list if c in pal]
    
    ax_right.legend(handles, [c for c in cat_order_list if c in pal], 
              loc='lower center', bbox_to_anchor=(0.5, 1.02), 
              ncol=4, frameon=False, fontsize=9)
    
    sns.despine(ax=ax_right, left=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to {output_path}")
    plt.close()

if __name__ == "__main__":
    base_dir = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final_viz"
    csv_path = os.path.join(base_dir, "val_full_final.csv")
    npz_path = os.path.join(base_dir, "top_features_data.npz")
    output_path = os.path.join(base_dir, "combined_analysis_style2_M.pdf")
    
    print(f"Loading data from {base_dir}...")
    if not os.path.exists(csv_path) or not os.path.exists(npz_path):
        print("Error: Data files not found.")
        print(f"CSV: {os.path.exists(csv_path)}")
        print(f"NPZ: {os.path.exists(npz_path)}")
        exit(1)
        
    df = pd.read_csv(csv_path)
    top_features = np.load(npz_path)
    
    generate_figure_from_saved_data(df, top_features, output_path, figsize=(8, 5))
