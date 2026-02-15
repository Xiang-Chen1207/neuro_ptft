
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.stats import gaussian_kde

# Style Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})

def get_feature_category(feature_name):
    """Map feature name to one of the 4 Main Categories."""
    name = feature_name.lower()
    if 'ratio' in name: return 'Ratios'
    if any(k in name for k in ['aperiodic', 'spectral', 'peak_frequency', 'individual_alpha']): return 'Structure'
    if any(k in name for k in ['power', 'alpha', 'beta', 'delta', 'theta', 'gamma']): return 'Power'
    return 'Time'

def plot_density_scatter_clean(ax, y_true, y_pred, feature_name, r2, pcc, category, color):
    """
    Cleaner scatter plot for half-column layout.
    - Title: Clean Feature Name (Large)
    - Category: Indicated by colored text tag or border
    """
    # Subsample
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
    
    # Scatter
    ax.scatter(x, y, c=z, s=8, cmap='mako', edgecolor='none', alpha=0.6, rasterized=True)
    
    # Diagonal
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, color='#333333', linestyle='--', lw=1, alpha=0.6)
    
    # Simplify Feature Name
    # e.g. alpha_relative_power -> Alpha Relative Power
    clean_name = feature_name.replace('_', ' ').title()
    # Remove redundant words if needed? e.g. "Relative Power" -> "Alpha Rel. Power"
    # For now, just Title Case
    
    # Title (Large Font)
    ax.set_title(clean_name, fontsize=12, fontweight='bold', pad=10)
    
    # Category Tag (Top Left, colored)
    ax.text(0.05, 0.95, category, transform=ax.transAxes,
            va='top', ha='left', fontsize=10, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.9, lw=1.5))
            
    # Stats (Bottom Right)
    stats_text = f"$R^2={r2:.2f}$\n$r={pcc:.2f}$"
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            va='bottom', ha='right', fontsize=10, fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#dddddd', alpha=0.8))

def main():
    base_dir = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final_viz"
    data_path = os.path.join(base_dir, "top_features_data.npz")
    csv_path = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/val_full_final.csv"
    output_path = os.path.join(base_dir, "combined_analysis_half_col.png")
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run eval_features.py once to generate it.")
        return
        
    data = np.load(data_path)
    df = pd.read_csv(csv_path)
    
    # Figure Setup (Half Column Optimized)
    # 8x5 is good for density, but maybe slightly taller for text space?
    fig = plt.figure(figsize=(9, 6)) 
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 0.7]) # Left part wider for scatters
    
    # Colors
    pal = {
        'Time': '#4c78a8', # Steel Blue
        'Power': '#72b7b2', # Teal
        'Structure': '#b279a2', # Muted Purple
        'Ratios': '#54a24b'  # Muted Green
    }
    
    categories = ['Time', 'Power', 'Structure', 'Ratios']
    
    # --- LEFT: 2x2 Scatters ---
    gs_left = gs[:, 0].subgridspec(2, 2, wspace=0.35, hspace=0.4)
    
    for i, cat in enumerate(categories):
        ax = fig.add_subplot(gs_left[i//2, i%2])
        
        feat_name = str(data[f"{cat}_name"])
        r2 = float(data[f"{cat}_r2"])
        pcc = float(data[f"{cat}_pcc"])
        y_true = data[f"{cat}_true"]
        y_pred = data[f"{cat}_pred"]
        
        plot_density_scatter_clean(ax, y_true, y_pred, feat_name, r2, pcc, cat, pal[cat])
        
        if i%2 == 0: ax.set_ylabel("Predicted", fontsize=10)
        if i//2 == 1: ax.set_xlabel("True", fontsize=10)
        
    # --- RIGHT: Bar Chart ---
    ax_right = fig.add_subplot(gs[:, 1])
    
    # Sort Logic
    df['Category'] = df['Feature Name'].apply(get_feature_category)
    cat_dtype = pd.CategoricalDtype(categories=categories[::-1], ordered=True)
    df['Category'] = df['Category'].astype(cat_dtype)
    df_sorted = df.sort_values(by=['Category', 'R2'], ascending=[True, True])
    
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
    ax_right.set_xlabel("R2 Score", fontsize=12) # Removed bold
    
    # X-Axis Limit & Line
    ax_right.set_xlim(0, 1.1)
    ax_right.axvline(1.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    
    # Legend (Category Blocks)
    handles = [plt.Rectangle((0,0),1,1, color=pal[c]) for c in categories if c in pal]
    
    # Move legend down slightly (was 1.02). 
    # If we put it at 0.98, it might overlap the top bar.
    # Let's try putting it INSIDE the plot at the bottom right? 
    # Or just closer to the top edge. 
    # User said "don't stick out" (凸出来). Maybe inside the top margin?
    # Let's try y=0.96 with no frame, assuming transparency handles overlap or bars aren't full width there.
    # Actually, if bars go to 1.0, top bars are full width. Overlap is bad.
    # Maybe move to Bottom?
    # Or make the plot title/legend integrated?
    # Let's try y=1.0 (flush with top) first.
    ax_right.legend(handles, [c for c in categories if c in pal], 
              loc='lower center', bbox_to_anchor=(0.5, 1.0), 
              ncol=2, frameon=False, fontsize=10)
    
    sns.despine(ax=ax_right, left=True)
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved optimized plot to {output_path}")
    
    # Save as PDF (Vector text, Raster scatter)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved optimized PDF to {pdf_path}")

if __name__ == "__main__":
    main()
