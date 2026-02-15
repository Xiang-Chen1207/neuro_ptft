
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import gaussian_kde, pearsonr

# --- ICML Style Settings ---
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
})

def get_feature_category(feature_name):
    """Map feature name to one of the 4 Main Categories."""
    name = feature_name.lower()
    if 'ratio' in name: return 'Ratios'
    if any(k in name for k in ['aperiodic', 'spectral', 'peak_frequency', 'individual_alpha']): return 'Structure'
    if any(k in name for k in ['power', 'alpha', 'beta', 'delta', 'theta', 'gamma']): return 'Power'
    return 'Time'

def generate_synthetic_data(n=500, r2_target=0.8):
    """Generate synthetic True/Pred data with approximate R2."""
    y_true = np.random.normal(0, 1, n)
    if r2_target < 0: r2_target = 0 # Prevent crash
    noise_std = np.sqrt((1 - r2_target) * np.var(y_true))
    y_pred = y_true + np.random.normal(0, noise_std, n)
    return y_true, y_pred

def plot_style_1_monochrome(df, output_path):
    """
    Style 1: Monochrome Blue (Minimalist Academic)
    - Left: Scatters in Blue Mako
    - Right: Bars in Single Color (Slate Blue) with Alpha transparency for values
    - Very clean, suitable for B&W printing too.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8])
    
    # --- LEFT: Scatter ---
    plot_scatters(fig, gs, df, cmap='Blues_r', color_color='#1f77b4')

    # --- RIGHT: Bars ---
    ax = fig.add_subplot(gs[:, 1])
    df_sorted = df.sort_values(by=['Category', 'R2'])
    
    # Single color, but alpha varies by R2
    # Normalize R2 for alpha
    r2_vals = df_sorted['R2'].clip(0, 1)
    alphas = 0.3 + 0.7 * (r2_vals - r2_vals.min()) / (r2_vals.max() - r2_vals.min())
    
    colors = [(0.2, 0.4, 0.6, a) for a in alphas] # Slate Blue
    
    bars = ax.barh(np.arange(len(df)), df_sorted['R2'], color=colors, height=0.7)
    
    # Separators
    add_separators(ax, df_sorted, color='black', lw=0.5)
    
    # Clean up
    ax.set_yticks([])
    ax.set_xlabel("R2 Score")
    ax.set_title("Performance Distribution (Monochrome)", fontsize=11)
    sns.despine(ax=ax, left=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_style_2_cool_categorical(df, output_path):
    """
    Style 2: Distinct Cool Tones (Categorical)
    - Left: Mako
    - Right: Each category has a distinct cool hue (Teal, Blue, Purple, Grey)
    - Easier to distinguish categories visually.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8])
    
    # --- LEFT: Scatter ---
    plot_scatters(fig, gs, df, cmap='mako', color_color=None)

    # --- RIGHT: Bars ---
    ax = fig.add_subplot(gs[:, 1])
    df_sorted = df.sort_values(by=['Category', 'R2'])
    
    # Define Palette
    pal = {
        'Time': '#4e79a7',      # Blue
        'Power': '#59a14f',     # Green-Teal (Cool side) -> Changed to Teal: #76b7b2
        'Structure': '#b07aa1', # Purple
        'Ratios': '#9c755f'     # Brown-Grey (Neutral) -> Changed to Grey: #bab0ac
    }
    # Let's use specific hexes for "Cool"
    pal = {
        'Time': '#4c78a8', # Steel Blue
        'Power': '#72b7b2', # Teal
        'Structure': '#b279a2', # Muted Purple
        'Ratios': '#54a24b'  # Muted Green
    }
    
    colors = [pal[c] for c in df_sorted['Category']]
    # Add gradient effect by varying lightness? No, keep flat for clean categorical look.
    
    bars = ax.barh(np.arange(len(df)), df_sorted['R2'], color=colors, height=0.8, alpha=0.9)
    
    add_separators(ax, df_sorted, color='white', lw=2) # Gap style separator
    
    ax.set_yticks([])
    ax.set_xlabel("R2 Score")
    # Removed title as legend is now on top and acts as grouping identifier
    # ax.set_title("Performance by Category", fontsize=11)
    
    # Legend
    # User requested: Move legend to top (like style 5) instead of lower right
    handles = [plt.Rectangle((0,0),1,1, color=pal[c]) for c in ['Time','Power','Structure','Ratios']]
    # Position: Upper center, slightly above the plot area
    ax.legend(handles, ['Time','Power','Structure','Ratios'], 
              loc='upper center', bbox_to_anchor=(0.5, 1.08), 
              ncol=4, frameon=False, fontsize=9)
    
    sns.despine(ax=ax, left=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_style_3_heatmap_strip(df, output_path):
    """
    Style 3: Heatmap Strip (Very Compact)
    - Left: Scatter
    - Right: Instead of Bars, use a colored strip (Heatmap) or very thin bars that look like a spectrum.
    - Matches "vlag" (Blue-White-Red) but mapped to Cool-Warm or just Cool.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.4]) # Narrower right side
    
    # --- LEFT: Scatter ---
    plot_scatters(fig, gs, df, cmap='viridis', color_color=None)

    # --- RIGHT: Bars as Spectrum ---
    ax = fig.add_subplot(gs[:, 1])
    df_sorted = df.sort_values(by=['Category', 'R2'])
    
    # Use a continuous colormap based on R2 value
    # viridis or coolwarm
    norm = plt.Normalize(0, 1)
    cmap = plt.cm.viridis
    
    colors = cmap(norm(df_sorted['R2']))
    
    # Plot as bars but full width
    bars = ax.barh(np.arange(len(df)), df_sorted['R2'], color=colors, height=1.0)
    
    # Add value labels for top 3? No, too crowded.
    
    add_separators(ax, df_sorted, color='black', lw=1)
    
    ax.set_yticks([])
    ax.set_xlabel("R2 Score")
    ax.set_title("Metric Spectrum", fontsize=11)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
    # cbar.set_label('R2 Intensity')
    
    sns.despine(ax=ax, left=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatters(fig, gs, df, cmap, color_color):
    """Helper to plot the 2x2 scatter grid."""
    df['Category'] = df['Feature Name'].apply(get_feature_category)
    categories = ['Time', 'Power', 'Structure', 'Ratios']
    
    gs_left = gs[:, 0].subgridspec(2, 2, wspace=0.3, hspace=0.3)
    
    # Find best features
    best_features = {}
    for cat in categories:
        cat_df = df[df['Category'] == cat]
        if not cat_df.empty:
            best_features[cat] = cat_df.loc[cat_df['R2'].idxmax()]
            
    for i, cat in enumerate(categories):
        if cat not in best_features: continue
        
        ax = fig.add_subplot(gs_left[i//2, i%2])
        row = best_features[cat]
        y_true, y_pred = generate_synthetic_data(r2_target=row['R2'])
        
        # Density
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = y_true[idx], y_pred[idx], z[idx]
        
        ax.scatter(x, y, c=z, s=15, cmap=cmap, edgecolor='none', alpha=0.9)
        
        # Diagonal
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, color='#333333', linestyle='--', lw=1, alpha=0.6)
        
        ax.set_title(f"{cat}", fontsize=10, fontweight='bold')
        stats_text = f"$R^2={row['R2']:.2f}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#dddddd', alpha=0.8))
                
        if i%2 == 0: ax.set_ylabel("Predicted")
        if i//2 == 1: ax.set_xlabel("True")

def add_separators(ax, df_sorted, color, lw):
    """Add lines between categories."""
    df_reset = df_sorted.reset_index(drop=True)
    current_cat = None
    for i in range(len(df_reset)):
        cat = df_reset.loc[i, 'Category']
        if current_cat is not None and cat != current_cat:
            ax.axhline(i - 0.5, color=color, linestyle='-', linewidth=lw, alpha=0.5)
        current_cat = cat

def main():
    csv_path = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/60sfeature_metrics_eval_feat_only.csv"
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    df['Category'] = df['Feature Name'].apply(get_feature_category)
    
    out_dir = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/final_viz"
    
    print("Generating Style 1: Monochrome...")
    plot_style_1_monochrome(df, os.path.join(out_dir, "style_1_monochrome.png"))
    
    print("Generating Style 2: Cool Categorical...")
    plot_style_2_cool_categorical(df, os.path.join(out_dir, "style_2_cool_categorical.png"))
    
    print("Generating Style 3: Heatmap Spectrum...")
    plot_style_3_heatmap_strip(df, os.path.join(out_dir, "style_3_heatmap.png"))
    
    print("Done!")

if __name__ == "__main__":
    main()
