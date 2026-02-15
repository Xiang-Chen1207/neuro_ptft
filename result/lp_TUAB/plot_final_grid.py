
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec

# --- DATA PREPARATION ---

# 1. Bar Chart Data (from plot_results.py)
# Note: Using the exact data structure from the user's existing script
bar_data_lp = {
    'Dataset': ['BCIC2A', 'BCIC2A', 'BCIC2A', 
                'SEED', 'SEED', 'SEED',
                'TUEP', 'TUEP', 'TUEP',
                'TUEV', 'TUEV', 'TUEV',
                'TUSZ', 'TUSZ', 'TUSZ',
                'TUAB', 'TUAB', 'TUAB'],
    'Model': ['Recon.', 'Feat.', 'Ours'] * 6,
    'BAcc': [27.34, 28.73, 30.21,
             36.43, 35.16, 41.90,
             58.99, 61.78, 62.65,
             33.75, 34.15, 43.33,
             31.97, 33.22, 34.02,
             67.12, 73.32, 74.70]
}
df_bar_lp = pd.DataFrame(bar_data_lp)
df_bar_ft = df_bar_lp.copy() # Placeholder for Full Finetuning (using same data as requested)

# 2. Curve Data (from plot_tuab_curve.py)
ratios = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
ratios_fraction = [r/100.0 for r in ratios]
ratios_labels = ['0.5', '1', '5', '10', '20', '50', '100']

lp_recon_scores = [60.25, 66.38, 66.43, 66.32, 65.62, 67.00, 67.12]
lp_ours_scores = [64.09, 69.53, 69.21, 72.24, 72.59, 74.10, 74.70]
lp_feat_scores = [64.44, 67.34, 67.94, 70.13, 70.37, 72.88, 73.32]

# Using real Full Finetuning data (Test BAcc)
ft_recon_scores = [63.44, 69.91, 71.88, 75.39, 75.69, 75.97, 77.66]
ft_ours_scores = [62.52, 69.47, 75.60, 78.87, 78.22, 78.12, 80.55]
ft_feat_scores = [67.58, 66.74, 73.72, 73.76, 75.64, 76.60, 77.95]

def create_curve_df(recon, ours, feat):
    data = {
        'Ratio': ratios_fraction * 3,
        'BAcc': recon + ours + feat,
        'Model': ['Recon.'] * 7 + ['Ours'] * 7 + ['Feat.'] * 7
    }
    df = pd.DataFrame(data)
    ratio_to_idx = {r: i for i, r in enumerate(ratios_fraction)}
    df['Step'] = df['Ratio'].map(ratio_to_idx)
    return df

df_curve_lp = create_curve_df(lp_recon_scores, lp_ours_scores, lp_feat_scores)
df_curve_ft = create_curve_df(ft_recon_scores, ft_ours_scores, ft_feat_scores)

# --- PLOTTING ---

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Create Figure
# Layout: 1 Row, 2 Columns (Side-by-side)
# "Thinner" aspect ratio: make the figure less wide relative to height.
# Standard 1-column width is small, so we generate a figure that fits well.
# Try figsize=(10, 6) -> Each plot is 5x6 (Tall/Thin)
fig, axes = plt.subplots(1, 2, figsize=(7.5, 6))
plt.subplots_adjust(wspace=0.3)

# Colors
palette = sns.color_palette("vlag", n_colors=3)
curve_palette = {
    'Recon.': palette[0],
    'Feat.': '#999999',
    'Ours': '#b52b2b'
}

def plot_curve(df, ax, label):
    sns.lineplot(data=df, x='Step', y='BAcc', hue='Model', 
                 style='Model', markers=['o', 's', 'D'], 
                 dashes=False, palette=curve_palette,
                 linewidth=3, markersize=10, ax=ax)
    
    # Adjust x-axis limits to give more space on the right
    ax.set_xlim(-0.5, len(ratios_labels) - 0.5)
    
    ax.set_xticks(range(len(ratios_labels)))
    ax.set_xticklabels(ratios_labels, rotation=0, ha='center') # No rotation needed now
    # Manually adjust the last label if needed or just rely on layout
    # Or make the font smaller for tick labels
    ax.tick_params(axis='x', labelsize=12) # Restore normal size
    
    ax.set_xlabel("Training Data Ratio (%)", fontweight='bold')
    ax.set_ylabel("Bal. Acc. (%)", fontweight='bold')
    
    # Add (a) / (b) label at top-left
    # ax.set_title(label, loc='left', fontsize=18, fontweight='bold', pad=10)
    # Alternatively, place it inside or just outside
    ax.text(-0.1, 1.05, label, transform=ax.transAxes, 
            ha='left', va='bottom', fontsize=20, fontweight='bold')
    
    ax.grid(True, which="major", axis='y', ls="--", alpha=0.3)
    
    # Force integer ticks on Y-axis
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Legend handling
    # We will remove individual legends and add a global one if needed, 
    # or keep them if they fit.
    # User said "Do not write linear probing and fine tuning" (Titles).
    # Assuming Legend (Model names) is still needed.
    ax.legend(title='', frameon=False, fontsize=14, loc='lower right')
    sns.despine(ax=ax)

# Plotting
plot_curve(df_curve_lp, axes[0], "(a)")
plot_curve(df_curve_ft, axes[1], "(b)")

# Remove individual legends to create a shared one?
# Or keep them if they are identical?
# Usually shared legend is better for side-by-side.
handles, labels = axes[0].get_legend_handles_labels()
axes[0].get_legend().remove()
axes[1].get_legend().remove()

# Add Shared Legend at the bottom or top
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=3, frameon=False, fontsize=16)

plt.tight_layout()
# Adjust for legend space
plt.subplots_adjust(bottom=0.15) 

plt.savefig('tuab_final_grid_plot_v2.png', dpi=300, bbox_inches='tight')
plt.savefig('tuab_final_grid_plot_v2.pdf', dpi=300, bbox_inches='tight')
print("Final grid plot generated.")
