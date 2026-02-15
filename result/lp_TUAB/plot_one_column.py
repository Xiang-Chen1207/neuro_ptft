
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np

# --- DATA ---
ratios = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
ratios_fraction = [r/100.0 for r in ratios]
ratios_labels = ['0.5%', '1%', '5%', '10%', '20%', '50%', '100%']

# LP Data (from lp_TUAB.md)
lp_recon = [60.25, 66.38, 66.43, 66.32, 65.62, 67.00, 67.12]
lp_ours = [64.09, 69.53, 69.21, 72.24, 72.59, 74.10, 74.70]
lp_feat = [64.44, 67.34, 67.94, 70.13, 70.37, 72.88, 73.32]

# FT Data (from ft_tuab.md)
ft_recon = [63.44, 69.91, 71.88, 75.39, 75.69, 75.97, 77.66]
ft_ours = [62.52, 69.47, 75.60, 78.87, 78.22, 78.12, 80.55]
ft_feat = [67.58, 66.74, 73.72, 73.76, 75.64, 76.60, 77.95]

def create_df(recon, ours, feat):
    data = {
        'Ratio': ratios_fraction * 3,
        'BAcc': recon + ours + feat,
        'Model': ['Recon.'] * 7 + ['Ours'] * 7 + ['Feat.'] * 7
    }
    df = pd.DataFrame(data)
    ratio_to_idx = {r: i for i, r in enumerate(ratios_fraction)}
    df['Step'] = df['Ratio'].map(ratio_to_idx)
    return df

df_lp = create_df(lp_recon, lp_ours, lp_feat)
df_ft = create_df(ft_recon, ft_ours, ft_feat)

# --- PLOTTING ---
sns.set_context("paper", font_scale=2.0) # Larger font as requested
sns.set_style("whitegrid")

# Stacked layout for single column width
# Using sharex=True to save space
fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

palette = sns.color_palette("vlag", n_colors=3)
curve_palette = {'Recon.': palette[0], 'Feat.': '#999999', 'Ours': '#b52b2b'}

def plot_ax(ax, df, title):
    sns.lineplot(data=df, x='Step', y='BAcc', hue='Model', 
                 style='Model', markers=['o', 's', 'D'], 
                 dashes=False, palette=curve_palette,
                 linewidth=3.5, markersize=12, ax=ax)
    
    ax.set_xticks(range(len(ratios_labels)))
    ax.set_xticklabels(ratios_labels)
    ax.set_ylabel("Bal. Acc. (%)", fontweight='bold')
    
    # Title centered
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    ax.grid(True, which="major", axis='y', ls="--", alpha=0.3)
    # ax.legend().remove() # Keep legend handles for later, but we will add it manually
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine(ax=ax)

plot_ax(axes[0], df_lp, "(a) Linear Probing")
plot_ax(axes[1], df_ft, "(b) Full Finetuning")

axes[1].set_xlabel("Training Data Ratio", fontweight='bold')
axes[0].set_xlabel("") # Hide x-label for top plot

# Shared Legend at the top right of the first plot, vertical
handles, labels = axes[0].get_legend_handles_labels()
# Filter unique handles/labels if necessary (but here they are from one plot)
# The order is Recon, Ours, Feat.
axes[0].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
           ncol=1, frameon=False, fontsize=16)

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(output_dir, 'tuab_combined_one_column.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'tuab_combined_one_column.png'), bbox_inches='tight')
print(f"One column plots generated in {output_dir}")
