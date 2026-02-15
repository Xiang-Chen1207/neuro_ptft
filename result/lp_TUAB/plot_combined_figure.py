
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data (Linear Probing)
ratios = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
ratios_fraction = [r/100.0 for r in ratios]
ratios_labels = ['0.5%', '1%', '5%', '10%', '20%', '50%', '100%']

# LP Data
lp_recon_scores = [60.25, 66.38, 66.43, 66.32, 65.62, 67.00, 67.12]
lp_ours_scores = [64.09, 69.53, 69.21, 72.24, 72.59, 74.10, 74.70]
lp_feat_scores = [64.44, 67.34, 67.94, 70.13, 70.37, 72.88, 73.32]

# FT Data (Currently placeholder - copy of LP data)
ft_recon_scores = list(lp_recon_scores)
ft_ours_scores = list(lp_ours_scores)
ft_feat_scores = list(lp_feat_scores)

# Create DataFrames
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

df_lp = create_df(lp_recon_scores, lp_ours_scores, lp_feat_scores)
df_ft = create_df(ft_recon_scores, ft_ours_scores, ft_feat_scores)

# Set Style
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")

# Colors
palette = sns.color_palette("vlag", n_colors=3)
custom_palette = {
    'Recon.': palette[0],      # Light Blue
    'Feat.': '#999999',        # Grey
    'Ours': '#b52b2b'          # Deep Red
}

def plot_combined(filename):
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # Plot (a) Linear Probing
    sns.lineplot(data=df_lp, x='Step', y='BAcc', hue='Model', 
                 style='Model', markers=['o', 's', 'D'], 
                 dashes=False, palette=custom_palette,
                 linewidth=2, markersize=8, ax=axes[0])
    
    axes[0].set_title('(a) Linear Probing', fontsize=16, pad=15, fontweight='bold')
    axes[0].set_xlabel("Training Data Ratio", fontsize=14)
    axes[0].set_ylabel("Balanced Accuracy (%)", fontsize=14)
    axes[0].set_xticks(range(len(ratios_labels)))
    axes[0].set_xticklabels(ratios_labels)
    axes[0].grid(True, which="major", axis='y', ls="--", alpha=0.3)
    axes[0].legend(title='', frameon=False, fontsize=12, loc='lower right')
    
    # Plot (b) Full Finetuning
    sns.lineplot(data=df_ft, x='Step', y='BAcc', hue='Model', 
                 style='Model', markers=['o', 's', 'D'], 
                 dashes=False, palette=custom_palette,
                 linewidth=2, markersize=8, ax=axes[1])
    
    axes[1].set_title('(b) Full Finetuning', fontsize=16, pad=15, fontweight='bold')
    axes[1].set_xlabel("Training Data Ratio", fontsize=14)
    axes[1].set_ylabel("") # Share y-label with left plot
    axes[1].set_xticks(range(len(ratios_labels)))
    axes[1].set_xticklabels(ratios_labels)
    axes[1].grid(True, which="major", axis='y', ls="--", alpha=0.3)
    axes[1].legend(title='', frameon=False, fontsize=12, loc='lower right')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

plot_combined('tuab_combined_comparison.png')
plot_combined('tuab_combined_comparison.pdf')
print("Combined plots generated.")
