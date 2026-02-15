
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data
ratios = [0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
# Convert to fraction for x-axis if needed, or keep as percentage
ratios_fraction = [r/100.0 for r in ratios]
ratios_labels = ['0.5%', '1%', '5%', '10%', '20%', '50%', '100%']

# BAcc Data
# Recon. (Baseline | EEG) - Full Finetuning (Test BAcc)
recon_scores = [63.44, 69.91, 71.88, 75.39, 75.69, 75.97, 77.66]

# Ours (Neuro-KE | eeg) - Full Finetuning (Test BAcc)
ours_scores = [62.52, 69.47, 75.60, 78.87, 78.22, 78.12, 80.55]

# Feat. (FeatOnly | feat) - Full Finetuning (Test BAcc)
feat_scores = [67.58, 66.74, 73.72, 73.76, 75.64, 76.60, 77.95]

data = {
    'Ratio': ratios_fraction * 3,
    'BAcc': recon_scores + ours_scores + feat_scores,
    'Model': ['Recon.'] * 7 + ['Ours'] * 7 + ['Feat.'] * 7
}

df = pd.DataFrame(data)
# Map ratios to uniform indices
ratio_to_idx = {r: i for i, r in enumerate(ratios_fraction)}
df['Step'] = df['Ratio'].map(ratio_to_idx)

# Set ICML style
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks") # or whitegrid

# Colors from vlag palette (n_colors=3)
# vlag with 3 colors: Blue, Grey, Red
# Ours should be Red (last one in vlag?), Recon Blue (first?), Feat Grey (middle?)
# Let's check palette
palette = sns.color_palette("vlag", n_colors=3)
# Usually vlag is Blue -> Grey -> Red.
# So index 0: Blue (Recon.), index 1: Grey (Feat.), index 2: Red (Ours)
# But we need to map specific model names to these colors.
model_colors = {
    'Recon.': palette[0],
    'Feat.': palette[1],
    'Ours': palette[2]
}

def plot_style_1(filename):
    """Standard Line Plot with markers"""
    plt.figure(figsize=(8, 6))
    # Use 'Step' for uniform x-axis spacing
    ax = sns.lineplot(data=df, x='Step', y='BAcc', hue='Model', style='Model', 
                      markers=True, dashes=False, palette=model_colors,
                      linewidth=2.5, markersize=9)
    
    # Custom ticks for uniform scale
    ax.set_xticks(range(len(ratios_labels)))
    ax.set_xticklabels(ratios_labels)
    
    plt.xlabel("Training Data Ratio", fontweight='bold')
    plt.ylabel("Balanced Accuracy (%)", fontweight='bold')
    
    # Legend
    plt.legend(frameon=False, loc='lower right')
    
    sns.despine()
    plt.grid(True, which="major", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_style_2(filename):
    """Linear scale x-axis (as requested '0.05 to 1') - Now Uniform Step"""
    # Filter data for >= 0.05 if strictly following user request
    # User said "0.05 to 1". 0.05 is 5%.
    df_filtered = df[df['Ratio'] >= 0.05]
    
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=df_filtered, x='Step', y='BAcc', hue='Model', style='Model', 
                      markers=True, dashes=False, palette=model_colors,
                      linewidth=3, markersize=10)
    
    # Uniform scale for subset
    # Ratios >= 0.05 are indices 2, 3, 4, 5, 6
    subset_indices = [i for i, r in enumerate(ratios_fraction) if r >= 0.05]
    subset_labels = [ratios_labels[i] for i in subset_indices]
    
    ax.set_xticks(subset_indices)
    ax.set_xticklabels(subset_labels)
    
    plt.xlabel("Training Data Ratio", fontweight='bold')
    plt.ylabel("Balanced Accuracy (%)", fontweight='bold')
    
    plt.legend(frameon=False, loc='lower right')
    
    sns.despine()
    plt.grid(True, axis='y', ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_style_3(filename):
    """All data but nicely formatted with log scale - The 'Paper' look"""
    # Make it smaller and more square-like (e.g., 5x4 or 5x5)
    plt.figure(figsize=(5, 4))
    
    # Custom palette assignment to ensure Ours is Red
    # vlag: Blue, White, Red
    # User wants: "Bottom to Top -> Light to Dark"
    # Recon (Bottom): Lightest (Light Blue)
    # Feat (Middle): Medium (Medium Grey)
    # Ours (Top): Darkest (Dark/Deep Red)
    
    # Define colors manually to achieve this gradient of depth while keeping hue
    light_blue = '#8DA0CB' # A nice light blue (from Set2 or similar)
    medium_grey = '#969696' # Medium grey, lighter than #555555
    deep_red = '#C44E52'    # A deeper red (seaborn deep palette red)
    
    # Or stick closer to vlag hues but adjust lightness
    # vlag[0] is light blue. vlag[2] is light red.
    # We keep Recon as vlag[0] (or similar), make Feat a balanced grey, make Ours a strong red.
    
    custom_palette = {
        'Recon.': palette[0],      # Keep the vlag light blue
        'Feat.': '#999999',        # Lighter grey (was #555555)
        'Ours': '#b52b2b'          # Deep red (darker than vlag[2])
    }
    
    ax = sns.lineplot(data=df, x='Step', y='BAcc', hue='Model', 
                      style='Model', markers=['o', 's', 'D'], 
                      dashes=False, palette=custom_palette,
                      linewidth=2, markersize=8)
    
    # ax.set_xscale('log') # Removed log scale
    # Custom ticks for uniform scale
    ax.set_xticks(range(len(ratios_labels)))
    ax.set_xticklabels(ratios_labels)
    
    plt.xlabel("Training Data Ratio", fontsize=14)
    plt.ylabel("Balanced Accuracy (%)", fontsize=14)
    
    # Add annotation for max gain
    # max_gain = ours_scores[-1] - recon_scores[-1]
    # plt.annotate(f'+{max_gain:.1f}%', xy=(1.0, ours_scores[-1]), xytext=(0.8, ours_scores[-1]+2),
    #              arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend(title='', frameon=False, fontsize=12)
    sns.despine()
    plt.grid(True, which="major", axis='y', ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
plot_style_1('tuab_learning_curve_log_all.pdf')
plot_style_1('tuab_learning_curve_log_all.png')

plot_style_2('tuab_learning_curve_linear_subset.pdf')
plot_style_2('tuab_learning_curve_linear_subset.png')

plot_style_3('tuab_learning_curve_icml.pdf')
plot_style_3('tuab_learning_curve_icml.png')

print("Plots generated.")
