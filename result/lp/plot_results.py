
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data from the provided markdown file with renamed models
data = {
    'Dataset': ['BCIC2A', 'BCIC2A', 'BCIC2A', 
                'SEED', 'SEED', 'SEED',
                'TUEP', 'TUEP', 'TUEP',
                'TUEV', 'TUEV', 'TUEV',
                'TUSZ', 'TUSZ', 'TUSZ',
                'TUAB', 'TUAB', 'TUAB'],
    'Model': ['Recon.', 'Feat.', 'Ours',
              'Recon.', 'Feat.', 'Ours',
              'Recon.', 'Feat.', 'Ours',
              'Recon.', 'Feat.', 'Ours',
              'Recon.', 'Feat.', 'Ours',
              'Recon.', 'Feat.', 'Ours'],
    'BAcc': [27.34, 28.73, 30.21,
             36.43, 35.16, 41.90,
             58.99, 61.78, 62.65,
             33.75, 34.15, 43.33,
             31.97, 33.22, 34.02,
             67.12, 73.32, 74.70]
}

df = pd.DataFrame(data)

# Set the context for ICML paper
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Get unique datasets
datasets = df['Dataset'].unique()
num_datasets = len(datasets)

# Create subplots: 1 row, N columns
# Make it wider and shorter (flatter)
fig, axes = plt.subplots(1, num_datasets, figsize=(18, 3.5), sharey=False)

# Define the order and palette
hue_order = ['Recon.', 'Feat.', 'Ours']
palette = sns.color_palette("vlag", n_colors=3)

# Iterate through datasets and plot
for i, dataset in enumerate(datasets):
    ax = axes[i]
    dataset_data = df[df['Dataset'] == dataset]
    
    # Create barplot for this dataset
    sns.barplot(
        x="Dataset", 
        y="BAcc", 
        hue="Model", 
        data=dataset_data, 
        ax=ax,
        hue_order=hue_order, 
        palette=palette, 
        edgecolor=".2"
    )
    
    # Customize axis
    ax.set_xlabel("")
    # ax.set_ylabel("Balanced Accuracy (%)" if i == 0 else "") # Only first one gets label? No, different scales needed.
    ax.set_ylabel("Bal. Acc. (%)") # All get label since scales differ
    
    # Set title as the dataset name at the bottom or top? 
    # Reference image has dataset name at the bottom (x-axis tick label).
    # Since x="Dataset", seaborn automatically puts the dataset name as the tick label.
    # But we only have one tick per plot.
    
    # Clean up legend (we'll add a common one later)
    ax.get_legend().remove()
    
    # Adjust y-axis limits to zoom in on the differences
    # Find min and max for this dataset
    y_min = dataset_data['BAcc'].min()
    y_max = dataset_data['BAcc'].max()
    margin = (y_max - y_min) * 0.5  # Add some margin
    # Set lower limit to somewhat below min, but not 0 to show differences, unless 0 is important.
    # The reference image shows y-axes starting not at 0 (e.g., 57.0 for FACED).
    ax.set_ylim(bottom=max(0, y_min - margin), top=y_max + margin * 0.5)
    
    # Despine
    sns.despine(ax=ax, left=False, bottom=False) # Keep left spine for y-axis

# Add a shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
output_path_pdf = 'icml_result_plot.pdf'
output_path_png = 'icml_result_plot.png'
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')

print(f"Plot saved to {output_path_pdf} and {output_path_png}")
