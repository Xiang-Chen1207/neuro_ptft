
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- ICML Style Settings ---
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

def plot_appendix_chart(df, metric, output_path, title):
    """
    Generate a large vertical bar chart (Appendix Style).
    - Grouped by Category
    - Sorted by Metric Descending within Category
    - Full Feature Names on Y-axis
    - Values annotated at the end of bars
    """
    df = df.copy()
    df['Category'] = df['Feature Name'].apply(get_feature_category)
    
    # Sort: Custom Category order then Metric Descending
    # Custom Order: Time -> Power -> Structure -> Ratios
    cat_order = {'Time': 0, 'Power': 1, 'Structure': 2, 'Ratios': 3}
    df['CatOrder'] = df['Category'].map(cat_order)
    
    # We want Metric Descending (Largest Top)
    # In barh(y_pos), index 0 is Bottom.
    # To have Category 'Time' at Top, it needs highest index.
    # To have Largest Metric at Top of 'Time', it needs highest index within 'Time'.
    
    # So we want to sort:
    # CatOrder Descending (3->0 means Ratios->Time at top? No wait)
    # y=0 (Bottom) -> y=N (Top)
    # We want Time at Top (Index N), Ratios at Bottom (Index 0).
    # So sort CatOrder Ascending? No, Ascending means 0 (Time) is at Bottom.
    # We want CatOrder Descending: 3 (Ratios) at Bottom, 0 (Time) at Top.
    # Wait, if we iterate list: [Time, Power...]
    # Plotting index 0..N:
    # Index 0 is Bottom. Index N is Top.
    # If list is [Time_Item1, Time_Item2, ..., Ratios_Item1...]
    # Then Time is at Bottom.
    # We want Time at Top.
    # So list should be [Ratios..., Structure..., Power..., Time...]
    # This means sorting CatOrder DESCENDING (3->2->1->0).
    
    # Within Category: We want Largest Metric at Top.
    # So within Time block (at top), the top-most bar should be largest.
    # This means list order for Time block should be [Smallest... Largest].
    # So sort Metric ASCENDING.
    
    df_sorted = df.sort_values(by=['CatOrder', metric], ascending=[False, True])
    
    # Setup Figure (Tall)
    height_per_bar = 0.25
    fig_height = max(10, len(df) * height_per_bar)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    y_pos = np.arange(len(df_sorted))
    
    # Colors (Cool Palette)
    pal = {
        'Time': '#4c78a8', # Steel Blue
        'Power': '#72b7b2', # Teal
        'Structure': '#b279a2', # Muted Purple
        'Ratios': '#54a24b'  # Muted Green
    }
    colors = [pal.get(c, '#888888') for c in df_sorted['Category']]
    
    # Plot Bars
    bars = ax.barh(y_pos, df_sorted[metric], color=colors, height=0.7, alpha=0.9)
    
    # Y-Axis Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['Feature Name'], fontsize=9)
    ax.set_ylim(-1, len(df_sorted))
    
    # X-Axis
    ax.set_xlabel(f"{metric} Score", fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Set x-axis limit to 1.0 (or slightly more for text space)
    ax.set_xlim(right=1.15) # Give space for text
    
    # Add vertical dashed line at 1.0
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    
    # Annotate Values
    x_max = df_sorted[metric].max()
    offset = x_max * 0.01
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        val = row[metric]
        ax.text(val + offset, i, f"{val:.3f}", va='center', ha='left', fontsize=8, color='#333333')
        
    # Separators between categories
    df_reset = df_sorted.reset_index(drop=True)
    current_cat = None
    cat_start_idx = 0
    
    for i in range(len(df_reset)):
        cat = df_reset.loc[i, 'Category']
        if current_cat is not None and cat != current_cat:
            # Line
            ax.axhline(i - 0.5, color='grey', linestyle='-', linewidth=0.8, alpha=0.5)
            
            # Label Previous Category on the right (optional, or just use legend)
            # Let's put a label on the far right y-axis
            # cat_center = (cat_start_idx + i - 1) / 2
            # ax.text(ax.get_xlim()[1]*1.15, cat_center, current_cat, 
            #         va='center', ha='left', fontsize=10, fontweight='bold', rotation=0)
            
            cat_start_idx = i
        current_cat = cat
        
    # Legend
    # Order legend to match plot (Time on top)
    legend_order = ['Time', 'Power', 'Structure', 'Ratios']
    handles = [plt.Rectangle((0,0),1,1, color=pal[c]) for c in legend_order if c in pal]
    
    # Move legend to top, horizontal
    ax.legend(handles, [c for c in legend_order if c in pal], 
              loc='upper center', bbox_to_anchor=(0.5, 1.02), 
              ncol=4, frameon=False, fontsize=10)
    
    # ax.set_title(title, fontsize=14, pad=10) # Removed title as requested
    
    sns.despine(ax=ax, left=False, bottom=False, right=True, top=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    csv_path = "/vePFS-0x0d/home/cx/ptft/experiments/feature_pred_validation/feature_metrics_eval_full.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    output_dir = os.path.dirname(csv_path)
    viz_dir = os.path.join(output_dir, "val_full_final_viz", "appendix")
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Generating Appendix R2 Chart...")
    plot_appendix_chart(df, 'R2', os.path.join(viz_dir, "appendix_R2_full.pdf"), "Full Feature R2 Performance")
    
    print("Generating Appendix PCC Chart...")
    plot_appendix_chart(df, 'PCC', os.path.join(viz_dir, "appendix_PCC_full.pdf"), "Full Feature Pearson Correlation")
    
    print(f"Done. Files saved to {viz_dir}")

if __name__ == "__main__":
    main()
