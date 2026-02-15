
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
import numpy as np

def main():
    # File paths
    # Use absolute path or assume current dir if running from there
    input_csv = 'tsne_embeddings.csv'
    output_dir = '.'
    
    # Load data
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Models
    models = df['model'].unique()
    
    # Set style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    
    # Colors
    # Normal: Blue, Abnormal: Red
    # Using specific hex codes for a "paper" look (muted but clear)
    colors = {
        0: '#4c72b0', # Blue (Normal)
        1: '#c44e52'  # Red (Abnormal)
    }
    label_map = {0: 'Normal', 1: 'Abnormal'}
    
    for model in models:
        print(f"Processing {model}...")
        df_model = df[df['model'] == model].copy()
        
        # Calculate metrics on 2D embeddings
        X = df_model[['x', 'y']].values
        labels = df_model['label'].values
        
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        # Prepare plot data
        df_model['Label_Str'] = df_model['label'].map(label_map)
        
        # Create figure
        plt.figure(figsize=(6, 6))
        
        # KDE (Density)
        sns.kdeplot(
            data=df_model, x='x', y='y', hue='Label_Str',
            palette={'Normal': colors[0], 'Abnormal': colors[1]},
            fill=True, alpha=0.1, levels=5, thresh=0.05,
            legend=False
        )
        
        # Scatter
        sns.scatterplot(
            data=df_model, x='x', y='y', hue='Label_Str',
            palette={'Normal': colors[0], 'Abnormal': colors[1]},
            s=40, alpha=0.7, edgecolor='white', linewidth=0.5,
            legend=True
        )
        
        # Remove axes details for cleaner look
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        
        # Despine
        sns.despine(left=True, bottom=True)
        
        # Title
        # Map model names to paper names if needed
        # Baseline (Recon) -> Recon.
        # Neuro-KE -> Ours
        # FeatOnly -> Feat.
        display_name = model
        if "Recon" in model: display_name = "Recon."
        elif "Neuro-KE" in model: display_name = "Ours"
        elif "FeatOnly" in model: display_name = "Feat."
        
        plt.title(display_name, fontsize=18, fontweight='bold', pad=15)
        
        # Add metrics text
        metrics_text = f"Silhouette: {sil:.3f}\nCH Score: {ch:.1f}"
        
        # Position text in a corner (e.g., top right or top left)
        # We can calculate bounds to place it safely? 
        # Or just put it in the legend area or specific corner.
        # Let's put it in the upper right
        # plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes,
        #          fontsize=12, verticalalignment='bottom', horizontalalignment='right',
        #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Or use legend title? No.
        # Let's put it at the bottom right.
        plt.text(0.95, 0.02, metrics_text, transform=plt.gca().transAxes,
                 fontsize=12, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
        
        # Legend settings
        plt.legend(title='', loc='upper right', frameon=False)
        
        # Save
        filename = f"tsne_{display_name.replace('.', '')}.png"
        save_path = os.path.join(output_dir, filename)
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
        plt.close()

def plot_combined():
    """Combine 3 plots into one row"""
    input_csv = 'tsne_embeddings.csv'
    output_dir = '.'
    df = pd.read_csv(input_csv)
    
    # Define order
    model_order = [
        ('Baseline (Recon)', 'Recon.'),
        ('FeatOnly', 'Feat.'),
        ('Neuro-KE', 'Ours')
    ]
    
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("white")
    
    colors = {0: '#4c72b0', 1: '#c44e52'}
    label_map = {0: 'Normal', 1: 'Abnormal'}
    
    for i, (model_name, display_name) in enumerate(model_order):
        ax = axes[i]
        df_model = df[df['model'] == model_name].copy()
        df_model['Label_Str'] = df_model['label'].map(label_map)
        
        # Calculate metrics
        X = df_model[['x', 'y']].values
        labels = df_model['label'].values
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        
        # Plot KDE
        sns.kdeplot(
            data=df_model, x='x', y='y', hue='Label_Str',
            palette={'Normal': colors[0], 'Abnormal': colors[1]},
            fill=True, alpha=0.1, levels=5, thresh=0.05,
            legend=False, ax=ax
        )
        
        # Plot Scatter
        sns.scatterplot(
            data=df_model, x='x', y='y', hue='Label_Str',
            palette={'Normal': colors[0], 'Abnormal': colors[1]},
            s=40, alpha=0.7, edgecolor='white', linewidth=0.5,
            legend=(i == 2), ax=ax # Only show legend on last plot? Or common legend?
        )
        
        # Clean axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        sns.despine(ax=ax, left=True, bottom=True)
        
        # Title
        ax.set_title(display_name, fontsize=20, fontweight='bold', pad=15)
        
        # Metrics
        metrics_text = f"Silhouette: {sil:.3f}\nCH Score: {ch:.1f}"
        ax.text(0.95, 0.02, metrics_text, transform=ax.transAxes,
                 fontsize=13, ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
                 
        # Legend handling
        if i == 2:
            # Fix legend title and position
            # sns scatterplot legend title is usually the hue column name
            handles, labels = ax.get_legend_handles_labels()
            # Usually we get title + labels.
            ax.legend(handles, labels, title='', loc='upper right', frameon=False, fontsize=12)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'tsne_combined.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {save_path}")

if __name__ == "__main__":
    # main() # Skip individual plots
    plot_combined()
