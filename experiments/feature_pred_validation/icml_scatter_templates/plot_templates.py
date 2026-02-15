
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import r2_score
import os

# --- ICML Style Settings ---
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
    'text.usetex': False, # Set to True if you have LaTeX installed
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def generate_synthetic_feature_data(n_samples=500, noise_level=0.3, bias=0.9):
    """
    Generate synthetic True vs Pred data for a single feature.
    Simulates a regression task where Pred ~ Bias * True + Noise
    """
    # True values (Standard Normal distribution)
    y_true = np.random.normal(0, 1, n_samples)
    
    # Predicted values: Linear relationship + Gaussian noise
    # bias=0.9 implies the model slightly underpredicts extremes (regression to mean)
    y_pred = bias * y_true + np.random.normal(0, noise_level, n_samples)
    
    return y_true, y_pred

def plot_single_fitting(y_true, y_pred, feature_name, output_path):
    """
    Template 1: High-quality Density Scatter for a Single Feature
    X-axis: True Value
    Y-axis: Predicted Value
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    
    # Calculate density for coloring (Gaussian KDE)
    # This makes the plot look much more professional than simple solid dots
    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
        # Sort points by density so dense points are on top
        idx = z.argsort()
        x, y, z = y_true[idx], y_pred[idx], z[idx]
        scatter = ax.scatter(x, y, c=z, s=30, cmap='Spectral_r', edgecolor='none', alpha=0.8)
    except:
        # Fallback if KDE fails (e.g. constant data)
        scatter = ax.scatter(y_true, y_pred, c='blue', s=30, alpha=0.5)

    # Add diagonal identity line (Perfect Prediction)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]), 
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, lw=1.5, label='Ideal (y=x)')
    
    # Add regression line (Best Fit)
    m, b = np.polyfit(y_true, y_pred, 1)
    ax.plot(y_true, m*y_true + b, color='red', alpha=0.6, lw=2, label='Fit')

    # Labels and Title
    ax.set_xlabel(f'True {feature_name}')
    ax.set_ylabel(f'Predicted {feature_name}')
    ax.set_title(f'{feature_name} Fitting Performance')
    
    # Metrics Text Box
    stats_text = (f"Pearson $r = {pcc:.3f}$\n"
                  f"$R^2 = {r2:.3f}$")
    
    # Place text in top-left corner
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))

    # Legend
    ax.legend(loc='lower right', frameon=True)
    
    # Equal aspect ratio ensures square plot
    ax.set_aspect('equal')
    
    plt.tight_layout()
    print(f"Saving {output_path}...")
    plt.savefig(output_path)
    plt.close()

def plot_multi_feature_grid(features_data, output_path, cmap_name='mako'):
    """
    Template 2: Grid of Scatter Plots for Multiple Features
    Good for showing consistency across different feature types.
    """
    n_feats = len(features_data)
    cols = 3
    rows = int(np.ceil(n_feats / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    
    for i, (name, (y_true, y_pred)) in enumerate(features_data.items()):
        ax = axes[i]
        
        # Calculate Metrics
        r2 = r2_score(y_true, y_pred)
        pcc, _ = pearsonr(y_true, y_pred)
        
        # Density Scatter
        xy = np.vstack([y_true, y_pred])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = y_true[idx], y_pred[idx], z[idx]
        
        # Plot
        ax.scatter(x, y, c=z, s=20, cmap=cmap_name, edgecolor='none', alpha=0.9)
        
        # Diagonal
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, color='#333333', linestyle='--', lw=1, alpha=0.6, label='Ideal')
        
        ax.set_title(f"{name}", fontsize=12, fontweight='bold', pad=10)
        
        # Only show labels on outer plots to save space
        if i % cols == 0:
            ax.set_ylabel("Predicted", fontsize=10)
        if i >= (rows-1)*cols:
            ax.set_xlabel("True", fontsize=10)
            
        # Stats box - consistent with ICML style
        # Using a slightly transparent white box
        stats_text = f"$R^2={r2:.2f}$\n$r={pcc:.2f}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=9, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.9))
        
        # Adjust ticks
        ax.tick_params(axis='both', which='major', labelsize=9)
        
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    print(f"Saving {output_path}...")
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_performance_summary(r2_scores, pearson_scores, output_path):
    """
    Template 3: Overall Performance Summary (R2 vs Pearson Distribution)
    Each point is one feature. Shows the global model performance.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot of metrics
    ax.scatter(r2_scores, pearson_scores, s=80, alpha=0.7, c='#2b7bba', edgecolor='white')
    
    ax.set_xlabel('Coefficient of Determination ($R^2$)')
    ax.set_ylabel('Pearson Correlation Coefficient ($r$)')
    ax.set_title('Global Feature Reconstruction Performance')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add mean lines
    mean_r2 = np.mean(r2_scores)
    mean_pcc = np.mean(pearson_scores)
    
    ax.axvline(mean_r2, color='red', linestyle='--', alpha=0.5, label=f'Mean $R^2$: {mean_r2:.2f}')
    ax.axhline(mean_pcc, color='green', linestyle='--', alpha=0.5, label=f'Mean $r$: {mean_pcc:.2f}')
    
    ax.legend()
    
    plt.tight_layout()
    print(f"Saving {output_path}...")
    plt.savefig(output_path)
    plt.close()

def main():
    # Setup Directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating Synthetic Data...")
    
    # 1. Generate data for a single "Hero" feature (e.g., Alpha Band Power)
    # High quality fit
    y_true_best, y_pred_best = generate_synthetic_feature_data(n_samples=1000, noise_level=0.2, bias=0.98)
    plot_single_fitting(y_true_best, y_pred_best, "Alpha Power", 
                        os.path.join(output_dir, "template_1_single_feature_fit.png"))
    
    # 2. Generate data for multiple features for the Grid Plot
    features = {
        'Delta Power': generate_synthetic_feature_data(noise_level=0.4),
        'Theta Power': generate_synthetic_feature_data(noise_level=0.35),
        'Alpha Power': (y_true_best, y_pred_best), # Reuse best
        'Beta Power': generate_synthetic_feature_data(noise_level=0.3),
        'Gamma Power': generate_synthetic_feature_data(noise_level=0.5),
        'Peak Freq': generate_synthetic_feature_data(noise_level=0.25),
    }
    # Option A: Blue-ish (Matches vlag's blue end)
    plot_multi_feature_grid(features, os.path.join(output_dir, "template_2_vlag_blue_mako.png"), cmap_name='mako')
    
    # Option B: Red-ish (Matches vlag's red end)
    plot_multi_feature_grid(features, os.path.join(output_dir, "template_2_vlag_red_rocket.png"), cmap_name='rocket')
    
    # Option C: Flare (More vibrant red/orange)
    plot_multi_feature_grid(features, os.path.join(output_dir, "template_2_vlag_red_flare.png"), cmap_name='flare')
    
    # 3. Generate distribution data for many features
    n_features = 50
    r2_list = []
    pcc_list = []
    for _ in range(n_features):
        yt, yp = generate_synthetic_feature_data(n_samples=200, 
                                                 noise_level=np.random.uniform(0.1, 0.8),
                                                 bias=np.random.uniform(0.8, 1.0))
        r2_list.append(r2_score(yt, yp))
        pcc_list.append(pearsonr(yt, yp)[0])
        
    plot_performance_summary(r2_list, pcc_list, os.path.join(output_dir, "template_3_performance_distribution.png"))
    
    print("Done! All images saved to:", output_dir)

if __name__ == "__main__":
    main()
