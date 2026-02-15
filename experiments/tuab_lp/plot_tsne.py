import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_tsne():
    input_csv = 'experiments/tuab_lp/tsne_embeddings.csv'
    output_dir = 'experiments/tuab_lp/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Map labels to names for better legend
    df['Label Name'] = df['label'].map({0: 'Normal', 1: 'Abnormal'})
    
    models = df['model'].unique()
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    for model in models:
        print(f"Plotting for {model}...")
        subset = df[df['model'] == model]
        
        plt.figure(figsize=(10, 8))
        
        # Plot
        sns.scatterplot(
            data=subset,
            x='x', y='y',
            hue='Label Name',
            palette={'Normal': 'blue', 'Abnormal': 'red'},
            alpha=0.7,
            s=50
        )
        
        plt.title(f't-SNE Visualization - {model}', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(title='Label', title_fontsize=12)
        
        # Save
        filename = f"tsne_{model.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {output_path}")

if __name__ == "__main__":
    plot_tsne()
