
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set context
sns.set_context("paper", font_scale=1.2)
sns.set_style("white")

fig, axes = plt.subplots(3, 1, figsize=(8, 6))

def plot_palette(ax, palette, title):
    # Palette is a list of RGB tuples
    n = len(palette)
    ax.imshow([palette], aspect='auto')
    ax.set_title(title, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Annotate with hex codes if few colors
    if n <= 10:
        for i, color in enumerate(palette):
            # Convert RGB to Hex
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0]*255), int(color[1]*255), int(color[2]*255)
            )
            # Calculate luminance to decide text color (black or white)
            lum = 0.2126*color[0] + 0.7152*color[1] + 0.0722*color[2]
            text_color = 'black' if lum > 0.5 else 'white'
            
            ax.text(i, 0, hex_color, ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=10)

# 1. Continuous-like (20 colors for smoother look)
palette_20 = sns.color_palette("vlag", n_colors=20)
plot_palette(axes[0], palette_20, "vlag (20 colors)")

# 2. Default (6 colors)
palette_default = sns.color_palette("vlag")
plot_palette(axes[1], palette_default, "vlag (Default / 6 colors)")

# 3. Used in previous plot (3 colors)
palette_3 = sns.color_palette("vlag", n_colors=3)
plot_palette(axes[2], palette_3, "vlag (Used in your plot / 3 colors)")

plt.tight_layout()
plt.savefig("vlag_palette_demo.png", dpi=300, bbox_inches='tight')
print("Saved vlag_palette_demo.png")
