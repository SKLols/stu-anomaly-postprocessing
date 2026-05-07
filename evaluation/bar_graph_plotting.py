import matplotlib.pyplot as plt
import numpy as np

# Data from your tables
methods = ['MaxLogit', 'MSP', 'SML', 'Energy', 'Entropy', 'RbA']
distances = ['0-10m', '10-20m', '20-30m', '30-40m', '40-50m']

# Bright, vibrant colors for each method
colors = [
    '#FF6B6B',  # Bright red
    '#4ECDC4',  # Turquoise
    '#FFD166',  # Bright yellow
    '#06D6A0',  # Emerald green
    '#118AB2',  # Bright blue
    '#9D4EDD'   # Purple
]

# AP data - Full queries (first value in F/Q)
ap_full = np.array([
    [7.82, 2.87, 1.51, 0.23, 0.02],  # MaxLogit
    [7.64, 3.24, 1.46, 0.23, 0.02],  # MSP
    [2.28, 2.11, 0.62, 0.17, 0.03],  # SML
    [7.29, 2.99, 1.42, 0.21, 0.02],  # Energy
    [7.79, 3.22, 1.55, 0.23, 0.02],  # Entropy
    [7.25, 2.96, 1.37, 0.20, 0.02]   # RbA
])

# AP data - Selected queries (second value in F/Q)
ap_selected = np.array([
    [7.79, 4.91, 1.67, 0.26, 0.02],
    [7.59, 5.17, 1.61, 0.25, 0.02],
    [2.28, 2.48, 0.62, 0.19, 0.03],
    [6.84, 2.24, 0.63, 0.12, 0.01],
    [7.74, 5.42, 1.72, 0.26, 0.02],
    [7.21, 5.67, 1.73, 0.26, 0.02]
])

# AUROC data - Full queries
auroc_full = np.array([
    [92.67, 81.58, 88.04, 84.43, 74.37],
    [92.68, 86.05, 88.49, 86.23, 74.99],
    [95.82, 87.16, 88.61, 85.84, 79.49],
    [91.29, 72.70, 86.23, 80.03, 69.70],
    [92.34, 78.73, 87.98, 84.30, 74.66],
    [83.22, 64.69, 79.61, 75.92, 64.28]
])

# AUROC data - Selected queries
auroc_selected = np.array([
    [92.71, 91.42, 93.65, 88.49, 76.02],
    [92.72, 93.16, 93.51, 88.38, 75.95],
    [95.84, 93.05, 92.48, 87.58, 81.10],
    [83.15, 65.25, 72.04, 69.99, 57.15],
    [92.37, 90.02, 93.74, 88.76, 76.23],
    [83.27, 81.39, 89.67, 86.74, 72.64]
])

def create_bar_plot(data_full, data_selected, ylabel, filename, log_scale=False):
    """Create side-by-side bar plots for full and selected queries with bright colors"""
    x = np.arange(len(distances))
    width = 0.13  # Width of each bar
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    # Set background color for better contrast
    fig.patch.set_facecolor('#F8F9FA')
    ax1.set_facecolor('#FFFFFF')
    ax2.set_facecolor('#FFFFFF')
    
    # Full queries plot
    bars_full = []
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2
        bars = ax1.bar(x + offset, data_full[i], width, 
                      label=method, color=colors[i], 
                      edgecolor='black', linewidth=0.8,
                      alpha=0.9)
        bars_full.append(bars)
        
        # Add value labels on top of bars (only for values > 1 for readability)
        for j, val in enumerate(data_full[i]):
            if val > 1:  # Only label significant values
                ax1.text(x[j] + offset, val + 0.1, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Distance Range', fontsize=13, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax1.set_title('Full Queries (100 queries)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(distances, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=11)
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylabel(f'{ylabel} (log scale)', fontsize=13, fontweight='bold')
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax1.set_axisbelow(True)
    
    # Add a subtle border
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    # Selected queries plot
    bars_selected = []
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width + width/2
        bars = ax2.bar(x + offset, data_selected[i], width, 
                      label=method, color=colors[i], 
                      edgecolor='black', linewidth=0.8,
                      alpha=0.9)
        bars_selected.append(bars)
        
        # Add value labels on top of bars (only for values > 1 for readability)
        for j, val in enumerate(data_selected[i]):
            if val > 1:  # Only label significant values
                ax2.text(x[j] + offset, val + 0.1, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Distance Range', fontsize=13, fontweight='bold')
    ax2.set_title('Selected Queries (~38 queries)', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(distances, fontsize=11, fontweight='bold')
    if log_scale:
        ax2.set_yscale('log')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax2.set_axisbelow(True)
    
    # Add a subtle border
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)
    
    # Create a custom legend with colored squares
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.8) 
                      for color in colors]
    
    # Add legend at the top
    fig.legend(legend_elements, methods, 
               loc='upper center', 
               ncol=6, 
               bbox_to_anchor=(0.5, 1.02),
               fontsize=12,
               title='Uncertainty Methods',
               title_fontsize=13,
               frameon=True,
               fancybox=True,
               shadow=True,
               borderpad=1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust for top legend
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {filename}")

# Create AP plots (log scale for better visualization)
print("Creating AP comparison plots...")
create_bar_plot(ap_full, ap_selected, 'Average Precision (%)', 'ap_distance_comparison.pdf', log_scale=True)

# Create AUROC plots
print("Creating AUROC comparison plots...")
create_bar_plot(auroc_full, auroc_selected, 'AUROC (%)', 'auroc_distance_comparison.pdf', log_scale=False)

# Create improvement visualization
def create_improvement_heatmap(data_full, data_selected, metric_name):
    """Create heatmap of percentage improvements with bright colors"""
    improvement = ((data_selected - data_full) / data_full * 100)
    improvement = np.nan_to_num(improvement, nan=0.0, posinf=0.0, neginf=0.0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    
    # Create custom colormap (red for negative, green for positive)
    from matplotlib.colors import LinearSegmentedColormap
    colors_cmap = ['#FF6B6B', '#FFFFFF', '#06D6A0']  # Red -> White -> Green
    cmap = LinearSegmentedColormap.from_list('improvement_cmap', colors_cmap, N=100)
    
    im = ax.imshow(improvement, cmap=cmap, aspect='auto', vmin=-30, vmax=30)
    
    # Add text annotations with better visibility
    for i in range(len(methods)):
        for j in range(len(distances)):
            val = improvement[i, j]
            if abs(val) > 0.1:  # Only show meaningful improvements
                color = 'white' if abs(val) > 10 else 'black'
                fontweight = 'bold' if abs(val) > 5 else 'normal'
                ax.text(j, i, f'{val:+.1f}%',
                       ha='center', va='center', color=color,
                       fontsize=10, fontweight=fontweight)
    
    ax.set_xlabel('Distance Range', fontsize=13, fontweight='bold')
    ax.set_ylabel('Uncertainty Method', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_name} Improvement with Query Selection', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(np.arange(len(distances)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(distances, fontsize=11, fontweight='bold')
    ax.set_yticklabels(methods, fontsize=11, fontweight='bold')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, len(distances), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Improvement (%)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    filename = f'{metric_name.lower()}_improvement_heatmap.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print(f"Saved: {filename}")

# Create improvement heatmaps
print("\nCreating improvement heatmaps...")
create_improvement_heatmap(auroc_full, auroc_selected, 'AUROC')
create_improvement_heatmap(ap_full, ap_selected, 'AP')

# Create a summary improvement bar chart
def create_summary_improvement_chart():
    """Create a summary chart showing average improvement per method"""
    # Calculate average improvement across all distances
    auroc_improvement = np.mean(((auroc_selected - auroc_full) / auroc_full * 100), axis=1)
    ap_improvement = np.mean(((ap_selected - ap_full) / ap_full * 100), axis=1)
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    
    bars1 = ax.bar(x - width/2, auroc_improvement, width, 
                  label='AUROC Improvement', color='#4ECDC4',
                  edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, ap_improvement, width,
                  label='AP Improvement', color='#FFD166',
                  edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.1:
                ax.text(bar.get_x() + bar.get_width()/2, height + (1 if height > 0 else -3),
                       f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Uncertainty Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Improvement (%)', fontsize=13, fontweight='bold')
    ax.set_title('Average Performance Improvement with Query Selection\nAcross All Distance Ranges', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('summary_improvement.pdf', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: summary_improvement.pdf")

# Create summary chart
print("\nCreating summary improvement chart...")
create_summary_improvement_chart()

print("\n✅ All graphs generated successfully!")
print("Files created:")
print("1. ap_distance_comparison.pdf - AP comparison")
print("2. auroc_distance_comparison.pdf - AUROC comparison")
print("3. auroc_improvement_heatmap.pdf - AUROC improvement heatmap")
print("4. ap_improvement_heatmap.pdf - AP improvement heatmap")
print("5. summary_improvement.pdf - Summary improvement chart")