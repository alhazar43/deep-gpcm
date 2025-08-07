#!/usr/bin/env python3
"""
Standalone Adaptive IRT Confusion Matrix Plotter
Creates confusion matrix plot saved directly to dataset directory
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix


def plot_adaptive_irt_confusion_matrix(dataset_path, results_file=None):
    """Create standalone confusion matrix for adaptive IRT"""
    
    dataset_path = Path(dataset_path)
    
    # Load adaptive IRT results
    if results_file is None:
        results_file = dataset_path / "results" / "test" / "adaptive_irt_results.json"
    
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    predictions = results['predictions']
    actual = results['actual']
    accuracy = results['evaluation_results']['accuracy']
    n_predictions = results['evaluation_results']['n_predictions']
    
    # Create confusion matrix
    n_cats = max(max(predictions), max(actual)) + 1
    conf_matrix = confusion_matrix(actual, predictions, labels=list(range(n_cats)))
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages for coloring (row-wise)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    percentage_matrix = conf_matrix / row_sums * 100
    
    # Create custom colormap (blue/gray theme for adaptive IRT)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', '#2E86AB']  # Blue gradient
    custom_cmap = LinearSegmentedColormap.from_list('adaptive_irt', colors, N=256)
    
    # Plot heatmap using percentage matrix for coloring
    im = plt.imshow(percentage_matrix, interpolation='nearest', cmap=custom_cmap, vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Row Percentage (%)', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations with absolute counts
    thresh = 50.0  # 50% threshold for text color
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            # Use percentage for text color determination
            color = 'white' if percentage_matrix[i, j] > thresh else 'black'
            # Display absolute count and percentage
            count = conf_matrix[i, j]
            pct = percentage_matrix[i, j]
            plt.text(j, i, f'{count}\n({pct:.1f}%)', 
                    ha='center', va='center', color=color, 
                    fontweight='bold', fontsize=11)
    
    # Customize plot
    plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
    plt.title(f'Adaptive IRT Confusion Matrix\n'
              f'Accuracy: {accuracy:.3f} | Predictions: {n_predictions:,}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set ticks and labels
    plt.xticks(range(n_cats), [f'Cat {i}' for i in range(n_cats)], fontsize=12)
    plt.yticks(range(n_cats), [f'Cat {i}' for i in range(n_cats)], fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, which='major', color='lightgray', linewidth=0.5, alpha=0.7)
    
    # Add border
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#2E86AB')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    # Save directly to dataset directory
    output_path = dataset_path / "adaptive_irt_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Adaptive IRT confusion matrix saved: {output_path}")
    
    # Also create a summary text file
    summary_path = dataset_path / "adaptive_irt_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ADAPTIVE IRT EVALUATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {dataset_path.name}\n")
        f.write(f"Total Predictions: {n_predictions:,}\n")
        f.write(f"Categorical Accuracy: {accuracy:.3f}\n")
        f.write(f"Ordinal Accuracy: {results['evaluation_results'].get('ordinal_accuracy', 'N/A')}\n")
        f.write(f"Theta MAE: {results['evaluation_results']['theta_mae']:.3f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        f.write("     ")
        for j in range(n_cats):
            f.write(f"Cat{j:2d} ")
        f.write("\n")
        
        for i in range(n_cats):
            f.write(f"Cat{i:2d}")
            for j in range(n_cats):
                f.write(f"{conf_matrix[i,j]:5d} ")
            f.write(f"  (Total: {conf_matrix[i,:].sum()})\n")
        
        f.write(f"\nTotal: {conf_matrix.sum()}\n")
    
    print(f"Summary saved: {summary_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Plot Adaptive IRT Confusion Matrix')
    parser.add_argument('--dataset', default='./data/synthetic_OC',
                       help='Path to dataset directory')
    parser.add_argument('--results_file', 
                       help='Path to adaptive IRT results JSON file')
    
    args = parser.parse_args()
    
    plot_adaptive_irt_confusion_matrix(args.dataset, args.results_file)


if __name__ == "__main__":
    main()