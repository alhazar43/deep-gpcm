#!/usr/bin/env python3
"""
Plot comprehensive CORN vs GPCM vs CORAL comparison with all 7 metrics.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_corn_gpcm_results(dataset_name="synthetic_OC"):
    """Load the latest CORN vs GPCM comparison results."""
    comparison_dir = "results/comparison"
    
    # Find the latest comparison file
    files = [f for f in os.listdir(comparison_dir) 
             if f.startswith(f"comprehensive_3way_benchmark_{dataset_name}") and f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError(f"No CORN vs GPCM vs CORAL comparison results found for {dataset_name}")
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(comparison_dir, latest_file)
    
    print(f"Loading results from: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data, latest_file

def prepare_comparison_data(data):
    """Prepare data for plotting."""
    results = data['results']
    
    # Extract metrics
    metrics = [
        'categorical_acc', 'ordinal_acc', 'prediction_consistency_acc', 
        'ordinal_ranking_acc', 'mae', 'distribution_consistency', 'qwk'
    ]
    
    plot_data = []
    
    for key, result in results.items():
        model_type = result['model_type']
        embedding_strategy = result['embedding_strategy']
        final_metrics = result['final_metrics']
        
        for metric in metrics:
            if metric in final_metrics:
                plot_data.append({
                    'model_type': model_type.upper(),
                    'embedding_strategy': embedding_strategy,
                    'metric': metric,
                    'value': final_metrics[metric]
                })
    
    return pd.DataFrame(plot_data), metrics

def create_comprehensive_comparison_plots(df, metrics, dataset_name, timestamp):
    """Create comprehensive comparison plots for GPCM, CORAL, and CORN."""
    
    # Create figure with subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    # Define colors
    colors = {'GPCM': '#1f77b4', 'CORAL': '#ff7f0e', 'CORN': '#2ca02c'}
    
    # Metric display names
    metric_names = {
        'categorical_acc': 'Categorical Accuracy',
        'ordinal_acc': 'Ordinal Accuracy', 
        'prediction_consistency_acc': 'Prediction Consistency',
        'ordinal_ranking_acc': 'Ordinal Ranking Accuracy',
        'mae': 'Mean Absolute Error',
        'distribution_consistency': 'Distribution Consistency',
        'qwk': 'Quadratic Weighted Kappa'
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = df[df['metric'] == metric]
        
        # Create grouped bar plot
        strategies = sorted(metric_data['embedding_strategy'].unique())
        x = np.arange(len(strategies))
        width = 0.25
        
        for j, model_type in enumerate(['GPCM', 'CORAL', 'CORN']):
            values = [metric_data[(metric_data['embedding_strategy'] == s) & (metric_data['model_type'] == model_type)]['value'].iloc[0] for s in strategies]
            ax.bar(x + (j - 1) * width, values, width, label=model_type, color=colors[model_type], alpha=0.8)
            for k, v in enumerate(values):
                ax.text(k + (j - 1) * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(metric_names.get(metric, metric.replace('_', ' ').title()), fontsize=12, fontweight='bold')
        ax.set_xlabel('Embedding Strategy', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        if metric == 'mae':
            ax.set_ylim(0, df[df['metric'] == 'mae']['value'].max() * 1.2)
        else:
            ax.set_ylim(0, 1.1)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'GPCM vs CORAL vs CORN Performance Comparison - {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plot_filename = f"corn_gpcm_coral_comparison_{len(metrics)}metrics_{dataset_name}_{timestamp}.png"
    plot_path = os.path.join("results/plots", plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def main():
    """Main function to create comparison plots."""
    dataset_name = "synthetic_OC"
    
    print(f"Creating GPCM vs CORAL vs CORN comparison plots for {dataset_name}...")
    
    try:
        data, results_file = load_corn_gpcm_results(dataset_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df, metrics = prepare_comparison_data(data)
        
        print(f"Found {len(metrics)} metrics: {', '.join(metrics)}")
        print(f"Embedding strategies: {', '.join(df['embedding_strategy'].unique())}")
        
        plot_filename = create_comprehensive_comparison_plots(df, metrics, dataset_name, timestamp)
        print(f"✅ Comparison plot saved: {plot_filename}")
        
    except Exception as e:
        print(f"❌ Error creating comparison plots: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ GPCM vs CORAL vs CORN comparison analysis completed successfully!")
    else:
        print(f"\n❌ GPCM vs CORAL vs CORN comparison analysis failed!")
