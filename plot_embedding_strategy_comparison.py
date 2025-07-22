#!/usr/bin/env python3
"""
Create 3-column plots comparing embedding strategies (ordered, unordered, linear_decay)
with focus on new prediction accuracy metrics and final epoch differences.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

def load_comparison_data(comparison_file):
    """Load the embedding strategy comparison data."""
    if not os.path.exists(comparison_file):
        print(f"Comparison file not found: {comparison_file}")
        return None
    
    with open(comparison_file, 'r') as f:
        return json.load(f)

import argparse

def get_metric_title(metric_name):
    """Get a human-readable title for a given metric name."""
    titles = {
        'categorical_acc': 'Categorical Accuracy\n(Exact Match)',
        'ordinal_acc': 'Ordinal Accuracy\n(±1 Tolerance)',
        'prediction_consistency_acc': 'Prediction Consistency\n(Cumulative Method)',
        'ordinal_ranking_acc': 'Ordinal Ranking\n(Spearman Corr)',
        'distribution_consistency': 'Distribution Consistency',
        'qwk': 'Quadratic Weighted Kappa',
        'mae': 'Mean Absolute Error'
    }
    return titles.get(metric_name, metric_name.replace("_", " ").title())

def create_embedding_comparison_plots(data, metrics_to_plot, save_path="results/plots"):
    """Create adaptive-column embedding strategy comparison plots."""
    if not data:
        return
    
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0:
        print("No metrics specified for plotting.")
        return

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, num_metrics, figsize=(6 * num_metrics, 12), squeeze=False)
    fig.suptitle('Deep-GPCM: Embedding Strategy Comparison', fontsize=16)
    
    # Colors for embedding strategies
    strategy_colors = {'ordered': 'blue', 'unordered': 'red', 'linear_decay': 'green', 'adjacent_weighted': 'purple'}
    strategy_markers = {'ordered': 'o', 'unordered': 's', 'linear_decay': '^', 'adjacent_weighted': 'D'}
    
    # Generate column titles
    column_titles = [get_metric_title(m) for m in metrics_to_plot]
    
    for col, title in enumerate(column_titles):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes, 
                         ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Row 1: Training progression over epochs
    for col, metric in enumerate(metrics_to_plot):
        ax = axes[0, col]
        
        for strategy, strategy_results in data.items():
            if strategy_results:
                # Check if metric exists in the data
                if metric not in strategy_results[0]:
                    print(f"Warning: Metric '{metric}' not found in data for strategy '{strategy}'. Skipping.")
                    continue
                
                epochs = [ep['epoch'] for ep in strategy_results]
                metric_values = [ep[metric] for ep in strategy_results]
                
                ax.plot(epochs, metric_values,
                       color=strategy_colors.get(strategy, 'gray'),
                       marker=strategy_markers.get(strategy, 'o'),
                       linewidth=3, alpha=0.8, markersize=6,
                       label=f'{strategy.title().replace("_", " ")} Embedding')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        if 'acc' in metric or 'qwk' in metric:
            ax.set_ylim(0, 1)
    
    # Row 2: Final epoch comparison showing differences
    strategies = list(data.keys())
    final_metrics = {metric: [] for metric in metrics_to_plot}
    
    # Extract final epoch values
    for strategy in strategies:
        if data.get(strategy):
            final_epoch = data[strategy][-1]
            for metric in metrics_to_plot:
                final_metrics[metric].append(final_epoch.get(metric, 0))
        else:
            for metric in metrics_to_plot:
                final_metrics[metric].append(0)
    
    # Create bar charts for final comparison
    for col, metric in enumerate(metrics_to_plot):
        ax = axes[1, col]
        
        x = np.arange(len(strategies))
        bars = ax.bar(x, final_metrics[metric],
                     color=[strategy_colors.get(s, 'gray') for s in strategies],
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, final_metrics[metric]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.3f}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
        
        title_parts = get_metric_title(metric).split('\n')
        ax.set_title(f'Final Epoch {title_parts[0]}', fontsize=12)
        ax.set_xlabel('Embedding Strategy')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels([s.title().replace('_', ' ') for s in strategies])
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        if 'acc' in metric or 'qwk' in metric or 'consistency' in metric or 'ranking' in metric:
            ax.set_ylim(0, 1)
        
        # Highlight the best performer
        if final_metrics[metric]:
            # For MAE, lower is better
            if metric == 'mae':
                best_idx = np.argmin(final_metrics[metric])
            else:
                best_idx = np.argmax(final_metrics[metric])
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"embedding_strategy_comparison_{num_metrics}col_{timestamp}.png"
    plot_path = os.path.join(save_path, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Adaptive {num_metrics}-column embedding strategy comparison plot saved to: {plot_path}")
    return plot_path


def create_performance_differences_analysis(data, save_path="results/plots"):
    """Create analysis of key accuracy differences between embedding strategies."""
    if not data:
        return
        
    strategies = list(data.keys())
    final_results = {}
    
    # Extract final epoch data
    for strategy in strategies:
        if data[strategy]:
            final_epoch = data[strategy][-1]
            final_results[strategy] = {
                'categorical_acc': final_epoch['categorical_acc'],
                'ordinal_acc': final_epoch['ordinal_acc'],
                'prediction_consistency_acc': final_epoch['prediction_consistency_acc'],
                'ordinal_ranking_acc': final_epoch['ordinal_ranking_acc'],
                'mae': final_epoch['mae']
            }
    
    if not final_results:
        return
        
    # Create summary table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = f"{save_path}/embedding_strategy_performance_{timestamp}.csv"
    
    # Convert to DataFrame for easy CSV export
    df_data = []
    for strategy, metrics in final_results.items():
        row = {'Strategy': strategy}
        row.update(metrics)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(table_path, index=False)
    
    print(f"Performance comparison table saved to: {table_path}")
    
    # Print analysis
    print(f"\n{'='*80}")
    print("EMBEDDING STRATEGY PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nFINAL EPOCH RESULTS:")
    for strategy, metrics in final_results.items():
        print(f"\n{strategy.upper()} EMBEDDING:")
        print(f"  Categorical Accuracy:     {metrics['categorical_acc']:.3f}")
        print(f"  Ordinal Accuracy:         {metrics['ordinal_acc']:.3f}")  
        print(f"  Prediction Consistency:   {metrics['prediction_consistency_acc']:.3f} (cumulative)")
        print(f"  Ordinal Ranking:          {metrics['ordinal_ranking_acc']:.3f}")
        print(f"  MAE:                      {metrics['mae']:.3f}")
    
    # Find best performers for each metric
    print(f"\nBEST PERFORMERS BY METRIC:")
    metrics_to_analyze = ['categorical_acc', 'ordinal_acc', 'prediction_consistency_acc', 'ordinal_ranking_acc']
    metric_names = ['Categorical Accuracy', 'Ordinal Accuracy', 'Prediction Consistency', 'Ordinal Ranking']
    
    for metric, name in zip(metrics_to_analyze, metric_names):
        best_strategy = max(final_results.items(), key=lambda x: x[1][metric])
        worst_strategy = min(final_results.items(), key=lambda x: x[1][metric])
        
        print(f"  {name:20}: Best = {best_strategy[0]:12} ({best_strategy[1][metric]:.3f}), "
              f"Worst = {worst_strategy[0]:12} ({worst_strategy[1][metric]:.3f})")
    
    # Strategy ranking
    print(f"\nOVERALL STRATEGY RANKING:")
    # Simple average ranking across key metrics (excluding MAE since lower is better)
    overall_scores = {}
    for strategy, metrics in final_results.items():
        score = (metrics['categorical_acc'] + metrics['ordinal_acc'] + 
                metrics['prediction_consistency_acc'] + metrics['ordinal_ranking_acc']) / 4
        overall_scores[strategy] = score
    
    ranked_strategies = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (strategy, score) in enumerate(ranked_strategies, 1):
        print(f"  {rank}. {strategy:12} (Average Score: {score:.3f})")
    
    return table_path

def main():
    """Main function to generate embedding strategy comparison plots."""
    parser = argparse.ArgumentParser(description='Generate adaptive plots for embedding strategy comparison.')
    parser.add_argument('--metrics', nargs='+', 
                        default=['categorical_acc', 'ordinal_acc', 'prediction_consistency_acc'],
                        help='List of metrics to plot. e.g., categorical_acc ordinal_acc mae')
    parser.add_argument('--file', type=str, default=None,
                        help='Specify a particular comparison JSON file to plot.')
    
    args = parser.parse_args()

    # Find the comparison file
    if args.file:
        comparison_path = args.file
        if not os.path.exists(comparison_path):
            print(f"Error: Specified file not found: {comparison_path}")
            return
    else:
        # Find the most recent comparison file
        comparison_dir = "results/comparison"
        if not os.path.exists(comparison_dir):
            print("No comparison data found. Run embedding strategy comparison first.")
            return
        
        comparison_files = [f for f in os.listdir(comparison_dir) 
                           if f.startswith("embedding_strategy_comparison_") and f.endswith(".json")]
        
        if not comparison_files:
            print("No embedding strategy comparison files found.")
            return
        
        # Use the most recent file
        latest_file = sorted(comparison_files)[-1]
        comparison_path = os.path.join(comparison_dir, latest_file)
    
    print(f"Loading comparison data from: {comparison_path}")
    
    # Load data
    with open(comparison_path, 'r') as f:
        data = json.load(f)
    
    # Create plots and analysis
    plot_path = create_embedding_comparison_plots(data, args.metrics)
    table_path = create_performance_differences_analysis(data)
    
    print(f"\nEmbedding strategy comparison analysis complete!")
    print(f"Adaptive visualization saved to: {plot_path}")
    print(f"Performance analysis table: {table_path}")

if __name__ == "__main__":
    main()