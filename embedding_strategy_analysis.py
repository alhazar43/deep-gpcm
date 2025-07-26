#!/usr/bin/env python3
"""
Deep-GPCM Embedding Strategy Analysis: Combined comparison and visualization tool.

This script combines training comparison and plotting functionality to analyze
different embedding strategies (ordered, unordered, linear_decay, adjacent_weighted)
for the Deep-GPCM model.
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Fix MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def run_training_experiment(dataset_name, embedding_strategy, epochs=10):
    """Run single training experiment with specified embedding strategy."""
    print(f"\n{'='*60}")
    print(f"Running {embedding_strategy} embedding strategy for {dataset_name}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset_name,
        '--epochs', str(epochs),
        '--embedding_strategy', embedding_strategy,
        '--batch_size', '64',
        '--learning_rate', '0.001',
        '--loss_type', 'ordinal'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {embedding_strategy}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    print(f"SUCCESS: {embedding_strategy} training completed")
    return result.stdout

def load_training_results(dataset_name):
    """Load training results from JSON file."""
    results_path = f"results/train/training_history_{dataset_name}.json"
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def save_comparison_results(dataset_name, all_results):
    """Save organized comparison results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison directory
    os.makedirs("results/comparison", exist_ok=True)
    
    # Save detailed results
    comparison_path = f"results/comparison/embedding_strategy_comparison_{dataset_name}_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Comparison results saved to: {comparison_path}")
    
    # Create summary
    summary = create_summary(all_results)
    summary_path = f"results/comparison/embedding_strategy_summary_{dataset_name}_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return comparison_path, summary_path

def create_summary(all_results):
    """Create summary of final epoch results."""
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'description': 'Embedding strategy comparison with new accuracy metrics',
            'strategies_tested': list(all_results.keys())
        },
        'results_by_strategy': {}
    }
    
    for strategy, strategy_results in all_results.items():
        if strategy_results:
            final_epoch = strategy_results[-1]  # Last epoch
            summary['results_by_strategy'][strategy] = {
                'final_epoch': final_epoch['epoch'],
                'train_loss': final_epoch['train_loss'],
                'valid_loss': final_epoch['valid_loss'],
                'categorical_acc': final_epoch['categorical_acc'],
                'ordinal_acc': final_epoch['ordinal_acc'],
                'mae': final_epoch['mae'],
                'qwk': final_epoch['qwk'],
                'prediction_consistency_acc': final_epoch['prediction_consistency_acc'],
                'ordinal_ranking_acc': final_epoch['ordinal_ranking_acc'],
                'distribution_consistency': final_epoch['distribution_consistency']
            }
    
    return summary

def get_metric_title(metric_name):
    """Get a human-readable title for a given metric name."""
    titles = {
        'categorical_acc': 'Exact Match\n(argmax)',
        'ordinal_acc': 'Match w/ Tolerance\n(argmax)',
        'prediction_consistency_acc': 'Exact Match\n(cumulative)',
        'ordinal_ranking_acc': 'Spearman correlation\n(expected value)',
        'distribution_consistency': 'Probability mass\nconcentration',
        'qwk': 'Quadratic Weighted Kappa\n(Agreement Measure)',
        'mae': 'Mean Absolute Error\n(Lower = Better)'
    }
    return titles.get(metric_name, metric_name.replace("_", " ").title())

def create_embedding_comparison_plots(data, metrics_to_plot, save_path="results/plots"):
    """Create adaptive-column embedding strategy comparison plots."""
    if not data:
        return None
    
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0:
        print("No metrics specified for plotting.")
        return None

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

def create_performance_analysis(data, save_path="results/plots"):
    """Create analysis of key accuracy differences between embedding strategies."""
    if not data:
        return None
        
    strategies = list(data.keys())
    final_results = {}
    
    # Extract final epoch data - dynamically get all available metrics
    available_metrics = set()
    for strategy in strategies:
        if data[strategy]:
            final_epoch = data[strategy][-1]
            available_metrics.update(final_epoch.keys())
    
    # Remove non-metric keys
    non_metric_keys = {'epoch', 'train_loss', 'valid_loss', 'embedding_strategy', 'learning_rate'}
    available_metrics = available_metrics - non_metric_keys
    
    for strategy in strategies:
        if data[strategy]:
            final_epoch = data[strategy][-1]
            final_results[strategy] = {metric: final_epoch.get(metric, 0.0) for metric in available_metrics}
    
    if not final_results:
        return None
        
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
        for metric_name, value in metrics.items():
            metric_title = get_metric_title(metric_name).replace('\n', ' ')
            print(f"  {metric_title:<30}: {value:.3f}")
    
    # Find best performers for each metric (dynamically)
    print(f"\nBEST PERFORMERS BY METRIC:")
    # Sort metrics for consistent output (put mae at end since lower is better)
    sorted_metrics = sorted([m for m in available_metrics if m != 'mae']) + (['mae'] if 'mae' in available_metrics else [])
    
    for metric in sorted_metrics:
        if metric in final_results[next(iter(final_results))]:  # Check if metric exists
            # For MAE, lower is better; for others, higher is better
            if metric == 'mae':
                best_strategy = min(final_results.items(), key=lambda x: x[1][metric])
                worst_strategy = max(final_results.items(), key=lambda x: x[1][metric])
            else:
                best_strategy = max(final_results.items(), key=lambda x: x[1][metric])
                worst_strategy = min(final_results.items(), key=lambda x: x[1][metric])
            
            metric_display_name = get_metric_title(metric).replace('\n', ' ')
            print(f"  {metric_display_name:<30}: Best = {best_strategy[0]:12} ({best_strategy[1][metric]:.3f}), "
                  f"Worst = {worst_strategy[0]:12} ({worst_strategy[1][metric]:.3f})")
    
    # Strategy ranking (dynamic, excluding MAE and other "lower is better" metrics)
    print(f"\nOVERALL STRATEGY RANKING:")
    # Use metrics where higher is better (exclude MAE and any other reverse metrics)
    positive_metrics = [m for m in available_metrics if m not in ['mae']]
    
    if positive_metrics:
        overall_scores = {}
        for strategy, metrics in final_results.items():
            # Average across positive metrics
            scores = [metrics[m] for m in positive_metrics if m in metrics]
            overall_scores[strategy] = sum(scores) / len(scores) if scores else 0.0
        
        ranked_strategies = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (strategy, score) in enumerate(ranked_strategies, 1):
            print(f"  {rank}. {strategy:12} (Average Score: {score:.3f})")
    
    return table_path

def run_comparison_experiment(dataset_name, strategies, epochs):
    """Run all embedding strategy comparison experiments."""
    print(f"Starting embedding strategy comparison experiment")
    print(f"Dataset: {dataset_name}")
    print(f"Epochs per experiment: {epochs}")
    print(f"Embedding strategies: {strategies}")
    
    # Run all experiments
    all_results = {}
    experiment_count = 0
    total_experiments = len(strategies)
    
    for strategy in strategies:
        experiment_count += 1
        print(f"\n>>> Experiment {experiment_count}/{total_experiments} <<<")
        
        # Run training
        output = run_training_experiment(dataset_name, strategy, epochs)
        
        if output is not None:
            # Load results immediately (before next experiment overwrites file)
            results = load_training_results(dataset_name)
            if results:
                # Add metadata
                for epoch_data in results:
                    epoch_data['embedding_strategy'] = strategy
                
                all_results[strategy] = results
                print(f"Results collected for {strategy}")
            else:
                print(f"WARNING: Could not load results for {strategy}")
                all_results[strategy] = None
        else:
            all_results[strategy] = None
    
    return all_results

def main():
    """Main experiment runner with integrated plotting."""
    parser = argparse.ArgumentParser(description='Deep-GPCM embedding strategy analysis')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for each experiment (default: 10)')
    parser.add_argument('--strategies', nargs='+', 
                        default=['ordered', 'unordered', 'linear_decay', 'adjacent_weighted'],
                        help='Embedding strategies to test (default: all)')
    parser.add_argument('--metrics', nargs='+', 
                        default=['categorical_acc', 'ordinal_acc', 'prediction_consistency_acc', 'ordinal_ranking_acc', 'distribution_consistency', 'mae', 'qwk'],
                        help='Metrics to plot (default: all 7 metrics)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing results (skip training)')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Specific results file to plot (for --plot-only mode)')
    
    args = parser.parse_args()
    
    if args.plot_only:
        # Plot-only mode
        if args.results_file:
            comparison_path = args.results_file
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
    else:
        # Run experiments and get results
        all_results = run_comparison_experiment(args.dataset, args.strategies, args.epochs)
        
        # Save organized results
        comparison_path, summary_path = save_comparison_results(args.dataset, all_results)
        data = all_results
    
    # Create plots and analysis
    plot_path = create_embedding_comparison_plots(data, args.metrics)
    table_path = create_performance_analysis(data)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EMBEDDING STRATEGY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    if not args.plot_only:
        print(f"Total experiments: {len(args.strategies)}")
        print(f"Dataset: {args.dataset}")
        print(f"Results saved to: {comparison_path}")
    
    if plot_path:
        print(f"Visualization saved to: {plot_path}")
    if table_path:
        print(f"Performance analysis table: {table_path}")

if __name__ == "__main__":
    main()