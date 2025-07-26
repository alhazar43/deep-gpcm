#!/usr/bin/env python3
"""
Plot comprehensive CORAL vs GPCM comparison with all 7 metrics.
Similar to embedding strategy comparison but comparing model types across strategies.
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

def load_coral_gpcm_results(dataset_name="synthetic_OC"):
    """Load the latest CORAL vs GPCM comparison results."""
    comparison_dir = "results/comparison"
    
    # Find the latest comparison file
    files = [f for f in os.listdir(comparison_dir) 
             if f.startswith(f"coral_gpcm_embedding_comparison_{dataset_name}") and f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError(f"No CORAL vs GPCM comparison results found for {dataset_name}")
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(comparison_dir, latest_file)
    
    print(f"Loading results from: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data, latest_file

def prepare_comparison_data(data):
    """Prepare data for plotting."""
    summary_results = data['summary_results']
    
    # Extract metrics
    metrics = [
        'categorical_acc', 'ordinal_acc', 'prediction_consistency_acc', 
        'ordinal_ranking_acc', 'mae', 'distribution_consistency'
    ]
    
    # Add quadratic weighted kappa if available
    sample_result = list(summary_results.values())[0]['final_metrics']
    if 'qwk' in sample_result:
        metrics.append('qwk')
    
    plot_data = []
    
    for key, result in summary_results.items():
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
    """Create comprehensive comparison plots similar to embedding strategy plots."""
    
    # Create figure with subplots
    n_metrics = len(metrics)
    n_cols = 3 if n_metrics >= 6 else 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define colors for GPCM vs CORAL
    colors = {'GPCM': '#2E86AB', 'CORAL': '#A23B72'}
    
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
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Filter data for this metric
        metric_data = df[df['metric'] == metric]
        
        # Create grouped bar plot
        strategies = metric_data['embedding_strategy'].unique()
        x = np.arange(len(strategies))
        width = 0.35
        
        gpcm_values = []
        coral_values = []
        
        for strategy in strategies:
            gpcm_val = metric_data[(metric_data['embedding_strategy'] == strategy) & 
                                  (metric_data['model_type'] == 'GPCM')]['value'].iloc[0]
            coral_val = metric_data[(metric_data['embedding_strategy'] == strategy) & 
                                   (metric_data['model_type'] == 'CORAL')]['value'].iloc[0]
            gpcm_values.append(gpcm_val)
            coral_values.append(coral_val)
        
        # Plot bars
        ax.bar(x - width/2, gpcm_values, width, label='GPCM', color=colors['GPCM'], alpha=0.8)
        ax.bar(x + width/2, coral_values, width, label='CORAL', color=colors['CORAL'], alpha=0.8)
        
        # Add value labels on bars
        for j, (gpcm_val, coral_val) in enumerate(zip(gpcm_values, coral_values)):
            ax.text(j - width/2, gpcm_val + 0.01, f'{gpcm_val:.3f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(j + width/2, coral_val + 0.01, f'{coral_val:.3f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize plot
        ax.set_title(metric_names.get(metric, metric.replace('_', ' ').title()), 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Embedding Strategy', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Set y-axis limits for better visualization
        if metric == 'mae':
            ax.set_ylim(0, max(max(gpcm_values), max(coral_values)) * 1.2)
        else:
            ax.set_ylim(0, 1.1)
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'CORAL vs GPCM Performance Comparison - {dataset_name.upper()}\n'
                f'All Embedding Strategies ({len(metrics)} Metrics)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plot_filename = f"coral_gpcm_comparison_{len(metrics)}metrics_{dataset_name}_{timestamp}.png"
    plot_path = os.path.join("results/plots", plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    
    return plot_filename

def create_performance_summary_table(df, metrics, dataset_name, timestamp):
    """Create and save performance summary table."""
    
    # Pivot data to create comparison table
    pivot_data = df.pivot_table(
        index=['embedding_strategy'], 
        columns=['model_type'], 
        values='value', 
        aggfunc='first'
    )
    
    # Create summary by metric
    summary_by_metric = {}
    
    for metric in metrics:
        metric_df = df[df['metric'] == metric].pivot_table(
            index=['embedding_strategy'], 
            columns=['model_type'], 
            values='value', 
            aggfunc='first'
        )
        
        # Calculate differences (GPCM - CORAL)
        metric_df['Difference'] = metric_df['GPCM'] - metric_df['CORAL']
        metric_df['GPCM_Better'] = metric_df['Difference'] > 0
        
        summary_by_metric[metric] = metric_df
    
    # Create comprehensive summary
    performance_summary = {
        'dataset': dataset_name,
        'timestamp': timestamp,
        'metrics_analyzed': len(metrics),
        'embedding_strategies': df['embedding_strategy'].unique().tolist(),
        'summary_by_metric': {}
    }
    
    for metric in metrics:
        metric_summary = summary_by_metric[metric]
        
        # Find best performers
        if metric == 'mae':  # Lower is better for MAE
            gpcm_best = metric_summary['GPCM'].idxmin()
            coral_best = metric_summary['CORAL'].idxmin()
            overall_best_strategy = gpcm_best if metric_summary.loc[gpcm_best, 'GPCM'] < metric_summary.loc[coral_best, 'CORAL'] else coral_best
            overall_best_model = 'GPCM' if metric_summary.loc[gpcm_best, 'GPCM'] < metric_summary.loc[coral_best, 'CORAL'] else 'CORAL'
        else:  # Higher is better for other metrics
            gpcm_best = metric_summary['GPCM'].idxmax()
            coral_best = metric_summary['CORAL'].idxmax()
            overall_best_strategy = gpcm_best if metric_summary.loc[gpcm_best, 'GPCM'] > metric_summary.loc[coral_best, 'CORAL'] else coral_best
            overall_best_model = 'GPCM' if metric_summary.loc[gpcm_best, 'GPCM'] > metric_summary.loc[coral_best, 'CORAL'] else 'CORAL'
        
        gpcm_wins = (metric_summary['GPCM_Better']).sum() if metric != 'mae' else (~metric_summary['GPCM_Better']).sum()
        
        performance_summary['summary_by_metric'][metric] = {
            'gpcm_best_strategy': gpcm_best,
            'gpcm_best_value': float(metric_summary.loc[gpcm_best, 'GPCM']),
            'coral_best_strategy': coral_best,
            'coral_best_value': float(metric_summary.loc[coral_best, 'CORAL']),
            'overall_best_strategy': overall_best_strategy,
            'overall_best_model': overall_best_model,
            'overall_best_value': float(metric_summary.loc[overall_best_strategy, overall_best_model]),
            'gpcm_wins': int(gpcm_wins),
            'coral_wins': int(len(metric_summary) - gpcm_wins),
            'avg_difference': float(metric_summary['Difference'].mean())
        }
    
    # Save comprehensive summary
    summary_filename = f"coral_gpcm_performance_summary_{dataset_name}_{timestamp}.json"
    summary_path = os.path.join("results/plots", summary_filename)
    
    with open(summary_path, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    # Save CSV for easy analysis
    csv_data = []
    for metric in metrics:
        metric_df = summary_by_metric[metric]
        for strategy in metric_df.index:
            csv_data.append({
                'metric': metric,
                'embedding_strategy': strategy,
                'GPCM': metric_df.loc[strategy, 'GPCM'],
                'CORAL': metric_df.loc[strategy, 'CORAL'],
                'difference': metric_df.loc[strategy, 'Difference'],
                'gpcm_better': metric_df.loc[strategy, 'GPCM_Better']
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_filename = f"coral_gpcm_comparison_data_{dataset_name}_{timestamp}.csv"
    csv_path = os.path.join("results/plots", csv_filename)
    csv_df.to_csv(csv_path, index=False)
    
    return summary_filename, csv_filename

def print_performance_analysis(performance_summary):
    """Print detailed performance analysis."""
    
    print(f"\n{'='*80}")
    print(f"CORAL vs GPCM PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Dataset: {performance_summary['dataset']}")
    print(f"Metrics Analyzed: {performance_summary['metrics_analyzed']}")
    print(f"Embedding Strategies: {', '.join(performance_summary['embedding_strategies'])}")
    
    print(f"\n📊 METRIC-BY-METRIC ANALYSIS:")
    print("-" * 80)
    
    gpcm_total_wins = 0
    coral_total_wins = 0
    
    for metric, summary in performance_summary['summary_by_metric'].items():
        print(f"\n🎯 {metric.upper().replace('_', ' ')}:")
        print(f"   Overall Best: {summary['overall_best_model']} with {summary['overall_best_strategy']} ({summary['overall_best_value']:.3f})")
        print(f"   GPCM Best: {summary['gpcm_best_strategy']} ({summary['gpcm_best_value']:.3f})")
        print(f"   CORAL Best: {summary['coral_best_strategy']} ({summary['coral_best_value']:.3f})")
        print(f"   Win Count: GPCM {summary['gpcm_wins']}, CORAL {summary['coral_wins']}")
        print(f"   Avg Difference: {summary['avg_difference']:.3f}")
        
        gpcm_total_wins += summary['gpcm_wins']
        coral_total_wins += summary['coral_wins']
    
    print(f"\n🏆 OVERALL SUMMARY:")
    print(f"   Total Strategy Wins: GPCM {gpcm_total_wins}, CORAL {coral_total_wins}")
    print(f"   Win Rate: GPCM {gpcm_total_wins/(gpcm_total_wins+coral_total_wins)*100:.1f}%, CORAL {coral_total_wins/(gpcm_total_wins+coral_total_wins)*100:.1f}%")
    
    if gpcm_total_wins > coral_total_wins:
        print(f"   🥇 GPCM shows superior overall performance across embedding strategies")
    elif coral_total_wins > gpcm_total_wins:
        print(f"   🥇 CORAL shows superior overall performance across embedding strategies")
    else:
        print(f"   🤝 GPCM and CORAL show comparable overall performance")

def main():
    """Main function to create CORAL vs GPCM comparison plots."""
    dataset_name = "synthetic_OC"
    
    print(f"Creating CORAL vs GPCM comparison plots for {dataset_name}...")
    
    try:
        # Load results
        data, results_file = load_coral_gpcm_results(dataset_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        df, metrics = prepare_comparison_data(data)
        
        print(f"Found {len(metrics)} metrics: {', '.join(metrics)}")
        print(f"Embedding strategies: {', '.join(df['embedding_strategy'].unique())}")
        
        # Create plots
        plot_filename = create_comprehensive_comparison_plots(df, metrics, dataset_name, timestamp)
        print(f"✅ Comparison plot saved: {plot_filename}")
        
        # Create performance summary
        summary_filename, csv_filename = create_performance_summary_table(df, metrics, dataset_name, timestamp)
        print(f"✅ Performance summary saved: {summary_filename}")
        print(f"✅ CSV data saved: {csv_filename}")
        
        # Load and print analysis
        summary_path = os.path.join("results/plots", summary_filename)
        with open(summary_path, 'r') as f:
            performance_summary = json.load(f)
        
        print_performance_analysis(performance_summary)
        
    except Exception as e:
        print(f"❌ Error creating comparison plots: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ CORAL vs GPCM comparison analysis completed successfully!")
    else:
        print(f"\n❌ CORAL vs GPCM comparison analysis failed!")