#!/usr/bin/env python3
"""
Plot comprehensive 3-way comparison: GPCM vs Original CORAL vs Improved CORAL
Shows all 7 metrics across all three approaches.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_3way_results(dataset_name="synthetic_OC"):
    """Load the latest 3-way comparison results."""
    comparison_dir = "results/comparison"
    
    # Find the latest comparison file
    files = [f for f in os.listdir(comparison_dir) 
             if f.startswith(f"comprehensive_3way_benchmark_{dataset_name}") and f.endswith('.json')]
    
    if not files:
        raise FileNotFoundError(f"No 3-way comparison results found for {dataset_name}")
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(comparison_dir, latest_file)
    
    print(f"Loading results from: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data, latest_file

def prepare_3way_data(data):
    """Prepare data for 3-way plotting."""
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
            metric_key = f'valid_{metric}' if f'valid_{metric}' in final_metrics else metric
            if metric_key in final_metrics:
                plot_data.append({
                    'model_type': model_type.upper().replace('_', ' '),
                    'embedding_strategy': embedding_strategy,
                    'metric': metric,
                    'value': final_metrics[metric_key]
                })
    
    return pd.DataFrame(plot_data), metrics

def create_3way_comparison_plots(df, metrics, dataset_name, timestamp):
    """Create comprehensive 3-way comparison plots."""
    
    # Create figure with subplots
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define colors for the three approaches
    colors = {
        'GPCM': '#2E86AB',
        'ORIGINAL CORAL': '#A23B72', 
        'IMPROVED CORAL': '#F18F01'
    }
    
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
        width = 0.25
        
        model_types = ['GPCM', 'ORIGINAL CORAL', 'IMPROVED CORAL']
        
        for j, model_type in enumerate(model_types):
            values = []
            for strategy in strategies:
                model_data = metric_data[
                    (metric_data['embedding_strategy'] == strategy) & 
                    (metric_data['model_type'] == model_type)
                ]
                if not model_data.empty:
                    values.append(model_data['value'].iloc[0])
                else:
                    values.append(0)
            
            # Plot bars
            bars = ax.bar(x + j*width, values, width, label=model_type, 
                         color=colors[model_type], alpha=0.8)
            
            # Add value labels on bars
            for k, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:  # Only label non-zero values
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize plot
        ax.set_title(metric_names.get(metric, metric.replace('_', ' ').title()), 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Embedding Strategy', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace('_', '\n') for s in strategies], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set y-axis limits for better visualization
        if metric == 'mae':
            max_val = max([max(values) for values in [
                [metric_data[(metric_data['embedding_strategy'] == s) & 
                           (metric_data['model_type'] == mt)]['value'].iloc[0] 
                 if not metric_data[(metric_data['embedding_strategy'] == s) & 
                                  (metric_data['model_type'] == mt)].empty else 0
                 for s in strategies] for mt in model_types
            ]])
            ax.set_ylim(0, max_val * 1.2)
        else:
            ax.set_ylim(0, 1.1)
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'3-Way Performance Comparison - {dataset_name.upper()}\n'
                f'GPCM vs Original CORAL vs Improved CORAL ({len(metrics)} Metrics)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    os.makedirs("results/plots", exist_ok=True)
    plot_filename = f"3way_comparison_{len(metrics)}metrics_{dataset_name}_{timestamp}.png"
    plot_path = os.path.join("results/plots", plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def analyze_3way_results(df, metrics):
    """Analyze the 3-way comparison results."""
    
    print(f"\n{'='*80}")
    print(f"3-WAY COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    # Overall performance by model type
    print(f"\n📊 AVERAGE PERFORMANCE BY MODEL TYPE:")
    print("-" * 50)
    
    model_types = df['model_type'].unique()
    
    for model_type in model_types:
        model_data = df[df['model_type'] == model_type]
        
        print(f"\n{model_type}:")
        for metric in metrics:
            metric_data = model_data[model_data['metric'] == metric]
            if not metric_data.empty:
                avg_score = metric_data['value'].mean()
                print(f"   {metric}: {avg_score:.3f}")
    
    # Best performer by metric
    print(f"\n🏆 BEST PERFORMER BY METRIC:")
    print("-" * 40)
    
    for metric in metrics:
        metric_data = df[df['metric'] == metric]
        if not metric_data.empty:
            if metric == 'mae':  # Lower is better for MAE
                best_idx = metric_data['value'].idxmin()
            else:  # Higher is better for other metrics
                best_idx = metric_data['value'].idxmax()
            
            best_row = metric_data.loc[best_idx]
            print(f"{metric}: {best_row['model_type']} with {best_row['embedding_strategy']} ({best_row['value']:.3f})")
    
    # Model type win counts
    print(f"\n📈 WIN COUNTS BY MODEL TYPE:")
    print("-" * 35)
    
    win_counts = {mt: 0 for mt in model_types}
    
    for metric in metrics:
        metric_data = df[df['metric'] == metric]
        if not metric_data.empty:
            if metric == 'mae':  # Lower is better for MAE
                winner = metric_data.loc[metric_data['value'].idxmin(), 'model_type']
            else:  # Higher is better for other metrics
                winner = metric_data.loc[metric_data['value'].idxmax(), 'model_type']
            win_counts[winner] += 1
    
    for model_type, wins in win_counts.items():
        print(f"{model_type}: {wins}/{len(metrics)} metrics")
    
    # Strategy analysis
    print(f"\n🎯 PERFORMANCE BY EMBEDDING STRATEGY:")
    print("-" * 45)
    
    strategies = df['embedding_strategy'].unique()
    
    for strategy in strategies:
        strategy_data = df[df['embedding_strategy'] == strategy]
        print(f"\n{strategy.upper()}:")
        
        for model_type in model_types:
            model_strategy_data = strategy_data[strategy_data['model_type'] == model_type]
            if not model_strategy_data.empty:
                # Get categorical accuracy as primary metric
                cat_acc_data = model_strategy_data[model_strategy_data['metric'] == 'categorical_acc']
                if not cat_acc_data.empty:
                    cat_acc = cat_acc_data['value'].iloc[0]
                    print(f"   {model_type}: {cat_acc:.3f}")

def identify_improved_coral_issues(df):
    """Identify issues with the improved CORAL implementation."""
    
    print(f"\n🔍 IMPROVED CORAL ANALYSIS:")
    print("-" * 40)
    
    improved_coral_data = df[df['model_type'] == 'IMPROVED CORAL']
    
    if improved_coral_data.empty:
        print("No improved CORAL data found!")
        return
    
    # Check for stuck values
    print(f"\n⚠️  IDENTIFIED ISSUES:")
    
    categorical_acc = improved_coral_data[improved_coral_data['metric'] == 'categorical_acc']['value'].iloc[0]
    if categorical_acc == 0.365:  # Seems stuck at this value
        print(f"   ❌ Categorical accuracy stuck at {categorical_acc:.3f}")
    
    ordinal_acc = improved_coral_data[improved_coral_data['metric'] == 'ordinal_acc']['value'].iloc[0]
    if ordinal_acc == 0.551:  # Seems stuck at this value
        print(f"   ❌ Ordinal accuracy stuck at {ordinal_acc:.3f}")
    
    qwk = improved_coral_data[improved_coral_data['metric'] == 'qwk']['value'].iloc[0]
    if qwk == 0.0:  # Exactly zero
        print(f"   ❌ QWK is exactly 0.0 (indicates constant predictions)")
    
    ord_rank = improved_coral_data[improved_coral_data['metric'] == 'ordinal_ranking_acc']['value'].iloc[0]
    if ord_rank == 0.0:  # Exactly zero
        print(f"   ❌ Ordinal ranking accuracy is 0.0 (indicates constant predictions)")
    
    mae = improved_coral_data[improved_coral_data['metric'] == 'mae']['value'].iloc[0]
    if mae > 1.3:  # Very high MAE
        print(f"   ❌ Very high MAE ({mae:.3f}) indicates poor predictions")
    
    print(f"\n💡 DIAGNOSIS:")
    print(f"   The improved CORAL model appears to be making constant predictions")
    print(f"   This suggests issues with:")
    print(f"   - CORAL threshold learning not working properly")
    print(f"   - Possible gradient flow problems")
    print(f"   - IRT parameter → CORAL feature transformation issues")

def main():
    """Main function to create 3-way comparison plots."""
    dataset_name = "synthetic_OC"
    
    print(f"Creating 3-way comparison plots for {dataset_name}...")
    
    try:
        # Load results
        data, results_file = load_3way_results(dataset_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        df, metrics = prepare_3way_data(data)
        
        print(f"Found {len(metrics)} metrics: {', '.join(metrics)}")
        print(f"Model types: {', '.join(df['model_type'].unique())}")
        print(f"Embedding strategies: {', '.join(df['embedding_strategy'].unique())}")
        
        # Create plots
        plot_filename = create_3way_comparison_plots(df, metrics, dataset_name, timestamp)
        print(f"✅ 3-way comparison plot saved: {plot_filename}")
        
        # Analyze results
        analyze_3way_results(df, metrics)
        
        # Identify improved CORAL issues
        identify_improved_coral_issues(df)
        
    except Exception as e:
        print(f"❌ Error creating 3-way comparison plots: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ 3-way comparison analysis completed successfully!")
    else:
        print(f"\n❌ 3-way comparison analysis failed!")