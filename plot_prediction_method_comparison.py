#!/usr/bin/env python3
"""
Create specialized plots showing differences between prediction methods
with focus on new prediction accuracy metrics over training progression.
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
    """Load the prediction method comparison data."""
    if not os.path.exists(comparison_file):
        print(f"Comparison file not found: {comparison_file}")
        return None
    
    with open(comparison_file, 'r') as f:
        return json.load(f)

def create_method_comparison_plots(data, save_path="results/plots"):
    """Create comprehensive prediction method comparison plots."""
    if not data:
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Deep-GPCM: Prediction Method Comparison - Raw Accuracy Analysis', fontsize=16)
    
    # Colors for methods
    method_colors = {'argmax': 'blue', 'cumulative': 'red', 'expected': 'green'}
    method_markers = {'argmax': 'o', 'cumulative': 's', 'expected': '^'}
    
    # Extract training curves for each method
    for strategy, strategy_results in data.items():
        if not strategy_results:
            continue
        
        # Plot 1: Categorical Accuracy (Raw Accuracy) over epochs
        ax = axes[0, 0]
        for method, method_data in strategy_results.items():
            if method_data:
                epochs = [ep['epoch'] for ep in method_data]
                cat_acc = [ep['categorical_acc'] for ep in method_data]
                ax.plot(epochs, cat_acc, 
                       color=method_colors.get(method, 'gray'),
                       marker=method_markers.get(method, 'o'),
                       linewidth=3, alpha=0.8, markersize=6,
                       label=f'{method.title()} Method')
        
        ax.set_title('Categorical Accuracy (Exact Match)\nDifferences by Prediction Method', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Categorical Accuracy')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 2: Ordinal Accuracy over epochs  
        ax = axes[0, 1]
        for method, method_data in strategy_results.items():
            if method_data:
                epochs = [ep['epoch'] for ep in method_data]
                ord_acc = [ep['ordinal_acc'] for ep in method_data]
                ax.plot(epochs, ord_acc,
                       color=method_colors.get(method, 'gray'),
                       marker=method_markers.get(method, 'o'),
                       linewidth=3, alpha=0.8, markersize=6,
                       label=f'{method.title()} Method')
        
        ax.set_title('Ordinal Accuracy (±1 Tolerance)\nEducational Assessment Standard', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ordinal Accuracy')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Plot 3: Training Loss over epochs
        ax = axes[1, 0] 
        for method, method_data in strategy_results.items():
            if method_data:
                epochs = [ep['epoch'] for ep in method_data]
                train_loss = [ep['train_loss'] for ep in method_data]
                valid_loss = [ep['valid_loss'] for ep in method_data]
                
                # Plot training loss
                ax.plot(epochs, train_loss,
                       color=method_colors.get(method, 'gray'),
                       linestyle='-', linewidth=3, alpha=0.8,
                       label=f'{method.title()} (Train)')
                
                # Plot validation loss
                ax.plot(epochs, valid_loss,
                       color=method_colors.get(method, 'gray'),
                       linestyle='--', linewidth=3, alpha=0.8,
                       label=f'{method.title()} (Valid)')
        
        ax.set_title('Training vs Validation Loss\n(Should be identical - same model)', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final Epoch Comparison - Focus on Key Differences
        ax = axes[1, 1]
        methods = sorted(list(strategy_results.keys()))
        
        # Extract final epoch metrics
        final_cat_acc = []
        final_ord_acc = []
        final_qwk = []
        final_mae = []
        
        for method in methods:
            method_data = strategy_results[method]
            if method_data:
                final_epoch = method_data[-1]
                final_cat_acc.append(final_epoch['categorical_acc'])
                final_ord_acc.append(final_epoch['ordinal_acc'])
                final_qwk.append(final_epoch['qwk'])
                final_mae.append(final_epoch['mae'])
            else:
                final_cat_acc.append(0)
                final_ord_acc.append(0)
                final_qwk.append(0)
                final_mae.append(0)
        
        x = np.arange(len(methods))
        width = 0.2
        
        # Create grouped bar chart
        bars1 = ax.bar(x - width, final_cat_acc, width, 
                      color=[method_colors[m] for m in methods], alpha=0.7,
                      label='Categorical Acc')
        bars2 = ax.bar(x, final_ord_acc, width,
                      color=[method_colors[m] for m in methods], alpha=0.5,
                      label='Ordinal Acc')
        bars3 = ax.bar(x + width, final_qwk, width,
                      color=[method_colors[m] for m in methods], alpha=0.3,
                      label='QWK')
        
        # Add value labels on bars
        for bars, values in [(bars1, final_cat_acc), (bars2, final_ord_acc), (bars3, final_qwk)]:
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title('Final Epoch Performance Comparison\nKey Accuracy Differences', fontsize=12)
        ax.set_xlabel('Prediction Method')
        ax.set_ylabel('Accuracy/Score')
        ax.set_xticks(x)
        ax.set_xticklabels([m.title() for m in methods])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(final_cat_acc), max(final_ord_acc), max(final_qwk)) + 0.1)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{save_path}/prediction_method_comparison_detailed_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Detailed prediction method comparison plot saved to: {plot_path}")
    return plot_path

def create_training_progression_table(data, save_path="results/plots"):
    """Create table showing training progression differences."""
    if not data:
        return
        
    # Create comparison table
    comparison_data = []
    
    for strategy, strategy_results in data.items():
        for method, method_data in strategy_results.items():
            if method_data:
                for epoch_data in method_data:
                    comparison_data.append({
                        'Strategy': strategy,
                        'Method': method,
                        'Epoch': epoch_data['epoch'],
                        'Train Loss': epoch_data['train_loss'],
                        'Valid Loss': epoch_data['valid_loss'],
                        'Cat Acc': epoch_data['categorical_acc'],
                        'Ord Acc': epoch_data['ordinal_acc'],
                        'MAE': epoch_data.get('mae', 0),
                        'QWK': epoch_data['qwk'],
                        'Pred Consistency': epoch_data['prediction_consistency_acc'],
                        'Ranking Acc': epoch_data['ordinal_ranking_acc'],
                        'Dist Consistency': epoch_data['distribution_consistency']
                    })
    
    if not comparison_data:
        return
        
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = f"{save_path}/prediction_method_training_progression_{timestamp}.csv"
    df.to_csv(table_path, index=False)
    
    print(f"Training progression table saved to: {table_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TRAINING PROGRESSION ANALYSIS")
    print("="*80)
    
    for strategy in df['Strategy'].unique():
        print(f"\n{strategy.upper()} EMBEDDING STRATEGY:")
        strategy_df = df[df['Strategy'] == strategy]
        
        # Final epoch comparison
        final_epoch_df = strategy_df[strategy_df['Epoch'] == strategy_df['Epoch'].max()]
        
        print("\nFINAL EPOCH RESULTS (Raw Accuracy):")
        for _, row in final_epoch_df.iterrows():
            method = row['Method']
            print(f"  {method:12}: CatAcc={row['Cat Acc']:.3f}, "
                  f"OrdAcc={row['Ord Acc']:.3f}, "
                  f"QWK={row['QWK']:.3f}, "
                  f"MAE={row['MAE']:.3f}")
        
        print(f"\nPREDICTION CONSISTENCY ANALYSIS:")
        for _, row in final_epoch_df.iterrows():
            method = row['Method']
            print(f"  {method:12}: PredCons={row['Pred Consistency']:.3f} "
                  f"(Consistency with ordinal training)")
        
        # Training improvement analysis
        print(f"\nTRAINING IMPROVEMENT (Epoch 1 → Final):")
        for method in strategy_df['Method'].unique():
            method_df = strategy_df[strategy_df['Method'] == method]
            if len(method_df) >= 2:
                first_epoch = method_df.iloc[0]
                last_epoch = method_df.iloc[-1]
                
                cat_improve = last_epoch['Cat Acc'] - first_epoch['Cat Acc']
                ord_improve = last_epoch['Ord Acc'] - first_epoch['Ord Acc']
                qwk_improve = last_epoch['QWK'] - first_epoch['QWK']
                
                print(f"  {method:12}: CatAcc={cat_improve:+.3f}, "
                      f"OrdAcc={ord_improve:+.3f}, "
                      f"QWK={qwk_improve:+.3f}")
        
        # Method comparison insights
        print(f"\nMETHOD COMPARISON INSIGHTS:")
        method_results = {}
        for _, row in final_epoch_df.iterrows():
            method_results[row['Method']] = {
                'cat_acc': row['Cat Acc'],
                'ord_acc': row['Ord Acc'],
                'qwk': row['QWK']
            }
        
        if len(method_results) >= 2:
            # Find best method for each metric
            best_cat = max(method_results.items(), key=lambda x: x[1]['cat_acc'])
            best_ord = max(method_results.items(), key=lambda x: x[1]['ord_acc'])
            best_qwk = max(method_results.items(), key=lambda x: x[1]['qwk'])
            
            print(f"  Best Categorical Accuracy: {best_cat[0]} ({best_cat[1]['cat_acc']:.3f})")
            print(f"  Best Ordinal Accuracy: {best_ord[0]} ({best_ord[1]['ord_acc']:.3f})")
            print(f"  Best QWK: {best_qwk[0]} ({best_qwk[1]['qwk']:.3f})")
    
    return table_path

def main():
    """Main function to generate prediction method comparison plots."""
    # Find the most recent comparison file
    comparison_dir = "results/comparison"
    if not os.path.exists(comparison_dir):
        print("No comparison data found. Run prediction method comparison first.")
        return
    
    comparison_files = [f for f in os.listdir(comparison_dir) 
                       if f.startswith("prediction_method_comparison_") and f.endswith(".json")]
    
    if not comparison_files:
        print("No prediction method comparison files found.")
        return
    
    # Use the most recent file
    latest_file = sorted(comparison_files)[-1]
    comparison_path = os.path.join(comparison_dir, latest_file)
    
    print(f"Loading comparison data from: {comparison_path}")
    
    # Load data
    with open(comparison_path, 'r') as f:
        data = json.load(f)
    
    # Create plots
    plot_path = create_method_comparison_plots(data)
    table_path = create_training_progression_table(data)
    
    print(f"\nPrediction method comparison analysis complete!")
    print(f"Visualization: {plot_path}")
    print(f"Data table: {table_path}")

if __name__ == "__main__":
    main()