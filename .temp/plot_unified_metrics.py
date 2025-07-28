#!/usr/bin/env python3
"""
Unified Training Metrics Plotting for Baseline vs AKVMN
Uses consistent data format and existing metrics from evaluation/metrics.py
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
from typing import Dict, List, Any, Optional

# Set style
plt.style.use('default')
sns.set_palette("husl")


def load_unified_training_history(model_type: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Load training history with unified filename format."""
    file_path = f"results/train/training_history_{model_type}_{dataset_name}.json"
    
    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
        print(f"✅ Loaded {model_type} training history: {len(history)} epochs")
        return history
    except FileNotFoundError:
        print(f"❌ Training history not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in training history: {file_path}")
        return []


def plot_unified_comparison(baseline_history: List[Dict], akvmn_history: List[Dict], 
                           dataset_name: str, output_dir: str = "results/plots") -> str:
    """
    Plot unified comparison using consistent metrics and format.
    """
    if not baseline_history and not akvmn_history:
        print("❌ No training data available for plotting")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot - 2x2 grid for the 4 main metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Define the 4 core metrics (consistent with evaluation/metrics.py)
    metrics = [
        ('categorical_acc', 'Categorical Accuracy', 'Accuracy', True, (0, 1)),
        ('ordinal_acc', 'Ordinal Accuracy', 'Accuracy', True, (0, 1.05)),
        ('qwk', 'Quadratic Weighted Kappa', 'QWK', True, (-0.1, 1.05)),
        ('mae', 'Mean Absolute Error', 'MAE', False, None)
    ]
    
    # Colors for models
    colors = {'baseline': '#1f77b4', 'akvmn': '#ff7f0e'}
    
    training_data = {}
    if baseline_history:
        training_data['baseline'] = baseline_history
    if akvmn_history:
        training_data['akvmn'] = akvmn_history
    
    for idx, (metric_key, title, ylabel, higher_is_better, ylim) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot each model's metrics
        for model_name, history in training_data.items():
            epochs = [epoch_data['epoch'] for epoch_data in history]
            values = [epoch_data.get(metric_key, 0) for epoch_data in history]
            
            # Skip if no data for this metric
            if all(v == 0 for v in values) and metric_key != 'mae':
                continue
            
            color = colors[model_name]
            ax.plot(epochs, values, marker='o', linewidth=2.5, markersize=4, 
                   label=model_name.upper(), color=color, alpha=0.8)
            
            # Add final value annotation
            if values:
                final_value = values[-1]
                ax.annotate(f'{final_value:.3f}', 
                           xy=(epochs[-1], final_value), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                           fontsize=9, fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits
        if ylim:
            ax.set_ylim(ylim)
    
    plt.tight_layout()
    
    # Save the plot with unified naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"unified_training_comparison_{dataset_name}_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Unified comparison plot saved: {output_file}")
    return output_file


def plot_unified_individual(history: List[Dict], model_type: str, dataset_name: str, 
                           output_dir: str = "results/plots") -> str:
    """
    Plot individual model metrics with unified format.
    """
    if not history:
        print(f"❌ No training history for {model_type}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_type.upper()} Training Metrics - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Extract data
    epochs = [epoch_data['epoch'] for epoch_data in history]
    
    # Define metrics (including loss metrics for individual plots)
    metrics_config = [
        ('train_loss', 'Training Loss', 'Loss', '#d62728'),
        ('valid_loss', 'Validation Loss', 'Loss', '#ff7f0e'),
        ('categorical_acc', 'Categorical Accuracy', 'Accuracy', '#2ca02c'),
        ('ordinal_acc', 'Ordinal Accuracy', 'Accuracy', '#1f77b4'),
        ('qwk', 'Quadratic Weighted Kappa', 'QWK', '#9467bd'),
        ('mae', 'Mean Absolute Error', 'MAE', '#8c564b')
    ]
    
    for idx, (metric_key, title, ylabel, color) in enumerate(metrics_config):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Extract values for this metric
        values = [epoch_data.get(metric_key, 0) for epoch_data in history]
        
        # Skip if no data for this metric
        if all(v == 0 for v in values) and metric_key not in ['train_loss', 'valid_loss']:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_title(title, fontweight='bold')
            continue
        
        # Plot the metric
        ax.plot(epochs, values, marker='o', linewidth=2.5, markersize=5, 
               color=color, alpha=0.8)
        
        # Add trend line
        if len(values) > 3:
            z = np.polyfit(epochs, values, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "--", alpha=0.5, color=color)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        if values:
            final_value = values[-1]
            ax.annotate(f'Final: {final_value:.3f}', 
                       xy=(epochs[-1], final_value), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                       fontsize=10, fontweight='bold')
        
        # Set appropriate y-axis limits
        if metric_key in ['categorical_acc', 'ordinal_acc']:
            ax.set_ylim(0, 1.05)
        elif metric_key == 'qwk':
            ax.set_ylim(-0.1, 1.05)
        elif 'loss' in metric_key and values:
            ax.set_ylim(0, max(values) * 1.1)
    
    plt.tight_layout()
    
    # Save with unified naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"unified_training_{model_type}_{dataset_name}_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {model_type.upper()} individual plot saved: {output_file}")
    return output_file


def create_unified_summary_table(baseline_history: List[Dict], akvmn_history: List[Dict]) -> None:
    """Create a unified performance summary table."""
    
    print("\n" + "="*60)
    print("                 UNIFIED PERFORMANCE SUMMARY")
    print("="*60)
    
    # Define metrics for summary
    summary_metrics = [
        ('categorical_acc', 'Categorical Accuracy', 3),
        ('ordinal_acc', 'Ordinal Accuracy', 3),
        ('qwk', 'QWK', 3),
        ('mae', 'MAE', 3)
    ]
    
    # Print header
    print(f"{'Metric':<20} {'Baseline':<12} {'AKVMN':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for metric_key, metric_name, decimals in summary_metrics:
        baseline_val = baseline_history[-1].get(metric_key, 0) if baseline_history else 0
        akvmn_val = akvmn_history[-1].get(metric_key, 0) if akvmn_history else 0
        
        # Calculate improvement (negative for MAE since lower is better)
        if metric_key == 'mae':
            improvement = baseline_val - akvmn_val  # Lower MAE is better
            improvement_sign = "+" if improvement > 0 else ""
        else:
            improvement = akvmn_val - baseline_val
            improvement_sign = "+" if improvement > 0 else ""
        
        print(f"{metric_name:<20} {baseline_val:<12.{decimals}f} {akvmn_val:<12.{decimals}f} {improvement_sign}{improvement:<12.{decimals}f}")
    
    # Print additional statistics
    if baseline_history and akvmn_history:
        print("\n" + "-" * 60)
        print("TRAINING STATISTICS:")
        print(f"Baseline epochs: {len(baseline_history)}")
        print(f"AKVMN epochs: {len(akvmn_history)}")
        
        # Calculate best metrics achieved
        print(f"\nBest QWK achieved:")
        baseline_best_qwk = max([e.get('qwk', 0) for e in baseline_history])
        akvmn_best_qwk = max([e.get('qwk', 0) for e in akvmn_history])
        print(f"  Baseline: {baseline_best_qwk:.3f}")
        print(f"  AKVMN: {akvmn_best_qwk:.3f}")


def main():
    """Main function for unified plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Training Metrics Plotting')
    parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'akvmn', 'both'], 
                       default=['both'], help='Models to plot')
    parser.add_argument('--output_dir', type=str, default='results/plots', help='Output directory')
    
    args = parser.parse_args()
    
    print("=== Unified Training Metrics Plotting ===")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    
    # Load training histories with unified format
    baseline_history = []
    akvmn_history = []
    
    if 'baseline' in args.models or 'both' in args.models:
        baseline_history = load_unified_training_history('baseline', args.dataset)
    
    if 'akvmn' in args.models or 'both' in args.models:
        akvmn_history = load_unified_training_history('akvmn', args.dataset)
    
    if not baseline_history and not akvmn_history:
        print("❌ No training data found. Run unified_training.py first.")
        return
    
    # Create plots
    if 'both' in args.models and baseline_history and akvmn_history:
        # Comparison plot
        comparison_plot = plot_unified_comparison(baseline_history, akvmn_history, args.dataset, args.output_dir)
        
        # Summary table
        create_unified_summary_table(baseline_history, akvmn_history)
    
    # Individual plots
    if baseline_history:
        plot_unified_individual(baseline_history, 'baseline', args.dataset, args.output_dir)
    
    if akvmn_history:
        plot_unified_individual(akvmn_history, 'akvmn', args.dataset, args.output_dir)


if __name__ == "__main__":
    main()