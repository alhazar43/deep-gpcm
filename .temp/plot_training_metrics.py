#!/usr/bin/env python3
"""
Plot training metrics over epochs for baseline and AKVMN models.
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

def load_training_history(file_path: str) -> List[Dict[str, Any]]:
    """Load training history from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Training history file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in training history file: {file_path}")
        return []

def plot_training_metrics(training_data: Dict[str, List[Dict]], output_dir: str = "results/plots") -> str:
    """
    Plot training metrics over epochs for multiple models.
    
    Args:
        training_data: Dictionary with model names as keys and training history as values
        output_dir: Directory to save plots
        
    Returns:
        Path to saved plot file
    """
    if not training_data:
        print("❌ No training data provided")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Metrics Over Epochs - Baseline vs AKVMN', fontsize=16, fontweight='bold')
    
    # Define metrics to plot
    metrics = [
        ('train_loss', 'Training Loss', 'Loss', False),
        ('valid_loss', 'Validation Loss', 'Loss', False),
        ('categorical_acc', 'Categorical Accuracy', 'Accuracy', True),
        ('ordinal_acc', 'Ordinal Accuracy', 'Accuracy', True),
        ('qwk', 'Quadratic Weighted Kappa', 'QWK', True),
        ('mae', 'Mean Absolute Error', 'MAE', False)
    ]
    
    # Colors for different models
    colors = {'baseline': '#1f77b4', 'akvmn': '#ff7f0e', 'AKVMN': '#ff7f0e', 'Baseline': '#1f77b4'}
    
    for idx, (metric_key, title, ylabel, higher_is_better) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot each model's metrics
        for model_name, history in training_data.items():
            if not history:
                continue
                
            epochs = [epoch_data['epoch'] for epoch_data in history]
            values = [epoch_data.get(metric_key, 0) for epoch_data in history]
            
            # Skip if no data for this metric
            if all(v == 0 for v in values):
                continue
            
            color = colors.get(model_name.lower(), colors.get(model_name, '#2ca02c'))
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=4, 
                   label=model_name.title(), color=color, alpha=0.8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set appropriate y-axis limits
        if metric_key in ['categorical_acc', 'ordinal_acc']:
            ax.set_ylim(0, 1.05)
        elif metric_key == 'qwk':
            ax.set_ylim(-0.1, 1.05)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"training_metrics_comparison_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training metrics plot saved: {output_file}")
    return output_file

def plot_single_model_metrics(history: List[Dict], model_name: str, output_dir: str = "results/plots") -> str:
    """
    Plot training metrics for a single model.
    
    Args:
        history: Training history data
        model_name: Name of the model
        output_dir: Directory to save plots
        
    Returns:
        Path to saved plot file
    """
    if not history:
        print(f"❌ No training history data for {model_name}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name.title()} Training Metrics Over Epochs', fontsize=16, fontweight='bold')
    
    # Extract data
    epochs = [epoch_data['epoch'] for epoch_data in history]
    
    # Define metrics with their properties
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
        if all(v == 0 for v in values):
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
            ax.set_title(title, fontweight='bold')
            continue
        
        # Plot the metric
        ax.plot(epochs, values, marker='o', linewidth=2.5, markersize=6, 
               color=color, alpha=0.8)
        
        # Add trend line for better visualization
        if len(values) > 2:
            z = np.polyfit(epochs, values, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "--", alpha=0.5, color=color)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_value = values[-1]
        ax.annotate(f'Final: {final_value:.3f}', 
                   xy=(epochs[-1], final_value), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                   fontsize=10)
        
        # Set appropriate y-axis limits
        if metric_key in ['categorical_acc', 'ordinal_acc']:
            ax.set_ylim(0, 1.05)
        elif metric_key == 'qwk':
            ax.set_ylim(-0.1, 1.05)
        elif 'loss' in metric_key:
            ax.set_ylim(0, max(values) * 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"training_metrics_{model_name}_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ {model_name.title()} training metrics plot saved: {output_file}")
    return output_file

def create_performance_summary(history: List[Dict], model_name: str) -> Dict[str, Any]:
    """Create a performance summary from training history."""
    if not history:
        return {}
    
    final_epoch = history[-1]
    best_metrics = {}
    
    # Find best values for each metric
    for metric in ['categorical_acc', 'ordinal_acc', 'qwk']:
        values = [epoch_data.get(metric, 0) for epoch_data in history]
        best_metrics[f'best_{metric}'] = max(values)
        best_metrics[f'best_{metric}_epoch'] = values.index(max(values)) + 1
    
    # Find best values for loss metrics (lower is better)
    for metric in ['train_loss', 'valid_loss', 'mae']:
        values = [epoch_data.get(metric, float('inf')) for epoch_data in history]
        valid_values = [v for v in values if v != float('inf')]
        if valid_values:
            best_metrics[f'best_{metric}'] = min(valid_values)
            best_metrics[f'best_{metric}_epoch'] = valid_values.index(min(valid_values)) + 1
    
    return {
        'model': model_name,
        'final_epoch': final_epoch,
        'best_metrics': best_metrics,
        'total_epochs': len(history)
    }

def main():
    """Main plotting function."""
    print("=== Training Metrics Plotting ===")
    
    # Try to load training histories
    training_data = {}
    
    # Load baseline training history
    baseline_file = "results/train/training_history_baseline_synthetic_OC.json"
    baseline_history = load_training_history(baseline_file)
    if baseline_history:
        training_data['baseline'] = baseline_history
        print(f"✅ Loaded baseline training history: {len(baseline_history)} epochs")
    
    # Look for AKVMN training history (multiple possible locations)
    akvmn_files = [
        "results/train/training_history_akvmn_synthetic_OC.json",
        "save_models/training_history_akvmn_synthetic_OC.json",
        "results/akvmn_training_history.json"
    ]
    
    for akvmn_file in akvmn_files:
        akvmn_history = load_training_history(akvmn_file)
        if akvmn_history:
            training_data['akvmn'] = akvmn_history
            print(f"✅ Loaded AKVMN training history: {len(akvmn_history)} epochs")
            break
    
    if not training_data:
        print("❌ No training history data found")
        return
    
    # Create plots
    if len(training_data) > 1:
        # Comparison plot
        comparison_plot = plot_training_metrics(training_data)
        
    # Individual model plots
    for model_name, history in training_data.items():
        single_plot = plot_single_model_metrics(history, model_name)
        
        # Print performance summary
        summary = create_performance_summary(history, model_name)
        if summary:
            print(f"\n📊 {model_name.title()} Performance Summary:")
            print(f"   Total epochs: {summary['total_epochs']}")
            final = summary['final_epoch']
            print(f"   Final metrics:")
            print(f"     Categorical Acc: {final.get('categorical_acc', 0):.3f}")
            print(f"     Ordinal Acc: {final.get('ordinal_acc', 0):.3f}")
            print(f"     QWK: {final.get('qwk', 0):.3f}")
            print(f"     MAE: {final.get('mae', 0):.3f}")
            
            best = summary['best_metrics']
            print(f"   Best metrics:")
            for metric in ['categorical_acc', 'ordinal_acc', 'qwk']:
                if f'best_{metric}' in best:
                    print(f"     Best {metric}: {best[f'best_{metric}']:.3f} (epoch {best[f'best_{metric}_epoch']})")

if __name__ == "__main__":
    main()