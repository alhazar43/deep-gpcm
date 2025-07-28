#!/usr/bin/env python3
"""
Unified Metrics Plotting Script for Deep-GPCM Models
Adaptive plotting for any number of models with training curves and final performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
import argparse
from datetime import datetime


def load_json_data(file_path):
    """Load JSON data with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON: {file_path}")
        return None


def extract_model_name(data):
    """Extract model name from data."""
    if 'config' in data and 'model' in data['config']:
        return data['config']['model']
    elif 'config' in data and 'model_type' in data['config']:
        return data['config']['model_type']
    elif 'metrics' in data and 'model_type' in data['metrics']:
        return data['metrics']['model_type']
    else:
        return "unknown"


def extract_training_history(data):
    """Extract training history from data."""
    if 'training_history' in data:
        return data['training_history']
    elif 'fold_results' in data:
        # For CV results, use the first fold's training history
        return data['fold_results'][0]['training_history']
    else:
        return None


def extract_final_metrics(data):
    """Extract final metrics from data."""
    if 'metrics' in data:
        return data['metrics']
    elif 'evaluation_results' in data and 'argmax' in data['evaluation_results']:
        return data['evaluation_results']['argmax']
    elif 'fold_results' in data:
        # For CV results, average across folds
        metrics = {}
        fold_results = data['fold_results']
        
        # Get common metrics
        common_metrics = ['categorical_accuracy', 'quadratic_weighted_kappa', 'ordinal_accuracy', 'mean_absolute_error']
        
        for metric in common_metrics:
            values = [fold['metrics'][metric] for fold in fold_results if metric in fold['metrics']]
            if values:
                metrics[metric] = np.mean(values)
        
        return metrics
    else:
        return None


def plot_final_performance(model_data, output_dir):
    """Plot final performance comparison."""
    if not model_data:
        print("❌ No model data available for final performance plot")
        return None
    
    models = list(model_data.keys())
    metrics_to_plot = [
        ('categorical_accuracy', 'Categorical Accuracy'),
        ('quadratic_weighted_kappa', 'Quadratic Weighted Kappa'),
        ('ordinal_accuracy', 'Ordinal Accuracy'),
        ('mean_absolute_error', 'Mean Absolute Error')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Final Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        values = []
        model_names = []
        
        for model_name in models:
            final_metrics = model_data[model_name]['final_metrics']
            if final_metrics and metric_key in final_metrics:
                values.append(final_metrics[metric_key])
                model_names.append(model_name.replace('_', ' ').title())
            else:
                print(f"⚠️  Missing {metric_key} for {model_name}")
        
        if values:
            bars = ax.bar(model_names, values, color=colors[:len(values)], alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(values) * 0.01),
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(metric_name, fontweight='bold')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Adjust y-axis for better visualization
            if metric_key == 'mean_absolute_error':
                ax.set_ylim(0, max(values) * 1.2)
            else:
                ax.set_ylim(0, max(1.0, max(values) * 1.1))
            
            # Rotate x-axis labels if needed
            if len(model_names) > 2:
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for\\n{metric_name}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(metric_name, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'final_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Final performance plot saved: {output_path}")
    
    return output_path


def plot_training_curves(model_data, output_dir):
    """Plot training curves comparison."""
    # Filter models that have training history
    models_with_history = {}
    for model_name, data in model_data.items():
        if data['training_history']:
            models_with_history[model_name] = data
    
    if not models_with_history:
        print("❌ No training history available for training curves")
        return None
    
    metrics_to_plot = [
        ('test_accuracy', 'Test Accuracy'),
        ('qwk', 'Quadratic Weighted Kappa'),
        ('ordinal_accuracy', 'Ordinal Accuracy'),
        ('mae', 'Mean Absolute Error'),
        ('train_loss', 'Training Loss'),
        ('gradient_norm', 'Gradient Norm')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_history)))
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for model_idx, (model_name, data) in enumerate(models_with_history.items()):
            history = data['training_history']
            
            # Extract metric values
            epochs = [h['epoch'] for h in history]
            if metric_key in history[0]:
                values = [h[metric_key] for h in history]
                
                # Plot with model-specific color
                ax.plot(epochs, values, 
                       color=colors[model_idx], 
                       linewidth=2, 
                       marker='o', 
                       markersize=3,
                       label=model_name.replace('_', ' ').title(),
                       alpha=0.8)
        
        ax.set_title(metric_name, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add best value annotation
        if models_with_history:
            first_model = list(models_with_history.values())[0]
            if metric_key in first_model['training_history'][0]:
                all_values = []
                for data in models_with_history.values():
                    history = data['training_history']
                    if metric_key in history[0]:
                        values = [h[metric_key] for h in history]
                        all_values.extend(values)
                
                if all_values:
                    if metric_key in ['mae', 'train_loss']:  # Lower is better
                        best_val = min(all_values)
                        best_text = f'Best: {best_val:.3f}'
                    else:  # Higher is better
                        best_val = max(all_values)
                        best_text = f'Best: {best_val:.3f}'
                    
                    ax.text(0.02, 0.98, best_text, 
                           transform=ax.transAxes, 
                           fontsize=10, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training curves plot saved: {output_path}")
    
    return output_path


def plot_confusion_matrices(model_data, output_dir):
    """Plot confusion matrices if available."""
    models_with_confusion = {}
    for model_name, data in model_data.items():
        if (data['test_results'] and 
            'confusion_matrix' in data['test_results']):
            models_with_confusion[model_name] = data['test_results']['confusion_matrix']
    
    if not models_with_confusion:
        print("❌ No confusion matrix data available")
        return None
    
    n_models = len(models_with_confusion)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    for idx, (model_name, confusion_matrix) in enumerate(models_with_confusion.items()):
        ax = axes[idx]
        
        cm = np.array(confusion_matrix)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_title(model_name.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Set tick labels
        n_classes = cm.shape[0]
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(range(n_classes))
        ax.set_yticklabels(range(n_classes))
        
        plt.colorbar(im, ax=ax)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"🎯 Confusion matrices plot saved: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Metrics Plotting')
    parser.add_argument('--train_results', nargs='+', help='Training result JSON files')
    parser.add_argument('--test_results', nargs='+', help='Test result JSON files')
    parser.add_argument('--output_dir', default='results/plots', help='Output directory for plots')
    parser.add_argument('--plot_type', choices=['all', 'final', 'curves', 'confusion'], 
                        default='all', help='Type of plots to generate')
    
    args = parser.parse_args()
    
    if not args.train_results and not args.test_results:
        print("❌ Error: Must provide either --train_results or --test_results")
        return 1
    
    print("=" * 80)
    print("UNIFIED DEEP-GPCM METRICS PLOTTING")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Plot type: {args.plot_type}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and organize data
    model_data = {}
    
    # Load training results
    if args.train_results:
        print(f"📊 Loading {len(args.train_results)} training result files...")
        for file_path in args.train_results:
            data = load_json_data(file_path)
            if data:
                model_name = extract_model_name(data)
                if model_name not in model_data:
                    model_data[model_name] = {
                        'training_history': None,
                        'final_metrics': None,
                        'test_results': None
                    }
                
                model_data[model_name]['training_history'] = extract_training_history(data)
                model_data[model_name]['final_metrics'] = extract_final_metrics(data)
                
                print(f"  ✅ Loaded training data for: {model_name}")
    
    # Load test results
    if args.test_results:
        print(f"🧪 Loading {len(args.test_results)} test result files...")
        for file_path in args.test_results:
            data = load_json_data(file_path)
            if data:
                model_name = extract_model_name(data)
                if model_name not in model_data:
                    model_data[model_name] = {
                        'training_history': None,
                        'final_metrics': None,
                        'test_results': None
                    }
                
                model_data[model_name]['test_results'] = data.get('evaluation_results', data)
                
                # Also extract final metrics from test results if not already available
                if not model_data[model_name]['final_metrics']:
                    model_data[model_name]['final_metrics'] = extract_final_metrics(data)
                
                print(f"  ✅ Loaded test data for: {model_name}")
    
    if not model_data:
        print("❌ No valid data loaded")
        return 1
    
    print(f"\\n📈 Generating plots for models: {list(model_data.keys())}")
    
    # Generate plots based on type
    generated_plots = []
    
    if args.plot_type in ['all', 'final']:
        plot_path = plot_final_performance(model_data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    if args.plot_type in ['all', 'curves']:
        plot_path = plot_training_curves(model_data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    if args.plot_type in ['all', 'confusion']:
        plot_path = plot_confusion_matrices(model_data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    # Summary
    print(f"\\n✅ Plotting completed!")
    print(f"📊 Generated {len(generated_plots)} plots:")
    for plot_path in generated_plots:
        print(f"  - {plot_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())