#!/usr/bin/env python3
"""
Adaptive Metrics Plotting System for Deep-GPCM Models
Automatically adjusts subplot layout based on available metrics.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
plt.style.use('default')
sns.set_palette("husl")


class AdaptivePlotter:
    """Adaptive plotting system that adjusts to available metrics."""
    
    def __init__(self, results_dir: str = "results", figsize_base: Tuple[int, int] = (4, 3)):
        """Initialize plotter with base figure size per subplot."""
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.figsize_base = figsize_base
        
        # Define metric categories and their display properties
        self.metric_categories = {
            'core': {
                'metrics': ['categorical_accuracy', 'ordinal_accuracy', 'quadratic_weighted_kappa', 
                           'mean_absolute_error', 'ordinal_loss'],
                'colors': ['#2E86AB', '#A23B72', '#F18F01', '#43AA8B', '#F77F00'],
                'higher_better': [True, True, True, False, False]
            },
            'correlation': {
                'metrics': ['kendall_tau', 'spearman_correlation', 'pearson_correlation', 'cohen_kappa'],
                'colors': ['#6A994E', '#386641', '#BC4749', '#D62828'],
                'higher_better': [True, True, True, True]
            },
            'probability': {
                'metrics': ['cross_entropy', 'mean_confidence', 'mean_entropy', 'expected_calibration_error'],
                'colors': ['#F72585', '#7209B7', '#4CC9F0', '#4361EE'],
                'higher_better': [False, True, False, False]
            },
            'detailed': {
                'metrics': ['macro_f1', 'weighted_f1', 'macro_precision', 'weighted_precision'],
                'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                'higher_better': [True, True, True, True]
            }
        }
    
    def calculate_subplot_layout(self, n_metrics: int) -> Tuple[int, int]:
        """Calculate optimal subplot layout for given number of metrics."""
        if n_metrics <= 1:
            return 1, 1
        elif n_metrics <= 2:
            return 1, 2
        elif n_metrics <= 4:
            return 2, 2
        elif n_metrics <= 6:
            return 2, 3
        elif n_metrics <= 9:
            return 3, 3
        elif n_metrics <= 12:
            return 3, 4
        elif n_metrics <= 16:
            return 4, 4
        else:
            # For very large numbers, use a more rectangular layout
            cols = int(np.ceil(np.sqrt(n_metrics)))
            rows = int(np.ceil(n_metrics / cols))
            return rows, cols
    
    def load_results_from_dir(self, result_type: str = "train") -> List[Dict[str, Any]]:
        """Load all results from a directory."""
        target_dir = self.results_dir / result_type
        results = []
        
        if not target_dir.exists():
            return results
        
        for file_path in target_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = file_path.name
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        return results
    
    def extract_metrics_from_results(self, results: List[Dict[str, Any]], 
                                   result_type: str = "training") -> Tuple[List[str], pd.DataFrame]:
        """Extract available metrics and create DataFrame."""
        if not results:
            return [], pd.DataFrame()
        
        all_metrics = set()
        all_data = []
        
        for result in results:
            if result_type == "training" and 'training_history' in result:
                # Training results - extract epoch-by-epoch data
                for epoch_data in result['training_history']:
                    row = epoch_data.copy()
                    
                    # Add model info from config
                    if 'config' in result:
                        model_type = result['config'].get('model', 'unknown')
                        row['model_type'] = model_type
                        row['dataset'] = result['config'].get('dataset', 'unknown')
                        if 'fold' in result['config']:
                            row['fold'] = result['config']['fold']
                    
                    row['source_file'] = result.get('source_file', 'unknown')
                    all_data.append(row)
                    all_metrics.update(row.keys())
            
            else:
                # Evaluation results - single-point data
                row = {}
                
                # Extract metrics from various locations
                if 'metrics' in result:
                    row.update(result['metrics'])
                if 'evaluation_results' in result:
                    row.update(result['evaluation_results'])
                
                # Add model info from config
                if 'config' in result:
                    model_type = result['config'].get('model_type', 
                                      result['config'].get('model', 'unknown'))
                    row['model_type'] = model_type
                    row['dataset'] = result['config'].get('dataset', 'unknown')
                    if 'fold' in result['config']:
                        row['fold'] = result['config']['fold']
                
                row['source_file'] = result.get('source_file', 'unknown')
                all_data.append(row)
                all_metrics.update(row.keys())
        
        # Filter out non-metric columns
        excluded_keys = {'epoch', 'source_file', 'model_type', 'dataset', 'fold', 
                        'timestamp', 'learning_rate', 'gradient_norm', 'train_loss', 'train_accuracy'}
        available_metrics = sorted([m for m in all_metrics if m not in excluded_keys])
        
        df = pd.DataFrame(all_data)
        return available_metrics, df
    
    def filter_metrics_for_plotting(self, available_metrics: List[str], 
                                   category: str = "auto") -> List[str]:
        """Filter metrics based on category and availability."""
        if category == "auto":
            # Use the default metrics from metrics.py
            from utils.metrics import DEFAULT_METRICS
            priority_metrics = DEFAULT_METRICS[:6]  # Top 6 for better layout
            filtered = [m for m in priority_metrics if m in available_metrics]
            
            # If we don't have enough, add category metrics
            if len(filtered) < 4:
                cat_metrics = [m for m in available_metrics if m.startswith('cat_') and m.endswith('_accuracy')]
                for metric in cat_metrics[:4-len(filtered)]:
                    filtered.append(metric)
            
            return filtered
        
        elif category in self.metric_categories:
            return [m for m in self.metric_categories[category]['metrics'] if m in available_metrics]
        
        else:
            return available_metrics
    
    def plot_training_metrics(self, results: List[Dict[str, Any]], 
                            metrics_to_plot: Optional[List[str]] = None,
                            save_path: Optional[str] = None) -> str:
        """Plot training metrics over epochs with adaptive layout."""
        available_metrics, df = self.extract_metrics_from_results(results, "training")
        
        if df.empty:
            print("No training data found.")
            return ""
        
        # Select metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = self.filter_metrics_for_plotting(available_metrics, "auto")
        
        if not metrics_to_plot:
            print("No suitable metrics found for plotting.")
            return ""
        
        # Calculate subplot layout
        n_metrics = len(metrics_to_plot)
        rows, cols = self.calculate_subplot_layout(n_metrics)
        
        # Create figure
        fig_width = cols * self.figsize_base[0]
        fig_height = rows * self.figsize_base[1]
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get unique models and datasets for grouping
        models = df['model_type'].unique() if 'model_type' in df.columns else ['unknown']
        datasets = df['dataset'].unique() if 'dataset' in df.columns else ['unknown']
        
        # Color palette for different models
        model_colors = sns.color_palette("husl", len(models))
        model_color_map = dict(zip(models, model_colors))
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'Metric\n{metric}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title())
                continue
            
            # Plot data for each model
            for model in models:
                model_data = df[df['model_type'] == model] if 'model_type' in df.columns else df
                
                if 'epoch' in model_data.columns and not model_data.empty:
                    # Group by fold if available
                    if 'fold' in model_data.columns:
                        for fold in model_data['fold'].unique():
                            fold_data = model_data[model_data['fold'] == fold]
                            if not fold_data.empty:
                                ax.plot(fold_data['epoch'], fold_data[metric], 
                                       color=model_color_map[model], alpha=0.3, linewidth=1)
                        
                        # Plot mean across folds
                        epoch_means = model_data.groupby('epoch')[metric].mean()
                        epoch_stds = model_data.groupby('epoch')[metric].std()
                        
                        ax.plot(epoch_means.index, epoch_means.values, 
                               color=model_color_map[model], linewidth=2, label=f'{model} (mean)')
                        ax.fill_between(epoch_means.index, 
                                       epoch_means.values - epoch_stds.values,
                                       epoch_means.values + epoch_stds.values,
                                       color=model_color_map[model], alpha=0.2)
                        
                        # Add final mean value annotation
                        if len(epoch_means) > 0:
                            final_epoch = epoch_means.index[-1]
                            final_mean = epoch_means.values[-1]
                            final_std = epoch_stds.values[-1]
                            ax.annotate(f'{final_mean:.4f}±{final_std:.4f}', 
                                      xy=(final_epoch, final_mean),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, color=model_color_map[model],
                                      bbox=dict(boxstyle='round,pad=0.2', 
                                              facecolor='white', alpha=0.8))
                    else:
                        # Single run
                        ax.plot(model_data['epoch'], model_data[metric], 
                               color=model_color_map[model], linewidth=2, label=model)
                        
                        # Add final value annotation
                        if not model_data.empty:
                            final_epoch = model_data['epoch'].iloc[-1]
                            final_value = model_data[metric].iloc[-1]
                            ax.annotate(f'{final_value:.4f}', 
                                      xy=(final_epoch, final_value),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=8, color=model_color_map[model],
                                      bbox=dict(boxstyle='round,pad=0.2', 
                                              facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Training Metrics Over Epochs ({len(results)} experiments)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / "training_metrics.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved: {save_path}")
        return str(save_path)
    
    def plot_evaluation_metrics(self, results: List[Dict[str, Any]], 
                              result_type: str = "test",
                              metrics_to_plot: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> str:
        """Plot evaluation metrics (final performance) with adaptive layout."""
        available_metrics, df = self.extract_metrics_from_results(results, "evaluation")
        
        if df.empty:
            print(f"No {result_type} data found.")
            return ""
        
        # Select metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = self.filter_metrics_for_plotting(available_metrics, "auto")
        
        if not metrics_to_plot:
            print("No suitable metrics found for plotting.")
            return ""
        
        # Calculate subplot layout
        n_metrics = len(metrics_to_plot)
        rows, cols = self.calculate_subplot_layout(n_metrics)
        
        # Create figure
        fig_width = cols * self.figsize_base[0]
        fig_height = rows * self.figsize_base[1]
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get unique models for comparison
        models = df['model_type'].unique() if 'model_type' in df.columns else ['unknown']
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'Metric\n{metric}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title())
                continue
            
            # Create bar plot for model comparison
            if 'fold' in df.columns:
                # Cross-validation results - show mean and error bars
                means = df.groupby('model_type')[metric].mean()
                stds = df.groupby('model_type')[metric].std()
                
                x_pos = np.arange(len(means))
                colors = sns.color_palette("husl", len(means))
                
                # Determine if higher is better for this metric
                higher_is_better = not any(keyword in metric.lower() for keyword in 
                                         ['error', 'loss', 'entropy']) 
                
                # Find best performing model
                if higher_is_better:
                    best_idx = means.argmax()
                else:
                    best_idx = means.argmin()
                
                # Create bars with highlight for best
                bars = ax.bar(x_pos, means.values, yerr=stds.values, 
                             capsize=5, alpha=0.7, color=colors)
                
                # Highlight best performing model
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('darkgoldenrod')
                bars[best_idx].set_linewidth(2)
                bars[best_idx].set_alpha(1.0)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(means.index, rotation=45)
                
                # Add value labels on bars with best marker
                for i, (mean, std) in enumerate(zip(means.values, stds.values)):
                    label = f'{mean:.4f}±{std:.4f}'
                    if i == best_idx:
                        label += ' ★'
                    ax.text(i, mean + std + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                           label, ha='center', va='bottom', fontsize=9,
                           fontweight='bold' if i == best_idx else 'normal')
            
            else:
                # Single evaluation - show individual values
                model_values = df.groupby('model_type')[metric].first()
                
                x_pos = np.arange(len(model_values))
                colors = sns.color_palette("husl", len(model_values))
                
                # Determine if higher is better for this metric
                higher_is_better = not any(keyword in metric.lower() for keyword in 
                                         ['error', 'loss', 'entropy']) 
                
                # Find best performing model
                if higher_is_better:
                    best_idx = model_values.argmax()
                else:
                    best_idx = model_values.argmin()
                
                # Create bars with highlight for best
                bars = ax.bar(x_pos, model_values.values, alpha=0.7, color=colors)
                
                # Highlight best performing model
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('darkgoldenrod')
                bars[best_idx].set_linewidth(2)
                bars[best_idx].set_alpha(1.0)
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(model_values.index, rotation=45)
                
                # Add value labels on bars with best marker
                for i, value in enumerate(model_values.values):
                    label = f'{value:.4f}'
                    if i == best_idx:
                        label += ' ★'
                    ax.text(i, value + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                           label, ha='center', va='bottom', fontsize=9, 
                           fontweight='bold' if i == best_idx else 'normal')
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{result_type.title()} Results Comparison ({len(results)} experiments)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"{result_type}_metrics.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{result_type.title()} metrics plot saved: {save_path}")
        return str(save_path)
    
    def plot_metric_comparison(self, train_results: List[Dict[str, Any]], 
                             test_results: List[Dict[str, Any]],
                             save_path: Optional[str] = None) -> str:
        """Create side-by-side comparison of training and test metrics."""
        # Extract final training metrics
        train_metrics, train_df = self.extract_metrics_from_results(train_results, "training")
        test_metrics, test_df = self.extract_metrics_from_results(test_results, "evaluation")
        
        # Find common metrics
        common_metrics = list(set(train_metrics) & set(test_metrics))
        common_metrics = self.filter_metrics_for_plotting(common_metrics, "auto")
        
        if not common_metrics:
            print("No common metrics found between training and test results.")
            return ""
        
        # Get final training values (last epoch)
        if 'epoch' in train_df.columns:
            if 'fold' in train_df.columns:
                # Group by model and fold for CV results
                final_train_df = train_df.loc[train_df.groupby(['model_type', 'fold'])['epoch'].idxmax()]
            else:
                # Group by model only for non-CV results
                final_train_df = train_df.loc[train_df.groupby(['model_type'])['epoch'].idxmax()]
        else:
            final_train_df = train_df
        
        # Create comparison plot
        n_metrics = len(common_metrics)
        rows, cols = self.calculate_subplot_layout(n_metrics)
        
        fig_width = cols * self.figsize_base[0]
        fig_height = rows * self.figsize_base[1]
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        models = set()
        if 'model_type' in final_train_df.columns:
            models.update(final_train_df['model_type'].unique())
        if 'model_type' in test_df.columns:
            models.update(test_df['model_type'].unique())
        models = sorted(list(models))
        
        # Plot comparison for each metric
        for idx, metric in enumerate(common_metrics):
            ax = axes[idx]
            
            x_pos = np.arange(len(models))
            width = 0.35
            
            train_values = []
            test_values = []
            
            for model in models:
                # Training values
                train_model_data = final_train_df[final_train_df['model_type'] == model] if 'model_type' in final_train_df.columns else final_train_df
                train_val = train_model_data[metric].mean() if not train_model_data.empty and metric in train_model_data.columns else 0
                train_values.append(train_val)
                
                # Test values
                test_model_data = test_df[test_df['model_type'] == model] if 'model_type' in test_df.columns else test_df
                test_val = test_model_data[metric].mean() if not test_model_data.empty and metric in test_model_data.columns else 0
                test_values.append(test_val)
            
            # Determine if higher is better for this metric
            higher_is_better = not any(keyword in metric.lower() for keyword in 
                                     ['error', 'loss', 'entropy']) 
            
            # Find best performing model for test values (most important)
            if test_values and any(v > 0 for v in test_values):
                if higher_is_better:
                    best_idx = np.argmax(test_values)
                else:
                    best_idx = np.argmin(test_values)
            else:
                best_idx = -1  # No highlighting
            
            # Create bars with highlighting
            train_bars = ax.bar(x_pos - width/2, train_values, width, label='Training', alpha=0.8)
            test_bars = ax.bar(x_pos + width/2, test_values, width, label='Test', alpha=0.8)
            
            # Highlight best performing model
            if best_idx >= 0:
                test_bars[best_idx].set_color('gold')
                test_bars[best_idx].set_edgecolor('darkgoldenrod')
                test_bars[best_idx].set_linewidth(2)
                test_bars[best_idx].set_alpha(1.0)
                # Also highlight corresponding training bar
                train_bars[best_idx].set_edgecolor('darkgoldenrod')
                train_bars[best_idx].set_linewidth(1.5)
            
            # Add value labels on bars
            for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
                # Training values
                if train_val > 0:
                    ax.text(i - width/2, train_val + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                           f'{train_val:.4f}', ha='center', va='bottom', fontsize=8)
                
                # Test values with star for best
                if test_val > 0:
                    test_label = f'{test_val:.4f}'
                    if i == best_idx:
                        test_label += ' ★'
                    ax.text(i + width/2, test_val + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                           test_label, ha='center', va='bottom', fontsize=8,
                           fontweight='bold' if i == best_idx else 'normal')
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Training vs Test Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / "training_vs_test_comparison.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training vs test comparison plot saved: {save_path}")
        return str(save_path)


def plot_all_results(results_dir: str = "results"):
    """Convenience function to plot all available results."""
    plotter = AdaptivePlotter(results_dir)
    
    # Load all results
    train_results = plotter.load_results_from_dir("train")
    test_results = plotter.load_results_from_dir("test")
    valid_results = plotter.load_results_from_dir("valid")
    
    generated_plots = []
    
    # Plot training metrics if available
    if train_results:
        plot_path = plotter.plot_training_metrics(train_results)
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot test metrics if available
    if test_results:
        plot_path = plotter.plot_evaluation_metrics(test_results, "test")
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot validation metrics if available
    if valid_results:
        plot_path = plotter.plot_evaluation_metrics(valid_results, "valid")
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot comparison if both training and test results available
    if train_results and test_results:
        plot_path = plotter.plot_metric_comparison(train_results, test_results)
        if plot_path:
            generated_plots.append(plot_path)
    
    print(f"\nGenerated {len(generated_plots)} plots:")
    for plot_path in generated_plots:
        print(f"  - {plot_path}")
    
    return generated_plots


if __name__ == "__main__":
    # Test plotting system
    plot_all_results()