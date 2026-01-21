#!/usr/bin/env python3
"""
Adaptive Metrics Plotting System for Deep-GPCM Models
Automatically adjusts subplot layout based on available metrics.
"""

import os
import sys
import json
import matplotlib

# Force a non-interactive backend unless the user explicitly supplied one.
# This prevents Qt from trying to load unavailable X11 plugins (e.g., xcb)
# on headless or minimal environments.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

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

# Configure matplotlib for better font rendering
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.weight'] = 'normal'  # Set default to normal, then override for titles

# Force matplotlib to use a font that supports bold
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Arial' in available_fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Arial')
elif 'Liberation Sans' in available_fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Liberation Sans')


class AdaptivePlotter:
    """Adaptive plotting system that adjusts to available metrics."""
    
    def __init__(self, results_dir: str = "results", figsize_base: Tuple[int, int] = (4, 3), dataset: Optional[str] = None):
        """Initialize plotter with base figure size per subplot."""
        self.results_dir = Path(results_dir)
        self.dataset = dataset
        
        # Create dataset-specific plots directory if dataset is provided
        if dataset:
            # NEW STRUCTURE: results/{dataset}/plots/
            self.plots_dir = self.results_dir / dataset / "plots"
        else:
            # Legacy structure: results/plots/
            self.plots_dir = self.results_dir / "plots"
        
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.figsize_base = figsize_base
        
        # Initialize consistent model color mapping
        self.model_colors = {}
        self.color_sequence = 0
        
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
                'colors': ['#6A994E', '#386641', '#BC4749', '#D62728'],
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
        
        # Build comprehensive higher_is_better lookup from metric categories
        self.higher_is_better_map = {}
        for category_data in self.metric_categories.values():
            for metric, higher_better in zip(category_data['metrics'], category_data['higher_better']):
                self.higher_is_better_map[metric] = higher_better
    
    def is_higher_better(self, metric: str) -> bool:
        """Determine if higher values are better for a given metric using comprehensive lookup."""
        # First check the comprehensive mapping
        if metric in self.higher_is_better_map:
            return self.higher_is_better_map[metric]
        
        # Fallback to keyword-based detection for unknown metrics
        # Lower is better for metrics containing these keywords
        lower_keywords = ['error', 'loss', 'entropy']
        return not any(keyword in metric.lower() for keyword in lower_keywords)
    
    def get_model_color(self, model_name: str) -> str:
        """Get consistent color for a model using factory-defined colors."""
        # First try to get color from model factory
        try:
            from models.factory import get_model_color as factory_get_color
            return factory_get_color(model_name)
        except:
            pass
        
        # Fallback to local color mapping for backward compatibility
        consistent_colors = {
            'deep_gpcm': '#ff7f0e',     # Orange
            'attn_gpcm': '#1f77b4',     # Blue
            'coral_gpcm': '#2ca02c',    # Green
            'coral_gpcm_proper': '#e377c2',  # Pink (proper IRT-CORAL separation)
            'ecoral_gpcm': '#d62728',   # Red (enhanced model)
            'adaptive_coral_gpcm': '#9467bd',  # Purple (adaptive research model)
            'full_adaptive_coral_gpcm': '#8c564b'  # Brown (full adaptive model)
        }
        
        if model_name in consistent_colors:
            return consistent_colors[model_name]
        
        # For unknown models, use dynamic assignment
        if model_name not in self.model_colors:
            # Use a diverse color palette that works well for up to 10 models
            colors = plt.cm.tab10.colors if hasattr(plt.cm.tab10, 'colors') else plt.cm.tab10(range(10))
            # If we have more than 10 models, cycle through with different shades
            base_idx = self.color_sequence % 10
            if self.color_sequence >= 10:
                # Darken or lighten the color for additional models
                factor = 0.7 if (self.color_sequence // 10) % 2 == 0 else 1.3
                color = tuple(min(1.0, c * factor) for c in colors[base_idx][:3])
                self.model_colors[model_name] = color
            else:
                self.model_colors[model_name] = colors[base_idx]
            self.color_sequence += 1
        return self.model_colors[model_name]
    
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
        """Load all results from a directory with support for new dataset-centric structure."""
        results = []
        
        if self.dataset:
            # NEW STRUCTURE: results/{dataset}/metrics/{result_type}_{model}.json
            new_structure_dir = self.results_dir / self.dataset / "metrics"
            if new_structure_dir.exists():
                pattern = f"{result_type}_*.json"
                for file_path in new_structure_dir.glob(pattern):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            data['source_file'] = file_path.name
                            results.append(data)
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
                
                # If we found files in new structure, return them
                if results:
                    return results
            
            # LEGACY STRUCTURE: results/{result_type}/{dataset}/*.json
            legacy_dir = self.results_dir / result_type / self.dataset
            if legacy_dir.exists():
                for file_path in legacy_dir.glob("*.json"):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            data['source_file'] = file_path.name
                            results.append(data)
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}: {e}")
        else:
            # No dataset specified, try legacy structure
            target_dir = self.results_dir / result_type
            if target_dir.exists():
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
                        # Try multiple ways to get model type
                        model_type = result['config'].get('model_type')
                        if not model_type:
                            model_type = result['config'].get('model')
                        if not model_type and 'models' in result['config']:
                            models_list = result['config']['models']
                            if models_list and len(models_list) > 0:
                                model_type = models_list[0]  # Take first model
                        if not model_type:
                            model_type = 'unknown'
                            
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
                
                # Also extract top-level metrics (for per-category accuracy, etc.)
                excluded_top_level = {'config', 'source_file', 'timestamp', 'predictions', 
                                    'probabilities', 'actual_labels', 'confusion_matrix',
                                    'ordinal_distances', 'category_transitions', 'training_history'}
                for key, value in result.items():
                    if key not in excluded_top_level and isinstance(value, (int, float, str)):
                        row[key] = value
                
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
                            save_path: Optional[str] = None,
                            zoom_in: bool = False) -> str:
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
        
        # Create figure with increased height for better spacing
        fig_width = cols * self.figsize_base[0]
        fig_height = rows * (self.figsize_base[1] + 1.2)  # Add 1.2 inches per row for better spacing
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get unique models and datasets for grouping
        models = [m for m in df['model_type'].unique() if m is not None] if 'model_type' in df.columns else ['unknown']
        datasets = df['dataset'].unique() if 'dataset' in df.columns else ['unknown']
        
        # Use consistent color mapping for models
        model_color_map = {model: self.get_model_color(model) for model in models}
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'Metric\n{metric}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title())
                continue
            
            # First, determine the best performing model based on final values
            final_values = {}
            for model in models:
                model_data = df[df['model_type'] == model] if 'model_type' in df.columns else df
                if 'epoch' in model_data.columns and not model_data.empty and metric in model_data.columns:
                    if 'fold' in model_data.columns:
                        # Get mean of final epoch across folds
                        max_epoch = model_data['epoch'].max()
                        final_epoch_data = model_data[model_data['epoch'] == max_epoch]
                        final_values[model] = final_epoch_data[metric].mean()
                    else:
                        # Get final value
                        final_values[model] = model_data[metric].iloc[-1]
            
            # Determine best model
            if final_values:
                higher_is_better = self.is_higher_better(metric)
                if higher_is_better:
                    best_model = max(final_values, key=final_values.get)
                else:
                    best_model = min(final_values, key=final_values.get)
            else:
                best_model = None
            
            # Plot data for each model
            for model in models:
                model_data = df[df['model_type'] == model] if 'model_type' in df.columns else df
                
                if 'epoch' in model_data.columns and not model_data.empty:
                    # Group by fold if available
                    if 'fold' in model_data.columns:
                        # Calculate statistics per epoch across folds
                        epoch_stats = model_data.groupby('epoch')[metric].agg(['mean', 'std', 'min', 'max', 'quantile']).reset_index()
                        epoch_stats.columns = ['epoch', 'mean', 'std', 'min', 'max', 'q50']
                        
                        # Calculate quartiles for better band representation
                        q25 = model_data.groupby('epoch')[metric].quantile(0.25)
                        q75 = model_data.groupby('epoch')[metric].quantile(0.75)
                        
                        # Create confidence band using IQR-style approach
                        # Use quartiles for main band and std for secondary band
                        
                        # Primary band (IQR): Q1 to Q3 - more visible (no legend)
                        ax.fill_between(epoch_stats['epoch'], q25.values, q75.values,
                                       color=model_color_map[model], alpha=0.3)
                        
                        # Secondary band (mean ± std) - subtle background (no legend)
                        ax.fill_between(epoch_stats['epoch'], 
                                       epoch_stats['mean'] - epoch_stats['std'],
                                       epoch_stats['mean'] + epoch_stats['std'],
                                       color=model_color_map[model], alpha=0.1)
                        
                        # Highlighted mean line - bold and prominent (only this in legend)
                        ax.plot(epoch_stats['epoch'], epoch_stats['mean'], 
                               color=model_color_map[model], linewidth=3, 
                               label=f'{model}', linestyle='-', 
                               marker='o', markersize=3, markevery=3)
                        
                        # Add final mean value annotation only for best model - position below curve
                        if len(epoch_stats) > 0 and model == best_model:
                            final_epoch = epoch_stats['epoch'].iloc[-1]
                            final_mean = epoch_stats['mean'].iloc[-1]
                            final_std = epoch_stats['std'].iloc[-1]
                            
                            # Handle NaN/invalid std values
                            if pd.isna(final_std) or np.isinf(final_std) or final_std <= 0:
                                final_std = 0.0
                            
                            # Position annotation below/above the curve endpoint based on metric type
                            higher_is_better = self.is_higher_better(metric)
                            
                            # Position below curve for "higher is better" metrics, above for "lower is better"
                            if higher_is_better:
                                ax.annotate(f'{final_mean:.2f} ★', 
                                          xy=(final_epoch, final_mean),
                                          xytext=(0, -15), textcoords='offset points',
                                          fontsize=9, color=model_color_map[model],
                                          fontweight='bold', ha='center', va='top',
                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                  facecolor='white', alpha=0.8, edgecolor='green'))
                            else:
                                ax.annotate(f'{final_mean:.2f} ★', 
                                          xy=(final_epoch, final_mean),
                                          xytext=(0, 15), textcoords='offset points',
                                          fontsize=9, color=model_color_map[model],
                                          fontweight='bold', ha='center', va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                  facecolor='white', alpha=0.8, edgecolor='green'))
                    else:
                        # Single run
                        ax.plot(model_data['epoch'], model_data[metric], 
                               color=model_color_map[model], linewidth=2, label=model)
                        
                        # Add final value annotation only for best model - position below curve
                        if not model_data.empty and model == best_model:
                            final_epoch = model_data['epoch'].iloc[-1]
                            final_value = model_data[metric].iloc[-1]
                            
                            # Position annotation below/above the curve endpoint based on metric type
                            higher_is_better = self.is_higher_better(metric)
                            
                            # Position below curve for "higher is better" metrics, above for "lower is better"
                            if higher_is_better:
                                ax.annotate(f'{final_value:.2f} ★', 
                                          xy=(final_epoch, final_value),
                                          xytext=(0, -15), textcoords='offset points',
                                          fontsize=9, color=model_color_map[model],
                                          fontweight='bold', ha='center', va='top',
                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                  facecolor='white', alpha=0.8, edgecolor='green'))
                            else:
                                ax.annotate(f'{final_value:.2f} ★', 
                                          xy=(final_epoch, final_value),
                                          xytext=(0, 15), textcoords='offset points',
                                          fontsize=9, color=model_color_map[model],
                                          fontweight='bold', ha='center', va='bottom',
                                          bbox=dict(boxstyle='round,pad=0.2', 
                                                  facecolor='white', alpha=0.8, edgecolor='green'))
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Add directional arrow based on metric ranking criteria
            higher_is_better = self.is_higher_better(metric)
            arrow = '↑' if higher_is_better else '↓'
            title_with_arrow = f"{metric.replace('_', ' ').title()} {arrow}"
            ax.set_title(title_with_arrow)
            ax.grid(True, alpha=0.3)
            
            # Apply zoom if requested
            if zoom_in:
                # Get all y-values from the plot to determine range
                all_y_values = []
                for line in ax.get_lines():
                    all_y_values.extend(line.get_ydata())
                
                valid_values = [v for v in all_y_values if np.isfinite(v)]
                if valid_values:
                    data_min = min(valid_values)
                    data_max = max(valid_values)
                    data_range = data_max - data_min
                    
                    # Add padding (10% on each side)
                    padding = max(data_range * 0.1, 0.001)
                    ax.set_ylim(max(0, data_min - padding), data_max + padding)
            
            # Figure-level legend will be added below
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Create figure-level legend with adaptive positioning
        unique_models = sorted([m for m in models if m is not None])
        legend_elements = [plt.Line2D([0], [0], color=self.get_model_color(model), 
                                     linewidth=3, label=model) for model in unique_models]
        
        # Adaptive legend positioning based on layout
        if rows <= 1:
            # For single row, place legend closer to title - reduced spacing
            legend_y = 0.93
            title_y = 0.97
        elif rows <= 2:
            # For 2 rows, moderate space between title and legend - reduced spacing
            legend_y = 0.91
            title_y = 0.96
        else:
            # For 3+ rows, more space but still reduced - reduced spacing
            legend_y = 0.89
            title_y = 0.95
            
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, legend_y), 
                  ncol=len(unique_models), fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        title = 'Training Metrics Over Epochs'
        if zoom_in:
            title += ' - Zoomed View'
        fig.suptitle(title, fontsize=18, weight='bold', y=title_y)
        
        # Adaptive tight_layout rect based on legend position with reduced spacing
        if rows <= 1:
            plt.tight_layout(rect=[0, 0, 1, 0.88], h_pad=1.2, w_pad=1.0)
        elif rows <= 2:
            plt.tight_layout(rect=[0, 0, 1, 0.86], h_pad=1.5, w_pad=1.2)
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.84], h_pad=1.8, w_pad=1.4)
        
        # Save plot
        if save_path is None:
            filename = "train_zoomed.png" if zoom_in else "train.png"
            save_path = self.plots_dir / filename
        else:
            save_path = Path(save_path)
            if zoom_in and not str(save_path).endswith('_zoomed.png'):
                save_path = save_path.parent / (save_path.stem + '_zoomed' + save_path.suffix)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training metrics plot saved: {save_path}")
        return str(save_path)
    
    def plot_evaluation_metrics(self, results: List[Dict[str, Any]], 
                              result_type: str = "test",
                              metrics_to_plot: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              zoom_in: bool = False) -> str:
        """Plot evaluation metrics (final performance) with adaptive layout."""
        available_metrics, df = self.extract_metrics_from_results(results, "evaluation")
        
        if df.empty:
            print(f"No {result_type} data found.")
            return ""
        
        # Select metrics to plot (exclude cohen_kappa for test results)
        if metrics_to_plot is None:
            filtered_metrics = self.filter_metrics_for_plotting(available_metrics, "auto")
            # Remove cohen_kappa from test results display
            if result_type == "test" and 'cohen_kappa' in filtered_metrics:
                filtered_metrics.remove('cohen_kappa')
            metrics_to_plot = filtered_metrics
        
        if not metrics_to_plot:
            print("No suitable metrics found for plotting.")
            return ""
        
        # Calculate subplot layout
        n_metrics = len(metrics_to_plot)
        rows, cols = self.calculate_subplot_layout(n_metrics)
        
        # Create figure with adaptive sizing for zoom
        fig_width = cols * self.figsize_base[0]
        if zoom_in:
            # Increase figure height and add extra space for zoomed plots
            fig_height = rows * (self.figsize_base[1] + 0.8)  # Add 0.8 inches per row
        else:
            fig_height = rows * self.figsize_base[1]
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get unique models for comparison
        models = [m for m in df['model_type'].unique() if m is not None] if 'model_type' in df.columns else ['unknown']
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'Metric\n{metric}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title())
                continue
            
            # Initialize best_value_info outside the if/else blocks
            best_value_info = None
            
            # Create bar plot for model comparison
            if 'fold' in df.columns:
                # Cross-validation results - show mean and error bars
                means = df.groupby('model_type')[metric].mean()
                stds = df.groupby('model_type')[metric].std()
                
                x_pos = np.arange(len(means))
                # Use consistent colors for models
                colors = [self.get_model_color(model) for model in means.index]
                
                # Determine if higher is better for this metric
                higher_is_better = self.is_higher_better(metric) 
                
                # Find best performing model
                if higher_is_better:
                    best_idx = means.argmax()
                else:
                    best_idx = means.argmin()
                
                # Create bars with highlight for best
                bars = ax.bar(x_pos, means.values, yerr=stds.values, 
                             capsize=5, alpha=0.7, color=colors)
                
                # Highlight best performing model with green edge
                bars[best_idx].set_edgecolor('green')
                bars[best_idx].set_linewidth(3)
                bars[best_idx].set_alpha(1.0)
                
                # Remove x-axis labels for test results since we have legend
                if result_type != "test":
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(means.index, rotation=45)
                else:
                    ax.set_xticks([])
                
                # Store best value info for later annotation
                if best_idx >= 0 and best_idx < len(means):
                    best_value_info = (best_idx, means.values[best_idx], stds.values[best_idx])
            
            else:
                # Single evaluation - show individual values
                model_values = df.groupby('model_type')[metric].first()
                
                x_pos = np.arange(len(model_values))
                # Use consistent colors for models
                colors = [self.get_model_color(model) for model in model_values.index]
                
                # Determine if higher is better for this metric
                higher_is_better = self.is_higher_better(metric) 
                
                # Find best performing model
                if higher_is_better:
                    best_idx = model_values.argmax()
                else:
                    best_idx = model_values.argmin()
                
                # Create bars with highlight for best
                bars = ax.bar(x_pos, model_values.values, alpha=0.7, color=colors)
                
                # Highlight best performing model with green edge
                bars[best_idx].set_edgecolor('green')
                bars[best_idx].set_linewidth(3)
                bars[best_idx].set_alpha(1.0)
                
                # Remove x-axis labels for test results since we have legend
                if result_type != "test":
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(model_values.index, rotation=45)
                else:
                    ax.set_xticks([])
                
                # Store best value info for later annotation
                if best_idx >= 0 and best_idx < len(model_values):
                    best_value_info = (best_idx, model_values.values[best_idx])
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits with zoom support FIRST
            y_min, y_max = ax.get_ylim()
            
            if zoom_in:
                # Get actual data range for zoom
                if 'model_values' in locals():
                    # Single evaluation case
                    data_values = model_values.values
                elif 'means' in locals():
                    # CV case
                    data_values = means.values
                else:
                    data_values = []
                
                valid_values = [v for v in data_values if np.isfinite(v) and v > 0]
                if valid_values:
                    data_min = min(valid_values)
                    data_max = max(valid_values)
                    data_range = data_max - data_min
                    
                    # Add padding (10% on each side)
                    padding = max(data_range * 0.1, 0.001)
                    y_min = max(0, data_min - padding)
                    y_max = data_max + padding
                    y_range = y_max - y_min
                else:
                    y_range = y_max - y_min
            else:
                y_range = y_max - y_min
            
            # For test results, add significant top padding to prevent annotation collision with suptitle
            if result_type == "test":
                # Add 25% extra space at top for annotations to not touch suptitle
                extra_top_space = y_range * 0.25
            else:
                extra_top_space = y_range * 0.1
            
            # Ensure enough space for text annotations
            if best_value_info is not None:
                annotation_offset = y_range * 0.05  # Use relative offset
                if len(best_value_info) == 3:
                    idx, mean, std = best_value_info
                    text_top = mean + std + annotation_offset
                else:
                    idx, value = best_value_info
                    text_top = value + annotation_offset
                
                # Adjust y_max if text would be clipped
                if text_top > y_max:
                    extra_top_space = max(extra_top_space, text_top - y_max + y_range * 0.1)
            
            ax.set_ylim(y_min - y_range * 0.05, y_max + extra_top_space)
            
            # Add best value annotation AFTER adjusting y-limits for proper positioning
            if best_value_info is not None:
                # Get current y-limits after zoom adjustment
                final_y_min, final_y_max = ax.get_ylim()
                final_y_range = final_y_max - final_y_min
                
                if len(best_value_info) == 3:  # CV case with std
                    idx, mean, std = best_value_info
                    # Handle NaN/invalid std values
                    if pd.isna(std) or np.isinf(std) or std <= 0:
                        std = 0.0
                    # Get best model color
                    best_model = means.index[idx]  # Use means for CV case
                    best_model_color = self.get_model_color(best_model)
                    # Position text above error bar with proper offset
                    text_y = mean + std + final_y_range * 0.05
                    # For test results, don't show std
                    if result_type == "test":
                        annotation_text = f'{mean:.2f} ★'
                    else:
                        annotation_text = f'{mean:.2f}±{std:.2f} ★'
                    ax.text(idx, text_y, 
                           annotation_text, ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color=best_model_color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=best_model_color, linewidth=1.5, alpha=0.9))
                else:  # Single evaluation case
                    idx, value = best_value_info
                    # Get best model color
                    best_model = model_values.index[idx]
                    best_model_color = self.get_model_color(best_model)
                    # Use proper relative positioning based on final y-range
                    text_y = value + final_y_range * 0.05
                    ax.text(idx, text_y, 
                           f'{value:.2f} ★', ha='center', va='bottom', 
                           fontsize=10, fontweight='bold', color=best_model_color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=best_model_color, linewidth=1.5, alpha=0.9))
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Add legend as figure-level legend to avoid blocking subplots
        if len(axes) > 0 and 'models' in locals() and len(models) > 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=self.get_model_color(model), 
                                   edgecolor='black', linewidth=1, 
                                   label=model) for model in models]
            
            # Adaptive legend positioning based on layout
            if rows <= 1:
                # For single row, place legend closer to title - reduced spacing
                legend_y = 0.93
                title_y = 0.97
                rect_top = 0.88
            elif rows <= 2:
                # For 2 rows, moderate space between title and legend - reduced spacing
                legend_y = 0.91
                title_y = 0.96
                rect_top = 0.86
            else:
                # For 3+ rows, more space but still reduced - reduced spacing
                legend_y = 0.89
                title_y = 0.95
                rect_top = 0.84
                
            # Place legend outside the plot area
            fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, legend_y), 
                      ncol=len(models), fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        title = f'{result_type.title()} Results Comparison'
        if zoom_in:
            title += ' - Zoomed View'
            
        # Use adaptive positioning if we have variables from legend setup
        if 'title_y' in locals() and 'rect_top' in locals():
            plt.suptitle(title, fontsize=14, fontweight='bold', y=title_y)
            if zoom_in:
                # Reduced spacing for zoomed plots
                if rows <= 1:
                    plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.5, w_pad=1.2)
                elif rows <= 2:
                    plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.8, w_pad=1.4)
                else:
                    plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=2.0, w_pad=1.5)
            else:
                # Reduced spacing for normal plots
                if rows <= 1:
                    plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.2, w_pad=1.0)
                elif rows <= 2:
                    plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.4, w_pad=1.2)
                else:
                    plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.6, w_pad=1.3)
        else:
            # Fallback to default positioning with reduced spacing
            plt.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
            if zoom_in:
                # Reduced spacing for zoomed plots
                plt.tight_layout(rect=[0, 0, 1, 0.86], h_pad=1.8, w_pad=1.4)
            else:
                plt.tight_layout(rect=[0, 0, 1, 0.88], h_pad=1.4, w_pad=1.2)
        
        # Save plot
        if save_path is None:
            filename = f"{result_type}_zoomed.png" if zoom_in else f"{result_type}.png"
            save_path = self.plots_dir / filename
        else:
            save_path = Path(save_path)
            if zoom_in and not str(save_path).endswith('_zoomed.png'):
                save_path = save_path.parent / (save_path.stem + '_zoomed' + save_path.suffix)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{result_type.title()} metrics plot saved: {save_path}")
        return str(save_path)
    
    def plot_metric_comparison(self, train_results: List[Dict[str, Any]], 
                             test_results: List[Dict[str, Any]],
                             validation_results: Optional[List[Dict[str, Any]]] = None,
                             save_path: Optional[str] = None,
                             zoom_in: bool = False) -> str:
        """Create side-by-side comparison of training and test metrics with highlighting.
        
        Args:
            zoom_in: If True, zoom in on actual data range to highlight small differences.
                    If False, use standard 0-1 range for most metrics.
        """
        # All available metrics
        ALL_METRICS = [
            'categorical_accuracy',
            'ordinal_accuracy',
            'quadratic_weighted_kappa',
            'mean_absolute_error',
            'kendall_tau',
            'spearman_correlation',
            'cohen_kappa',
            'cross_entropy'
        ]
        
        # Note: Using self.is_higher_better() method for consistent ranking logic
        
        # Extract data from results
        train_data = {}
        test_data = {}
        
        # Process training results - try validation results first (CV summaries)
        if validation_results:
            for result in validation_results:
                if 'config' in result and 'cv_summary' in result:
                    # Try multiple ways to get model type
                    model = result['config'].get('model_type')
                    if not model:
                        model = result['config'].get('model')
                    if not model and 'models' in result['config']:
                        models_list = result['config']['models']
                        if models_list and len(models_list) > 0:
                            model = models_list[0]
                    if not model:
                        model = 'unknown'
                    train_data[model] = result['cv_summary']
        
        # Fall back to train_results if no validation results
        if not train_data:
            for result in train_results:
                if 'config' in result:
                    # Try multiple ways to get model type (same logic as in extract_metrics_from_results)
                    model = result['config'].get('model_type')
                    if not model:
                        model = result['config'].get('model')
                    if not model and 'models' in result['config']:
                        models_list = result['config']['models']
                        if models_list and len(models_list) > 0:
                            model = models_list[0]  # Take first model
                    if not model:
                        model = 'unknown'
                        
                    # Use metrics field instead of cv_summary
                    if 'metrics' in result:
                        train_data[model] = result['metrics']
                    elif 'cv_summary' in result:
                        train_data[model] = result['cv_summary']
        
        # Process test results
        for result in test_results:
            if 'config' in result:
                # Try multiple ways to get model type
                model = result['config'].get('model_type')
                if not model:
                    model = result['config'].get('model')
                if not model and 'models' in result['config']:
                    models_list = result['config']['models']
                    if models_list and len(models_list) > 0:
                        model = models_list[0]
                if not model:
                    model = 'unknown'
                test_data[model] = result
        
        if not train_data or not test_data:
            print("Insufficient data for training vs test comparison.")
            return ""
        
        # Common models
        models = sorted(list(set(train_data.keys()) & set(test_data.keys())))
        if not models:
            print("No common models found between training and test results.")
            return ""
        
        # Use consistent model colors from get_model_color method
        
        # Helper function to highlight best value
        def highlight_best_value(ax, bars, values, metric):
            """Highlight the best performing model."""
            if not self.is_higher_better(metric):
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            
            # Star annotation is already included in the value annotation above
            # Add border to highlight best bar
            best_bar = bars[best_idx]
            best_bar.set_edgecolor('black')
            best_bar.set_linewidth(2)
        
        # Create figure with proper layout for all metrics
        n_metrics = len(ALL_METRICS)
        cols = 4
        rows = (n_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.5))
        axes = axes.flatten()
        
        for idx, metric in enumerate(ALL_METRICS):
            ax = axes[idx]
            
            # Skip if metric not available
            has_metric = False
            for model in models:
                if (model in train_data and metric in train_data[model]) or                    (model in test_data and metric in test_data[model]):
                    has_metric = True
                    break
            
            if not has_metric:
                ax.set_visible(False)
                continue
            
            # Set up positions for grouped bars
            n_models = len(models)
            x = np.arange(2)  # Two groups: Train and Test
            width = 0.25 if n_models <= 3 else 0.2
            
            train_values = []
            test_values = []
            
            # Plot each model
            for i, model in enumerate(models):
                # Get values - handle both nested and flat structures
                if model in train_data and metric in train_data[model]:
                    # Check if it's a nested structure with 'mean' or a flat value
                    if isinstance(train_data[model][metric], dict) and 'mean' in train_data[model][metric]:
                        train_val = train_data[model][metric]['mean']
                    else:
                        # Flat structure - direct value
                        train_val = train_data[model][metric]
                else:
                    train_val = 0
                    
                # Look for test metric in evaluation_results first, then top level
                if model in test_data:
                    if 'evaluation_results' in test_data[model] and metric in test_data[model]['evaluation_results']:
                        test_val = test_data[model]['evaluation_results'][metric]
                    else:
                        test_val = test_data[model].get(metric, 0)
                else:
                    test_val = 0
                
                # Handle NaN/Inf values
                train_val = train_val if np.isfinite(train_val) else 0
                test_val = test_val if np.isfinite(test_val) else 0
                
                train_values.append(train_val)
                test_values.append(test_val)
                
                # Calculate positions
                positions = x + (i - n_models/2 + 0.5) * width
                
                # Plot bars
                values = [train_val, test_val]
                color = self.get_model_color(model)
                bars = ax.bar(positions, values, width, 
                              label=model.capitalize(), 
                              color=color,
                              alpha=0.8)
                
            # Find and annotate best performers for both train and test (no duplicate values on bars)
            if not self.is_higher_better(metric):
                train_best_idx = np.argmin(train_values) if any(v > 0 for v in train_values) else None
                test_best_idx = np.argmin(test_values) if any(v > 0 for v in test_values) else None
            else:
                train_best_idx = np.argmax(train_values) if any(v > 0 for v in train_values) else None
                test_best_idx = np.argmax(test_values) if any(v > 0 for v in test_values) else None
            
            # Store best performer info for later annotation (after y-limits are set)
            train_best_info = None
            test_best_info = None
            
            if train_best_idx is not None:
                train_best_val = train_values[train_best_idx]
                train_best_model = models[train_best_idx]
                train_model_color = self.get_model_color(train_best_model)
                train_x_pos = (train_best_idx - n_models/2 + 0.5) * width
                train_best_info = (train_x_pos, train_best_val, train_model_color, train_best_model)
            
            if test_best_idx is not None:
                test_best_val = test_values[test_best_idx]
                test_best_model = models[test_best_idx]
                test_model_color = self.get_model_color(test_best_model)
                test_x_pos = 1 + (test_best_idx - n_models/2 + 0.5) * width
                test_best_info = (test_x_pos, test_best_val, test_model_color, test_best_model)
            
            # Highlight best performers on both training AND test data
            # Collect bars in the correct order by sorting by x position
            all_patches = list(ax.patches)
            
            # Separate train and test bars based on x position
            train_patches = sorted([p for p in all_patches if p.get_x() < 0.5], key=lambda p: p.get_x())
            test_patches = sorted([p for p in all_patches if p.get_x() >= 0.5], key=lambda p: p.get_x())
            
            # Highlight best training performer
            if len(train_patches) == len(train_values) and any(v > 0 for v in train_values):
                highlight_best_value(ax, train_patches, train_values, metric)
            
            # Highlight best test performer  
            if len(test_patches) == len(test_values) and any(v > 0 for v in test_values):
                highlight_best_value(ax, test_patches, test_values, metric)
            
            # Customize subplot
            title = metric.replace('_', ' ').title()
            if not self.is_higher_better(metric):
                title += ' ↓'
            else:
                title += ' ↑'
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Train', 'Test'], fontsize=10)
            ax.set_ylabel('Score', fontsize=9)
            
            # Set y-axis limits based on metric with NaN/Inf handling
            all_values = train_values + test_values
            # Filter out NaN and Inf values
            valid_values = [v for v in all_values if np.isfinite(v) and v > 0]
            
            if zoom_in and valid_values:
                # Zoom-in mode: tight range around actual values
                min_val = min(valid_values)
                max_val = max(valid_values)
                value_range = max_val - min_val
                
                # Add padding (10% on each side) + extra space for annotations
                padding = max(value_range * 0.1, 0.001)  # minimum padding
                # Reserve extra space at top for annotations to avoid touching suptitle
                annotation_space = value_range * 0.15  # 15% extra space for annotations
                ax.set_ylim(max(0, min_val - padding), max_val + padding + annotation_space)
            else:
                # Normal mode: standard range with space for annotations
                if metric in ['mean_absolute_error', 'cross_entropy']:
                    max_val = max(valid_values) if valid_values else 1
                    ax.set_ylim(0, max_val * 1.3)  # Increased from 1.2 to 1.3 for annotation space
                else:
                    ax.set_ylim(0, 1.15)  # Increased from 1.05 to 1.15 for annotation space
            
            ax.grid(axis='y', alpha=0.3)
            
            # Add annotations AFTER y-limits are set for proper positioning
            if train_best_info is not None or test_best_info is not None:
                # Get final y-limits after zoom adjustment
                final_y_min, final_y_max = ax.get_ylim()
                final_y_range = final_y_max - final_y_min
                annotation_offset = final_y_range * 0.05  # 5% of final y-axis range
                
                if train_best_info is not None:
                    train_x_pos, train_best_val, train_model_color, train_best_model = train_best_info
                    ax.text(train_x_pos, train_best_val + annotation_offset,
                           f'{train_best_val:.3f} ★',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=train_model_color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=train_model_color, linewidth=1.5, alpha=0.9))
                
                if test_best_info is not None:
                    test_x_pos, test_best_val, test_model_color, test_best_model = test_best_info
                    ax.text(test_x_pos, test_best_val + annotation_offset,
                           f'{test_best_val:.3f} ★',
                           ha='center', va='bottom', fontsize=9, fontweight='bold',
                           color=test_model_color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=test_model_color, linewidth=1.5, alpha=0.9))
            
            # Legend will be added at figure level instead of subplot level
        
        # Hide empty subplots
        for idx in range(len(ALL_METRICS), len(axes)):
            axes[idx].set_visible(False)
        
        # Adaptive legend positioning for comparison plot  
        # Calculate layout dimensions from figure setup
        n_metrics = len(ALL_METRICS)
        cols = 4
        rows = (n_metrics + cols - 1) // cols
        
        if rows <= 1:
            # For single row, place legend closer to title - reduced spacing
            legend_y = 0.93
            title_y = 0.97
            rect_top = 0.88
        elif rows <= 2:
            # For 2 rows, moderate space between title and legend - reduced spacing
            legend_y = 0.91
            title_y = 0.96
            rect_top = 0.86
        else:
            # For 3+ rows, more space but still reduced - reduced spacing
            legend_y = 0.89
            title_y = 0.95
            rect_top = 0.84
        
        # Overall title with adaptive positioning
        title = 'Training vs Test Performance Comparison (★ = Best)'
        if zoom_in:
            title += ' - Zoomed View'
        fig.suptitle(title, fontsize=18, weight='bold', y=title_y)
        
        # Add figure-level legend with adaptive positioning
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.get_model_color(model), 
                               edgecolor='black', linewidth=1, 
                               label=model.replace('_', ' ').title()) for model in models]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, legend_y), 
                  ncol=len(models), fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        if zoom_in:
            # Reduced spacing for zoomed comparison plots
            if rows <= 1:
                plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.8, w_pad=1.4)
            elif rows <= 2:
                plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=2.0, w_pad=1.6)
            else:
                plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=2.2, w_pad=1.8)
        else:
            # Reduced spacing for normal comparison plots
            if rows <= 1:
                plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.4, w_pad=1.2)
            elif rows <= 2:
                plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.6, w_pad=1.3)
            else:
                plt.tight_layout(rect=[0, 0, 1, rect_top], h_pad=1.8, w_pad=1.4)
        
        # Save plot
        if save_path is None:
            filename = "train_v_test_zoomed.png" if zoom_in else "train_v_test.png"
            save_path = self.plots_dir / filename
        else:
            save_path = Path(save_path)
            if zoom_in and not str(save_path).endswith('_zoomed.png'):
                save_path = save_path.parent / (save_path.stem + '_zoomed' + save_path.suffix)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        view_type = "zoomed " if zoom_in else ""
        print(f"Training vs test {view_type}comparison plot saved: {save_path}")
        return str(save_path)

    def plot_categorical_breakdown(self, results: List[Dict[str, Any]], 
                                 result_type: str = "test",
                                 save_path: Optional[str] = None) -> str:
        """Plot per-category accuracy breakdown with consistent model coloring."""
        available_metrics, df = self.extract_metrics_from_results(results, "evaluation")
        
        if df.empty:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for per-category metrics
        category_metrics = [m for m in available_metrics if m.startswith('cat_') and m.endswith('_accuracy')]
        
        if not category_metrics:
            print("No per-category accuracy metrics found.")
            return ""
        
        # Sort category metrics by category number
        category_metrics.sort(key=lambda x: int(x.split('_')[1]))
        
        # Get unique models
        models = [m for m in df['model_type'].unique() if m is not None] if 'model_type' in df.columns else ['unknown']
        
        # Create figure
        fig_width = 10
        fig_height = 6
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        
        n_categories = len(category_metrics)
        n_models = len(models)
        
        # Setup bar positions
        x = np.arange(n_categories)
        width = 0.8 / n_models
        
        # Plot bars for each model
        for i, model in enumerate(models):
            model_data = df[df['model_type'] == model] if 'model_type' in df.columns else df
            
            category_values = []
            category_stds = []
            
            for metric in category_metrics:
                if metric in model_data.columns:
                    if 'fold' in model_data.columns:
                        # CV results - show mean and std
                        mean_val = model_data[metric].mean()
                        std_val = model_data[metric].std()
                        category_values.append(mean_val)
                        category_stds.append(std_val)
                    else:
                        # Single evaluation
                        category_values.append(model_data[metric].iloc[0])
                        category_stds.append(0)
                else:
                    category_values.append(0)
                    category_stds.append(0)
            
            # Calculate positions
            positions = x + (i - n_models/2 + 0.5) * width
            
            # Use consistent model color
            color = self.get_model_color(model)
            
            # Create bars
            bars = ax.bar(positions, category_values, width, 
                          yerr=category_stds if any(category_stds) else None,
                          label=model, color=color, alpha=0.8,
                          capsize=5 if any(category_stds) else 0)
            
            # Add value labels on top of bars
            for j, (pos, val, std) in enumerate(zip(positions, category_values, category_stds)):
                if val > 0:
                    if std > 0:
                        ax.text(pos, val + std + 0.01, f'{val:.2f}', 
                               ha='center', va='bottom', fontsize=8)
                    else:
                        ax.text(pos, val + 0.01, f'{val:.2f}', 
                               ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('Response Category', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Category Accuracy Breakdown', fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        category_labels = [f'Category {i}' for i in range(n_categories)]
        ax.set_xticks(x)
        ax.set_xticklabels(category_labels)
        
        # Create figure-level legend
        legend_elements = [mpatches.Patch(facecolor=self.get_model_color(model), 
                                         edgecolor='black', label=model) for model in models]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), 
                  ncol=len(models), fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / "cat_breakdown.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Categorical breakdown plot saved: {save_path}")
        return str(save_path)
    
    def plot_confusion_matrices(self, results: List[Dict[str, Any]], 
                               result_type: str = "test",
                               save_path: Optional[str] = None) -> str:
        """Plot confusion matrices for each model with consistent coloring."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for confusion matrix data
        matrices_found = []
        models = []
        
        # Deduplicate models for confusion matrices to avoid showing same model twice
        seen_models = set()
        
        for result in results:
            # Check multiple possible locations for confusion matrix
            conf_matrix = None
            if 'confusion_matrix' in result:
                conf_matrix = result['confusion_matrix']
            elif 'evaluation_results' in result and 'confusion_matrix' in result['evaluation_results']:
                conf_matrix = result['evaluation_results']['confusion_matrix']
            
            if conf_matrix is not None:
                model_name = 'unknown'
                if 'config' in result:
                    model_name = result['config'].get('model_type', 
                                  result['config'].get('model', 'unknown'))
                
                # Skip if we've already seen this model (avoid duplicates)
                if model_name not in seen_models:
                    matrices_found.append(conf_matrix)
                    models.append(model_name)
                    seen_models.add(model_name)
        
        if not matrices_found:
            print("No confusion matrices found in results.")
            return ""
        
        n_models = len(matrices_found)
        
        # Create subplots
        if n_models == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            axes = [ax]
        else:
            cols = min(3, n_models)
            rows = int(np.ceil(n_models / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            if n_models > 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx, (matrix, model) in enumerate(zip(matrices_found, models)):
            ax = axes[idx]
            matrix = np.array(matrix)
            
            # Create custom colormap based on model color
            model_color = self.get_model_color(model)
            if isinstance(model_color, tuple):
                model_color = model_color[:3]  # RGB only
            
            # Convert to percentage matrix for coloring (row-wise percentages)
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            percentage_matrix = matrix / row_sums * 100
            
            # Create colormap from white to model color
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['white', model_color]
            custom_cmap = LinearSegmentedColormap.from_list(f'{model}_cmap', colors, N=256)
            
            # Plot heatmap using percentage matrix for coloring
            im = ax.imshow(percentage_matrix, interpolation='nearest', cmap=custom_cmap, vmin=0, vmax=100)
            
            # Add colorbar showing percentages
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Row Percentage (%)', rotation=270, labelpad=15)
            
            # Add text annotations with absolute counts
            thresh = 50.0  # 50% threshold for text color (white vs black)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    # Use percentage for text color determination
                    color = 'white' if percentage_matrix[i, j] > thresh else 'black'
                    # Display absolute count
                    ax.text(j, i, f'{matrix[i, j]}', 
                           ha='center', va='center', color=color, fontweight='bold')
            
            # Customize plot
            ax.set_xlabel('Predicted Category', fontsize=11)
            ax.set_ylabel('Actual Category', fontsize=11)
            ax.set_title(f'{model.title()} Confusion Matrix', fontsize=12, 
                        fontweight='bold', color=model_color)
            
            # Set ticks
            n_categories = matrix.shape[0]
            ax.set_xticks(range(n_categories))
            ax.set_yticks(range(n_categories))
            ax.set_xticklabels([f'Cat {i}' for i in range(n_categories)])
            ax.set_yticklabels([f'Cat {i}' for i in range(n_categories)])
            
            # Add border in model color
            for spine in ax.spines.values():
                spine.set_edgecolor(model_color)
                spine.set_linewidth(2)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Confusion Matrices - {result_type.title()} Results', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"confusion_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrices plot saved: {save_path}")
        return str(save_path)
    
    def plot_ordinal_distance_distribution(self, results: List[Dict[str, Any]], 
                                         result_type: str = "test",
                                         save_path: Optional[str] = None) -> str:
        """Plot distribution of ordinal prediction distances |actual - predicted|."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for prediction distance data
        distance_data = {}
        
        for result in results:
            model_name = 'unknown'
            if 'config' in result:
                model_name = result['config'].get('model_type', 
                              result['config'].get('model', 'unknown'))
            
            # Look for ordinal distances in various locations
            distances = None
            if 'ordinal_distances' in result:
                distances = result['ordinal_distances']
            elif 'evaluation_results' in result and 'ordinal_distances' in result['evaluation_results']:
                distances = result['evaluation_results']['ordinal_distances']
            elif 'predictions' in result and 'actual' in result:
                # Calculate distances from predictions and actual values
                pred = np.array(result['predictions'])
                actual = np.array(result['actual'])
                if len(pred) == len(actual):
                    distances = np.abs(pred - actual).tolist()
            
            if distances is not None:
                if model_name not in distance_data:
                    distance_data[model_name] = []
                distance_data[model_name].extend(distances)
        
        if not distance_data:
            print("No ordinal distance data found in results.")
            return ""
        
        # Create figure with extra height for legend and statistics
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate individual histograms for side-by-side bars like categorical breakdown
        max_distance = max(max(distances) for distances in distance_data.values())
        distance_values = list(range(int(max_distance) + 1))
        
        # Calculate counts for each model and distance
        model_counts = {}
        for model_name, distances in distance_data.items():
            counts = []
            for dist_val in distance_values:
                count = sum(1 for d in distances if int(d) == dist_val)
                counts.append(count)
            model_counts[model_name] = counts
        
        # Plot side-by-side bars like categorical breakdown
        n_models = len(distance_data)
        n_distances = len(distance_values)
        width = 0.8 / n_models
        x = np.arange(n_distances)
        
        for i, (model_name, counts) in enumerate(model_counts.items()):
            color = self.get_model_color(model_name)
            positions = x + (i - n_models/2 + 0.5) * width
            ax.bar(positions, counts, width, label=model_name, color=color, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
        
        # Use table-style annotation below the plot to avoid any conflicts
        # Calculate statistics for all models
        stats_data = []
        for model_name, distances in distance_data.items():
            color = self.get_model_color(model_name)
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            perfect_pct = (np.array(distances) == 0).mean() * 100
            stats_data.append((model_name, mean_dist, std_dist, perfect_pct, color))
        
        # Create a table below the plot
        table_y_start = -0.25
        row_height = 0.08
        col_width = 1.0 / len(stats_data)
        
        for i, (model_name, mean_dist, std_dist, perfect_pct, color) in enumerate(stats_data):
            x_center = (i + 0.5) * col_width
            
            # Model name (header)
            ax.text(x_center, table_y_start, f'{model_name}', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   color=color, transform=ax.transAxes)
            
            # Statistics
            stats_text = f'Mean: {mean_dist:.2f}\nStd: {std_dist:.2f}\nPerfect: {perfect_pct:.1f}%'
            ax.text(x_center, table_y_start - row_height, stats_text, 
                   ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2),
                   transform=ax.transAxes)
        
        # Customize plot
        ax.set_xlabel('Ordinal Distance |Actual - Predicted|', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Ordinal Prediction Distances', fontsize=14, fontweight='bold')
        
        # Set x-axis to show integer values
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i}' for i in distance_values])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Move legend under title with proper spacing (lower position to avoid blocking title)
        ax.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 0.98), ncol=len(distance_data), 
                  frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0.2, 1, 0.88])  # Leave more space for title above legend and bottom statistics table
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"ord_dist_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Ordinal distance distribution plot saved: {save_path}")
        return str(save_path)
    
    def plot_learning_curves_with_confidence(self, train_results: List[Dict[str, Any]], 
                                           valid_results: Optional[List[Dict[str, Any]]] = None,
                                           metrics_to_plot: Optional[List[str]] = None,
                                           save_path: Optional[str] = None) -> str:
        """Plot learning curves with confidence intervals for training and validation."""
        available_metrics, train_df = self.extract_metrics_from_results(train_results, "training")
        
        if train_df.empty:
            print("No training data found.")
            return ""
        
        # Select metrics to plot
        if metrics_to_plot is None:
            metrics_to_plot = self.filter_metrics_for_plotting(available_metrics, "auto")[:4]  # Top 4 for readability
        
        if not metrics_to_plot:
            print("No suitable metrics found for plotting.")
            return ""
        
        # Process validation data if provided
        valid_df = pd.DataFrame()
        if valid_results:
            _, valid_df = self.extract_metrics_from_results(valid_results, "training")
        
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
        
        # Get unique models
        models = train_df['model_type'].unique() if 'model_type' in train_df.columns else ['unknown']
        
        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            if metric not in train_df.columns:
                ax.text(0.5, 0.5, f'Metric\\n{metric}\\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title())
                continue
            
            # Plot for each model
            for model in models:
                model_color = self.get_model_color(model)
                model_train_data = train_df[train_df['model_type'] == model] if 'model_type' in train_df.columns else train_df
                
                if 'epoch' in model_train_data.columns and not model_train_data.empty:
                    # Calculate mean and confidence intervals across folds
                    if 'fold' in model_train_data.columns:
                        # Cross-validation: mean and std across folds
                        epoch_stats = model_train_data.groupby('epoch')[metric].agg(['mean', 'std', 'count']).reset_index()
                        
                        # Calculate 95% confidence intervals
                        from scipy import stats
                        confidence_interval = 0.95
                        alpha = 1 - confidence_interval
                        
                        ci_lower = []
                        ci_upper = []
                        for _, row in epoch_stats.iterrows():
                            if row['count'] > 1:
                                t_val = stats.t.ppf(1 - alpha/2, row['count'] - 1)
                                margin = t_val * (row['std'] / np.sqrt(row['count']))
                                ci_lower.append(row['mean'] - margin)
                                ci_upper.append(row['mean'] + margin)
                            else:
                                ci_lower.append(row['mean'])
                                ci_upper.append(row['mean'])
                        
                        # Plot mean line
                        ax.plot(epoch_stats['epoch'], epoch_stats['mean'], 
                               color=model_color, linewidth=2, label=f'{model} (train)',
                               linestyle='-')
                        
                        # Plot confidence interval
                        ax.fill_between(epoch_stats['epoch'], ci_lower, ci_upper,
                                       color=model_color, alpha=0.2)
                    else:
                        # Single run
                        ax.plot(model_train_data['epoch'], model_train_data[metric], 
                               color=model_color, linewidth=2, label=f'{model} (train)',
                               linestyle='-')
                
                # Plot validation data if available
                if not valid_df.empty and 'model_type' in valid_df.columns:
                    model_valid_data = valid_df[valid_df['model_type'] == model]
                    
                    if 'epoch' in model_valid_data.columns and not model_valid_data.empty and metric in model_valid_data.columns:
                        if 'fold' in model_valid_data.columns:
                            # Cross-validation validation data
                            epoch_stats = model_valid_data.groupby('epoch')[metric].agg(['mean', 'std', 'count']).reset_index()
                            
                            # Calculate confidence intervals
                            ci_lower = []
                            ci_upper = []
                            for _, row in epoch_stats.iterrows():
                                if row['count'] > 1:
                                    t_val = stats.t.ppf(1 - alpha/2, row['count'] - 1)
                                    margin = t_val * (row['std'] / np.sqrt(row['count']))
                                    ci_lower.append(row['mean'] - margin)
                                    ci_upper.append(row['mean'] + margin)
                                else:
                                    ci_lower.append(row['mean'])
                                    ci_upper.append(row['mean'])
                            
                            # Plot validation mean line (dashed)
                            ax.plot(epoch_stats['epoch'], epoch_stats['mean'], 
                                   color=model_color, linewidth=2, label=f'{model} (valid)',
                                   linestyle='--')
                            
                            # Plot confidence interval
                            ax.fill_between(epoch_stats['epoch'], ci_lower, ci_upper,
                                           color=model_color, alpha=0.1)
                        else:
                            # Single validation run
                            ax.plot(model_valid_data['epoch'], model_valid_data[metric], 
                                   color=model_color, linewidth=2, label=f'{model} (valid)',
                                   linestyle='--')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            # Figure-level legend will be added below
        
        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Create figure-level legend
        unique_models = sorted([m for m in models if m is not None])
        legend_elements = []
        for model in unique_models:
            legend_elements.extend([
                plt.Line2D([0], [0], color=self.get_model_color(model), linewidth=2, 
                          label=f'{model} (train)', linestyle='-'),
                plt.Line2D([0], [0], color=self.get_model_color(model), linewidth=2, 
                          label=f'{model} (valid)', linestyle='--')
            ])
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), 
                  ncol=len(unique_models), fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle('Learning Curves with 95% Confidence Intervals', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / "learn_curves.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Learning curves with confidence intervals plot saved: {save_path}")
        return str(save_path)
    
    def plot_category_transition_matrix(self, results: List[Dict[str, Any]], 
                                       result_type: str = "test",
                                       save_path: Optional[str] = None) -> str:
        """Plot category transition matrices showing student improvement/degradation patterns."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for transition matrix data or raw sequence data
        transition_data = {}
        
        for result in results:
            model_name = 'unknown'
            if 'config' in result:
                model_name = result['config'].get('model_type', 
                              result['config'].get('model', 'unknown'))
            
            # Look for transition matrices in various locations
            transitions = None
            if 'transition_matrix' in result:
                transitions = result['transition_matrix']
            elif 'evaluation_results' in result and 'transition_matrix' in result['evaluation_results']:
                transitions = result['evaluation_results']['transition_matrix']
            elif 'sequences' in result and 'responses' in result:
                # Calculate transitions from sequence data
                sequences = result['sequences']
                responses = result['responses']
                if sequences and responses:
                    transitions = self._calculate_transition_matrix(sequences, responses)
            
            if transitions is not None:
                transition_data[model_name] = np.array(transitions)
        
        if not transition_data:
            print("No transition matrix data found in results.")
            return ""
        
        n_models = len(transition_data)
        
        # Create subplots
        if n_models == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            axes = [ax]
        else:
            cols = min(3, n_models)
            rows = int(np.ceil(n_models / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
            if n_models > 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx, (model_name, matrix) in enumerate(transition_data.items()):
            ax = axes[idx]
            
            # Create custom colormap based on model color
            model_color = self.get_model_color(model_name)
            if isinstance(model_color, tuple):
                model_color = model_color[:3]  # RGB only
            
            # Create colormap from white to model color
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['white', model_color]
            custom_cmap = LinearSegmentedColormap.from_list(f'{model_name}_transition_cmap', colors, N=256)
            
            # Normalize matrix to probabilities (row-wise)
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            prob_matrix = matrix / row_sums
            
            # Plot heatmap
            im = ax.imshow(prob_matrix, interpolation='nearest', cmap=custom_cmap, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Transition Probability', rotation=270, labelpad=15)
            
            # Add text annotations
            thresh = prob_matrix.max() / 2.0
            for i in range(prob_matrix.shape[0]):
                for j in range(prob_matrix.shape[1]):
                    color = 'white' if prob_matrix[i, j] > thresh else 'black'
                    ax.text(j, i, f'{prob_matrix[i, j]:.2f}', 
                           ha='center', va='center', color=color, fontweight='bold', fontsize=9)
            
            # Customize plot
            ax.set_xlabel('Next Response Category', fontsize=11)
            ax.set_ylabel('Current Response Category', fontsize=11)
            ax.set_title(f'{model_name.title()} - Category Transitions', fontsize=12, 
                        fontweight='bold', color=model_color)
            
            # Set ticks
            n_categories = prob_matrix.shape[0]
            ax.set_xticks(range(n_categories))
            ax.set_yticks(range(n_categories))
            ax.set_xticklabels([f'Cat {i}' for i in range(n_categories)])
            ax.set_yticklabels([f'Cat {i}' for i in range(n_categories)])
            
            # Add border in model color
            for spine in ax.spines.values():
                spine.set_edgecolor(model_color)
                spine.set_linewidth(2)
            
            # Add diagonal highlighting for stability
            for i in range(n_categories):
                ax.add_patch(plt.Rectangle((i-0.4, i-0.4), 0.8, 0.8, 
                                         fill=False, edgecolor='red', linewidth=2, alpha=0.7))
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Category Transition Matrices - {result_type.title()} Results', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"cat_trans_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Category transition matrices plot saved: {save_path}")
        return str(save_path)
    
    def _calculate_transition_matrix(self, sequences: List, responses: List) -> np.ndarray:
        """Calculate transition matrix from sequence and response data."""
        # Determine number of categories
        all_responses = []
        for seq_responses in responses:
            all_responses.extend(seq_responses)
        n_categories = max(all_responses) + 1
        
        # Initialize transition matrix
        transitions = np.zeros((n_categories, n_categories))
        
        # Count transitions
        for seq_responses in responses:
            for i in range(len(seq_responses) - 1):
                current_cat = seq_responses[i]
                next_cat = seq_responses[i + 1]
                transitions[current_cat, next_cat] += 1
        
        return transitions
    
    def plot_roc_curves_per_category(self, results: List[Dict[str, Any]], 
                                    result_type: str = "test",
                                    save_path: Optional[str] = None) -> str:
        """Plot ROC curves for each category (one-vs-rest) with model comparison."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for probability predictions and actual labels
        roc_data = {}
        
        for result in results:
            model_name = 'unknown'
            if 'config' in result:
                model_name = result['config'].get('model_type', 
                              result['config'].get('model', 'unknown'))
            
            # Look for probability predictions and actual labels
            probabilities = None
            actual_labels = None
            
            if 'probabilities' in result and 'actual' in result:
                probabilities = np.array(result['probabilities'])
                actual_labels = np.array(result['actual'])
            elif 'evaluation_results' in result:
                eval_res = result['evaluation_results']
                if 'probabilities' in eval_res and 'actual' in eval_res:
                    probabilities = np.array(eval_res['probabilities'])
                    actual_labels = np.array(eval_res['actual'])
            
            if probabilities is not None and actual_labels is not None:
                roc_data[model_name] = (probabilities, actual_labels)
        
        if not roc_data:
            print("No probability predictions found for ROC analysis.")
            return ""
        
        # Determine number of categories
        all_labels = []
        for _, (_, labels) in roc_data.items():
            all_labels.extend(labels.tolist())
        n_categories = max(all_labels) + 1
        
        # Create subplots for each category
        cols = min(3, n_categories)
        rows = int(np.ceil(n_categories / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        
        if n_categories == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Import sklearn for ROC calculations
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
        except ImportError:
            print("sklearn not available for ROC analysis.")
            return ""
        
        # Plot ROC curves for each category
        for cat_idx in range(n_categories):
            ax = axes[cat_idx]
            
            for model_name, (probabilities, actual_labels) in roc_data.items():
                model_color = self.get_model_color(model_name)
                
                # Create binary labels for one-vs-rest
                binary_labels = (actual_labels == cat_idx).astype(int)
                
                # Get probabilities for this category
                if probabilities.ndim == 2 and probabilities.shape[1] > cat_idx:
                    cat_probs = probabilities[:, cat_idx]
                else:
                    # Fallback: use binary prediction probabilities
                    cat_probs = (np.argmax(probabilities, axis=1) == cat_idx).astype(float)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(binary_labels, cat_probs)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, color=model_color, linewidth=2, 
                       label=f'{model_name} (AUC = {roc_auc:.2f})')
            
            # Plot diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
            
            # Customize plot
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'ROC Curve - Category {cat_idx}', fontsize=11, fontweight='bold')
            # Figure-level legend will be added below
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide empty subplots
        for idx in range(n_categories, len(axes)):
            axes[idx].set_visible(False)
        
        # Create figure-level legend
        models = list(roc_data.keys())
        legend_elements = []
        for model in models:
            legend_elements.append(plt.Line2D([0], [0], color=self.get_model_color(model), 
                                            linewidth=2, label=model))
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=1, 
                                        linestyle='--', label='Random'))
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), 
                  ncol=len(models)+1, fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        plt.suptitle(f'ROC Curves per Category (One-vs-Rest) - {result_type.title()} Results', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"roc_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves per category plot saved: {save_path}")
        return str(save_path)
    
    def plot_calibration_curves(self, results: List[Dict[str, Any]], 
                               result_type: str = "test",
                               save_path: Optional[str] = None) -> str:
        """Plot calibration curves showing predicted probabilities vs actual frequencies."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for probability predictions and actual labels
        calibration_data = {}
        
        for result in results:
            model_name = 'unknown'
            if 'config' in result:
                model_name = result['config'].get('model_type', 
                              result['config'].get('model', 'unknown'))
            
            # Look for probability predictions and actual labels
            probabilities = None
            actual_labels = None
            
            if 'probabilities' in result and 'actual' in result:
                probabilities = np.array(result['probabilities'])
                actual_labels = np.array(result['actual'])
            elif 'evaluation_results' in result:
                eval_res = result['evaluation_results']
                if 'probabilities' in eval_res and 'actual' in eval_res:
                    probabilities = np.array(eval_res['probabilities'])
                    actual_labels = np.array(eval_res['actual'])
            
            if probabilities is not None and actual_labels is not None:
                calibration_data[model_name] = (probabilities, actual_labels)
        
        if not calibration_data:
            print("No probability predictions found for calibration analysis.")
            return ""
        
        # Import sklearn for calibration calculations
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            print("sklearn not available for calibration analysis.")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
        
        # Storage for calibration errors
        calibration_errors = []
        
        # Plot calibration curves for each model
        for model_name, (probabilities, actual_labels) in calibration_data.items():
            model_color = self.get_model_color(model_name)
            
            # Get predicted probabilities (for multiclass, use max probability)
            if probabilities.ndim == 2:
                # Multiclass case: use predicted probabilities and convert to binary
                predicted_probs = np.max(probabilities, axis=1)
                predicted_classes = np.argmax(probabilities, axis=1)
                binary_labels = (predicted_classes == actual_labels).astype(int)
            else:
                # Binary case
                predicted_probs = probabilities
                binary_labels = actual_labels
            
            # Calculate calibration curve
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    binary_labels, predicted_probs, n_bins=10, strategy='uniform'
                )
                
                # Plot calibration curve
                ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
                       color=model_color, linewidth=2, markersize=6,
                       label=f'{model_name}')
                
                # Calculate calibration error (Brier score approximation)
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                
                # Store calibration error for later annotation
                calibration_errors.append((model_name, calibration_error, model_color))
                
            except Exception as e:
                print(f"Could not calculate calibration for {model_name}: {e}")
                continue
        
        # Customize plot
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Reliability Diagram (Calibration Plot)', fontsize=14, fontweight='bold')
        # Create figure-level legend
        models = list(calibration_data.keys())
        legend_elements = []
        for model in models:
            legend_elements.append(plt.Line2D([0], [0], color=self.get_model_color(model), 
                                            linewidth=2, marker='o', markersize=6, label=model))
        legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=1, 
                                        linestyle='--', label='Perfect Calibration'))
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), 
                  ncol=len(models)+1, fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add calibration error annotations without overlap
        if calibration_errors:
            y_positions = np.linspace(0.95, 0.80, len(calibration_errors))
            for i, (model_name, cal_error, model_color) in enumerate(calibration_errors):
                ax.text(0.02, y_positions[i], f'{model_name} Cal Error: {cal_error:.2f}',
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=model_color, alpha=0.3),
                       verticalalignment='top')
        
        # Add explanation text
        explanation = ("Perfect calibration: predicted probabilities match actual frequencies.\n"
                      "Above diagonal: overconfident, Below diagonal: underconfident.")
        ax.text(0.02, 0.15, explanation, transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
               verticalalignment='bottom')
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"calib_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Calibration curves plot saved: {save_path}")
        return str(save_path)
    
    def plot_attention_weights(self, results: List[Dict[str, Any]], 
                              result_type: str = "test",
                              save_path: Optional[str] = None) -> str:
        """Plot attention weights or feature importance if available."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for attention/importance data
        attention_data = {}
        
        for result in results:
            model_name = 'unknown'
            if 'config' in result:
                model_name = result['config'].get('model_type', 
                              result['config'].get('model', 'unknown'))
            
            # Look for attention weights in various locations
            weights = None
            if 'attention_weights' in result:
                weights = result['attention_weights']
            elif 'feature_importance' in result:
                weights = result['feature_importance'] 
            elif 'evaluation_results' in result:
                eval_res = result['evaluation_results']
                if 'attention_weights' in eval_res:
                    weights = eval_res['attention_weights']
                elif 'feature_importance' in eval_res:
                    weights = eval_res['feature_importance']
            
            if weights is not None:
                attention_data[model_name] = np.array(weights)
        
        if not attention_data:
            print("No attention weights or feature importance found.")
            return ""
        
        # Determine plot layout
        n_models = len(attention_data)
        if n_models == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        else:
            cols = min(2, n_models)
            rows = int(np.ceil(n_models / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
            if n_models > 1:
                axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        for idx, (model_name, weights) in enumerate(attention_data.items()):
            ax = axes[idx]
            model_color = self.get_model_color(model_name)
            
            # Handle different weight shapes
            if weights.ndim == 1:
                # 1D feature importance
                features = range(len(weights))
                ax.bar(features, weights, color=model_color, alpha=0.7)
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Importance/Weight')
            elif weights.ndim == 2:
                # 2D attention matrix
                im = ax.imshow(weights, cmap='Blues', aspect='auto')
                plt.colorbar(im, ax=ax)
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
            
            ax.set_title(f'{model_name.title()} - Attention/Importance', 
                        fontweight='bold', color=model_color)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Feature Importance/Attention Weights - {result_type.title()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"attention_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention weights plot saved: {save_path}")
        return str(save_path)
    
    def plot_cv_score_comparison(self, valid_results: List[Dict[str, Any]],
                                save_path: Optional[str] = None) -> str:
        """Plot cross-validation score comparison from cv_summary files."""
        # Filter only CV summary results
        cv_summaries = []
        for result in valid_results:
            if 'cv_summary' in result and 'config' in result:
                cv_summaries.append(result)
        
        if not cv_summaries:
            print("No CV summary data found.")
            return ""
        
        # Metrics to compare
        metrics = [
            'categorical_accuracy',
            'ordinal_accuracy',
            'quadratic_weighted_kappa',
            'mean_absolute_error'
        ]
        
        # Extract data
        models = []
        data = {metric: {'means': [], 'stds': []} for metric in metrics}
        
        for result in cv_summaries:
            model_name = result['config'].get('model', 'unknown')
            models.append(model_name)
            
            cv_summary = result['cv_summary']
            for metric in metrics:
                if metric in cv_summary:
                    data[metric]['means'].append(cv_summary[metric]['mean'])
                    data[metric]['stds'].append(cv_summary[metric]['std'])
                else:
                    data[metric]['means'].append(0)
                    data[metric]['stds'].append(0)
        
        if not models:
            print("No models found in CV summaries.")
            return ""
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            means = np.array(data[metric]['means'])
            stds = np.array(data[metric]['stds'])
            
            # Handle NaN/invalid std values
            stds = np.where(np.isnan(stds) | np.isinf(stds) | (stds <= 0), 0, stds)
            
            x_pos = np.arange(len(models))
            
            # Create bars
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                          color=[self.get_model_color(m) for m in models],
                          alpha=0.8, edgecolor='black', linewidth=1)
            
            # Highlight best performer
            higher_is_better = self.is_higher_better(metric)
            best_idx = np.argmax(means) if higher_is_better else np.argmin(means)
            
            # Add value annotations
            for i, (mean, std) in enumerate(zip(means, stds)):
                if std > 0:
                    text = f'{mean:.2f}\n±{std:.2f}'
                else:
                    text = f'{mean:.2f}'
                
                y_pos = mean + std + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                
                if i == best_idx:
                    ax.text(i, y_pos, text + ' ★', ha='center', va='bottom',
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                                   alpha=0.5, edgecolor='green'))
                else:
                    ax.text(i, y_pos, text, ha='center', va='bottom', fontsize=8)
            
            # Customize subplot
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} {"↑" if higher_is_better else "↓"}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        # Overall title
        fig.suptitle('Cross-Validation Score Comparison (Mean ± Std)', fontsize=16, weight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / "cv_scores.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"CV score comparison plot saved: {save_path}")
        return str(save_path)
    
    def plot_time_series_performance(self, results: List[Dict[str, Any]], 
                                   result_type: str = "test",
                                   save_path: Optional[str] = None) -> str:
        """Plot performance metrics over student interaction sequences."""
        if not results:
            print(f"No {result_type} data found.")
            return ""
        
        # Look for sequence-level performance data
        sequence_data = {}
        
        for result in results:
            model_name = 'unknown'
            if 'config' in result:
                model_name = result['config'].get('model_type', 
                              result['config'].get('model', 'unknown'))
            
            # Look for sequence performance data
            seq_performance = None
            if 'sequence_performance' in result:
                seq_performance = result['sequence_performance']
            elif 'evaluation_results' in result and 'sequence_performance' in result['evaluation_results']:
                seq_performance = result['evaluation_results']['sequence_performance']
            elif 'sequences' in result and 'accuracy_by_position' in result:
                seq_performance = result['accuracy_by_position']
            
            if seq_performance is not None:
                sequence_data[model_name] = seq_performance
        
        if not sequence_data:
            print("No time-series performance data found.")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot performance over sequence positions
        for model_name, performance in sequence_data.items():
            model_color = self.get_model_color(model_name)
            
            if isinstance(performance, dict):
                # Performance by position
                positions = sorted(performance.keys())
                accuracies = [performance[pos] for pos in positions]
            elif isinstance(performance, (list, np.ndarray)):
                # Direct performance array
                positions = range(len(performance))
                accuracies = performance
            else:
                continue
            
            # Plot line with confidence intervals if available
            ax.plot(positions, accuracies, 'o-', color=model_color, 
                   linewidth=2, markersize=4, label=f'{model_name}')
        
        # Customize plot
        ax.set_xlabel('Sequence Position', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Performance Over Student Interaction Sequences', 
                    fontsize=14, fontweight='bold')
        
        # Create figure-level legend
        models = list(sequence_data.keys())
        legend_elements = [plt.Line2D([0], [0], color=self.get_model_color(model), 
                                    linewidth=2, marker='o', markersize=4, label=model) 
                          for model in models]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.92), 
                  ncol=len(models), fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"time_series_{result_type}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Time-series performance plot saved: {save_path}")
        return str(save_path)


def plot_all_results(results_dir: str = "results", dataset: Optional[str] = None):
    """Convenience function to plot all available results.
    
    Generates both normal and zoomed plots automatically.
    """
    plotter = AdaptivePlotter(results_dir, dataset=dataset)
    
    # Load all results
    train_results = plotter.load_results_from_dir("train")
    test_results = plotter.load_results_from_dir("test")
    valid_results = plotter.load_results_from_dir("valid")
    
    # Also load validation results for CV summaries
    validation_results = plotter.load_results_from_dir("validation")
    
    generated_plots = []
    
    # Plot training metrics if available
    if train_results:
        # Normal view
        plot_path = plotter.plot_training_metrics(train_results)
        if plot_path:
            generated_plots.append(plot_path)
        # Zoomed view
        plot_path = plotter.plot_training_metrics(train_results, zoom_in=True)
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot test metrics if available
    if test_results:
        # Plot with specific metrics for better visualization
        metrics_to_plot = [
            'categorical_accuracy',
            'ordinal_accuracy', 
            'quadratic_weighted_kappa',
            'mean_absolute_error',
            'kendall_tau',
            'spearman_correlation',
            'cohen_kappa'
        ]
        # Normal view
        plot_path = plotter.plot_evaluation_metrics(test_results, "test", metrics_to_plot)
        if plot_path:
            generated_plots.append(plot_path)
        # Zoomed view
        plot_path = plotter.plot_evaluation_metrics(test_results, "test", metrics_to_plot, zoom_in=True)
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot validation metrics if available
    if valid_results:
        # Normal view
        plot_path = plotter.plot_evaluation_metrics(valid_results, "valid")
        if plot_path:
            generated_plots.append(plot_path)
        # Zoomed view
        plot_path = plotter.plot_evaluation_metrics(valid_results, "valid", zoom_in=True)
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot comparison if both training and test results available
    if (train_results or validation_results) and test_results:
        # Normal view
        plot_path = plotter.plot_metric_comparison(train_results, test_results, validation_results)
        if plot_path:
            generated_plots.append(plot_path)
        # Zoomed view
        plot_path = plotter.plot_metric_comparison(train_results, test_results, validation_results, zoom_in=True)
        if plot_path:
            generated_plots.append(plot_path)
    
    # Plot categorical breakdown if test results available
    if test_results:
        plot_path = plotter.plot_categorical_breakdown(test_results, "test")
        if plot_path:
            generated_plots.append(plot_path)
    
    # Advanced visualizations (try each, skip if data not available)
    
    # Confusion matrices
    if test_results:
        try:
            plot_path = plotter.plot_confusion_matrices(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping confusion matrices: {e}")
    
    # Ordinal distance distribution
    if test_results:
        try:
            plot_path = plotter.plot_ordinal_distance_distribution(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping ordinal distance distribution: {e}")
    
    # Learning curves with confidence intervals - DISABLED (redundant with training_metrics)
    # if train_results:
    #     try:
    #         plot_path = plotter.plot_learning_curves_with_confidence(train_results, valid_results)
    #         if plot_path:
    #             generated_plots.append(plot_path)
    #     except Exception as e:
    #         print(f"Skipping learning curves with confidence: {e}")
    
    # Category transition matrices
    if test_results:
        try:
            plot_path = plotter.plot_category_transition_matrix(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping category transition matrices: {e}")
    
    # ROC curves per category
    if test_results:
        try:
            plot_path = plotter.plot_roc_curves_per_category(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping ROC curves per category: {e}")
    
    # Calibration curves
    if test_results:
        try:
            plot_path = plotter.plot_calibration_curves(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping calibration curves: {e}")
    
    # Cross-validation score comparison
    if validation_results:
        try:
            plot_path = plotter.plot_cv_score_comparison(validation_results)
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping CV score comparison: {e}")
    
    # Attention weights (if available)
    if test_results:
        try:
            plot_path = plotter.plot_attention_weights(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping attention weights: {e}")
    
    # Time-series performance (if available)
    if test_results:
        try:
            plot_path = plotter.plot_time_series_performance(test_results, "test")
            if plot_path:
                generated_plots.append(plot_path)
        except Exception as e:
            print(f"Skipping time-series performance: {e}")
    
    print(f"\nGenerated {len(generated_plots)} plots:")
    for plot_path in generated_plots:
        print(f"  - {plot_path}")
    
    return generated_plots


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots for Deep-GPCM results')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name for organizing plots (e.g., synthetic_OC)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Results directory path')
    
    args = parser.parse_args()
    
    # Test plotting system
    plot_all_results(results_dir=args.results_dir, dataset=args.dataset)
