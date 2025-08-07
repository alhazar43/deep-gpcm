#!/usr/bin/env python3
"""
Optimized Plotting System for Deep-GPCM Pipeline
Maintains all original plotting functionality with factory integration and configuration system.
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
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import original plotting functionality
from utils.plot_metrics import AdaptivePlotter as OriginalAdaptivePlotter

# Import optimized configuration system with explicit local path
from config.evaluation import PlottingConfig
from config.pipeline import PipelineConfig
from config.parser import SmartArgumentParser

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for better font rendering
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.weight'] = 'normal'

# Force matplotlib to use a font that supports bold
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Arial' in available_fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Arial')
elif 'Liberation Sans' in available_fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Liberation Sans')


class OptimizedPlotter(OriginalAdaptivePlotter):
    """
    Optimized plotter that extends the original with configuration system integration.
    Maintains all original plotting functionality while adding intelligent defaults.
    """
    
    def __init__(self, config: PlottingConfig):
        """Initialize with optimized configuration."""
        self.config = config
        
        # Initialize parent with configuration values
        super().__init__(
            results_dir=str(config.results_dir),
            figsize_base=config.figsize_base,
            dataset=config.dataset
        )
        
        # Override settings with config values
        self.enable_detailed_plots = config.enable_detailed_plots
        self.enable_comparison_plots = config.enable_comparison_plots
        self.enable_categorical_breakdown = config.enable_categorical_breakdown
        self.enable_confusion_matrices = config.enable_confusion_matrices
        self.enable_learning_curves = config.enable_learning_curves
        self.enable_roc_curves = config.enable_roc_curves
        self.enable_calibration_plots = config.enable_calibration_plots
        self.custom_metrics = config.custom_metrics
        self.plot_quality = config.plot_quality
        
        # Set DPI based on quality
        quality_dpi = {'low': 100, 'medium': 150, 'high': 300}
        self.dpi = quality_dpi.get(config.plot_quality, 150)
        
        # Configure save options
        self.save_formats = config.save_formats
        
    def plot_all_results_optimized(self) -> List[Path]:
        """
        Optimized version of plot_all_results with configuration-driven execution.
        Maintains all original functionality while respecting configuration settings.
        """
        print("🎨 Starting optimized plotting system...")
        print(f"📊 Dataset: {self.dataset}")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"🎯 Quality: {self.plot_quality} (DPI: {self.dpi})")
        print(f"💾 Formats: {', '.join(self.save_formats)}")
        
        # Load all results (using original functionality)
        train_results = self.load_results_from_dir("train")
        test_results = self.load_results_from_dir("test")
        valid_results = self.load_results_from_dir("valid")
        validation_results = self.load_results_from_dir("validation")
        
        print(f"📈 Found {len(train_results)} training results")
        print(f"🧪 Found {len(test_results)} test results")
        print(f"✅ Found {len(valid_results)} validation results")
        
        generated_plots = []
        
        try:
            # Core plots (always generated)
            generated_plots.extend(self._generate_core_plots(
                train_results, test_results, valid_results, validation_results
            ))
            
            # Optional detailed plots based on configuration
            if self.enable_detailed_plots:
                generated_plots.extend(self._generate_detailed_plots(test_results))
            
            # Comparison plots
            if self.enable_comparison_plots and (train_results or validation_results) and test_results:
                generated_plots.extend(self._generate_comparison_plots(
                    train_results, test_results, validation_results
                ))
            
            # Categorical analysis plots
            if self.enable_categorical_breakdown and test_results:
                generated_plots.extend(self._generate_categorical_plots(test_results))
            
            # Advanced analysis plots
            if test_results:
                generated_plots.extend(self._generate_advanced_plots(test_results))
            
        except Exception as e:
            print(f"⚠️  Error during plotting: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print(f"\n✅ Generated {len(generated_plots)} plots:")
        for plot_path in generated_plots:
            print(f"  📊 {plot_path}")
        
        return generated_plots
    
    def _generate_core_plots(self, train_results, test_results, valid_results, validation_results) -> List[Path]:
        """Generate core essential plots."""
        plots = []
        
        # Training metrics
        if train_results:
            print("📈 Generating training metrics plot...")
            plot_path = self.plot_training_metrics(train_results)
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        
        # Test metrics
        if test_results:
            print("🧪 Generating test metrics plot...")
            metrics_to_plot = self.custom_metrics or [
                'categorical_accuracy',
                'ordinal_accuracy', 
                'quadratic_weighted_kappa',
                'mean_absolute_error',
                'kendall_tau',
                'spearman_correlation',
                'cohen_kappa'
            ]
            plot_path = self.plot_evaluation_metrics(test_results, "test", metrics_to_plot)
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        
        # Validation metrics
        if valid_results:
            print("✅ Generating validation metrics plot...")
            plot_path = self.plot_evaluation_metrics(valid_results, "valid")
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        
        return plots
    
    def _generate_detailed_plots(self, test_results) -> List[Path]:
        """Generate detailed analysis plots."""
        plots = []
        
        if not test_results:
            return plots
        
        print("🔍 Generating detailed analysis plots...")
        
        try:
            # Ordinal distance distribution
            plot_path = self.plot_ordinal_distance_distribution(test_results)
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        except Exception as e:
            print(f"⚠️  Skipping ordinal distance plot: {e}")
        
        try:
            # Category transition matrix
            plot_path = self.plot_category_transition_matrix(test_results)
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        except Exception as e:
            print(f"⚠️  Skipping transition matrix: {e}")
        
        return plots
    
    def _generate_comparison_plots(self, train_results, test_results, validation_results) -> List[Path]:
        """Generate comparison plots."""
        plots = []
        
        print("📊 Generating comparison plots...")
        
        try:
            # Metric comparison
            plot_path = self.plot_metric_comparison(train_results, test_results, validation_results)
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        except Exception as e:
            print(f"⚠️  Skipping metric comparison: {e}")
        
        if self.enable_learning_curves and train_results:
            try:
                # Learning curves with confidence
                plot_path = self.plot_learning_curves_with_confidence(train_results, test_results)
                if plot_path:
                    plots.extend(self._save_multiple_formats(plot_path))
            except Exception as e:
                print(f"⚠️  Skipping learning curves: {e}")
        
        return plots
    
    def _generate_categorical_plots(self, test_results) -> List[Path]:
        """Generate categorical analysis plots."""
        plots = []
        
        print("📊 Generating categorical analysis plots...")
        
        try:
            # Categorical breakdown
            plot_path = self.plot_categorical_breakdown(test_results)
            if plot_path:
                plots.extend(self._save_multiple_formats(plot_path))
        except Exception as e:
            print(f"⚠️  Skipping categorical breakdown: {e}")
        
        if self.enable_confusion_matrices:
            try:
                # Confusion matrices
                plot_path = self.plot_confusion_matrices(test_results)
                if plot_path:
                    plots.extend(self._save_multiple_formats(plot_path))
            except Exception as e:
                print(f"⚠️  Skipping confusion matrices: {e}")
        
        return plots
    
    def _generate_advanced_plots(self, test_results) -> List[Path]:
        """Generate advanced analysis plots."""
        plots = []
        
        print("🔬 Generating advanced analysis plots...")
        
        if self.enable_roc_curves:
            try:
                # ROC curves per category
                plot_path = self.plot_roc_curves_per_category(test_results)
                if plot_path:
                    plots.extend(self._save_multiple_formats(plot_path))
            except Exception as e:
                print(f"⚠️  Skipping ROC curves: {e}")
        
        if self.enable_calibration_plots:
            try:
                # Calibration curves
                plot_path = self.plot_calibration_curves(test_results)
                if plot_path:
                    plots.extend(self._save_multiple_formats(plot_path))
            except Exception as e:
                print(f"⚠️  Skipping calibration curves: {e}")
        
        return plots
    
    def _save_multiple_formats(self, original_path: Path) -> List[Path]:
        """Save plot in multiple formats based on configuration."""
        saved_paths = []
        
        if isinstance(original_path, str):
            original_path = Path(original_path)
        
        base_path = original_path.with_suffix('')
        
        for fmt in self.save_formats:
            if fmt == 'png':
                new_path = base_path.with_suffix('.png')
                if new_path != original_path:
                    # Save with optimized DPI
                    plt.savefig(new_path, dpi=self.dpi, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                saved_paths.append(new_path)
            elif fmt == 'pdf':
                new_path = base_path.with_suffix('.pdf')
                plt.savefig(new_path, format='pdf', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                saved_paths.append(new_path)
            elif fmt == 'svg':
                new_path = base_path.with_suffix('.svg')
                plt.savefig(new_path, format='svg', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                saved_paths.append(new_path)
        
        return saved_paths


class PlottingOrchestrator:
    """Orchestrates plotting workflows with configuration management."""
    
    def __init__(self, config: Optional[PlottingConfig] = None):
        self.config = config or PlottingConfig()
        self.plotter = OptimizedPlotter(self.config)
    
    def plot_model_results(self, model_name: str, dataset: str) -> List[Path]:
        """Plot results for a specific model."""
        print(f"🎨 Plotting results for {model_name} on {dataset}")
        
        # Update config for specific model/dataset
        model_config = PlottingConfig(
            dataset=dataset,
            results_dir=self.config.results_dir,
            enable_detailed_plots=True,
            enable_comparison_plots=True
        )
        
        plotter = OptimizedPlotter(model_config)
        return plotter.plot_all_results_optimized()
    
    def plot_batch_comparison(self, datasets: List[str], models: List[str]) -> List[Path]:
        """Plot comparison across multiple datasets and models."""
        print(f"📊 Creating batch comparison plots")
        print(f"   Datasets: {', '.join(datasets)}")
        print(f"   Models: {', '.join(models)}")
        
        all_plots = []
        
        for dataset in datasets:
            dataset_config = PlottingConfig(
                dataset=dataset,
                results_dir=self.config.results_dir,
                enable_comparison_plots=True,
                enable_categorical_breakdown=True,
                plot_quality='high'
            )
            
            plotter = OptimizedPlotter(dataset_config)
            plots = plotter.plot_all_results_optimized()
            all_plots.extend(plots)
        
        return all_plots
    
    def generate_summary_report(self, datasets: List[str]) -> Path:
        """Generate a summary report with all plots."""
        print(f"📋 Generating summary report for {len(datasets)} datasets")
        
        # This would create an HTML or PDF report with all plots
        # For now, just return a placeholder
        report_path = Path("results/plots/summary_report.html")
        
        # Create a simple HTML report
        html_content = self._create_html_report(datasets)
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Report saved to: {report_path}")
        return report_path
    
    def _create_html_report(self, datasets: List[str]) -> str:
        """Create HTML summary report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deep-GPCM Results Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2E86AB; }
                h2 { color: #43AA8B; }
                .dataset { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .plot { margin: 10px; }
                img { max-width: 800px; height: auto; }
            </style>
        </head>
        <body>
            <h1>🎯 Deep-GPCM Results Summary</h1>
            <p>Generated with optimized plotting pipeline</p>
        """
        
        for dataset in datasets:
            html += f"""
            <div class="dataset">
                <h2>📊 Dataset: {dataset}</h2>
                <p>Results and visualizations for {dataset}</p>
                <!-- Plot images would be embedded here -->
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


def plot_all_results_optimized(results_dir: str = "results", dataset: Optional[str] = None,
                              config: Optional[PlottingConfig] = None) -> List[Path]:
    """
    Optimized convenience function to plot all available results.
    Maintains backward compatibility while adding configuration support.
    """
    
    if config is None:
        config = PlottingConfig(
            dataset=dataset,
            results_dir=Path(results_dir),
            enable_detailed_plots=True,
            enable_comparison_plots=True,
            enable_categorical_breakdown=True,
            plot_quality='medium'
        )
    
    plotter = OptimizedPlotter(config)
    return plotter.plot_all_results_optimized()


def main():
    """Main entry point with optimized argument parsing."""
    
    try:
        # Try to parse as plotting configuration
        parser = SmartArgumentParser.create_plotting_parser()
        config = parser.parse()
        
        # If dataset not provided in config, try command line
        if not config.dataset:
            import argparse
            cli_parser = argparse.ArgumentParser(description='Generate optimized plots for Deep-GPCM results')
            cli_parser.add_argument('--dataset', type=str, help='Dataset name')
            cli_parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
            cli_parser.add_argument('--models', nargs='+', help='Models to include in plots')
            cli_parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='medium',
                                  help='Plot quality')
            cli_parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], 
                                  default=['png'], help='Output formats')
            
            cli_args = cli_parser.parse_args()
            
            # Update config with CLI arguments
            config.dataset = cli_args.dataset
            config.results_dir = Path(cli_args.results_dir)
            config.plot_quality = cli_args.quality
            config.save_formats = cli_args.formats
        
        # Run optimized plotting
        plots = plot_all_results_optimized(config=config)
        
        print(f"\n🎉 Successfully generated {len(plots)} plots!")
        
        # Optionally create summary report
        if config.dataset:
            orchestrator = PlottingOrchestrator(config)
            report = orchestrator.generate_summary_report([config.dataset])
            print(f"📋 Summary report: {report}")
        
    except Exception as e:
        print(f"❌ Error in optimized plotting: {e}")
        
        # Fallback to original plotting system
        print("🔄 Falling back to original plotting system...")
        
        import argparse
        parser = argparse.ArgumentParser(description='Generate plots for Deep-GPCM results')
        parser.add_argument('--dataset', type=str, default=None,
                           help='Dataset name for organizing plots (e.g., synthetic_OC)')
        parser.add_argument('--results_dir', type=str, default='results',
                           help='Results directory path')
        
        args = parser.parse_args()
        
        # Use original plotting system as fallback
        from utils.plot_metrics import plot_all_results
        plot_all_results(results_dir=args.results_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()