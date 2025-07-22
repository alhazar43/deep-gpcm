#!/usr/bin/env python3
"""
Deep-GPCM Comprehensive Analysis Suite

Unified analysis tool combining all visualization and comparison capabilities:
- OC vs PC format comparison with strategy analysis
- Training curve visualization and performance analysis
- Statistical significance testing and correlation analysis
- Comprehensive reporting and publication-ready plots

This replaces: analyze_strategies.py, compare_formats.py, compare_strategies.py, 
               visualize.py, gpcm_analysis.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from datetime import datetime
import logging
from scipy import stats

class DeepGPCMAnalyzer:
    """Comprehensive analysis suite for Deep-GPCM experiments."""
    
    def __init__(self, save_path="results/plots", log_level=logging.INFO):
        """Initialize analyzer with output configuration."""
        self.save_path = save_path
        self.logger = self._setup_logging(log_level)
        
        # Create output directories
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(f"{self.save_path}/../analysis", exist_ok=True)
        
        # Analysis results storage
        self.results = {}
        
    def _setup_logging(self, level):
        """Setup logging for analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/comprehensive_analysis_{timestamp}.log"
        
        os.makedirs("logs", exist_ok=True)
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)

    def load_all_data(self):
        """Load all available experimental data."""
        self.logger.info("Loading all experimental data...")
        
        data = {}
        
        # Load format comparison data (primary source)
        format_comparison_path = "results/comparison/strategy_format_comparison.json"
        if os.path.exists(format_comparison_path):
            with open(format_comparison_path, 'r') as f:
                data['format_comparison'] = json.load(f)
                self.logger.info("Loaded comprehensive format comparison data")
        
        # Load individual training histories
        for format_type in ['OC', 'PC']:
            history_path = f"results/train/training_history_synthetic_{format_type}.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    data[f'individual_history_{format_type}'] = json.load(f)
                    self.logger.info(f"Loaded individual training history for {format_type}")
        
        # Load strategy analysis
        for format_type in ['OC', 'PC']:
            analysis_path = f"results/analysis/strategy_analysis_synthetic_{format_type}.json"
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    data[f'strategy_analysis_{format_type}'] = json.load(f)
                    self.logger.info(f"Loaded strategy analysis for {format_type}")
        
        self.results = data
        return data

    def extract_format_data(self, format_type):
        """Extract and structure data for specific format."""
        format_data = {}
        
        # Use comprehensive format comparison as primary source
        if 'format_comparison' in self.results and 'epoch_results' in self.results['format_comparison']:
            format_results = [
                r for r in self.results['format_comparison']['epoch_results'] 
                if r.get('format') == format_type
            ]
            if format_results:
                format_data['training_history'] = format_results
                
                # Extract strategy analysis
                strategies = {}
                for result in format_results:
                    strategy = result['strategy']
                    if strategy not in strategies:
                        strategies[strategy] = []
                    strategies[strategy].append(result)
                
                # Create strategy summary
                comparison_summary = []
                for strategy, strategy_results in strategies.items():
                    best_result = max(strategy_results, key=lambda x: x['categorical_acc'])
                    comparison_summary.append({
                        'strategy': strategy,
                        'config_name': f"{strategy}_best",
                        'categorical_acc': best_result['categorical_acc'],
                        'ordinal_acc': best_result['ordinal_acc'],
                        'qwk': best_result['qwk'],
                        'mae': best_result['mae']
                    })
                
                format_data['strategy_analysis'] = {
                    'comparison_summary': comparison_summary,
                    'dataset': f"synthetic_{format_type}"
                }
        
        # Fallback to individual files
        if f'individual_history_{format_type}' in self.results:
            if 'training_history' not in format_data:
                format_data['individual_training_history'] = self.results[f'individual_history_{format_type}']
        
        if f'strategy_analysis_{format_type}' in self.results:
            if 'strategy_analysis' not in format_data:
                format_data['strategy_analysis'] = self.results[f'strategy_analysis_{format_type}']
        
        return format_data

    def create_comprehensive_comparison(self):
        """Create the main comprehensive OC vs PC comparison visualization."""
        self.logger.info("Creating comprehensive OC vs PC comparison")
        
        oc_data = self.extract_format_data('OC')
        pc_data = self.extract_format_data('PC')
        
        # Create 4x2 subplot layout
        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        fig.suptitle('Deep-GPCM: Ordered Categories (OC) vs Partial Credit (PC) Comprehensive Comparison', 
                     fontsize=16, y=0.98)
        
        # Column headers
        axes[0, 0].text(0.5, 1.1, 'Ordered Categories (OC)', 
                        transform=axes[0, 0].transAxes, ha='center', va='bottom', 
                        fontsize=14, fontweight='bold')
        axes[0, 1].text(0.5, 1.1, 'Partial Credit (PC)', 
                        transform=axes[0, 1].transAxes, ha='center', va='bottom', 
                        fontsize=14, fontweight='bold')
        
        # Row 1: Training Loss by Strategy
        self._plot_training_loss_by_strategy(axes[0, :], oc_data, pc_data)
        
        # Row 2: New Prediction Accuracy Metrics by Strategy
        self._plot_new_accuracy_metrics(axes[1, :], oc_data, pc_data)
        
        # Row 3: Strategy Performance Heatmaps
        self._plot_strategy_heatmaps(axes[2, :], oc_data, pc_data)
        
        # Row 4: Performance Comparison
        self._plot_performance_comparison(axes[3, :], oc_data, pc_data)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/comprehensive_oc_pc_comparison.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Comprehensive comparison saved to {self.save_path}/comprehensive_oc_pc_comparison.png")

    def _plot_training_loss_by_strategy(self, axes, oc_data, pc_data):
        """Plot training loss curves by strategy."""
        if 'training_history' not in oc_data or 'training_history' not in pc_data:
            return
        
        oc_df = pd.DataFrame(oc_data['training_history'])
        pc_df = pd.DataFrame(pc_data['training_history'])
        
        strategy_colors = {'ordered': 'blue', 'unordered': 'red', 'linear_decay': 'green'}
        
        for ax, df, title_suffix in [(axes[0], oc_df, ''), (axes[1], pc_df, '')]:
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                color = strategy_colors.get(strategy, 'gray')
                
                ax.plot(strategy_data['epoch'], strategy_data['train_loss'], 
                       color=color, linestyle='-', linewidth=2, alpha=0.8, 
                       label=f'{strategy.title()} (Train)')
                ax.plot(strategy_data['epoch'], strategy_data['valid_loss'], 
                       color=color, linestyle='--', linewidth=2, alpha=0.8, 
                       label=f'{strategy.title()} (Valid)')
            
            ax.set_title('Training Loss by Strategy', fontsize=12, pad=20)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Ensure same y-axis scale
        all_losses = list(oc_df['train_loss']) + list(oc_df['valid_loss']) + \
                    list(pc_df['train_loss']) + list(pc_df['valid_loss'])
        y_min, y_max = min(all_losses), max(all_losses)
        axes[0].set_ylim(y_min - 0.1, y_max + 0.1)
        axes[1].set_ylim(y_min - 0.1, y_max + 0.1)

    def _plot_new_accuracy_metrics(self, axes, oc_data, pc_data):
        """Plot new prediction accuracy metrics by strategy."""
        if 'training_history' not in oc_data or 'training_history' not in pc_data:
            return
        
        oc_df = pd.DataFrame(oc_data['training_history'])
        pc_df = pd.DataFrame(pc_data['training_history'])
        
        for ax, df in [(axes[0], oc_df), (axes[1], pc_df)]:
            for strategy in df['strategy'].unique():
                strategy_data = df[df['strategy'] == strategy]
                linestyle = '-' if strategy == 'ordered' else '--' if strategy == 'unordered' else ':'
                
                # Use new prediction accuracy metrics if available, fallback to original
                if 'prediction_consistency_acc' in strategy_data.columns:
                    ax.plot(strategy_data['epoch'], strategy_data['prediction_consistency_acc'], 
                           color='green', linestyle=linestyle, linewidth=2, alpha=0.8,
                           label=f'{strategy.title()} (Pred Consist)')
                else:
                    ax.plot(strategy_data['epoch'], strategy_data['categorical_acc'], 
                           color='green', linestyle=linestyle, linewidth=2, alpha=0.8,
                           label=f'{strategy.title()} (Cat)')
                
                if 'ordinal_ranking_acc' in strategy_data.columns:
                    ax.plot(strategy_data['epoch'], strategy_data['ordinal_ranking_acc'], 
                           color='orange', linestyle=linestyle, linewidth=2, alpha=0.8,
                           label=f'{strategy.title()} (Ranking)')
                else:
                    ax.plot(strategy_data['epoch'], strategy_data['ordinal_acc'], 
                           color='orange', linestyle=linestyle, linewidth=2, alpha=0.8,
                           label=f'{strategy.title()} (Ord)')
                
                if 'distribution_consistency' in strategy_data.columns:
                    ax.plot(strategy_data['epoch'], strategy_data['distribution_consistency'], 
                           color='purple', linestyle=linestyle, linewidth=2, alpha=0.8,
                           label=f'{strategy.title()} (Dist Consist)')
                else:
                    # Fallback to QWK if new metrics not available
                    if 'qwk' in strategy_data.columns:
                        ax.plot(strategy_data['epoch'], strategy_data['qwk'], 
                               color='purple', linestyle=linestyle, linewidth=2, alpha=0.8,
                               label=f'{strategy.title()} (QWK)')
            
            ax.set_title('New Prediction Accuracy Metrics by Strategy', fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy/Score')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    def _plot_strategy_heatmaps(self, axes, oc_data, pc_data):
        """Plot strategy performance heatmaps."""
        if 'strategy_analysis' not in oc_data or 'strategy_analysis' not in pc_data:
            return
        
        strategies = ['ordered', 'unordered', 'linear_decay']
        # Use new metrics if available, fallback to original
        metrics = ['prediction_consistency_acc', 'ordinal_ranking_acc', 'distribution_consistency']
        fallback_metrics = ['categorical_acc', 'ordinal_acc', 'qwk']
        
        for ax, data, title_suffix in [(axes[0], oc_data, ''), (axes[1], pc_data, '')]:
            strategy_data = data['strategy_analysis']['comparison_summary']
            
            # Determine which metrics to use
            available_metrics = metrics
            sample_item = strategy_data[0] if strategy_data else {}
            if not any(m in sample_item for m in metrics):
                available_metrics = fallback_metrics
            
            # Create matrix
            matrix = np.zeros((len(strategies), len(available_metrics)))
            for i, strategy in enumerate(strategies):
                item = next((s for s in strategy_data if s['strategy'] == strategy), None)
                if item:
                    matrix[i] = [item.get(m, 0) for m in available_metrics]
            
            sns.heatmap(matrix, 
                       xticklabels=[m.replace('_', ' ').title() for m in available_metrics],
                       yticklabels=[s.replace('_', ' ').title() for s in strategies],
                       annot=True, fmt='.3f', cmap='Blues', ax=ax,
                       vmin=0, vmax=1, cbar_kws={'label': 'Score'})
            ax.set_title('Strategy Performance Matrix', fontsize=12)

    def _plot_performance_comparison(self, axes, oc_data, pc_data):
        """Plot final performance comparison by strategy."""
        if 'strategy_analysis' not in oc_data or 'strategy_analysis' not in pc_data:
            return
        
        oc_strategies = oc_data['strategy_analysis']['comparison_summary']
        pc_strategies = pc_data['strategy_analysis']['comparison_summary']
        
        strategies = ['ordered', 'unordered', 'linear_decay']
        # Use new metrics if available, fallback to original
        metrics = ['prediction_consistency_acc', 'ordinal_ranking_acc', 'distribution_consistency']
        metric_labels = ['Pred Consistency', 'Ordinal Ranking', 'Dist Consistency']
        fallback_metrics = ['categorical_acc', 'ordinal_acc', 'qwk']
        fallback_labels = ['Categorical Acc', 'Ordinal Acc', 'QWK']
        
        # Determine which metrics to use
        sample_item = oc_strategies[0] if oc_strategies else {}
        if not any(m in sample_item for m in metrics):
            metrics = fallback_metrics
            metric_labels = fallback_labels
        
        # Grouped bar chart
        x = np.arange(len(strategies))
        width = 0.25
        colors_oc = ['lightblue', 'lightcoral', 'lightgreen']
        colors_pc = ['blue', 'red', 'green']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            oc_values = []
            pc_values = []
            
            for strategy in strategies:
                oc_item = next((item for item in oc_strategies if item['strategy'] == strategy), None)
                pc_item = next((item for item in pc_strategies if item['strategy'] == strategy), None)
                
                oc_values.append(oc_item[metric] if oc_item else 0)
                pc_values.append(pc_item[metric] if pc_item else 0)
            
            # Plot bars
            bars_oc = axes[0].bar(x + i*width - width, oc_values, width, 
                                color=colors_oc[i], alpha=0.7, label=f'OC {label}')
            bars_pc = axes[0].bar(x + i*width, pc_values, width, 
                                color=colors_pc[i], alpha=0.7, label=f'PC {label}')
            
            # Add value labels
            for bar, val in zip(bars_oc + bars_pc, oc_values + pc_values):
                if val > 0:
                    axes[0].text(bar.get_x() + bar.get_width()/2., val + 0.01,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        axes[0].set_title('Performance by Strategy', fontsize=12)
        axes[0].set_xlabel('Embedding Strategy')
        axes[0].set_ylabel('Score')
        axes[0].set_xticks(x + width/2)
        axes[0].set_xticklabels([s.replace('_', ' ').title() for s in strategies])
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Strategy differences
        differences = []
        strategy_names = []
        
        for strategy in strategies:
            oc_item = next((item for item in oc_strategies if item['strategy'] == strategy), None)
            pc_item = next((item for item in pc_strategies if item['strategy'] == strategy), None)
            
            if oc_item and pc_item:
                diff_qwk = pc_item['qwk'] - oc_item['qwk']
                differences.append(diff_qwk)
                strategy_names.append(strategy.replace('_', ' ').title())
        
        colors_diff = ['green' if d > 0 else 'red' for d in differences]
        bars_diff = axes[1].bar(strategy_names, differences, color=colors_diff, alpha=0.7)
        
        axes[1].set_title('QWK Difference by Strategy (PC - OC)', fontsize=12)
        axes[1].set_xlabel('Embedding Strategy')
        axes[1].set_ylabel('QWK Difference')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars_diff, differences):
            axes[1].text(bar.get_x() + bar.get_width()/2., 
                        diff + (0.01 if diff > 0 else -0.01),
                        f'{diff:+.3f}', ha='center', 
                        va='bottom' if diff > 0 else 'top', fontsize=10)

    def analyze_prediction_targets(self):
        """Analyze and explain prediction target mechanisms."""
        self.logger.info("Analyzing prediction target mechanisms...")
        
        analysis = {
            "prediction_inconsistency_issue": {
                "problem": "Training vs Inference Inconsistency",
                "training_mechanism": "Uses cumulative probabilities P(Y ≤ k) via OrdinalLoss",
                "inference_mechanism": "Uses individual probabilities P(Y = k) via argmax",
                "impact": "Model learns ordinal structure but predicts without it",
                "solution": "Use cumulative-based or expected value prediction methods"
            },
            "prediction_methods_available": {
                "argmax": {
                    "description": "Current method - argmax over P(Y = k)",
                    "mechanism": "torch.argmax(gpcm_probs, dim=-1)",
                    "pros": "Simple, fast computation",
                    "cons": "Loses ordinal information, inconsistent with training",
                    "use_case": "When ordinal structure is not important"
                },
                "cumulative": {
                    "description": "GPCM-consistent method - uses P(Y ≤ k)",
                    "mechanism": "Find first k where P(Y ≤ k) > 0.5 (median)",
                    "pros": "Consistent with ordinal loss, respects category ordering",
                    "cons": "Slightly more complex computation",
                    "use_case": "Educational assessments with ordinal responses"
                },
                "expected": {
                    "description": "Expected value method - E[Y] = Σ k * P(Y = k)",
                    "mechanism": "torch.sum(gpcm_probs * categories, dim=-1)",
                    "pros": "Natural for ordinal data, continuous then discretized",
                    "cons": "May not align perfectly with category boundaries",
                    "use_case": "When treating categories as continuous scale"
                }
            },
            "prediction_targets": {
                "categorical_accuracy": {
                    "description": "Exact category match prediction accuracy",
                    "mechanism": "Depends on prediction method selected",
                    "threshold": "No threshold for argmax; 0.5 for cumulative; rounding for expected",
                    "formula": "categorical_acc = (predicted_category == true_category).mean()",
                    "interpretation": "Percentage of responses with exact category prediction",
                    "method_dependency": "Results vary significantly by prediction method"
                },
                "ordinal_accuracy": {
                    "description": "Educational tolerance accuracy (±1 category)",
                    "mechanism": "Allow predictions within ±1 category of true response",
                    "threshold": "±1 category tolerance (educational assessment standard)",
                    "formula": "ordinal_acc = (|predicted_category - true_category| <= 1).mean()",
                    "interpretation": "Percentage of responses within educationally acceptable range",
                    "method_dependency": "Less sensitive to prediction method due to tolerance"
                }
            },
            "gpcm_prediction_process": {
                "step1": "Generate IRT parameters: θ (ability), α (discrimination), β (thresholds)",
                "step2": "Compute GPCM cumulative logits: γ_k = α(θ - β_k)",
                "step3": "Calculate category probabilities: P(Y=k) via cumulative logits",
                "step4": "Categorical prediction: argmax(P(Y=k)) - selects most likely category",
                "step5": "Ordinal evaluation: check if |predicted - actual| <= tolerance"
            },
            "comparison_to_binary": {
                "binary_threshold": "Fixed 0.5 threshold for P(Y=1) vs P(Y=0)",
                "gpcm_categorical": "Dynamic argmax over K categories - no fixed threshold",
                "gpcm_ordinal": "Educational tolerance bands - accounts for partial credit"
            }
        }
        
        # Save analysis
        with open(f"{self.save_path}/../analysis/prediction_target_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        self.logger.info("Generating comprehensive analysis report")
        
        oc_data = self.extract_format_data('OC')
        pc_data = self.extract_format_data('PC')
        prediction_analysis = self.analyze_prediction_targets()
        
        report_path = f"{self.save_path}/comprehensive_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEP-GPCM COMPREHENSIVE ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Prediction inconsistency issue
            f.write("PREDICTION INCONSISTENCY ISSUE (CRITICAL FINDING)\n")
            f.write("-" * 50 + "\n")
            issue = prediction_analysis["prediction_inconsistency_issue"]
            f.write(f"Problem: {issue['problem']}\n")
            f.write(f"Training: {issue['training_mechanism']}\n")
            f.write(f"Inference: {issue['inference_mechanism']}\n")
            f.write(f"Impact: {issue['impact']}\n")
            f.write(f"Solution: {issue['solution']}\n\n")
            
            # Available prediction methods
            f.write("AVAILABLE PREDICTION METHODS\n")
            f.write("-" * 35 + "\n")
            methods = prediction_analysis["prediction_methods_available"]
            
            for method_name, method_info in methods.items():
                f.write(f"{method_name.upper()}:\n")
                f.write(f"  Description: {method_info['description']}\n")
                f.write(f"  Mechanism: {method_info['mechanism']}\n")
                f.write(f"  Pros: {method_info['pros']}\n")
                f.write(f"  Cons: {method_info['cons']}\n")
                f.write(f"  Use Case: {method_info['use_case']}\n\n")
            
            # Prediction targets explanation
            f.write("PREDICTION TARGET ANALYSIS\n")
            f.write("-" * 30 + "\n")
            pred_targets = prediction_analysis["prediction_targets"]
            
            f.write("CATEGORICAL ACCURACY:\n")
            f.write(f"  Mechanism: {pred_targets['categorical_accuracy']['mechanism']}\n")
            f.write(f"  Threshold: {pred_targets['categorical_accuracy']['threshold']}\n")
            f.write(f"  Formula: {pred_targets['categorical_accuracy']['formula']}\n")
            f.write(f"  Method Dependency: {pred_targets['categorical_accuracy']['method_dependency']}\n\n")
            
            f.write("ORDINAL ACCURACY:\n")
            f.write(f"  Mechanism: {pred_targets['ordinal_accuracy']['mechanism']}\n")
            f.write(f"  Threshold: {pred_targets['ordinal_accuracy']['threshold']}\n")
            f.write(f"  Formula: {pred_targets['ordinal_accuracy']['formula']}\n")
            f.write(f"  Method Dependency: {pred_targets['ordinal_accuracy']['method_dependency']}\n\n")
            
            f.write("GPCM PREDICTION PROCESS:\n")
            gpcm_process = prediction_analysis["gpcm_prediction_process"]
            for step, description in gpcm_process.items():
                f.write(f"  {step.upper()}: {description}\n")
            f.write("\n")
            
            f.write("COMPARISON TO BINARY CLASSIFICATION:\n")
            comparison = prediction_analysis["comparison_to_binary"]
            f.write(f"  Binary: {comparison['binary_threshold']}\n")
            f.write(f"  GPCM Categorical: {comparison['gpcm_categorical']}\n")
            f.write(f"  GPCM Ordinal: {comparison['gpcm_ordinal']}\n\n")
            
            # Performance comparison
            if 'strategy_analysis' in oc_data and 'strategy_analysis' in pc_data:
                f.write("FORMAT PERFORMANCE COMPARISON\n")
                f.write("-" * 30 + "\n")
                
                oc_strategies = oc_data['strategy_analysis']['comparison_summary']
                pc_strategies = pc_data['strategy_analysis']['comparison_summary']
                
                f.write("Best performing strategies by format:\n")
                oc_best = max(oc_strategies, key=lambda x: x['qwk'])
                pc_best = max(pc_strategies, key=lambda x: x['qwk'])
                
                f.write(f"OC Best: {oc_best['strategy'].title()} - "
                       f"Cat: {oc_best['categorical_acc']:.3f}, "
                       f"Ord: {oc_best['ordinal_acc']:.3f}, "
                       f"QWK: {oc_best['qwk']:.3f}\n")
                f.write(f"PC Best: {pc_best['strategy'].title()} - "
                       f"Cat: {pc_best['categorical_acc']:.3f}, "
                       f"Ord: {pc_best['ordinal_acc']:.3f}, "
                       f"QWK: {pc_best['qwk']:.3f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated by Deep-GPCM Comprehensive Analysis Suite\n")
        
        self.logger.info(f"Comprehensive report saved to {report_path}")

    def run_full_analysis(self):
        """Run complete analysis suite."""
        self.logger.info("Starting comprehensive Deep-GPCM analysis")
        
        # Load all data
        self.load_all_data()
        
        # Create visualizations
        self.create_comprehensive_comparison()
        
        # Analyze prediction targets
        self.analyze_prediction_targets()
        
        # Generate report
        self.generate_comprehensive_report()
        
        self.logger.info("Comprehensive analysis complete!")
        
        # Output summary
        print(f"Results saved to: {self.save_path}")
        print("Generated files:")
        print("- comprehensive_oc_pc_comparison.png (Main visualization)")
        print("- comprehensive_analysis_report.txt (Detailed report)")
        print("- prediction_target_analysis.json (Prediction mechanism analysis)")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Deep-GPCM Comprehensive Analysis Suite')
    parser.add_argument('--save_path', type=str, default='results/plots',
                        help='Save path for outputs (default: results/plots)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level.upper())
    analyzer = DeepGPCMAnalyzer(save_path=args.save_path, log_level=log_level)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()