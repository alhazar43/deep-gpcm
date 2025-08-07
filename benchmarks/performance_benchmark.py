"""
Comprehensive performance benchmarking for ordinal-aware attention mechanisms.
Evaluates improvements across different datasets, metrics, and configurations.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import training components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ordinal_trainer import OrdinalTrainer
from models.metrics.ordinal_metrics import calculate_ordinal_improvement


class PerformanceBenchmark:
    """Comprehensive benchmark suite for ordinal attention mechanisms."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
        # Benchmark configurations
        self.dataset_configs = [
            {'n_samples': 400, 'n_questions': 20, 'seq_len': 10, 'name': 'small'},
            {'n_samples': 800, 'n_questions': 50, 'seq_len': 15, 'name': 'medium'},
            {'n_samples': 1200, 'n_questions': 100, 'seq_len': 20, 'name': 'large'}
        ]
        
        self.attention_configs = {
            'baseline': {
                'use_ordinal_attention': False,
                'attention_types': None,
                'loss_type': 'standard_ce'
            },
            'ordinal_aware': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware'],
                'loss_type': 'ordinal_ce'
            },
            'qwk_aligned': {
                'use_ordinal_attention': True,
                'attention_types': ['qwk_aligned'],
                'loss_type': 'combined'
            },
            'response_conditioned': {
                'use_ordinal_attention': True,
                'attention_types': ['response_conditioned'],
                'loss_type': 'ordinal_ce'
            },
            'combined_basic': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned'],
                'loss_type': 'combined'
            },
            'full_suite': {
                'use_ordinal_attention': True,
                'attention_types': ['ordinal_aware', 'qwk_aligned', 'response_conditioned'],
                'loss_type': 'combined'
            }
        }
    
    def run_dataset_benchmark(self, dataset_config: Dict, n_epochs: int = 15, 
                            n_runs: int = 3) -> Dict:
        """Run benchmark on a specific dataset configuration."""
        print(f"\nBenchmarking {dataset_config['name']} dataset:")
        print(f"  Samples: {dataset_config['n_samples']}, "
              f"Questions: {dataset_config['n_questions']}, "
              f"Seq Length: {dataset_config['seq_len']}")
        
        all_results = {}
        
        for run_idx in range(n_runs):
            print(f"\n  Run {run_idx + 1}/{n_runs}")
            
            # Create trainer with current config
            trainer = OrdinalTrainer(
                n_questions=dataset_config['n_questions'],
                n_cats=4,
                device=self.device
            )
            
            # Override configs
            trainer.configs = self.attention_configs
            
            # Run training
            run_results = trainer.run_comparison_experiment(
                n_samples=dataset_config['n_samples'],
                n_epochs=n_epochs,
                lr=0.001
            )
            
            # Store results
            for config_name, config_result in run_results.items():
                if config_name not in all_results:
                    all_results[config_name] = []
                all_results[config_name].append(config_result)
        
        # Aggregate results across runs
        aggregated_results = self._aggregate_runs(all_results)
        
        return aggregated_results
    
    def _aggregate_runs(self, all_results: Dict) -> Dict:
        """Aggregate results across multiple runs."""
        aggregated = {}
        
        for config_name, run_list in all_results.items():
            # Extract metrics from all runs
            qwks = [run['best_qwk'] for run in run_list]
            accuracies = [run['final_val_metrics']['accuracy'] for run in run_list]
            maes = [run['final_val_metrics']['mae'] for run in run_list]
            ord_accs = [run['final_val_metrics']['ordinal_accuracy'] for run in run_list]
            times = [run['training_time'] for run in run_list]
            
            aggregated[config_name] = {
                'qwk_mean': np.mean(qwks),
                'qwk_std': np.std(qwks),
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
                'ordinal_accuracy_mean': np.mean(ord_accs),
                'ordinal_accuracy_std': np.std(ord_accs),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
                'n_runs': len(run_list)
            }
        
        return aggregated
    
    def run_comprehensive_benchmark(self, n_epochs: int = 15, n_runs: int = 3) -> Dict:
        """Run comprehensive benchmark across all dataset sizes."""
        print("Starting comprehensive performance benchmark...")
        print(f"Configurations: {list(self.attention_configs.keys())}")
        print(f"Datasets: {[config['name'] for config in self.dataset_configs]}")
        print(f"Epochs per training: {n_epochs}, Runs per config: {n_runs}")
        
        benchmark_results = {}
        
        for dataset_config in self.dataset_configs:
            dataset_results = self.run_dataset_benchmark(dataset_config, n_epochs, n_runs)
            benchmark_results[dataset_config['name']] = dataset_results
        
        self.results = benchmark_results
        return benchmark_results
    
    def calculate_improvements(self) -> Dict:
        """Calculate improvements over baseline for all configurations."""
        improvements = {}
        
        for dataset_name, dataset_results in self.results.items():
            baseline_metrics = dataset_results.get('baseline', {})
            if not baseline_metrics:
                continue
            
            dataset_improvements = {}
            
            for config_name, config_results in dataset_results.items():
                if config_name == 'baseline':
                    continue
                
                # Calculate improvements
                qwk_imp = ((config_results['qwk_mean'] - baseline_metrics['qwk_mean']) / 
                          baseline_metrics['qwk_mean'] * 100) if baseline_metrics['qwk_mean'] > 0 else 0
                
                acc_imp = ((config_results['accuracy_mean'] - baseline_metrics['accuracy_mean']) / 
                          baseline_metrics['accuracy_mean'] * 100) if baseline_metrics['accuracy_mean'] > 0 else 0
                
                mae_imp = ((baseline_metrics['mae_mean'] - config_results['mae_mean']) / 
                          baseline_metrics['mae_mean'] * 100) if baseline_metrics['mae_mean'] > 0 else 0
                
                ord_acc_imp = ((config_results['ordinal_accuracy_mean'] - baseline_metrics['ordinal_accuracy_mean']) / 
                              baseline_metrics['ordinal_accuracy_mean'] * 100) if baseline_metrics['ordinal_accuracy_mean'] > 0 else 0
                
                time_overhead = ((config_results['time_mean'] - baseline_metrics['time_mean']) / 
                               baseline_metrics['time_mean'] * 100) if baseline_metrics['time_mean'] > 0 else 0
                
                dataset_improvements[config_name] = {
                    'qwk_improvement': qwk_imp,
                    'accuracy_improvement': acc_imp,
                    'mae_improvement': mae_imp,
                    'ordinal_accuracy_improvement': ord_acc_imp,
                    'time_overhead': time_overhead
                }
            
            improvements[dataset_name] = dataset_improvements
        
        return improvements
    
    def print_benchmark_summary(self):
        """Print comprehensive benchmark summary."""
        if not self.results:
            print("No benchmark results available. Run benchmark first.")
            return
        
        print("\n" + "="*90)
        print("COMPREHENSIVE ORDINAL ATTENTION BENCHMARK RESULTS")
        print("="*90)
        
        # Results table
        print(f"\nPerformance Results (Mean ± Std):")
        
        for dataset_name, dataset_results in self.results.items():
            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"{'Model':<20} {'QWK':<15} {'Accuracy':<15} {'MAE':<15} {'Ord.Acc':<15} {'Time(s)':<10}")
            print("-" * 90)
            
            for config_name, metrics in dataset_results.items():
                qwk_str = f"{metrics['qwk_mean']:.3f}±{metrics['qwk_std']:.3f}"
                acc_str = f"{metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}"
                mae_str = f"{metrics['mae_mean']:.3f}±{metrics['mae_std']:.3f}"
                ord_acc_str = f"{metrics['ordinal_accuracy_mean']:.3f}±{metrics['ordinal_accuracy_std']:.3f}"
                time_str = f"{metrics['time_mean']:.1f}±{metrics['time_std']:.1f}"
                
                print(f"{config_name:<20} {qwk_str:<15} {acc_str:<15} {mae_str:<15} {ord_acc_str:<15} {time_str:<10}")
        
        # Improvements table
        improvements = self.calculate_improvements()
        print(f"\n\nImprovement over Baseline (%):")
        
        for dataset_name, dataset_improvements in improvements.items():
            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"{'Model':<20} {'QWK':<10} {'Accuracy':<10} {'MAE':<10} {'Ord.Acc':<10} {'Time OH':<10}")
            print("-" * 70)
            
            for config_name, imp_metrics in dataset_improvements.items():
                qwk_imp = f"{imp_metrics['qwk_improvement']:+.1f}%"
                acc_imp = f"{imp_metrics['accuracy_improvement']:+.1f}%"
                mae_imp = f"{imp_metrics['mae_improvement']:+.1f}%"
                ord_acc_imp = f"{imp_metrics['ordinal_accuracy_improvement']:+.1f}%"
                time_oh = f"{imp_metrics['time_overhead']:+.1f}%"
                
                print(f"{config_name:<20} {qwk_imp:<10} {acc_imp:<10} {mae_imp:<10} {ord_acc_imp:<10} {time_oh:<10}")
        
        # Best performers
        print(f"\n\nBest Performing Configurations:")
        for dataset_name in self.results.keys():
            dataset_improvements = improvements.get(dataset_name, {})
            if not dataset_improvements:
                continue
            
            best_qwk = max(dataset_improvements.items(), key=lambda x: x[1]['qwk_improvement'])
            best_acc = max(dataset_improvements.items(), key=lambda x: x[1]['accuracy_improvement'])
            
            print(f"{dataset_name.upper()}:")
            print(f"  Best QWK: {best_qwk[0]} ({best_qwk[1]['qwk_improvement']:+.1f}%)")
            print(f"  Best Accuracy: {best_acc[0]} ({best_acc[1]['accuracy_improvement']:+.1f}%)")
        
        print("\n" + "="*90)
    
    def create_visualization(self, save_path: str = None):
        """Create visualization of benchmark results."""
        if not self.results:
            print("No results to visualize. Run benchmark first.")
            return
        
        improvements = self.calculate_improvements()
        
        # Prepare data for plotting
        plot_data = []
        for dataset_name, dataset_improvements in improvements.items():
            for config_name, metrics in dataset_improvements.items():
                plot_data.append({
                    'Dataset': dataset_name,
                    'Model': config_name,
                    'QWK_Improvement': metrics['qwk_improvement'],
                    'Ordinal_Accuracy_Improvement': metrics['ordinal_accuracy_improvement'],
                    'Time_Overhead': metrics['time_overhead']
                })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # QWK improvement by dataset
        sns.barplot(data=df, x='Dataset', y='QWK_Improvement', hue='Model', ax=axes[0,0])
        axes[0,0].set_title('QWK Improvement by Dataset')
        axes[0,0].set_ylabel('QWK Improvement (%)')
        
        # Ordinal accuracy improvement by dataset
        sns.barplot(data=df, x='Dataset', y='Ordinal_Accuracy_Improvement', hue='Model', ax=axes[0,1])
        axes[0,1].set_title('Ordinal Accuracy Improvement by Dataset')
        axes[0,1].set_ylabel('Ordinal Accuracy Improvement (%)')
        
        # Time overhead by model
        sns.barplot(data=df, x='Model', y='Time_Overhead', ax=axes[1,0])
        axes[1,0].set_title('Training Time Overhead by Model')
        axes[1,0].set_ylabel('Time Overhead (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # QWK vs Time overhead scatter
        axes[1,1].scatter(df['Time_Overhead'], df['QWK_Improvement'], 
                         c=[plt.cm.viridis(i/len(df)) for i in range(len(df))])
        for i, row in df.iterrows():
            axes[1,1].annotate(f"{row['Model'][:3]}", 
                             (row['Time_Overhead'], row['QWK_Improvement']),
                             fontsize=8)
        axes[1,1].set_xlabel('Time Overhead (%)')
        axes[1,1].set_ylabel('QWK Improvement (%)')
        axes[1,1].set_title('QWK Improvement vs Time Overhead')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Add improvements to results
        results_with_improvements = self.results.copy()
        improvements = self.calculate_improvements()
        results_with_improvements['improvements'] = improvements
        
        with open(filepath, 'w') as f:
            json.dump(results_with_improvements, f, indent=2)
        
        print(f"Benchmark results saved to {filepath}")


def run_quick_benchmark():
    """Run a quick benchmark for demonstration."""
    print("Running quick performance benchmark...")
    
    benchmark = PerformanceBenchmark()
    
    # Use smaller configs for quick demo
    benchmark.dataset_configs = [
        {'n_samples': 300, 'n_questions': 20, 'seq_len': 10, 'name': 'small'},
        {'n_samples': 500, 'n_questions': 30, 'seq_len': 12, 'name': 'medium'}
    ]
    
    # Run benchmark
    results = benchmark.run_comprehensive_benchmark(n_epochs=8, n_runs=2)
    benchmark.print_benchmark_summary()
    
    return benchmark


if __name__ == "__main__":
    benchmark = run_quick_benchmark()