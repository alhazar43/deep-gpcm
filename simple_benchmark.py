#!/usr/bin/env python3
"""
Simple Benchmark Comparison Script

Compares performance metrics between baseline, AKVMN, and Bayesian models.
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_results(save_dir: Path):
    """Load results from existing model checkpoints and JSON result files."""
    results = {}
    
    # Helper function to load from CV summary or JSON results
    def load_from_json(model_name, checkpoint_data=None):
        # First try to load from CV summary (most complete results)
        cv_summary_path = Path('logs') / f'train_results_{model_name}_synthetic_OC_cv_summary.json'
        if cv_summary_path.exists():
            with open(cv_summary_path, 'r') as f:
                cv_data = json.load(f)
            cv_summary = cv_data.get('cv_summary', {})
            return {
                'test_accuracy': cv_summary.get('categorical_accuracy', {}).get('mean', 0.0),
                'test_qwk': cv_summary.get('quadratic_weighted_kappa', {}).get('mean', 0.0),
                'test_ordinal_accuracy': cv_summary.get('ordinal_accuracy', {}).get('mean', 0.0),
                'test_mae': cv_summary.get('mean_absolute_error', {}).get('mean', 0.0)
            }
        
        # Fallback to regular JSON results
        json_path = save_dir / f'results_{model_name}_synthetic_OC.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            # Get best metrics from JSON
            best_metrics = json_data.get('best_metrics', {})
            # Use argmax method as primary metric
            argmax_metrics = best_metrics.get('argmax', {})
            return {
                'test_accuracy': argmax_metrics.get('categorical_accuracy', 0.0),
                'test_qwk': argmax_metrics.get('quadratic_weighted_kappa', 0.0),
                'test_ordinal_accuracy': argmax_metrics.get('ordinal_accuracy', 0.0),
                'test_mae': argmax_metrics.get('mean_absolute_error', 0.0)
            }
        elif checkpoint_data:
            return {
                'test_accuracy': checkpoint_data.get('test_accuracy', 0.0),
                'test_qwk': checkpoint_data.get('test_qwk', 0.0),
                'test_ordinal_accuracy': checkpoint_data.get('test_ordinal_accuracy', 0.0),
                'test_mae': checkpoint_data.get('test_mae', 0.0)
            }
        else:
            return {
                'test_accuracy': 0.0,
                'test_qwk': 0.0,
                'test_ordinal_accuracy': 0.0,
                'test_mae': 0.0
            }
    
    # Baseline model
    baseline_path = save_dir / 'best_baseline_synthetic_OC.pth'
    if baseline_path.exists():
        checkpoint = torch.load(baseline_path, map_location='cpu')
        metrics = load_from_json('baseline', checkpoint)
        results['baseline'] = {
            **metrics,
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'model_type': 'baseline'
        }
    
    # AKVMN model
    akvmn_path = save_dir / 'best_akvmn_synthetic_OC.pth'
    if akvmn_path.exists():
        checkpoint = torch.load(akvmn_path, map_location='cpu')
        metrics = load_from_json('akvmn', checkpoint)
        results['akvmn'] = {
            **metrics,
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'model_type': 'akvmn'
        }
    
    # Bayesian model
    bayesian_path = save_dir / 'best_bayesian_synthetic_OC.pth'
    if bayesian_path.exists():
        checkpoint = torch.load(bayesian_path, map_location='cpu')
        metrics = load_from_json('bayesian', checkpoint)
        # For Bayesian, fallback to checkpoint data if JSON loading fails
        if all(v == 0.0 for v in metrics.values()):
            metrics = {
                'test_accuracy': checkpoint.get('test_accuracy', 0.0),
                'test_qwk': checkpoint.get('test_qwk', 0.0),
                'test_ordinal_accuracy': checkpoint.get('test_ordinal_accuracy', 0.0),
                'test_mae': checkpoint.get('test_mae', 0.0)
            }
        results['bayesian'] = {
            **metrics,
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'model_type': 'bayesian',
            'has_irt_comparison': 'irt_comparison' in checkpoint.get('args', {})
        }
    
    return results


def create_benchmark_plots(results: dict, output_dir: Path):
    """Create benchmark comparison plots using predefined 6 metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(results.keys())
    model_names = [model.replace('_', ' ').title() for model in models]
    
    # Prepare data for 6 predefined metrics
    accuracies = [results[m]['test_accuracy'] for m in models]
    qwks = [results[m]['test_qwk'] for m in models]
    ordinal_accuracies = [results[m].get('test_ordinal_accuracy', 0.0) for m in models]
    maes = [results[m].get('test_mae', 0.0) for m in models]
    parameters = [results[m]['parameters'] / 1000 for m in models]  # In thousands
    efficiencies = [acc / (param if param > 0 else 1) for acc, param in zip(accuracies, parameters)]
    
    # Create figure with 2x3 subplots for 6 metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define colors like in your plots
    colors = ['steelblue', 'mediumpurple', 'orange'][:len(models)]
    
    metrics_data = [
        (accuracies, 'Categorical Accuracy', 'Model Performance Comparison', 0, 1),
        (qwks, 'Quadratic Weighted Kappa', 'Ordinal Performance (QWK)', 0, 1),
        (ordinal_accuracies, 'Ordinal Accuracy', 'Ordinal Accuracy', 0, 1),
        (maes, 'Mean Absolute Error', 'Mean Absolute Error', 0, None),
        (parameters, 'Parameters (K)', 'Model Complexity', 0, None),
        (efficiencies, 'Accuracy per 1K Parameters', 'Model Efficiency', 0, None)
    ]
    
    for idx, (values, ylabel, title, y_min, y_max) in enumerate(metrics_data):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            label = f'{value:.3f}' if idx < 4 else f'{value:.0f}K' if idx == 4 else f'{value:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(values) * 0.01),
                   label, ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits
        if y_max is not None:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, max(values) * 1.15)
        
        # Rotate x-axis labels if needed
        if len(model_names) > 2:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_data = []
    for model in models:
        summary_data.append({
            'Model': model.replace('_', ' ').title(),
            'Categorical Accuracy': f"{results[model]['test_accuracy']:.3f}",
            'QWK': f"{results[model]['test_qwk']:.3f}",
            'Ordinal Accuracy': f"{results[model].get('test_ordinal_accuracy', 0.0):.3f}",
            'MAE': f"{results[model].get('test_mae', 0.0):.3f}",
            'Parameters': f"{results[model]['parameters']:,}",
            'Efficiency': f"{results[model]['test_accuracy'] / (results[model]['parameters'] / 1000):.4f}",
            'Type': results[model]['model_type']
        })
    
    # Save as text table
    with open(output_dir / 'benchmark_summary.txt', 'w') as f:
        f.write("Deep-GPCM Model Benchmark Comparison\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Model':<12} {'Cat.Acc':<9} {'QWK':<8} {'Ord.Acc':<9} {'MAE':<8} {'Params':<12} {'Efficiency':<11} {'Type':<10}\n")
        f.write("-" * 100 + "\n")
        
        for data in summary_data:
            f.write(f"{data['Model']:<12} {data['Categorical Accuracy']:<9} {data['QWK']:<8} "
                   f"{data['Ordinal Accuracy']:<9} {data['MAE']:<8} {data['Parameters']:<12} "
                   f"{data['Efficiency']:<11} {data['Type']:<10}\n")
    
    return summary_data


def analyze_irt_recovery(results: dict):
    """Analyze IRT parameter recovery if available."""
    bayesian_results = results.get('bayesian', {})
    
    if bayesian_results.get('has_irt_comparison'):
        print("\n🧠 IRT Parameter Recovery Analysis:")
        print("- Bayesian model includes IRT parameter comparison with ground truth")
        print("- Prior distributions incorporated: θ ~ N(0,1), α ~ LogN(0,0.3), β ~ Ordered Normal")
        print("- Variational inference with ELBO optimization")
        print("- See training logs for detailed recovery metrics")
    else:
        print("\n⚠️  IRT parameter recovery analysis not available")
        print("- Train Bayesian model with ground truth comparison for full analysis")


def main():
    parser = argparse.ArgumentParser(description='Simple benchmark comparison')
    parser.add_argument('--save_dir', type=str, default='save_models',
                       help='Directory containing saved models')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("DEEP-GPCM MODEL BENCHMARK COMPARISON")
    print("=" * 70)
    
    # Load results
    results = load_model_results(save_dir)
    
    if not results:
        print("❌ No model checkpoints found!")
        print(f"   Looking in: {save_dir}")
        print("   Expected files: best_baseline_synthetic_OC.pth, best_akvmn_synthetic_OC.pth, best_bayesian_synthetic_OC.pth")
        return
    
    print(f"\n📊 Found {len(results)} trained models:")
    for model_name, result in results.items():
        print(f"   • {model_name.title()}: {result['test_accuracy']:.3f} accuracy, "
              f"{result['parameters']:,} parameters")
    
    # Create benchmark plots
    print(f"\n📈 Creating benchmark comparison plots...")
    summary_data = create_benchmark_plots(results, output_dir)
    
    # Print detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 120)
    
    # Find best performing model
    best_acc_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_qwk_model = max(results.keys(), key=lambda k: results[k]['test_qwk'])
    
    for data in summary_data:
        print(f"{data['Model']:<12} | Acc: {data['Categorical Accuracy']:<8} | QWK: {data['QWK']:<8} | "
              f"Ord: {data['Ordinal Accuracy']:<8} | MAE: {data['MAE']:<8} | "
              f"Params: {data['Parameters']:<10} | Eff: {data['Efficiency']:<8} | Type: {data['Type']}")
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Highest Accuracy: {best_acc_model.title()} ({results[best_acc_model]['test_accuracy']:.3f})")
    print(f"   Highest QWK: {best_qwk_model.title()} ({results[best_qwk_model]['test_qwk']:.3f})")
    
    # Parameter efficiency analysis
    if len(results) > 1:
        efficiency_scores = {}
        for model, result in results.items():
            # Simple efficiency: accuracy per parameter (scaled)
            efficiency = result['test_accuracy'] / (result['parameters'] / 100000)
            efficiency_scores[model] = efficiency
        
        best_efficiency = max(efficiency_scores.keys(), key=lambda k: efficiency_scores[k])
        print(f"   Most Efficient: {best_efficiency.title()} (accuracy/complexity ratio)")
    
    # IRT analysis
    analyze_irt_recovery(results)
    
    # Model-specific insights
    print(f"\n💡 MODEL INSIGHTS:")
    if 'baseline' in results:
        print("   • Baseline GPCM: Standard DKVMN + GPCM implementation")
    if 'akvmn' in results:
        print("   • AKVMN: Enhanced with multi-head attention and deep integration")
    if 'bayesian' in results:
        print("   • Bayesian: Variational approach with proper IRT parameter priors")
        print("     - Incorporates uncertainty quantification")
        print("     - Enables parameter recovery analysis")
    
    print(f"\n📂 Results saved to: {output_dir}")
    print(f"   • model_benchmark_comparison.png")
    print(f"   • benchmark_summary.txt")
    
    # Save results as JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'benchmark_results_{timestamp}.json'
    
    # Prepare JSON serializable data
    json_results = {}
    for model, result in results.items():
        json_results[model] = {
            'test_accuracy': float(result['test_accuracy']),
            'test_qwk': float(result['test_qwk']),
            'parameters': int(result['parameters']),
            'model_type': result['model_type']
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'dataset': 'synthetic_OC',
            'results': json_results,
            'summary': {
                'best_accuracy': {'model': best_acc_model, 'value': float(results[best_acc_model]['test_accuracy'])},
                'best_qwk': {'model': best_qwk_model, 'value': float(results[best_qwk_model]['test_qwk'])},
                'model_count': len(results)
            }
        }, f, indent=2)
    
    print(f"   • benchmark_results_{timestamp}.json")
    print("\n✅ Benchmark comparison complete!")


if __name__ == '__main__':
    main()