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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_model_results(save_dir: Path):
    """Load results from existing model checkpoints."""
    results = {}
    
    # Baseline model
    baseline_path = save_dir / 'best_baseline_synthetic_OC.pth'
    if baseline_path.exists():
        checkpoint = torch.load(baseline_path, map_location='cpu')
        results['baseline'] = {
            'test_accuracy': checkpoint.get('test_accuracy', 0.0),
            'test_qwk': checkpoint.get('test_qwk', 0.0),
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'model_type': 'baseline'
        }
    
    # AKVMN model
    akvmn_path = save_dir / 'best_akvmn_synthetic_OC.pth'
    if akvmn_path.exists():
        checkpoint = torch.load(akvmn_path, map_location='cpu')
        results['akvmn'] = {
            'test_accuracy': checkpoint.get('test_accuracy', 0.0),
            'test_qwk': checkpoint.get('test_qwk', 0.0),
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'model_type': 'akvmn'
        }
    
    # Bayesian model
    bayesian_path = save_dir / 'best_bayesian_synthetic_OC.pth'
    if bayesian_path.exists():
        checkpoint = torch.load(bayesian_path, map_location='cpu')
        results['bayesian'] = {
            'test_accuracy': checkpoint.get('test_accuracy', 0.0),
            'test_qwk': checkpoint.get('test_qwk', 0.0),
            'parameters': sum(p.numel() for p in checkpoint['model_state_dict'].values()),
            'model_type': 'bayesian',
            'has_irt_comparison': 'irt_comparison' in checkpoint.get('args', {})
        }
    
    return results


def create_benchmark_plots(results: dict, output_dir: Path):
    """Create benchmark comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    models = list(results.keys())
    accuracies = [results[m]['test_accuracy'] for m in models]
    qwks = [results[m]['test_qwk'] for m in models]
    parameters = [results[m]['parameters'] / 1000 for m in models]  # In thousands
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy comparison
    ax = axes[0, 0]
    bars = ax.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Performance Comparison - Categorical Accuracy')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # QWK comparison
    ax = axes[0, 1]
    bars = ax.bar(models, qwks, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
    ax.set_ylabel('Quadratic Weighted Kappa')
    ax.set_title('Model Performance Comparison - QWK')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, qwk in zip(bars, qwks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{qwk:.3f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # Parameter count comparison
    ax = axes[1, 0]
    bars = ax.bar(models, parameters, color=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
    ax.set_ylabel('Parameters (thousands)')
    ax.set_title('Model Complexity Comparison')
    
    # Add value labels on bars
    for bar, param in zip(bars, parameters):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{param:.0f}K', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    # Efficiency plot (accuracy vs parameters)
    ax = axes[1, 1]
    scatter = ax.scatter(parameters, accuracies, s=100, alpha=0.7,
                        c=['skyblue', 'lightcoral', 'lightgreen'][:len(models)])
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (parameters[i], accuracies[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Parameters (thousands)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Efficiency - Accuracy vs Complexity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_benchmark_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_data = []
    for model in models:
        summary_data.append({
            'Model': model.title(),
            'Test Accuracy': f"{results[model]['test_accuracy']:.3f}",
            'QWK': f"{results[model]['test_qwk']:.3f}",
            'Parameters': f"{results[model]['parameters']:,}",
            'Type': results[model]['model_type']
        })
    
    # Save as text table
    with open(output_dir / 'benchmark_summary.txt', 'w') as f:
        f.write("Deep-GPCM Model Benchmark Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"{'Model':<10} {'Accuracy':<10} {'QWK':<10} {'Parameters':<12} {'Type':<10}\n")
        f.write("-" * 60 + "\n")
        
        for data in summary_data:
            f.write(f"{data['Model']:<10} {data['Test Accuracy']:<10} {data['QWK']:<10} "
                   f"{data['Parameters']:<12} {data['Type']:<10}\n")
    
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
    print("-" * 70)
    
    # Find best performing model
    best_acc_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_qwk_model = max(results.keys(), key=lambda k: results[k]['test_qwk'])
    
    for data in summary_data:
        print(f"{data['Model']:<12} | Acc: {data['Test Accuracy']:<8} | QWK: {data['QWK']:<8} | "
              f"Params: {data['Parameters']:<10} | Type: {data['Type']}")
    
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