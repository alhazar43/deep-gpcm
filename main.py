"""
Main Script for Deep-GPCM
Unified interface for training, evaluation, and benchmarking.
"""

import os
import argparse
import json
from datetime import datetime
from typing import Dict, Any, List

from config import get_preset_configs
from train import train_model
from evaluate import load_trained_model, evaluate_model_comprehensive
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader
from utils.loss_utils import create_loss_function
from evaluation.metrics import GpcmMetrics
import torch


def benchmark_models(model_types: List[str], dataset_name: str, 
                    epochs: int = 15, batch_size: int = 32,
                    device: torch.device = None) -> Dict[str, Any]:
    """
    Benchmark multiple models side-by-side.
    
    Args:
        model_types: List of model types to benchmark
        dataset_name: Dataset to use for benchmarking
        epochs: Number of training epochs
        batch_size: Training batch size
        device: Computing device
        
    Returns:
        Comprehensive benchmark results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🏁 BENCHMARKING {len(model_types)} MODELS")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Models: {', '.join(model_types)}")
    print("="*60)
    
    preset_configs = get_preset_configs()
    results = {}
    
    for model_type in model_types:
        if model_type not in preset_configs:
            print(f"❌ Unknown model type: {model_type}")
            continue
        
        print(f"\n🚀 Benchmarking {model_type.upper()}...")
        
        # Get configuration
        config = preset_configs[model_type]
        config.epochs = epochs
        config.batch_size = batch_size
        config.dataset_name = dataset_name
        
        try:
            # Train model
            train_results = train_model(config, dataset_name, device)
            results[model_type] = train_results
            
            print(f"✅ {model_type.upper()} completed - Accuracy: {train_results['final_accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} failed: {e}")
            continue
    
    return results


def create_benchmark_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create benchmark summary and comparison."""
    if len(results) < 2:
        return {"summary": "Need at least 2 models for comparison"}
    
    summary = {
        'models': {},
        'comparison': {},
        'winner': None
    }
    
    baseline_acc = None
    best_model = None
    best_accuracy = 0.0
    
    # Extract key metrics for each model
    for model_name, model_results in results.items():
        best_metrics = model_results['best_metrics']
        
        summary['models'][model_name] = {
            'categorical_accuracy': best_metrics['categorical_acc'],
            'ordinal_accuracy': best_metrics['ordinal_acc'],
            'qwk': best_metrics['qwk'],
            'mae': best_metrics['mae'],
            'inference_time': best_metrics['avg_inference_time'],
            'parameters': sum(p.numel() for p in results[model_name].get('model', {}).parameters()) if 'model' in results[model_name] else 0
        }
        
        # Track baseline and best
        if 'baseline' in model_name.lower():
            baseline_acc = best_metrics['categorical_acc']
        
        if best_metrics['categorical_acc'] > best_accuracy:
            best_accuracy = best_metrics['categorical_acc']
            best_model = model_name
    
    # Calculate improvements vs baseline
    if baseline_acc is not None:
        for model_name in summary['models']:
            if 'baseline' not in model_name.lower():
                acc_delta = summary['models'][model_name]['categorical_accuracy'] - baseline_acc
                summary['models'][model_name]['improvement_vs_baseline'] = (acc_delta / baseline_acc) * 100
    
    summary['winner'] = best_model
    
    return summary


def print_benchmark_results(results: Dict[str, Any], summary: Dict[str, Any]):
    """Print comprehensive benchmark results."""
    print("\n" + "="*80)
    print("🏆 BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Model performance table
    print(f"\n{'Model':<15} {'Cat.Acc':<10} {'Ord.Acc':<10} {'QWK':<8} {'MAE':<8} {'Speed(ms)':<12} {'Status':<15}")
    print("-" * 80)
    
    for model_name, metrics in summary['models'].items():
        cat_acc = f"{metrics['categorical_accuracy']:.3f}"
        ord_acc = f"{metrics['ordinal_accuracy']:.3f}"
        qwk = f"{metrics['qwk']:.3f}"
        mae = f"{metrics['mae']:.3f}"
        speed = f"{metrics['inference_time']*1000:.1f}"
        
        # Status based on performance
        if model_name == summary.get('winner'):
            status = "🏆 WINNER"
        elif 'baseline' in model_name.lower():
            status = "✅ Reference"
        else:
            improvement = metrics.get('improvement_vs_baseline', 0)
            if improvement > 0:
                status = f"↑ +{improvement:.1f}%"
            else:
                status = f"↓ {improvement:.1f}%"
        
        print(f"{model_name:<15} {cat_acc:<10} {ord_acc:<10} {qwk:<8} {mae:<8} {speed:<12} {status:<15}")
    
    # Key insights
    if summary.get('winner'):
        print(f"\n🎯 WINNER: {summary['winner'].upper()}")
        winner_metrics = summary['models'][summary['winner']]
        
        if 'improvement_vs_baseline' in winner_metrics:
            improvement = winner_metrics['improvement_vs_baseline']
            print(f"📈 Improvement vs Baseline: {improvement:+.1f}%")
        
        print(f"🎪 Ordinal Accuracy: {winner_metrics['ordinal_accuracy']:.1%}")
        print(f"🎯 QWK Score: {winner_metrics['qwk']:.3f}")
    
    print("\n" + "="*80)


def main():
    """Main function with comprehensive CLI interface."""
    parser = argparse.ArgumentParser(description='Deep-GPCM: Unified Training, Evaluation & Benchmarking')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'benchmark', 'all'],
                        help='Operation mode')
    
    # Model and data selection
    parser.add_argument('--models', type=str, nargs='+', default=['baseline', 'akvmn'],
                        help='Models to use (baseline, akvmn)')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    
    # Evaluation parameters
    parser.add_argument('--model_paths', type=str, nargs='+',
                        help='Paths to trained models for evaluation')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    # Options
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after completion')
    parser.add_argument('--save_models', action='store_true', default=True,
                        help='Save trained models')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Using device: {device}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'train':
        print(f"🚂 TRAINING MODE: {', '.join(args.models)}")
        
        preset_configs = get_preset_configs()
        for model_type in args.models:
            if model_type not in preset_configs:
                print(f"❌ Unknown model type: {model_type}")
                continue
            
            config = preset_configs[model_type]
            config.epochs = args.epochs
            config.batch_size = args.batch_size
            config.learning_rate = args.learning_rate
            
            results = train_model(config, args.dataset, device)
            
            # Save results
            results_file = os.path.join(args.output_dir, f"train_{model_type}_{args.dataset}.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"💾 Results saved: {results_file}")
    
    elif args.mode == 'eval':
        print(f"📊 EVALUATION MODE")
        
        if not args.model_paths:
            print("❌ Please provide --model_paths for evaluation")
            return
        
        # Use evaluate.py functionality
        from evaluate import main as eval_main
        eval_main()
    
    elif args.mode == 'benchmark':
        print(f"🏁 BENCHMARK MODE")
        
        # Run benchmark
        results = benchmark_models(
            args.models, args.dataset, args.epochs, args.batch_size, device
        )
        
        # Create summary
        summary = create_benchmark_summary(results)
        
        # Print results
        print_benchmark_results(results, summary)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_file = os.path.join(args.output_dir, f"benchmark_{args.dataset}_{timestamp}.json")
        
        benchmark_data = {
            'timestamp': timestamp,
            'dataset': args.dataset,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'models': args.models
            },
            'results': results,
            'summary': summary
        }
        
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"💾 Benchmark results saved: {benchmark_file}")
        
        # Generate plots if requested
        if args.plot:
            try:
                from plot_unified import create_benchmark_plot
                plot_file = create_benchmark_plot(benchmark_data, args.output_dir)
                print(f"📊 Plot saved: {plot_file}")
            except ImportError:
                print("📊 Plotting functionality not available")
    
    elif args.mode == 'all':
        print(f"🎯 COMPLETE WORKFLOW MODE")
        
        # 1. Benchmark all models
        print("\n1️⃣ Running benchmark...")
        results = benchmark_models(
            args.models, args.dataset, args.epochs, args.batch_size, device
        )
        
        # 2. Create summary
        summary = create_benchmark_summary(results)
        print_benchmark_results(results, summary)
        
        # 3. Save everything
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"complete_workflow_{args.dataset}_{timestamp}.json")
        
        workflow_data = {
            'timestamp': timestamp,
            'dataset': args.dataset,
            'workflow': 'complete',
            'results': results,
            'summary': summary
        }
        
        with open(output_file, 'w') as f:
            json.dump(workflow_data, f, indent=2)
        
        print(f"💾 Complete workflow results saved: {output_file}")
        
        # 4. Generate plots if requested
        if args.plot:
            try:
                from plot_unified import create_comprehensive_plot
                plot_file = create_comprehensive_plot(workflow_data, args.output_dir)
                print(f"📊 Comprehensive plot saved: {plot_file}")
            except ImportError:
                print("📊 Plotting functionality not available")
    
    print(f"\n✅ {args.mode.upper()} mode completed successfully!")


if __name__ == "__main__":
    main()