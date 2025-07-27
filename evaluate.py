"""
Unified Evaluation Script for Deep-GPCM
Evaluates trained baseline and AKVMN models.
"""

import os
import torch
import numpy as np
import argparse
import json
from typing import Dict, Any

from config import get_model_config
from model_factory import create_model, count_parameters
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader
from utils.loss_utils import create_loss_function
from evaluation.metrics import GpcmMetrics


def load_trained_model(model_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate config
    config_dict = checkpoint['config']
    config = get_model_config(config_dict['model_type'], **config_dict)
    
    # Recreate model
    model = create_model(config, config.n_questions, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config, checkpoint.get('results', {})


def evaluate_model_comprehensive(model: torch.nn.Module, test_loader: UnifiedDataLoader,
                                loss_fn: torch.nn.Module, metrics: GpcmMetrics,
                                device: torch.device, n_cats: int) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    inference_times = []
    total_loss = 0.0
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in test_loader:
            # Time inference
            import time
            start_time = time.time()
            _, _, _, gpcm_probs = model(q_batch, r_batch)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Process outputs
            if mask_batch is not None:
                valid_probs = gpcm_probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = gpcm_probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            # Compute loss
            if hasattr(loss_fn, '__class__') and 'Ordinal' in loss_fn.__class__.__name__:
                loss = loss_fn(valid_probs, valid_targets)
            else:
                loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
            
            total_loss += loss.item()
            
            # Collect data
            predictions = torch.argmax(valid_probs, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(valid_targets.cpu().numpy())
            all_probs.extend(valid_probs.cpu().numpy())
    
    # Convert to numpy
    predictions_np = np.array(all_predictions)
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)
    
    # Compute comprehensive metrics
    eval_metrics = metrics.benchmark_prediction_methods(
        torch.tensor(probs_np), torch.tensor(targets_np), n_cats
    )
    
    # Add additional metrics
    eval_metrics.update({
        'test_loss': total_loss / len(test_loader),
        'avg_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'predictions_distribution': np.bincount(predictions_np, minlength=n_cats).tolist(),
        'targets_distribution': np.bincount(targets_np, minlength=n_cats).tolist()
    })
    
    return eval_metrics


def compare_models(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compare multiple model results."""
    if len(results) < 2:
        return {"comparison": "Need at least 2 models for comparison"}
    
    # Get baseline performance
    baseline_name = None
    baseline_metrics = None
    
    for model_name, model_results in results.items():
        if 'baseline' in model_name.lower():
            baseline_name = model_name
            baseline_metrics = model_results['evaluation']
            break
    
    if baseline_metrics is None:
        # Use first model as baseline
        baseline_name = list(results.keys())[0]
        baseline_metrics = results[baseline_name]['evaluation']
    
    comparison = {
        'baseline_model': baseline_name,
        'comparisons': {}
    }
    
    # Compare each model to baseline
    for model_name, model_results in results.items():
        if model_name == baseline_name:
            continue
        
        model_metrics = model_results['evaluation']
        model_comparison = {}
        
        # Key metrics comparison
        key_metrics = ['categorical_acc', 'ordinal_acc', 'qwk', 'mae', 'avg_inference_time', 'total_parameters']
        
        for metric in key_metrics:
            if metric in baseline_metrics and metric in model_metrics:
                baseline_val = baseline_metrics[metric]
                model_val = model_metrics[metric]
                
                if metric in ['mae']:  # Lower is better
                    improvement = ((baseline_val - model_val) / baseline_val) * 100
                else:  # Higher is better (except parameters and time)
                    if metric in ['avg_inference_time', 'total_parameters']:
                        improvement = ((baseline_val - model_val) / baseline_val) * 100
                    else:
                        improvement = ((model_val - baseline_val) / baseline_val) * 100
                
                model_comparison[metric] = {
                    'baseline': baseline_val,
                    'model': model_val,
                    'improvement_percent': improvement,
                    'better': improvement > 0
                }
        
        comparison['comparisons'][model_name] = model_comparison
    
    return comparison


def print_evaluation_summary(results: Dict[str, Dict[str, Any]], comparison: Dict[str, Any]):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("="*80)
    
    # Individual model results
    for model_name, model_results in results.items():
        eval_metrics = model_results['evaluation']
        
        print(f"\n📊 {model_name.upper()} PERFORMANCE:")
        print("-" * 50)
        print(f"Categorical Accuracy: {eval_metrics['categorical_acc']:.4f}")
        print(f"Ordinal Accuracy: {eval_metrics['ordinal_acc']:.4f}")
        print(f"QWK Score: {eval_metrics['qwk']:.4f}")
        print(f"MAE: {eval_metrics['mae']:.4f}")
        print(f"Parameters: {eval_metrics['total_parameters']:,}")
        print(f"Inference Time: {eval_metrics['avg_inference_time']*1000:.1f}ms")
        
        if 'features' in model_results.get('model_info', {}):
            features = model_results['model_info']['features']
            print(f"Features: {', '.join(features)}")
    
    # Comparison results
    if 'comparisons' in comparison:
        print(f"\n🔬 COMPARATIVE ANALYSIS:")
        print("-" * 50)
        baseline_name = comparison['baseline_model']
        print(f"Baseline: {baseline_name}")
        
        for model_name, comp_data in comparison['comparisons'].items():
            print(f"\n{model_name} vs {baseline_name}:")
            
            for metric, metric_data in comp_data.items():
                improvement = metric_data['improvement_percent']
                symbol = "↑" if metric_data['better'] else "↓"
                color_indicator = "✅" if metric_data['better'] else "❌"
                
                print(f"  {metric}: {symbol} {improvement:+.1f}% {color_indicator}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Evaluation')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Paths to trained model checkpoints')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison analysis')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load test data (assuming same for all models)
    test_path = f"data/{args.dataset}/{args.dataset.lower()}_test.txt"
    test_seqs, test_questions, test_responses, n_cats = load_gpcm_data(test_path)
    
    print(f"Test data: {len(test_seqs)} sequences, {n_cats} categories")
    
    # Create test data loader
    test_loader = UnifiedDataLoader(test_questions, test_responses,
                                  args.batch_size, shuffle=False, device=device)
    
    # Evaluate each model
    results = {}
    metrics = GpcmMetrics()
    
    for model_path in args.models:
        model_name = os.path.basename(model_path).replace('.pth', '')
        
        print(f"\n=== Evaluating {model_name} ===")
        
        try:
            # Load model
            model, config, training_results = load_trained_model(model_path, device)
            
            # Create loss function
            loss_fn = create_loss_function(config.loss_type, n_cats)
            
            # Evaluate
            eval_metrics = evaluate_model_comprehensive(
                model, test_loader, loss_fn, metrics, device, n_cats
            )
            
            # Get model info
            if hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
            else:
                model_info = {"name": config.model_type, "type": config.model_type}
            
            results[model_name] = {
                'model_path': model_path,
                'config': config.__dict__,
                'model_info': model_info,
                'evaluation': eval_metrics,
                'training_results': training_results
            }
            
            print(f"✅ Evaluation completed for {model_name}")
            print(f"   Categorical Accuracy: {eval_metrics['categorical_acc']:.4f}")
            print(f"   Ordinal Accuracy: {eval_metrics['ordinal_acc']:.4f}")
            print(f"   QWK Score: {eval_metrics['qwk']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_name}: {e}")
            continue
    
    # Generate comparison if requested
    comparison = {}
    if args.compare and len(results) > 1:
        comparison = compare_models(results)
    
    # Print summary
    print_evaluation_summary(results, comparison)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    output_file = os.path.join(args.output, f"evaluation_{args.dataset}.json")
    final_results = {
        'dataset': args.dataset,
        'device': str(device),
        'models': results,
        'comparison': comparison
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()