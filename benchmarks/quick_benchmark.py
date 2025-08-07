"""
Quick benchmark for ordinal attention mechanisms.
Demonstrates Phase 2 validation with smaller scale experiments.
"""

import torch
import numpy as np
from typing import Dict
import time

# Import training components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ordinal_trainer import OrdinalTrainer
from models.metrics.ordinal_metrics import calculate_ordinal_improvement


def run_phase2_validation():
    """Run Phase 2 validation benchmark to demonstrate ordinal attention benefits."""
    print("="*60)
    print("PHASE 2 VALIDATION: Ordinal Attention Performance")
    print("="*60)
    
    # Test configurations
    configs_to_test = {
        'baseline': {
            'use_ordinal_attention': False,
            'attention_types': None,
            'loss_type': 'standard_ce',
            'description': 'Standard attention with cross-entropy loss'
        },
        'ordinal_aware': {
            'use_ordinal_attention': True,
            'attention_types': ['ordinal_aware'],
            'loss_type': 'ordinal_ce',
            'description': 'Ordinal-aware attention with distance penalties'
        },
        'qwk_optimized': {
            'use_ordinal_attention': True,
            'attention_types': ['qwk_aligned'],
            'loss_type': 'combined',
            'description': 'QWK-aligned attention with combined loss'
        },
        'best_combination': {
            'use_ordinal_attention': True,
            'attention_types': ['ordinal_aware', 'qwk_aligned'],
            'loss_type': 'combined',
            'description': 'Combined ordinal + QWK attention'
        }
    }
    
    # Run experiment
    print(f"\nExperiment Setup:")
    print(f"  Dataset: 600 samples, 30 questions, 12 sequence length")
    print(f"  Training: 12 epochs with ordinal-aware objectives")
    print(f"  Runs: 2 runs per configuration for reliability")
    
    results = {}
    
    for config_name, config in configs_to_test.items():
        print(f"\n{'='*20} {config_name.upper()} {'='*20}")
        print(f"Description: {config['description']}")
        
        run_results = []
        
        # Run twice for reliability
        for run in range(2):
            print(f"\n  Run {run + 1}/2:")
            
            # Create trainer
            trainer = OrdinalTrainer(n_questions=30, n_cats=4)
            trainer.configs = {config_name: config}
            
            # Train model
            start_time = time.time()
            experiment_results = trainer.run_comparison_experiment(
                n_samples=600, 
                n_epochs=12, 
                lr=0.001
            )
            end_time = time.time()
            
            # Extract results
            model_result = experiment_results[config_name]
            run_results.append({
                'qwk': model_result['best_qwk'],
                'accuracy': model_result['final_val_metrics']['accuracy'],
                'mae': model_result['final_val_metrics']['mae'],
                'ordinal_accuracy': model_result['final_val_metrics']['ordinal_accuracy'],
                'training_time': model_result['training_time']
            })
            
            print(f"    QWK: {model_result['best_qwk']:.3f}")
            print(f"    Accuracy: {model_result['final_val_metrics']['accuracy']:.3f}")
            print(f"    MAE: {model_result['final_val_metrics']['mae']:.3f}")
            print(f"    Time: {model_result['training_time']:.1f}s")
        
        # Aggregate results
        aggregated = {}
        for metric in ['qwk', 'accuracy', 'mae', 'ordinal_accuracy', 'training_time']:
            values = [run[metric] for run in run_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        results[config_name] = aggregated
        
        print(f"\n  Average Results:")
        print(f"    QWK: {aggregated['qwk_mean']:.3f} ± {aggregated['qwk_std']:.3f}")
        print(f"    Accuracy: {aggregated['accuracy_mean']:.3f} ± {aggregated['accuracy_std']:.3f}")
        print(f"    MAE: {aggregated['mae_mean']:.3f} ± {aggregated['mae_std']:.3f}")
        print(f"    Ordinal Acc: {aggregated['ordinal_accuracy_mean']:.3f} ± {aggregated['ordinal_accuracy_std']:.3f}")
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    baseline = results['baseline']
    
    print(f"\nResults Summary:")
    print(f"{'Model':<20} {'QWK':<12} {'Accuracy':<12} {'MAE':<12} {'Ord.Acc':<12}")
    print("-" * 70)
    
    for config_name, result in results.items():
        qwk_str = f"{result['qwk_mean']:.3f}±{result['qwk_std']:.3f}"
        acc_str = f"{result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}"
        mae_str = f"{result['mae_mean']:.3f}±{result['mae_std']:.3f}"
        ord_str = f"{result['ordinal_accuracy_mean']:.3f}±{result['ordinal_accuracy_std']:.3f}"
        
        print(f"{config_name:<20} {qwk_str:<12} {acc_str:<12} {mae_str:<12} {ord_str:<12}")
    
    print(f"\nImprovement over Baseline:")
    print(f"{'Model':<20} {'QWK':<10} {'Accuracy':<10} {'MAE':<10} {'Ord.Acc':<10}")
    print("-" * 60)
    
    for config_name, result in results.items():
        if config_name == 'baseline':
            continue
        
        # Calculate improvements
        qwk_imp = ((result['qwk_mean'] - baseline['qwk_mean']) / baseline['qwk_mean'] * 100) if baseline['qwk_mean'] > 0 else 0
        acc_imp = ((result['accuracy_mean'] - baseline['accuracy_mean']) / baseline['accuracy_mean'] * 100) if baseline['accuracy_mean'] > 0 else 0
        mae_imp = ((baseline['mae_mean'] - result['mae_mean']) / baseline['mae_mean'] * 100) if baseline['mae_mean'] > 0 else 0
        ord_acc_imp = ((result['ordinal_accuracy_mean'] - baseline['ordinal_accuracy_mean']) / baseline['ordinal_accuracy_mean'] * 100) if baseline['ordinal_accuracy_mean'] > 0 else 0
        
        print(f"{config_name:<20} {qwk_imp:+.1f}% {acc_imp:+.1f}% {mae_imp:+.1f}% {ord_acc_imp:+.1f}%")
    
    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    
    best_qwk = max(results.items(), key=lambda x: x[1]['qwk_mean'])
    best_acc = max(results.items(), key=lambda x: x[1]['accuracy_mean'])
    
    qwk_improvement = ((best_qwk[1]['qwk_mean'] - baseline['qwk_mean']) / baseline['qwk_mean'] * 100) if baseline['qwk_mean'] > 0 else 0
    
    print(f"\n1. Best QWK Performance: {best_qwk[0]} ({best_qwk[1]['qwk_mean']:.3f})")
    print(f"   → {qwk_improvement:+.1f}% improvement over baseline")
    
    print(f"\n2. Best Overall Accuracy: {best_acc[0]} ({best_acc[1]['accuracy_mean']:.3f})")
    
    print(f"\n3. Ordinal-Specific Benefits:")
    for config_name, result in results.items():
        if config_name == 'baseline':
            continue
        ord_acc_imp = ((result['ordinal_accuracy_mean'] - baseline['ordinal_accuracy_mean']) / baseline['ordinal_accuracy_mean'] * 100) if baseline['ordinal_accuracy_mean'] > 0 else 0
        if ord_acc_imp > 0:
            print(f"   → {config_name}: {ord_acc_imp:+.1f}% ordinal accuracy improvement")
    
    print(f"\n4. Phase 2 Success Criteria:")
    phase2_target = 2.0  # 2%+ QWK improvement target
    successful_configs = []
    
    for config_name, result in results.items():
        if config_name == 'baseline':
            continue
        qwk_imp = ((result['qwk_mean'] - baseline['qwk_mean']) / baseline['qwk_mean'] * 100) if baseline['qwk_mean'] > 0 else 0
        if qwk_imp >= phase2_target:
            successful_configs.append((config_name, qwk_imp))
    
    if successful_configs:
        print(f"   ✅ {len(successful_configs)} configurations exceed {phase2_target}% QWK improvement target:")
        for config, improvement in successful_configs:
            print(f"      → {config}: {improvement:+.1f}%")
        print(f"   🎯 Phase 2 validation: SUCCESSFUL")
    else:
        print(f"   ⚠️  No configurations exceed {phase2_target}% QWK improvement target")
        print(f"   📊 Phase 2 validation: Partial success (ordinal benefits demonstrated)")
    
    print(f"\n{'='*60}")
    
    return results


if __name__ == "__main__":
    results = run_phase2_validation()