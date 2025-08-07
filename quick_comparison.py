#!/usr/bin/env python3
"""
Quick performance comparison across all models on synthetic_OC dataset
"""
import json
from pathlib import Path

def extract_metrics(file_path):
    """Extract key metrics from test results JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Get final evaluation results
    eval_results = data.get('evaluation_results', {})
    
    return {
        'categorical_accuracy': eval_results.get('categorical_accuracy', 0),
        'ordinal_accuracy': eval_results.get('ordinal_accuracy', 0),
        'qwk': eval_results.get('quadratic_weighted_kappa', 0),
        'mae': eval_results.get('mean_absolute_error', 0),
        'cohen_kappa': eval_results.get('cohen_kappa', 0)
    }

def main():
    # Model results from synthetic_OC dataset
    results_dir = Path('/home/steph/dirt-new/deep-gpcm/results/test/synthetic_OC')
    
    models = {
        'Deep-GPCM': 'test_deep_gpcm.json',
        'Attention-GPCM': 'test_attn_gpcm.json',
        'CORAL-GPCM': 'test_coral_gpcm_proper.json',
        'CORAL-GPCM-Fixed': 'test_coral_gpcm_fixed.json'
    }
    
    comparison_data = []
    
    print("Performance Comparison on synthetic_OC Dataset")
    print("=" * 60)
    
    for model_name, filename in models.items():
        file_path = results_dir / filename
        if file_path.exists():
            metrics = extract_metrics(file_path)
            comparison_data.append({
                'Model': model_name,
                **metrics
            })
    
    # Sort by QWK (primary ordinal metric)
    comparison_data.sort(key=lambda x: x['qwk'], reverse=True)
    
    print(f"{'Model':<20} {'Cat Acc':<8} {'Ord Acc':<8} {'QWK':<8} {'MAE':<8} {'Cohen κ':<8}")
    print("-" * 70)
    
    for row in comparison_data:
        print(f"{row['Model']:<20} {row['categorical_accuracy']:.3f}    {row['ordinal_accuracy']:.3f}    {row['qwk']:.3f}    {row['mae']:.3f}    {row['cohen_kappa']:.3f}")
    
    print("\nKey Findings:")
    print("-" * 30)
    
    if comparison_data:
        # Find best performers
        best_qwk = max(comparison_data, key=lambda x: x['qwk'])
        best_ord_acc = max(comparison_data, key=lambda x: x['ordinal_accuracy'])
        best_cat_acc = max(comparison_data, key=lambda x: x['categorical_accuracy'])
        
        print(f"• Best QWK (ordinal ranking): {best_qwk['Model']} ({best_qwk['qwk']:.3f})")
        print(f"• Best Ordinal Accuracy: {best_ord_acc['Model']} ({best_ord_acc['ordinal_accuracy']:.3f})")
        print(f"• Best Categorical Accuracy: {best_cat_acc['Model']} ({best_cat_acc['categorical_accuracy']:.3f})")
        
        # Ordinal vs baseline comparison
        baseline_models = ['Deep-GPCM', 'Attention-GPCM']
        ordinal_models = ['Attention-GPCM-New', 'CORAL-GPCM', 'CORAL-GPCM-Fixed', 'Test-GPCM']
        
        baseline_qwks = [row['qwk'] for row in comparison_data if row['Model'] in baseline_models]
        ordinal_qwks = [row['qwk'] for row in comparison_data if row['Model'] in ordinal_models]
        
        if baseline_qwks and ordinal_qwks:
            baseline_avg_qwk = sum(baseline_qwks) / len(baseline_qwks)
            ordinal_avg_qwk = sum(ordinal_qwks) / len(ordinal_qwks)
            
            improvement = ((ordinal_avg_qwk - baseline_avg_qwk) / baseline_avg_qwk) * 100
            
            print(f"\nOrdinal Attention Impact:")
            print(f"• Baseline QWK (Deep-GPCM, Attn-GPCM): {baseline_avg_qwk:.3f}")
            print(f"• Ordinal Models QWK: {ordinal_avg_qwk:.3f}")
            print(f"• Improvement: {improvement:+.1f}%")

if __name__ == "__main__":
    main()