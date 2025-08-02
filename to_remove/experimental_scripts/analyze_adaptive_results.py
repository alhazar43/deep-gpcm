"""
Phase 2 Analysis: Compare Adaptive vs Baseline Results

This script performs comprehensive statistical analysis comparing the enhanced_coral_gpcm_adaptive
results against the baseline coral_gpcm to validate the effectiveness of threshold-distance-based
dynamic blending for solving middle category prediction imbalance.
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def load_results():
    """Load baseline and adaptive results for comparison."""
    
    # Load baseline coral_gpcm results
    with open('results/test/test_results_coral_gpcm_synthetic_OC.json', 'r') as f:
        baseline_results = json.load(f)
    
    # Load adaptive results (will be created after training completes)
    try:
        with open('results/test/test_results_enhanced_coral_gpcm_adaptive_synthetic_OC.json', 'r') as f:
            adaptive_results = json.load(f)
    except FileNotFoundError:
        print("⚠️  Adaptive results not found. Run train_adaptive_experiment.py first.")
        return None, None
    
    return baseline_results, adaptive_results


def extract_key_metrics(results: Dict) -> Dict:
    """Extract key metrics for comparison."""
    if 'final_metrics' in results:
        # Adaptive results format
        metrics = results['final_metrics']
    else:
        # Baseline results format
        metrics = results
    
    return {
        'categorical_accuracy': metrics['categorical_accuracy'],
        'ordinal_accuracy': metrics['ordinal_accuracy'],
        'quadratic_weighted_kappa': metrics['quadratic_weighted_kappa'],
        'mean_absolute_error': metrics['mean_absolute_error'],
        'cat_0_accuracy': metrics['cat_0_accuracy'],
        'cat_1_accuracy': metrics['cat_1_accuracy'],
        'cat_2_accuracy': metrics['cat_2_accuracy'],
        'cat_3_accuracy': metrics['cat_3_accuracy']
    }


def analyze_category_improvements(baseline_metrics: Dict, adaptive_metrics: Dict) -> Dict:
    """Analyze per-category improvements."""
    improvements = {}
    
    for i in range(4):
        cat_key = f'cat_{i}_accuracy'
        baseline_val = baseline_metrics[cat_key]
        adaptive_val = adaptive_metrics[cat_key]
        
        absolute_improvement = adaptive_val - baseline_val
        relative_improvement = (absolute_improvement / baseline_val) * 100
        
        improvements[f'cat_{i}'] = {
            'baseline': baseline_val,
            'adaptive': adaptive_val,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement
        }
    
    return improvements


def check_success_criteria(improvements: Dict) -> Dict:
    """Check if success criteria are met."""
    # Target improvements: Cat 1 >25%, Cat 2 >30%
    cat1_target = 25.0  # 25% relative improvement
    cat2_target = 30.0  # 30% relative improvement
    
    cat1_achieved = improvements['cat_1']['relative_improvement'] >= cat1_target
    cat2_achieved = improvements['cat_2']['relative_improvement'] >= cat2_target
    
    # Check that Cat 0 and Cat 3 don't degrade significantly (>20% loss)
    cat0_stable = improvements['cat_0']['relative_improvement'] >= -20.0
    cat3_stable = improvements['cat_3']['relative_improvement'] >= -20.0
    
    return {
        'cat_1_success': cat1_achieved,
        'cat_2_success': cat2_achieved,
        'cat_0_stable': cat0_stable,
        'cat_3_stable': cat3_stable,
        'overall_success': cat1_achieved and cat2_achieved and cat0_stable and cat3_stable
    }


def statistical_significance_test(baseline_metrics: Dict, adaptive_metrics: Dict) -> Dict:
    """Perform statistical significance testing."""
    # Note: For proper statistical testing, we'd need multiple runs or confidence intervals
    # For now, we'll calculate effect sizes and provide guidelines
    
    results = {}
    
    for metric in ['categorical_accuracy', 'ordinal_accuracy', 'quadratic_weighted_kappa']:
        baseline_val = baseline_metrics[metric]
        adaptive_val = adaptive_metrics[metric]
        
        # Calculate Cohen's d (effect size)
        # For single comparisons, we estimate pooled std from the values
        pooled_std = np.sqrt((baseline_val * (1 - baseline_val) + adaptive_val * (1 - adaptive_val)) / 2)
        if pooled_std > 0:
            cohens_d = abs(adaptive_val - baseline_val) / pooled_std
        else:
            cohens_d = 0
        
        # Effect size interpretation
        if cohens_d < 0.2:
            effect_size = "negligible"
        elif cohens_d < 0.5:
            effect_size = "small"
        elif cohens_d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        results[metric] = {
            'baseline': baseline_val,
            'adaptive': adaptive_val,
            'difference': adaptive_val - baseline_val,
            'cohens_d': cohens_d,
            'effect_size': effect_size
        }
    
    return results


def generate_comparison_report(baseline_results: Dict, adaptive_results: Dict):
    """Generate comprehensive comparison report."""
    
    print("📊 PHASE 2 EXPERIMENTAL VALIDATION RESULTS")
    print("=" * 70)
    
    # Extract metrics
    baseline_metrics = extract_key_metrics(baseline_results)
    adaptive_metrics = extract_key_metrics(adaptive_results)
    
    # Analyze improvements
    improvements = analyze_category_improvements(baseline_metrics, adaptive_metrics)
    
    print("\\n🎯 PER-CATEGORY PERFORMANCE COMPARISON")
    print("-" * 50)
    
    for i in range(4):
        cat_data = improvements[f'cat_{i}']
        baseline_pct = cat_data['baseline'] * 100
        adaptive_pct = cat_data['adaptive'] * 100
        rel_change = cat_data['relative_improvement']
        
        status = "✅" if rel_change > 0 else "❌" if rel_change < -5 else "➖"
        
        print(f"Category {i}:")
        print(f"  Baseline:  {baseline_pct:6.2f}%")
        print(f"  Adaptive:  {adaptive_pct:6.2f}% ({rel_change:+6.2f}%) {status}")
    
    # Check success criteria
    success_criteria = check_success_criteria(improvements)
    
    print("\\n🏆 SUCCESS CRITERIA EVALUATION")
    print("-" * 50)
    
    cat1_improvement = improvements['cat_1']['relative_improvement']
    cat2_improvement = improvements['cat_2']['relative_improvement']
    
    print(f"Cat 1 Target: ≥+25.0% → Achieved: {cat1_improvement:+.1f}% {'✅' if success_criteria['cat_1_success'] else '❌'}")
    print(f"Cat 2 Target: ≥+30.0% → Achieved: {cat2_improvement:+.1f}% {'✅' if success_criteria['cat_2_success'] else '❌'}")
    print(f"Cat 0 Stability: ≥-20.0% → Achieved: {improvements['cat_0']['relative_improvement']:+.1f}% {'✅' if success_criteria['cat_0_stable'] else '❌'}")
    print(f"Cat 3 Stability: ≥-20.0% → Achieved: {improvements['cat_3']['relative_improvement']:+.1f}% {'✅' if success_criteria['cat_3_stable'] else '❌'}")
    
    overall_success = success_criteria['overall_success']
    print(f"\\n🎊 OVERALL SUCCESS: {'✅ ACHIEVED' if overall_success else '❌ NOT ACHIEVED'}")
    
    # Statistical analysis
    print("\\n📈 STATISTICAL ANALYSIS")
    print("-" * 50)
    
    stat_results = statistical_significance_test(baseline_metrics, adaptive_metrics)
    
    for metric, data in stat_results.items():
        print(f"{metric}:")
        print(f"  Baseline: {data['baseline']:.4f}")
        print(f"  Adaptive: {data['adaptive']:.4f}")
        print(f"  Change: {data['difference']:+.4f}")
        print(f"  Effect size: {data['effect_size']} (Cohen's d = {data['cohens_d']:.3f})")
    
    # Adaptive blending analysis
    if 'adaptive_analysis' in adaptive_results and adaptive_results['adaptive_analysis']:
        print("\\n🎛️  ADAPTIVE BLENDING ANALYSIS")
        print("-" * 50)
        
        adaptive_analysis = adaptive_results['adaptive_analysis']
        if adaptive_analysis.get('analysis_available'):
            params = adaptive_analysis['learnable_parameters']
            print(f"Final Learned Parameters:")
            print(f"  Range sensitivity: {params['range_sensitivity']:.4f}")
            print(f"  Distance sensitivity: {params['distance_sensitivity']:.4f}")  
            print(f"  Baseline bias: {params['baseline_bias']:.4f}")
            
            # Check if parameters learned meaningfully
            range_meaningful = abs(params['range_sensitivity'] - 1.0) > 0.1
            dist_meaningful = abs(params['distance_sensitivity'] - 1.0) > 0.1
            bias_meaningful = abs(params['baseline_bias']) > 0.1
            
            print(f"\\nParameter Learning Validation:")
            print(f"  Range sensitivity adapted: {'✅' if range_meaningful else '❌'}")
            print(f"  Distance sensitivity adapted: {'✅' if dist_meaningful else '❌'}")
            print(f"  Baseline bias adapted: {'✅' if bias_meaningful else '❌'}")
            
            learning_success = range_meaningful or dist_meaningful or bias_meaningful
            print(f"  Adaptive learning successful: {'✅' if learning_success else '❌'}")
    
    # Final verdict
    print("\\n" + "=" * 70)
    if overall_success:
        print("🎉 EXPERIMENTAL VALIDATION: SUCCESS!")
        print("✅ Threshold-distance-based dynamic blending successfully solved")
        print("   the middle category prediction imbalance problem!")
        print("\\n📋 Next Steps:")
        print("  • Proceed to Phase 3: Multi-dataset validation")
        print("  • Measure computational overhead")
        print("  • Document results for publication")
    else:
        print("🔬 EXPERIMENTAL VALIDATION: NEEDS REFINEMENT")
        print("❌ Target improvements not fully achieved.")
        print("\\n📋 Recommended Actions:")
        print("  • Analyze learned parameters for insights")
        print("  • Consider hyperparameter tuning")
        print("  • Investigate alternative blending strategies")
        print("  • Extend training duration if needed")
    
    return overall_success, improvements, stat_results


def main():
    """Main analysis function."""
    print("🔬 Loading experimental results for analysis...")
    
    baseline_results, adaptive_results = load_results()
    
    if baseline_results is None or adaptive_results is None:
        return False
    
    print("✅ Results loaded successfully")
    
    # Generate comprehensive comparison report
    success, improvements, statistical_results = generate_comparison_report(baseline_results, adaptive_results)
    
    # Save analysis results
    analysis_output = {
        'timestamp': baseline_results.get('timestamp', 'unknown'),
        'success_achieved': success,
        'category_improvements': improvements,
        'statistical_analysis': statistical_results,
        'baseline_model': 'coral_gpcm (HybridCORALGPCM)',
        'adaptive_model': 'enhanced_coral_gpcm_adaptive',
        'analysis_version': 'Phase_2_Experimental_Validation'
    }
    
    with open('results/analysis_adaptive_vs_baseline_comparison.json', 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)
    
    print(f"\\n💾 Analysis saved to: results/analysis_adaptive_vs_baseline_comparison.json")
    
    return success


if __name__ == "__main__":
    success = main()