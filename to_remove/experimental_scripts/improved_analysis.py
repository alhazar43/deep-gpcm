#!/usr/bin/env python3
"""
Improved analysis of adaptive vs baseline results.
"""

def main():
    print("📊 IMPROVED PHASE 2 ANALYSIS")
    print("=" * 50)
    
    # Baseline coral_gpcm results
    baseline = {
        'cat_0_accuracy': 76.10,
        'cat_1_accuracy': 28.11, 
        'cat_2_accuracy': 27.37,
        'cat_3_accuracy': 72.31,
        'quadratic_weighted_kappa': 0.7077
    }
    
    # Improved Adaptive coral_gpcm results (10 epochs)
    adaptive = {
        'cat_0_accuracy': 79.19,
        'cat_1_accuracy': 19.71,
        'cat_2_accuracy': 20.03,
        'cat_3_accuracy': 73.94,
        'quadratic_weighted_kappa': 0.6680
    }
    
    print("\n🎯 PER-CATEGORY COMPARISON (Baseline vs Improved Adaptive @ 10 epochs)")
    print("-" * 65)
    
    for i in range(4):
        cat_key = f'cat_{i}_accuracy'
        baseline_val = baseline[cat_key]
        adaptive_val = adaptive[cat_key]
        
        abs_diff = adaptive_val - baseline_val
        rel_diff = (abs_diff / baseline_val) * 100
        
        status = "✅" if abs_diff > 0 else "❌" if abs_diff < -2 else "➖"
        
        print(f"Category {i}:")
        print(f"  Baseline:  {baseline_val:6.2f}%")
        print(f"  Adaptive:  {adaptive_val:6.2f}% ({rel_diff:+6.2f}%) {status}")
    
    print(f"\nQWK: Baseline {baseline['quadratic_weighted_kappa']:.3f} vs Adaptive {adaptive['quadratic_weighted_kappa']:.3f}")
    
    print("\n🔬 ANALYSIS")
    print("-" * 15)
    print("🔄 MIDDLE CATEGORIES STILL STRUGGLING:")
    print("   - Cat 1: 28.11% → 19.71% (-29.9% relative decline)")
    print("   - Cat 2: 27.37% → 20.03% (-26.8% relative decline)")
    print("✅ End categories improved:")
    print("   - Cat 0: 76.10% → 79.19% (+4.1% relative)")
    print("   - Cat 3: 72.31% → 73.94% (+2.3% relative)")
    print("⚠️  Overall QWK still declined: 0.708 → 0.668 (-5.6%)")
    
    print("\n💡 INSIGHTS")
    print("-" * 12)
    print("📈 PROGRESS: Adaptive blending is category-specific (not uniform 0.5)")
    print("❌ ISSUE: Categories 1&2 are still getting worse despite more CORAL weight")
    print("🤔 HYPOTHESIS: Maybe CORAL itself struggles with middle categories")
    print("   - CORAL ordinal assumption may not match the data distribution")
    print("   - Middle categories might need different strategy entirely")
    
    print("\n🎯 NEXT STEPS")
    print("-" * 15)
    print("1. Analyze CORAL vs GPCM predictions individually for middle cats")
    print("2. Consider inverting the strategy: MORE GPCM for middle categories")
    print("3. Or implement category-specific loss weighting instead of blending")
    print("4. Train for full 30 epochs to see if trend continues")

if __name__ == "__main__":
    main()