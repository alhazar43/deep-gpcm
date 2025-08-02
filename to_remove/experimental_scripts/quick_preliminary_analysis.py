#!/usr/bin/env python3
"""
Quick preliminary analysis of adaptive vs baseline results.
"""

def main():
    print("📊 PRELIMINARY PHASE 2 ANALYSIS")
    print("=" * 50)
    
    # Baseline coral_gpcm results
    baseline = {
        'cat_0_accuracy': 0.7610,
        'cat_1_accuracy': 0.2811, 
        'cat_2_accuracy': 0.2737,
        'cat_3_accuracy': 0.7231,
        'quadratic_weighted_kappa': 0.7077
    }
    
    # Adaptive coral_gpcm results (5 epochs only)
    adaptive = {
        'cat_0_accuracy': 0.7937,
        'cat_1_accuracy': 0.1680,
        'cat_2_accuracy': 0.1938,
        'cat_3_accuracy': 0.7460,
        'quadratic_weighted_kappa': 0.6573
    }
    
    print("\n🎯 PER-CATEGORY COMPARISON (Baseline vs Adaptive @ 5 epochs)")
    print("-" * 60)
    
    for i in range(4):
        cat_key = f'cat_{i}_accuracy'
        baseline_val = baseline[cat_key] * 100
        adaptive_val = adaptive[cat_key] * 100
        
        abs_diff = adaptive_val - baseline_val
        rel_diff = (abs_diff / baseline_val) * 100
        
        status = "✅" if abs_diff > 0 else "❌" if abs_diff < -2 else "➖"
        
        print(f"Category {i}:")
        print(f"  Baseline:  {baseline_val:6.2f}%")
        print(f"  Adaptive:  {adaptive_val:6.2f}% ({rel_diff:+6.2f}%) {status}")
    
    print(f"\nQWK: Baseline {baseline['quadratic_weighted_kappa']:.3f} vs Adaptive {adaptive['quadratic_weighted_kappa']:.3f}")
    
    print("\n🔬 INITIAL OBSERVATIONS")
    print("-" * 30)
    print("❌ Middle categories (1,2) got WORSE with adaptive blending")
    print("   - Cat 1: 28.11% → 16.80% (-40.2% relative decline)")
    print("   - Cat 2: 27.37% → 19.38% (-29.2% relative decline)")
    print("✅ End categories slightly improved")
    print("   - Cat 0: 76.10% → 79.37% (+4.3% relative)")
    print("   - Cat 3: 72.31% → 74.60% (+3.2% relative)")
    print("⚠️  Overall QWK declined: 0.708 → 0.657 (-7.2%)")
    
    print("\n📋 PRELIMINARY CONCLUSION")
    print("-" * 30)
    print("🚨 The current SafeThresholdDistanceBlender is TOO CONSERVATIVE")
    print("   - It's blending uniformly (0.5) across all categories")
    print("   - This makes the model behave more like baseline, not adaptive")
    print("   - Middle categories need HIGHER CORAL weight, not equal blend")
    
    print("\n🎯 NEXT STEPS")
    print("-" * 15)
    print("1. Make the SafeThresholdDistanceBlender actually adaptive")
    print("2. Design it to give middle categories MORE CORAL influence")
    print("3. Re-train and compare against true baseline performance")

if __name__ == "__main__":
    main()