#!/usr/bin/env python3
"""
Example usage of threshold coupling in CORAL-GPCM models.

This example demonstrates how to use the clean threshold coupling system
implemented following the TODO.md simplification principles.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.factory import create_model
from models.threshold_coupling import ThresholdCouplingConfig


def example_coral_gpcm_with_coupling():
    """Example: CORAL-GPCM with threshold coupling enabled by default."""
    print("Example 1: CORAL-GPCM with Default Threshold Coupling")
    print("=" * 55)
    
    # coral_gpcm enables threshold coupling by default
    model = create_model(
        model_type='coral_gpcm',
        n_questions=100,
        n_cats=4,
        memory_size=50,
        final_fc_dim=50
    )
    
    # Check coupling configuration
    coupling_info = model.get_coupling_info()
    print(f"Coupling enabled: {coupling_info['coupling_enabled']}")
    print(f"Coupling type: {coupling_info['coupling_type']}")
    print(f"GPCM weight: {coupling_info['config']['gpcm_weight']}")
    print(f"CORAL weight: {coupling_info['config']['coral_weight']}")
    
    if 'current_weights' in coupling_info:
        current = coupling_info['current_weights']
        print(f"Current learned weights: GPCM={current['gpcm_weight']:.3f}, CORAL={current['coral_weight']:.3f}")
    
    return model


def example_coral_with_custom_coupling():
    """Example: CORAL model with custom coupling configuration."""
    print("\nExample 2: CORAL Model with Custom Coupling Configuration")
    print("=" * 58)
    
    # CORAL model with explicit coupling configuration
    model = create_model(
        model_type='coral',
        n_questions=100,
        n_cats=4,
        enable_threshold_coupling=True,
        coupling_type='linear',
        gpcm_weight=0.8,  # Give more weight to GPCM thresholds
        coral_weight=0.2   # Less weight to CORAL thresholds
    )
    
    coupling_info = model.get_coupling_info()
    print(f"Custom GPCM weight: {coupling_info['config']['gpcm_weight']}")
    print(f"Custom CORAL weight: {coupling_info['config']['coral_weight']}")
    
    return model


def example_coral_without_coupling():
    """Example: CORAL model without threshold coupling."""
    print("\nExample 3: CORAL Model without Threshold Coupling")
    print("=" * 50)
    
    # CORAL model with coupling explicitly disabled
    model = create_model(
        model_type='coral',
        n_questions=100,
        n_cats=4,
        enable_threshold_coupling=False
    )
    
    coupling_info = model.get_coupling_info()
    print(f"Coupling enabled: {coupling_info is not None}")
    
    return model


def example_training_integration():
    """Example: How to use in training pipeline."""
    print("\nExample 4: Training Pipeline Integration")
    print("=" * 42)
    
    print("Training with threshold coupling:")
    print("python train.py --model coral_gpcm --dataset synthetic_OC --epochs 30")
    print("  → Threshold coupling enabled by default")
    print()
    
    print("Training CORAL model with custom coupling:")
    print("python train.py --model coral --dataset synthetic_OC --epochs 30 \\")
    print("    --enable_threshold_coupling --gpcm_weight 0.8 --coral_weight 0.2")
    print("  → Custom coupling weights")
    print()
    
    print("Training CORAL model without coupling:")
    print("python train.py --model coral --dataset synthetic_OC --epochs 30")
    print("  → No coupling by default for 'coral' model type")


def main():
    """Run all examples."""
    try:
        model1 = example_coral_gpcm_with_coupling()
        model2 = example_coral_with_custom_coupling()
        model3 = example_coral_without_coupling()
        example_training_integration()
        
        print("\n" + "=" * 60)
        print("🎉 All threshold coupling examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Clean LinearThresholdCoupler implementation")
        print("  • Configuration-driven factory pattern")
        print("  • Default coupling for coral_gpcm model")
        print("  • Custom coupling weights for coral model")
        print("  • Backward compatibility (coral without coupling)")
        print("  • Simple training pipeline integration")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()