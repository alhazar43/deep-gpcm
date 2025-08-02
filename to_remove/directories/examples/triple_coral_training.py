#!/usr/bin/env python3
"""
Example script demonstrating Triple CORAL Loss training for Enhanced CORAL-GPCM models.

This script shows how to train CORAL-enhanced models using the sophisticated triple loss:
1. Cross-Entropy (CE) Loss - for basic categorical prediction
2. Quadratic Weighted Kappa (QWK) Loss - for ordinal consistency  
3. CORAL Loss - for rank consistency in ordinal predictions

The triple loss approach ensures both categorical accuracy and ordinal structure preservation.
"""

import os
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_triple_coral_examples():
    """Run examples of Triple CORAL Loss training."""
    
    print("=" * 80)
    print("TRIPLE CORAL LOSS TRAINING EXAMPLES")
    print("=" * 80)
    print()
    
    print("Triple CORAL Loss combines three complementary objectives:")
    print("  1. Cross-Entropy (CE): Categorical prediction accuracy")
    print("  2. Quadratic Weighted Kappa (QWK): Ordinal consistency") 
    print("  3. CORAL: Rank consistency in ordinal predictions")
    print()
    
    examples = [
        {
            "name": "Standard CORAL-GPCM with Triple Loss",
            "model": "coral_gpcm",
            "description": "Direct IRT-CORAL integration with triple loss",
            "command": [
                "python", "train.py",
                "--model", "coral_gpcm",
                "--dataset", "synthetic_OC",
                "--epochs", "10",
                "--no_cv",
                "--loss", "triple_coral",
                "--triple_ce_weight", "0.33",
                "--triple_qwk_weight", "0.33", 
                "--triple_coral_weight", "0.34"
            ]
        },
        {
            "name": "Enhanced CORAL-GPCM with Adaptive Coupling + Triple Loss",
            "model": "ecoral_gpcm", 
            "description": "Sophisticated threshold coupling with triple loss optimization",
            "command": [
                "python", "train.py",
                "--model", "ecoral_gpcm",
                "--dataset", "synthetic_OC", 
                "--epochs", "10",
                "--no_cv",
                "--loss", "triple_coral",
                "--enable_threshold_coupling",
                "--triple_ce_weight", "0.4",
                "--triple_qwk_weight", "0.3",
                "--triple_coral_weight", "0.3"
            ]
        },
        {
            "name": "Cross-Validation Triple CORAL Training",
            "model": "ecoral_gpcm",
            "description": "5-fold CV with triple loss for robust evaluation",
            "command": [
                "python", "train.py",
                "--model", "ecoral_gpcm",
                "--dataset", "synthetic_OC",
                "--epochs", "15", 
                "--n_folds", "5",
                "--loss", "triple_coral",
                "--enable_threshold_coupling",
                "--triple_ce_weight", "0.33",
                "--triple_qwk_weight", "0.33",
                "--triple_coral_weight", "0.34"
            ]
        },
        {
            "name": "Balanced Triple Loss Configuration",
            "model": "ecoral_gpcm",
            "description": "Equal weighting for all loss components",
            "command": [
                "python", "train.py",
                "--model", "ecoral_gpcm", 
                "--dataset", "synthetic_OC",
                "--epochs", "10",
                "--no_cv",
                "--loss", "triple_coral",
                "--enable_threshold_coupling",
                "--triple_ce_weight", "0.333",
                "--triple_qwk_weight", "0.333",
                "--triple_coral_weight", "0.334"
            ]
        },
        {
            "name": "CORAL-Focused Triple Loss",
            "model": "ecoral_gpcm",
            "description": "Higher CORAL weight for stronger rank consistency",
            "command": [
                "python", "train.py",
                "--model", "ecoral_gpcm",
                "--dataset", "synthetic_OC",
                "--epochs", "10", 
                "--no_cv",
                "--loss", "triple_coral",
                "--enable_threshold_coupling",
                "--triple_ce_weight", "0.25",
                "--triple_qwk_weight", "0.25", 
                "--triple_coral_weight", "0.5"
            ]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Model: {example['model']}")
        print(f"   Description: {example['description']}")
        print(f"   Command:")
        print(f"   {' '.join(example['command'])}")
        print()
    
    print("=" * 80)
    print("LOSS WEIGHT GUIDELINES")
    print("=" * 80)
    print()
    
    print("Recommended weight configurations:")
    print()
    
    print("• Balanced (default): CE=0.33, QWK=0.33, CORAL=0.34")
    print("  - Good starting point for most ordinal tasks")
    print("  - Equal emphasis on all objectives")
    print()
    
    print("• Accuracy-focused: CE=0.5, QWK=0.3, CORAL=0.2") 
    print("  - Prioritizes categorical accuracy")
    print("  - Use when exact category prediction is most important")
    print()
    
    print("• Ordinal-focused: CE=0.2, QWK=0.4, CORAL=0.4")
    print("  - Emphasizes ordinal structure preservation")
    print("  - Use when ordinal relationships are critical")
    print()
    
    print("• Rank-focused: CE=0.25, QWK=0.25, CORAL=0.5")
    print("  - Strongest emphasis on rank consistency")
    print("  - Use when relative ordering is most important")
    print()
    
    print("=" * 80)
    print("ADVANCED FEATURES")
    print("=" * 80)
    print()
    
    print("✅ Adaptive Threshold Coupling (--enable_threshold_coupling)")
    print("   - Sophisticated integration of GPCM β and CORAL τ thresholds")
    print("   - Learnable coupling weights for dynamic balance")
    print("   - NOT simple circumvention but true adaptive integration")
    print()
    
    print("✅ Triple Loss Architecture")
    print("   - CE Loss: Standard categorical cross-entropy")
    print("   - QWK Loss: Differentiable quadratic weighted kappa")
    print("   - CORAL Loss: Rank consistency with cumulative logits")
    print()
    
    print("✅ Model Compatibility")
    print("   - coral_gpcm: Direct IRT-CORAL integration")
    print("   - ecoral_gpcm: Enhanced with adaptive threshold coupling")
    print("   - Both support triple CORAL loss training")
    print()
    
    print("=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print()
    
    print("# Quick start with enhanced model and adaptive coupling:")
    print("python train.py --model ecoral_gpcm --dataset synthetic_OC \\\\")
    print("                --epochs 20 --loss triple_coral \\\\")
    print("                --enable_threshold_coupling")
    print()
    
    print("# Custom loss weights for ordinal focus:")
    print("python train.py --model ecoral_gpcm --dataset synthetic_OC \\\\") 
    print("                --loss triple_coral --enable_threshold_coupling \\\\")
    print("                --triple_ce_weight 0.2 --triple_qwk_weight 0.4 \\\\")
    print("                --triple_coral_weight 0.4")
    print()
    
    print("# Cross-validation for robust evaluation:")
    print("python train.py --model ecoral_gpcm --dataset synthetic_OC \\\\")
    print("                --epochs 15 --n_folds 5 --loss triple_coral \\\\")
    print("                --enable_threshold_coupling")
    print()
    
    print("For more options, run: python train.py --help")

if __name__ == "__main__":
    run_triple_coral_examples()