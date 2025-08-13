#!/usr/bin/env python3
"""
Debug script to analyze parameter initialization differences between 
working debug tests and failing training.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import create_model
import numpy as np

def analyze_model_parameters():
    """Compare parameter initialization between different model creation scenarios."""
    print("🔍 Analyzing Model Parameter Initialization")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Create model like debug tests (works)
    print("1️⃣ Creating model like debug tests (WORKS)...")
    model1 = create_model(
        model_type="adaptive_coral_gpcm",
        n_questions=50,
        n_cats=4,
        memory_size=20,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy="linear_decay",
        ability_scale=1.0,
        use_discrimination=True,
        dropout_rate=0.1
    ).to(device)
    
    print("✓ Model 1 created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model1.parameters())}")
    
    # Analyze BGT parameters
    if hasattr(model1, 'threshold_blender'):
        blender1 = model1.threshold_blender
        print(f"  - BGT range_sensitivity: {blender1.range_sensitivity.item():.6f}")
        print(f"  - BGT distance_sensitivity: {blender1.distance_sensitivity.item():.6f}")
        print(f"  - BGT baseline_bias: {blender1.baseline_bias.item():.6f}")
        
        # Check parameter bounds
        print(f"  - Range bounds: [{blender1.range_bounds[0]:.1f}, {blender1.range_bounds[1]:.1f}]")
        print(f"  - Distance bounds: [{blender1.distance_bounds[0]:.1f}, {blender1.distance_bounds[1]:.1f}]")
        print(f"  - Bias bounds: [{blender1.bias_bounds[0]:.1f}, {blender1.bias_bounds[1]:.1f}]")
    
    # Test 2: Create model like train.py (fails)
    print(f"\\n2️⃣ Creating model like train.py (FAILS)...")
    
    # Check if there are any differences in the factory call
    # Looking at train.py to see if there are different parameters
    model2 = create_model(
        model_type="adaptive_coral_gpcm",
        n_questions=50,  # Same as loaded data
        n_cats=4,        # Same as loaded data
        memory_size=20,  # Default from train.py
        key_dim=50,      # Default from train.py
        value_dim=200,   # Default from train.py
        final_fc_dim=50, # Default from train.py
        embedding_strategy="linear_decay",  # Default from train.py
        ability_scale=1.0,        # Default from train.py
        use_discrimination=True,  # Default from train.py
        dropout_rate=0.1         # Default from train.py
    ).to(device)
    
    print("✓ Model 2 created successfully")
    print(f"  - Total parameters: {sum(p.numel() for p in model2.parameters())}")
    
    # Compare BGT parameters
    if hasattr(model2, 'threshold_blender'):
        blender2 = model2.threshold_blender
        print(f"  - BGT range_sensitivity: {blender2.range_sensitivity.item():.6f}")
        print(f"  - BGT distance_sensitivity: {blender2.distance_sensitivity.item():.6f}")
        print(f"  - BGT baseline_bias: {blender2.baseline_bias.item():.6f}")
    
    # Test 3: Check parameter ranges and any potential issues
    print(f"\\n3️⃣ Parameter Analysis...")
    
    # Check all model parameters for unusual values
    param_stats = {}
    for name, param in model1.named_parameters():
        param_data = param.data.cpu().numpy()
        param_stats[name] = {
            'mean': np.mean(param_data),
            'std': np.std(param_data),
            'min': np.min(param_data),
            'max': np.max(param_data),
            'shape': param.shape
        }
    
    # Look for parameters with extreme values
    print("Parameters with potentially problematic ranges:")
    for name, stats in param_stats.items():
        if abs(stats['max']) > 10 or abs(stats['min']) > 10 or stats['std'] > 5:
            print(f"  ⚠️  {name}: range=[{stats['min']:.6f}, {stats['max']:.6f}], std={stats['std']:.6f}")
        elif 'threshold_blender' in name:
            print(f"  🎯 {name}: {stats['mean']:.6f} (BGT parameter)")
    
    # Test 4: Check initialization seed differences
    print(f"\\n4️⃣ Testing with different random seeds...")
    
    seeds_to_test = [42, 0, 123, 999]
    for seed in seeds_to_test:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model_seed = create_model(
            model_type="adaptive_coral_gpcm",
            n_questions=50,
            n_cats=4,
            memory_size=20,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50,
            embedding_strategy="linear_decay",
            ability_scale=1.0,
            use_discrimination=True,
            dropout_rate=0.1
        ).to(device)
        
        if hasattr(model_seed, 'threshold_blender'):
            blender_seed = model_seed.threshold_blender
            range_val = blender_seed.range_sensitivity.item()
            dist_val = blender_seed.distance_sensitivity.item()
            bias_val = blender_seed.baseline_bias.item()
            
            print(f"  Seed {seed}: range={range_val:.6f}, dist={dist_val:.6f}, bias={bias_val:.6f}")
            
            # Check if any values are outside expected bounds
            if range_val < 0.01 or range_val > 2.0:
                print(f"    ⚠️  Range sensitivity out of bounds!")
            if dist_val < 0.01 or dist_val > 3.0:
                print(f"    ⚠️  Distance sensitivity out of bounds!")
            if bias_val < -1.0 or bias_val > 1.0:
                print(f"    ⚠️  Baseline bias out of bounds!")
    
    # Test 5: Check if the model architecture is identical
    print(f"\\n5️⃣ Model Architecture Comparison...")
    
    arch1 = str(model1).replace('\\n', '').replace(' ', '')
    arch2 = str(model2).replace('\\n', '').replace(' ', '')
    
    if arch1 == arch2:
        print("✅ Model architectures are identical")
    else:
        print("❌ Model architectures differ!")
        print("This could explain the training differences.")
    
    # Test 6: Simple forward pass to check for immediate numerical issues
    print(f"\\n6️⃣ Quick Forward Pass Test...")
    
    # Create small test batch
    test_questions = torch.randint(0, 50, (2, 10), device=device)
    test_responses = torch.randint(0, 4, (2, 10), device=device).float()
    
    model1.eval()
    try:
        with torch.no_grad():
            output1 = model1(test_questions, test_responses)
            print("✅ Model 1 forward pass successful")
            
            # Check output ranges
            _, _, _, probs1 = output1
            print(f"  - Probabilities range: [{probs1.min():.6f}, {probs1.max():.6f}]")
            
            if torch.isnan(probs1).any() or torch.isinf(probs1).any():
                print("  ⚠️  NaN/Inf detected in output!")
            else:
                print("  ✅ Output is numerically stable")
                
    except Exception as e:
        print(f"❌ Model 1 forward pass failed: {e}")
    
    return param_stats

if __name__ == "__main__":
    param_stats = analyze_model_parameters()
    
    print(f"\\n📊 Parameter Analysis Complete")
    print(f"Check above for any parameters with unusual initialization values.")