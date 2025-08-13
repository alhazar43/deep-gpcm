#!/usr/bin/env python3
"""
Debug script to test the exact parameter configuration used in adaptive_coral_gpcm
when enable_threshold_coupling=False.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.stable_threshold_blender import StableThresholdDistanceBlender

def test_model_integration_scenario():
    """Test the exact scenario used in coral_gpcm.py lines 398-403."""
    print("🔍 Testing Model Integration Scenario (enable_threshold_coupling=False)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create blender with exact model parameters
    blender = StableThresholdDistanceBlender(
        n_categories=4,
        range_sensitivity_init=0.01,
        distance_sensitivity_init=0.01,
        baseline_bias_init=0.0
    ).to(device)
    
    # Realistic model dimensions from actual training
    batch_size, seq_len, n_cats = 64, 32, 4
    n_thresholds = n_cats - 1
    
    # Generate parameters matching the model architecture
    # These match what the model actually produces during training
    item_thresholds = torch.randn(batch_size, seq_len, n_thresholds, device=device, requires_grad=True) * 0.5
    coral_bias = torch.randn(n_thresholds, device=device, requires_grad=True) * 0.5  # coral_projection.bias
    student_abilities = torch.randn(batch_size, seq_len, device=device, requires_grad=True) * 0.5
    discrimination_params = torch.ones(batch_size, seq_len, device=device, requires_grad=True)
    
    print(f"✓ Generated model parameters:")
    print(f"  - item_thresholds: {item_thresholds.shape}, range [{item_thresholds.min():.3f}, {item_thresholds.max():.3f}]")
    print(f"  - coral_bias: {coral_bias.shape}, range [{coral_bias.min():.3f}, {coral_bias.max():.3f}]")
    print(f"  - student_abilities: {student_abilities.shape}, range [{student_abilities.min():.3f}, {student_abilities.max():.3f}]")
    print(f"  - discrimination_params: {discrimination_params.shape}, range [{discrimination_params.min():.3f}, {discrimination_params.max():.3f}]")
    
    # Test the exact function call used in the model (line 398-403)
    try:
        print("\\n🧪 Testing BGT function call with model parameters...")
        
        blend_weights = blender.calculate_blend_weights_stable(
            item_betas=item_thresholds,                 # GPCM β parameters
            ordinal_taus=coral_bias,                    # CORAL τ parameters  
            student_abilities=student_abilities,        # θ parameters
            discrimination_alphas=discrimination_params  # α parameters
        )
        
        print(f"✓ BGT function call succeeded")
        print(f"  - blend_weights: {blend_weights.shape}, range [{blend_weights.min():.6f}, {blend_weights.max():.6f}]")
        
        # Check for numerical issues
        if torch.isnan(blend_weights).any() or torch.isinf(blend_weights).any():
            print("🚨 ERROR: NaN/Inf detected in blend_weights!")
            return False
        
        print("\\n🧪 Testing gradient computation with realistic loss...")
        
        # Simulate the full blending process as done in the model
        # Generate fake GPCM and CORAL probabilities 
        gpcm_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
        coral_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
        
        # Apply adaptive blending (exact same logic as _apply_adaptive_blending)
        final_probs = (1 - blend_weights) * gpcm_probs + blend_weights * coral_probs
        
        # Robust probability normalization 
        prob_sums = final_probs.sum(dim=-1, keepdim=True)
        final_probs = final_probs / torch.clamp(prob_sums, min=1e-7)
        final_probs = torch.clamp(final_probs, min=1e-7, max=1.0 - 1e-7)
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        
        # Compute loss (cross-entropy as used in training)
        targets = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, device=device).bool()
        
        criterion = nn.CrossEntropyLoss()
        probs_flat = final_probs.view(-1, n_cats)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        valid_probs = probs_flat[mask_flat]
        valid_targets = targets_flat[mask_flat]
        
        loss = criterion(valid_probs, valid_targets)
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Analyze gradients
        print(f"\\n📊 Gradient Analysis:")
        
        total_grad_norm = 0
        param_count = 0
        large_grad_params = []
        
        # Check blender gradients
        for name, param in blender.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                if grad_norm > 100:
                    large_grad_params.append((name, grad_norm))
        
        # Check model parameter gradients
        model_params = [
            (item_thresholds, "item_thresholds"),
            (coral_bias, "coral_bias"),
            (student_abilities, "student_abilities"),
            (discrimination_params, "discrimination_params")
        ]
        
        for param, name in model_params:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                if grad_norm > 1000:
                    large_grad_params.append((name, grad_norm))
        
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        
        print(f"\\n✓ Gradient computation completed")
        print(f"  - Total blender gradient norm: {total_grad_norm:.3f}")
        print(f"  - Average blender gradient norm: {avg_grad_norm:.3f}")
        
        if large_grad_params:
            print(f"🚨 Large gradients detected ({len(large_grad_params)}):")
            for name, grad_norm in sorted(large_grad_params, key=lambda x: x[1], reverse=True):
                print(f"    {name}: {grad_norm:.3f}")
        
        # Determine stability
        if avg_grad_norm > 1000 or any(grad_norm > 10000 for _, grad_norm in large_grad_params):
            print("🚨 GRADIENT EXPLOSION DETECTED!")
            return False
        elif avg_grad_norm > 10:
            print("⚠️  Large gradients detected but not explosive")
            return True
        else:
            print("✅ Gradients are stable!")
            return True
            
    except Exception as e:
        print(f"❌ Error during model integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_integration_scenario()
    
    if success:
        print("\\n✅ MODEL INTEGRATION TEST PASSED")
        print("The BGT solution works correctly with the model parameters.")
    else:
        print("\\n❌ MODEL INTEGRATION TEST FAILED") 
        print("Issue found in BGT integration with model parameters.")