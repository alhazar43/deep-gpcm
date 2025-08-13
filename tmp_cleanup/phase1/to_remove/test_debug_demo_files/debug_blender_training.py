#!/usr/bin/env python3
"""
Debug script to isolate the gradient explosion issue in StableThresholdDistanceBlender
during actual training vs. unit tests.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.stable_threshold_blender import StableThresholdDistanceBlender

def test_blender_with_training_conditions():
    """Test the blender under actual training conditions."""
    print("🔍 Testing StableThresholdDistanceBlender Under Training Conditions")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create blender with very conservative settings
    blender = StableThresholdDistanceBlender(
        n_categories=4,
        range_sensitivity_init=0.01,  # Very conservative
        distance_sensitivity_init=0.01,  # Very conservative
        baseline_bias_init=0.0
    ).to(device)
    
    print(f"✓ Blender created with ultra-conservative parameters")
    
    # Test with realistic training data patterns
    batch_size, seq_len, n_thresholds = 64, 32, 3  # Realistic training batch
    
    # Generate data similar to what the model would produce during training
    # These values are based on actual GPCM model outputs
    item_betas = torch.randn(batch_size, seq_len, n_thresholds, device=device) * 0.5  # Conservative scale
    ordinal_taus = torch.randn(n_thresholds, device=device) * 0.5  # Conservative scale  
    student_abilities = torch.randn(batch_size, seq_len, device=device) * 0.5  # Conservative scale
    
    print(f"✓ Generated realistic training data:")
    print(f"  - item_betas range: [{item_betas.min():.3f}, {item_betas.max():.3f}]")
    print(f"  - ordinal_taus range: [{ordinal_taus.min():.3f}, {ordinal_taus.max():.3f}]")
    print(f"  - student_abilities range: [{student_abilities.min():.3f}, {student_abilities.max():.3f}]")
    
    # Enable gradients (training mode)
    item_betas.requires_grad_(True)
    ordinal_taus.requires_grad_(True)
    for param in blender.parameters():
        param.requires_grad_(True)
    
    # Create optimizer (training scenario)
    optimizer = torch.optim.Adam(blender.parameters(), lr=0.001)
    
    try:
        print("\n🧪 Testing Forward Pass with Training Data...")
        
        # Forward pass
        blend_weights = blender.calculate_blend_weights_stable(
            item_betas=item_betas,
            ordinal_taus=ordinal_taus,
            student_abilities=student_abilities
        )
        
        print(f"✓ Forward pass completed")
        print(f"  - blend_weights shape: {blend_weights.shape}")
        print(f"  - blend_weights range: [{blend_weights.min():.6f}, {blend_weights.max():.6f}]")
        
        # Check for NaN/Inf
        if torch.isnan(blend_weights).any() or torch.isinf(blend_weights).any():
            print("🚨 ERROR: NaN/Inf detected in blend_weights!")
            return False
        
        print("\n🧪 Testing Backward Pass with Cross-Entropy Loss...")
        
        # Simulate realistic loss computation (like in training)
        # Create fake GPCM and CORAL probabilities
        gpcm_probs = torch.softmax(torch.randn(batch_size, seq_len, 4, device=device), dim=-1)
        coral_probs = torch.softmax(torch.randn(batch_size, seq_len, 4, device=device), dim=-1)
        
        # Apply adaptive blending (this is what happens in the actual model)
        final_probs = (1 - blend_weights) * gpcm_probs + blend_weights * coral_probs
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        
        # Simulate actual training target
        targets = torch.randint(0, 4, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, device=device)
        
        # Cross-entropy loss (like in training)
        criterion = nn.CrossEntropyLoss()
        probs_flat = final_probs.view(-1, 4)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1).bool()
        
        valid_probs = probs_flat[mask_flat]
        valid_targets = targets_flat[mask_flat]
        
        loss = criterion(valid_probs, valid_targets)
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Analyze gradients
        total_grad_norm = 0
        param_count = 0
        large_grad_params = []
        
        print(f"\n📊 Gradient Analysis:")
        
        # Check blender parameter gradients
        for name, param in blender.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                
                if grad_norm > 100:
                    large_grad_params.append((name, grad_norm))
        
        # Check input tensor gradients
        if item_betas.grad is not None:
            beta_grad_norm = item_betas.grad.norm().item()
            print(f"  item_betas: grad_norm = {beta_grad_norm:.6f}")
            if beta_grad_norm > 1000:
                large_grad_params.append(("item_betas", beta_grad_norm))
        
        if ordinal_taus.grad is not None:
            tau_grad_norm = ordinal_taus.grad.norm().item()
            print(f"  ordinal_taus: grad_norm = {tau_grad_norm:.6f}")
            if tau_grad_norm > 1000:
                large_grad_params.append(("ordinal_taus", tau_grad_norm))
        
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        
        print(f"\n✓ Gradient computation completed")
        print(f"  - Total gradient norm: {total_grad_norm:.3f}")
        print(f"  - Average gradient norm: {avg_grad_norm:.3f}")
        print(f"  - Parameters with gradients: {param_count}")
        
        if large_grad_params:
            print(f"🚨 Large gradient parameters ({len(large_grad_params)}):")
            for name, grad_norm in sorted(large_grad_params, key=lambda x: x[1], reverse=True):
                print(f"    {name}: {grad_norm:.3f}")
        
        # Determine if gradients are stable
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
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_blender_step_by_step():
    """Test each component of the blender separately."""
    print("\n🔍 Testing StableThresholdDistanceBlender Components Step-by-Step")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    blender = StableThresholdDistanceBlender(
        n_categories=4,
        range_sensitivity_init=0.01,
        distance_sensitivity_init=0.01,
        baseline_bias_init=0.0
    ).to(device)
    
    # Test data
    batch_size, seq_len, n_thresholds = 4, 8, 3
    item_betas = torch.randn(batch_size, seq_len, n_thresholds, device=device, requires_grad=True) * 0.5
    ordinal_taus = torch.randn(n_thresholds, device=device, requires_grad=True) * 0.5
    student_abilities = torch.randn(batch_size, seq_len, device=device) * 0.5
    
    try:
        print("📊 Step 1: Testing threshold geometry analysis...")
        
        geometry = blender.analyze_threshold_geometry_stable(item_betas, ordinal_taus)
        
        print(f"✓ Geometry analysis completed")
        for key, value in geometry.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, range=[{value.min():.6f}, {value.max():.6f}]")
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"    🚨 {key} contains NaN/Inf!")
                    return False
        
        print("\n📊 Step 2: Testing BGT transforms...")
        
        from models.stable_threshold_blender import BoundedGeometricTransform
        bgt = BoundedGeometricTransform()
        
        # Test each transform with geometry outputs
        range_term = bgt.stable_range_transform(geometry['range_divergence'])
        distance_term = bgt.stable_distance_transform(geometry['min_distance'])
        correlation_term = bgt.stable_correlation_transform(geometry['threshold_correlation'])
        spread_term = bgt.stable_spread_transform(geometry['distance_spread'])
        
        transforms = {
            'range_term': range_term,
            'distance_term': distance_term, 
            'correlation_term': correlation_term,
            'spread_term': spread_term
        }
        
        for name, tensor in transforms.items():
            print(f"  {name}: range=[{tensor.min():.6f}, {tensor.max():.6f}]")
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"    🚨 {name} contains NaN/Inf!")
                return False
        
        print("\n📊 Step 3: Testing final blend weight computation...")
        
        # Test the final computation step
        blend_weights = blender.calculate_blend_weights_stable(
            item_betas, ordinal_taus, student_abilities
        )
        
        print(f"✓ Blend weights computed")
        print(f"  shape: {blend_weights.shape}")
        print(f"  range: [{blend_weights.min():.6f}, {blend_weights.max():.6f}]")
        
        if torch.isnan(blend_weights).any() or torch.isinf(blend_weights).any():
            print(f"🚨 blend_weights contains NaN/Inf!")
            return False
        
        print("\n📊 Step 4: Testing gradient flow...")
        
        loss = blend_weights.sum()
        loss.backward()
        
        # Check gradient magnitudes
        for name, param in blender.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                if grad_norm > 100:
                    print(f"    🚨 Large gradient in {name}!")
                    return False
        
        print("✅ All components working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error in step-by-step testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_blender_with_training_conditions()
    success2 = test_blender_step_by_step()
    
    if success1 and success2:
        print("\n✅ DEBUG PASSED: StableThresholdDistanceBlender works correctly")
    else:
        print("\n❌ DEBUG FAILED: Issue found in StableThresholdDistanceBlender")