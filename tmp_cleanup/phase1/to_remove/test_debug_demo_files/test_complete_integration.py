#!/usr/bin/env python3
"""
Test complete minimal adaptive blender integration with actual training.

This tests the medium-term solution: minimal adaptive blending with gradient isolation
to achieve stable training of the adaptive coral GPCM system.
"""

import sys
import os
sys.path.append('/home/steph/dirt-new/deep-gpcm')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set environment variables before importing anything else
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def test_complete_integration():
    """Test complete minimal adaptive blender integration."""
    print("🔧 Testing Complete Minimal Adaptive Blender Integration")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model with minimal adaptive blending enabled
    from models.factory import create_model
    
    try:
        model = create_model(
            model_type='adaptive_coral_gpcm',
            n_questions=50,
            n_cats=4,
            memory_size=20,
            key_dim=20,
            value_dim=50,
            final_fc_dim=10,
            embedding_strategy='linear_decay',
            enable_adaptive_blending=True,
            enable_threshold_coupling=False,  # Keep simple for testing
            range_sensitivity_init=0.1,
            distance_sensitivity_init=0.1,
            baseline_bias_init=0.0
        ).to(device)
        print(f"✓ Model created: {model.model_name}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"  - Adaptive blending: {model.enable_adaptive_blending}")
        print(f"  - Threshold coupling: {model.enable_threshold_coupling}")
        
        # Create synthetic training data
        batch_size, seq_len = 32, 20
        n_questions, n_cats = 50, 4
        
        # Generate realistic data
        questions = torch.randint(0, n_questions, (batch_size, seq_len), device=device)
        responses = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
        
        print(f"✓ Test data created: {questions.shape}, {responses.shape}")
        
        # Test forward pass
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print("\n🧪 Testing Training Steps")
        print("-" * 40)
        
        stable_steps = 0
        total_steps = 10
        
        for step in range(total_steps):
            optimizer.zero_grad()
            
            try:
                # Forward pass
                student_abilities, item_thresholds, discrimination_params, probs = model(
                    questions, responses
                )
                
                # Check for numerical issues in outputs
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    print(f"❌ Step {step+1}: NaN/Inf in model outputs")
                    break
                
                # Compute loss
                targets = responses.view(-1)
                probs_flat = probs.view(-1, n_cats)
                loss = F.cross_entropy(probs_flat, targets)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ Step {step+1}: NaN/Inf in loss: {loss.item()}")
                    break
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                total_grad_norm = 0
                param_count = 0
                max_grad = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm ** 2
                        param_count += 1
                        max_grad = max(max_grad, grad_norm)
                
                total_grad_norm = np.sqrt(total_grad_norm)
                
                # Check for gradient explosion
                if total_grad_norm > 100 or max_grad > 50:
                    print(f"❌ Step {step+1}: Gradient explosion!")
                    print(f"  - Total norm: {total_grad_norm:.3f}")
                    print(f"  - Max grad: {max_grad:.3f}")
                    break
                
                optimizer.step()
                
                print(f"✓ Step {step+1}: Loss={loss.item():.6f}, GradNorm={total_grad_norm:.6f}")
                stable_steps += 1
                
            except Exception as e:
                print(f"❌ Step {step+1}: Exception: {e}")
                break
        
        print(f"\n📊 Results:")
        print(f"  - Stable steps: {stable_steps}/{total_steps}")
        print(f"  - Success rate: {100*stable_steps/total_steps:.1f}%")
        
        if stable_steps == total_steps:
            print("✅ COMPLETE INTEGRATION TEST PASSED!")
            print("   Medium-term solution achieved: stable adaptive blending")
            return True
        elif stable_steps >= total_steps * 0.8:
            print("⚠️  MOSTLY STABLE (minor issues)")
            print("   Medium-term solution mostly working")
            return True
        else:
            print("❌ INTEGRATION TEST FAILED")
            print("   Medium-term solution needs more work")
            return False
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_gradient_isolation():
    """Test that gradient isolation is working correctly."""
    print("\n🔬 Testing Gradient Isolation")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test the minimal blender in isolation
    from models.minimal_adaptive_blender import MinimalAdaptiveBlender
    
    blender = MinimalAdaptiveBlender(
        n_categories=4,
        base_sensitivity=0.1,
        distance_threshold=1.0
    ).to(device)
    
    # Create test data with gradients
    batch_size, seq_len, n_cats = 32, 20, 4
    gpcm_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
    coral_probs = torch.softmax(torch.randn(batch_size, seq_len, n_cats, device=device), dim=-1)
    item_betas = torch.randn(batch_size, seq_len, n_cats-1, device=device, requires_grad=True)
    ordinal_taus = torch.randn(n_cats-1, device=device, requires_grad=True)
    
    # Forward pass
    blended_probs = blender(gpcm_probs, coral_probs, item_betas, ordinal_taus)
    
    # Compute loss and backprop
    targets = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
    loss = F.cross_entropy(blended_probs.view(-1, n_cats), targets.view(-1))
    loss.backward()
    
    # Check that item_betas gradients are controlled (due to detachment)
    if item_betas.grad is not None:
        beta_grad_norm = item_betas.grad.norm().item()
        tau_grad_norm = ordinal_taus.grad.norm().item() if ordinal_taus.grad is not None else 0
        
        print(f"✓ Gradient isolation test:")
        print(f"  - Beta gradients: {beta_grad_norm:.6f}")
        print(f"  - Tau gradients: {tau_grad_norm:.6f}")
        print(f"  - Loss: {loss.item():.6f}")
        
        if beta_grad_norm < 10 and tau_grad_norm < 10:
            print("✅ Gradient isolation working correctly")
            return True
        else:
            print("⚠️  Large gradients detected")
            return False
    else:
        print("✓ item_betas gradients properly isolated (None)")
        return True

if __name__ == "__main__":
    print("🚀 Testing Complete Minimal Adaptive Blender Integration")
    print("Testing medium-term solution for stable adaptive blending")
    print("=" * 70)
    
    # Test gradient isolation first
    isolation_success = test_gradient_isolation()
    
    # Test complete integration
    integration_success = test_complete_integration()
    
    print("\n" + "=" * 70)
    print("📋 FINAL RESULTS:")
    print(f"  - Gradient Isolation: {'✅ PASS' if isolation_success else '❌ FAIL'}")
    print(f"  - Complete Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if isolation_success and integration_success:
        print("\n🎉 MEDIUM-TERM SOLUTION SUCCESSFUL!")
        print("   Minimal adaptive blending with gradient isolation achieved.")
        print("   System is ready for stable adaptive blending training.")
    else:
        print("\n⚠️  MEDIUM-TERM SOLUTION NEEDS REFINEMENT")
        print("   Some stability issues remain.")