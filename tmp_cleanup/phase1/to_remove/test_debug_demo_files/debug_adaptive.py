#!/usr/bin/env python3
"""
Debug script to isolate the gradient explosion issue in adaptive_coral_gpcm.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import create_model

def debug_adaptive_model():
    """Debug the adaptive model step by step."""
    print("🔍 Debugging Adaptive CORAL GPCM Model")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = create_model('adaptive_coral_gpcm', n_questions=50, n_cats=4)
    model = model.to(device)
    
    print(f"✓ Model created: {type(model).__name__}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create small test data
    batch_size, seq_len = 4, 8
    questions = torch.randint(0, 50, (batch_size, seq_len)).to(device)
    responses = torch.randint(0, 4, (batch_size, seq_len)).to(device)
    
    print(f"✓ Test data: batch_size={batch_size}, seq_len={seq_len}")
    
    # Enable gradient tracking for all parameters
    for param in model.parameters():
        param.requires_grad_(True)
    
    try:
        print("\n🧪 Testing forward pass...")
        
        # Forward pass
        student_abilities, item_thresholds, discrimination_params, final_probs = model(questions, responses)
        
        print(f"✓ Forward pass successful")
        print(f"  - student_abilities shape: {student_abilities.shape}")
        print(f"  - item_thresholds shape: {item_thresholds.shape}")
        print(f"  - final_probs shape: {final_probs.shape}")
        print(f"  - final_probs range: [{final_probs.min():.6f}, {final_probs.max():.6f}]")
        
        # Check for NaN/Inf in outputs
        if torch.isnan(final_probs).any() or torch.isinf(final_probs).any():
            print("🚨 WARNING: NaN/Inf detected in model outputs!")
            return False
        
        print("\n🧪 Testing backward pass...")
        
        # Simple loss for gradient computation
        loss = final_probs.sum()
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        param_count = 0
        large_grad_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                
                if grad_norm > 100:
                    large_grad_params.append((name, grad_norm))
        
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        
        print(f"✓ Gradient computation completed")
        print(f"  - Total gradient norm: {total_grad_norm:.3f}")
        print(f"  - Average gradient norm: {avg_grad_norm:.3f}")
        print(f"  - Parameters with gradients: {param_count}")
        
        if large_grad_params:
            print(f"🚨 Large gradient parameters ({len(large_grad_params)}):")
            for name, grad_norm in sorted(large_grad_params, key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {name}: {grad_norm:.3f}")
        
        if avg_grad_norm > 1000:
            print("🚨 GRADIENT EXPLOSION DETECTED!")
            return False
        else:
            print("✅ Gradients are stable!")
            return True
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def compare_with_base_model():
    """Compare with a stable base model."""
    print("\n🔍 Comparing with Base CORAL GPCM Model")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base model for comparison
    base_model = create_model('coral_gpcm', n_questions=50, n_cats=4)
    base_model = base_model.to(device)
    
    batch_size, seq_len = 4, 8
    questions = torch.randint(0, 50, (batch_size, seq_len)).to(device)
    responses = torch.randint(0, 4, (batch_size, seq_len)).to(device)
    
    try:
        # Test base model
        for param in base_model.parameters():
            param.requires_grad_(True)
        
        student_abilities, item_thresholds, discrimination_params, final_probs = base_model(questions, responses)
        loss = final_probs.sum()
        loss.backward()
        
        total_grad_norm = sum(param.grad.norm().item() for param in base_model.parameters() if param.grad is not None)
        param_count = sum(1 for param in base_model.parameters() if param.grad is not None)
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        
        print(f"✓ Base model gradient norm: {avg_grad_norm:.3f}")
        
        if avg_grad_norm < 100:
            print("✅ Base model is stable!")
            return True
        else:
            print("🚨 Base model also has gradient issues!")
            return False
            
    except Exception as e:
        print(f"❌ Base model error: {e}")
        return False

if __name__ == "__main__":
    success1 = debug_adaptive_model()
    success2 = compare_with_base_model()
    
    if success1 and success2:
        print("\n✅ DEBUG PASSED: Both models are stable")
    else:
        print("\n❌ DEBUG FAILED: Gradient instability detected")