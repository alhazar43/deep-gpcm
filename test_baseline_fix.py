#!/usr/bin/env python3
"""
Test script to verify baseline model functionality after memory interface fixes.
"""

import sys
import torch
import torch.nn as nn

# Add current directory to path for imports
sys.path.append('.')

from models.baseline import BaselineGPCM, create_baseline_gpcm

def test_baseline_creation():
    """Test baseline model creation."""
    print("Testing baseline model creation...")
    
    model = create_baseline_gpcm(n_questions=50, n_cats=4)
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"✅ Baseline model created successfully")
    print(f"   Parameters: {param_count:,}")
    print(f"   Model name: {model.model_name}")
    
    return model, param_count

def test_forward_pass(model):
    """Test forward pass functionality."""
    print("\nTesting forward pass...")
    
    batch_size, seq_len = 2, 5
    questions = torch.randint(1, 51, (batch_size, seq_len))
    responses = torch.randint(0, 4, (batch_size, seq_len))
    
    try:
        outputs = model(questions, responses)
        print(f"✅ Forward pass successful")
        print(f"   Outputs: {len(outputs)} tensors")
        print(f"   Output shapes: {[tuple(out.shape) for out in outputs]}")
        
        # Check output types
        student_abilities, item_thresholds, discrimination_params, gpcm_probs = outputs
        print(f"   Student abilities: {student_abilities.shape}")
        print(f"   Item thresholds: {item_thresholds.shape}")
        print(f"   Discrimination: {discrimination_params.shape}")
        print(f"   GPCM probs: {gpcm_probs.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_interface(model):
    """Test memory interface functionality."""
    print("\nTesting memory interface...")
    
    # Access the DKVMN memory
    memory = model.gpcm_model.memory
    batch_size = 2
    
    try:
        # Test memory initialization
        init_memory = model.gpcm_model.init_value_memory
        memory.init_value_memory(batch_size, init_memory)
        print(f"✅ Memory initialization successful")
        print(f"   Memory shape: {memory.value_memory_matrix.shape}")
        
        # Test attention
        query = torch.randn(batch_size, memory.key_dim)
        correlation_weight = memory.attention(query)
        print(f"✅ Attention computation successful")
        print(f"   Correlation weight shape: {correlation_weight.shape}")
        
        # Test read
        read_content = memory.read(correlation_weight)
        print(f"✅ Memory read successful")
        print(f"   Read content shape: {read_content.shape}")
        
        # Test write
        value_embed = torch.randn(batch_size, memory.value_dim)
        memory.write(correlation_weight, value_embed)
        print(f"✅ Memory write successful")
        
        return True
    except Exception as e:
        print(f"❌ Memory interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=== Baseline Model Functionality Test ===")
    
    # Test model creation
    model, param_count = test_baseline_creation()
    
    # Test forward pass
    forward_success = test_forward_pass(model)
    
    # Test memory interface
    memory_success = test_memory_interface(model)
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Model creation: ✅")
    print(f"Forward pass: {'✅' if forward_success else '❌'}")
    print(f"Memory interface: {'✅' if memory_success else '❌'}")
    print(f"Parameter count: {param_count:,}")
    
    # Expected parameter count comparison
    target_param_count = 130655  # From historical results
    print(f"Target parameter count: {target_param_count:,}")
    print(f"Parameter match: {'✅' if abs(param_count - target_param_count) < 1000 else '❌'}")
    
    if forward_success and memory_success:
        print("\n🎉 All tests passed! Baseline model is functional.")
        return True
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    main()