#!/usr/bin/env python3
"""Minimal test for ordinal attention integration."""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    from train import load_simple_data
    
    # Test with smaller dataset
    train_data, test_data, n_questions, n_cats = load_simple_data(
        "data/synthetic_4000_200_2/synthetic_4000_200_2_train.txt",
        "data/synthetic_4000_200_2/synthetic_4000_200_2_test.txt"
    )
    
    print(f"Loaded: {len(train_data)} train, {len(test_data)} test")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Check first sequence
    if train_data:
        q, r = train_data[0]
        print(f"First sequence: {len(q)} questions, {len(r)} responses")
    
    return train_data, test_data, n_questions, n_cats

def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    from models.implementations.attention_gpcm import AttentionGPCM
    
    # Create simple model
    model = AttentionGPCM(
        n_questions=200,
        n_cats=2,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=1,
        use_ordinal_attention=True,
        attention_types=['ordinal_aware']
    )
    
    print(f"Model created: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_forward_pass():
    """Test forward pass with sample data."""
    print("Testing forward pass...")
    
    # Create sample data
    batch_size, seq_len = 2, 10
    questions = torch.randint(0, 200, (batch_size, seq_len))
    responses = torch.randint(0, 2, (batch_size, seq_len))
    
    # Create model
    model = test_model_creation()
    
    # Forward pass
    try:
        memory, abilities, difficulty, probs = model(questions, responses)
        print(f"Forward pass successful:")
        print(f"  Memory: {memory.shape}")
        print(f"  Abilities: {abilities.shape}")
        print(f"  Difficulty: {difficulty.shape}")
        print(f"  Probs: {probs.shape}")
        return True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MINIMAL INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test 1: Data loading
        train_data, test_data, n_questions, n_cats = test_data_loading()
        print("✅ Data loading: SUCCESS")
        
        # Test 2: Model creation
        model = test_model_creation()
        print("✅ Model creation: SUCCESS")
        
        # Test 3: Forward pass
        success = test_forward_pass()
        if success:
            print("✅ Forward pass: SUCCESS")
        else:
            print("❌ Forward pass: FAILED")
        
        print("\n" + "=" * 50)
        print("MINIMAL TEST COMPLETED")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()