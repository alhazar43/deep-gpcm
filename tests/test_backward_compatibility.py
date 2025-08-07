"""
Test backward compatibility of ordinal attention integration.
Ensures existing models continue to work without modification.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations.attention_gpcm import AttentionGPCM
from models.implementations.deep_gpcm import DeepGPCM


def test_attention_gpcm_backward_compatibility():
    """Test that AttentionGPCM works without ordinal attention."""
    # Create model with default settings (no ordinal attention)
    model = AttentionGPCM(
        n_questions=100,
        n_cats=4,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=2
    )
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    questions = torch.randint(1, 100, (batch_size, seq_len))
    responses = torch.randint(0, 4, (batch_size, seq_len))
    
    # Forward pass should work
    abilities, thresholds, discriminations, probs = model(questions, responses)
    
    # Check output shapes
    assert abilities.shape == (batch_size, seq_len)
    assert thresholds.shape == (batch_size, seq_len, 3)  # n_cats - 1
    assert discriminations.shape == (batch_size, seq_len)
    assert probs.shape == (batch_size, seq_len, 4)
    
    # Check that probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)
    
    print("✓ AttentionGPCM backward compatibility test passed")


def test_attention_gpcm_with_ordinal():
    """Test AttentionGPCM with ordinal attention enabled."""
    # Create model with ordinal attention
    model = AttentionGPCM(
        n_questions=100,
        n_cats=4,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=2,
        use_ordinal_attention=True,
        attention_types=["ordinal_aware"]
    )
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    questions = torch.randint(1, 100, (batch_size, seq_len))
    responses = torch.randint(0, 4, (batch_size, seq_len))
    
    # Forward pass should work
    abilities, thresholds, discriminations, probs = model(questions, responses)
    
    # Check output shapes
    assert abilities.shape == (batch_size, seq_len)
    assert thresholds.shape == (batch_size, seq_len, 3)
    assert discriminations.shape == (batch_size, seq_len)
    assert probs.shape == (batch_size, seq_len, 4)
    
    print("✓ AttentionGPCM with ordinal attention test passed")


def test_deep_gpcm_unaffected():
    """Test that base DeepGPCM is unaffected by changes."""
    # Create base model
    model = DeepGPCM(
        n_questions=100,
        n_cats=4,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50
    )
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    questions = torch.randint(1, 100, (batch_size, seq_len))
    responses = torch.randint(0, 4, (batch_size, seq_len))
    
    # Forward pass should work
    abilities, thresholds, discriminations, probs = model(questions, responses)
    
    # Check output shapes
    assert abilities.shape == (batch_size, seq_len)
    assert thresholds.shape == (batch_size, seq_len, 3)
    assert discriminations.shape == (batch_size, seq_len)
    assert probs.shape == (batch_size, seq_len, 4)
    
    print("✓ DeepGPCM unaffected test passed")


def test_mixed_attention_types():
    """Test using multiple attention types together."""
    # Create model with multiple attention types
    model = AttentionGPCM(
        n_questions=100,
        n_cats=4,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=2,
        use_ordinal_attention=True,
        attention_types=["ordinal_aware", "response_conditioned"]
    )
    
    # Create dummy data
    batch_size = 2
    seq_len = 10
    questions = torch.randint(1, 100, (batch_size, seq_len))
    responses = torch.randint(0, 4, (batch_size, seq_len))
    
    # Forward pass should work
    abilities, thresholds, discriminations, probs = model(questions, responses)
    
    # Check output shapes
    assert abilities.shape == (batch_size, seq_len)
    assert thresholds.shape == (batch_size, seq_len, 3)
    assert discriminations.shape == (batch_size, seq_len)
    assert probs.shape == (batch_size, seq_len, 4)
    
    print("✓ Mixed attention types test passed")


if __name__ == "__main__":
    test_attention_gpcm_backward_compatibility()
    test_attention_gpcm_with_ordinal()
    test_deep_gpcm_unaffected()
    test_mixed_attention_types()
    print("\n✅ All backward compatibility tests passed!")