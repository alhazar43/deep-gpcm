#!/usr/bin/env python3
"""Quick test of integrated AKVMN model."""

import torch
from core.integrated_attention import IntegratedAttentionGPCM

# Test model creation
try:
    model = IntegratedAttentionGPCM(
        n_questions=400,
        n_cats=4,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=2,
        embedding_strategy="linear_decay",
        ability_scale=2.0,
        dropout_rate=0.1
    )
    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 5
    questions = torch.randint(0, 400, (batch_size, seq_len))
    responses = torch.randint(0, 4, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        outputs = model(questions, responses)
        print(f"✓ Forward pass successful")
        print(f"  Outputs: {len(outputs)} tensors")
        print(f"  Shapes: {[o.shape for o in outputs]}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()