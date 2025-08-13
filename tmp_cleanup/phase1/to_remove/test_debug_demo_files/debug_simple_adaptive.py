#!/usr/bin/env python3
"""
Debug script to test simple adaptive blending without complex threshold geometry.
This will help isolate whether the issue is in the ThresholdDistanceBlender or elsewhere.
"""
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torch.nn as nn
from models.coral_gpcm import EnhancedCORALGPCM

class SimpleThresholdBlender(nn.Module):
    """Simplified threshold blender for debugging."""
    
    def __init__(self, n_categories=4):
        super().__init__()
        self.n_categories = n_categories
        # Single learnable parameter instead of complex geometry
        self.blend_param = nn.Parameter(torch.tensor(0.0))
    
    def calculate_blend_weights(self, item_betas, ordinal_taus, student_abilities, discrimination_alphas=None):
        """Simple fixed blending weights."""
        batch_size, seq_len = student_abilities.shape
        # Fixed 0.5 blend weight for all categories - should be stable
        return torch.full((batch_size, seq_len, self.n_categories), 0.5, 
                         device=student_abilities.device, dtype=student_abilities.dtype)

# Test simple creation
print("Creating simple adaptive model...")
model = EnhancedCORALGPCM(
    n_questions=400,
    n_cats=4,
    memory_size=50,
    key_dim=50,
    value_dim=200,
    final_fc_dim=50,
    embedding_strategy='linear_decay',
    dropout_rate=0.0,
    ability_scale=1.0,
    use_discrimination=True,
    enable_threshold_coupling=True,
    coupling_type='linear',
    gpcm_weight=0.7,
    coral_weight=0.3,
    enable_adaptive_blending=True,
    blend_weight=0.5,
    range_sensitivity_init=0.1,
    distance_sensitivity_init=0.1,
    baseline_bias_init=0.0
)

# Replace complex blender with simple one
model.threshold_blender = SimpleThresholdBlender(n_categories=4)

print(f"Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
print("Testing forward pass...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Simple test data
batch_size, seq_len = 2, 5
questions = torch.randint(0, 400, (batch_size, seq_len)).to(device)
responses = torch.randint(0, 4, (batch_size, seq_len)).to(device)

try:
    with torch.no_grad():
        student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
    print("✅ Forward pass successful!")
    print(f"Output shapes: {gpcm_probs.shape}")
    print(f"Output range: [{gpcm_probs.min():.4f}, {gpcm_probs.max():.4f}]")
    print(f"Contains NaN: {torch.isnan(gpcm_probs).any()}")
    print(f"Contains Inf: {torch.isinf(gpcm_probs).any()}")
    
    # Test gradients
    print("Testing gradients...")
    model.train()
    student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
    loss = torch.nn.functional.cross_entropy(gpcm_probs.view(-1, 4), responses.view(-1))
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            total_grad_norm += param_grad_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"✅ Gradient computation successful!")
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    
except Exception as e:
    print(f"❌ Error during forward/backward: {e}")
    import traceback
    traceback.print_exc()