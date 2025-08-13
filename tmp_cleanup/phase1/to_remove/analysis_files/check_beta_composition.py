#!/usr/bin/env python3
"""
Check Beta Composition for full_adaptive_coral_gpcm
Determines whether the extracted betas are pure beta or beta+tau combinations.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.coral_gpcm import EnhancedCORALGPCM
from train import load_simple_data, create_data_loaders


def check_beta_composition():
    """Check what beta values are actually extracted from full_adaptive_coral_gpcm."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = "save_models/best_full_adaptive_coral_gpcm_synthetic_OC.pth"
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Get correct dimensions
    state_dict = checkpoint['model_state_dict']
    memory_size = state_dict['memory.key_memory_matrix'].shape[0]
    key_dim = state_dict['q_embed.weight'].shape[1]
    value_dim = state_dict['init_value_memory'].shape[1]
    final_fc_dim = state_dict['summary_network.0.bias'].shape[0]
    
    print("=" * 80)
    print("BETA COMPOSITION ANALYSIS FOR FULL_ADAPTIVE_CORAL_GPCM")
    print("=" * 80)
    print(f"Model dimensions: memory_size={memory_size}, key_dim={key_dim}, value_dim={value_dim}")
    
    # Create model
    model = EnhancedCORALGPCM(
        n_questions=200,
        n_cats=4,
        memory_size=memory_size,
        key_dim=key_dim,
        value_dim=value_dim,
        final_fc_dim=final_fc_dim,
        enable_adaptive_blending=True,
        use_full_blender=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # Check model configuration
    print(f"\n🔍 MODEL CONFIGURATION:")
    print(f"  enable_threshold_coupling: {model.enable_threshold_coupling}")
    print(f"  enable_adaptive_blending: {model.enable_adaptive_blending}")
    print(f"  threshold_coupler: {model.threshold_coupler}")
    print(f"  threshold_blender: {type(model.threshold_blender).__name__ if model.threshold_blender else None}")
    
    # Load some test data
    train_path = "data/synthetic_OC/synthetic_oc_train.txt"
    test_path = "data/synthetic_OC/synthetic_oc_test.txt"
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    _, test_loader = create_data_loaders(train_data, test_data, batch_size=8)
    
    # Get one batch for analysis
    questions, responses, mask = next(iter(test_loader))
    questions = questions.to(device)
    responses = responses.to(device)
    
    print(f"\n🧮 ANALYZING BETA EXTRACTION:")
    print(f"  Batch shape: {questions.shape}")
    
    with torch.no_grad():
        # Get model outputs
        student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
        
        print(f"  Beta shape: {item_thresholds.shape}")
        print(f"  Beta sample (first sequence, first 3 timesteps):")
        for t in range(3):
            beta_t = item_thresholds[0, t, :].cpu().numpy()
            print(f"    t={t}: {beta_t}")
        
        # Check if we have CORAL tau parameters
        if hasattr(model, 'coral_layer') and model.coral_layer is not None:
            coral_taus = model.coral_layer.rank_classifier.bias.data.cpu().numpy()
            coral_weights = model.coral_layer.rank_classifier.weight.data.cpu().numpy()
            print(f"\n📊 CORAL TAU PARAMETERS:")
            print(f"  CORAL τ (bias): {coral_taus}")
            print(f"  CORAL weights shape: {coral_weights.shape}")
        
        # Check what the model actually does in forward pass
        print(f"\n🔍 FORWARD PASS ANALYSIS:")
        
        # Manually trace through the forward pass logic
        # First get base HybridCORALGPCM results
        base_abilities, base_thresholds, base_discrimination, base_probs = model.__class__.__bases__[0].forward(
            model, questions, responses
        )
        
        print(f"  Base thresholds (from HybridCORALGPCM):")
        print(f"    Shape: {base_thresholds.shape}")
        print(f"    Sample: {base_thresholds[0, 0, :].cpu().numpy()}")
        
        # Check if threshold coupling was applied
        coupling_applied = False
        if model.enable_threshold_coupling and model.threshold_coupler is not None:
            print(f"  ⚡ THRESHOLD COUPLING ENABLED - betas are β+τ combinations")
            coupling_applied = True
        else:
            print(f"  ⚡ THRESHOLD COUPLING DISABLED - betas are pure β values")
        
        # Compare the returned thresholds with base thresholds
        threshold_diff = torch.abs(item_thresholds - base_thresholds).max().item()
        print(f"  Difference between returned and base thresholds: {threshold_diff:.10f}")
        
        if threshold_diff < 1e-8:
            print(f"  ✅ CONCLUSION: Extracted betas are PURE β values (no τ coupling)")
        else:
            print(f"  ⚠️  CONCLUSION: Extracted betas are MODIFIED (possibly β+τ combinations)")
    
    print(f"\n" + "=" * 80)
    print(f"FINAL DETERMINATION:")
    
    if not model.enable_threshold_coupling:
        print(f"✅ For full_adaptive_coral_gpcm with threshold_coupling=False:")
        print(f"   Beta = pure β values (GPCM thresholds)")
        print(f"   CORAL τ parameters are separate and available via model.coral_layer")
        print(f"   The model uses β for GPCM computation and τ for CORAL computation")
        print(f"   No coupling/combination occurs between β and τ")
    else:
        print(f"⚠️  For full_adaptive_coral_gpcm with threshold_coupling=True:")
        print(f"   Beta = coupled β+τ combinations")
        print(f"   The returned thresholds are adaptively coupled combinations")
    
    print(f"=" * 80)


if __name__ == "__main__":
    check_beta_composition()