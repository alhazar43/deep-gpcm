#!/usr/bin/env python3
"""
Comprehensive Analysis of τ (Tau) Usage in full_adaptive_coral_gpcm
Shows exactly how τ parameters are used in probability computation.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.coral_gpcm import EnhancedCORALGPCM
from train import load_simple_data, create_data_loaders


def analyze_tau_usage():
    """Comprehensive analysis of how τ parameters are used in probability computation."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("COMPREHENSIVE τ (TAU) USAGE ANALYSIS")
    print("=" * 80)
    
    # Load model
    model_path = "save_models/best_full_adaptive_coral_gpcm_synthetic_OC.pth"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get dimensions and create model
    state_dict = checkpoint['model_state_dict']
    memory_size = state_dict['memory.key_memory_matrix'].shape[0]
    key_dim = state_dict['q_embed.weight'].shape[1]
    value_dim = state_dict['init_value_memory'].shape[1]
    final_fc_dim = state_dict['summary_network.0.bias'].shape[0]
    
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
    
    # Load test data
    train_path = "data/synthetic_OC/synthetic_oc_train.txt"
    test_path = "data/synthetic_OC/synthetic_oc_test.txt"
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    _, test_loader = create_data_loaders(train_data, test_data, batch_size=4)
    
    # Get one batch
    questions, responses, mask = next(iter(test_loader))
    questions = questions.to(device)
    responses = responses.to(device)
    
    print(f"🔍 ANALYSIS SETUP:")
    print(f"  Batch shape: {questions.shape}")
    print(f"  Model configuration:")
    print(f"    enable_threshold_coupling: {model.enable_threshold_coupling}")
    print(f"    enable_adaptive_blending: {model.enable_adaptive_blending}")
    print(f"    threshold_blender: {type(model.threshold_blender).__name__}")
    
    with torch.no_grad():
        # Extract τ parameters
        coral_taus = model.coral_layer.rank_classifier.bias.data.cpu().numpy()
        coral_weights = model.coral_layer.rank_classifier.weight.data.cpu().numpy()
        
        print(f"\n📊 τ (TAU) PARAMETERS:")
        print(f"  CORAL τ (bias): {coral_taus}")
        print(f"  CORAL weights shape: {coral_weights.shape}")
        print(f"  CORAL weights (first few): {coral_weights[:, :5]}")
        
        # Step 1: Full forward pass
        print(f"\n🔄 STEP-BY-STEP COMPUTATION ANALYSIS:")
        student_abilities, item_thresholds, discrimination_params, final_probs = model(questions, responses)
        
        print(f"  1. Forward pass completed")
        print(f"     Final probabilities shape: {final_probs.shape}")
        
        # Step 2: Trace through the adaptive blending computation
        # Get base GPCM probabilities (without CORAL blending)
        base_abilities, base_thresholds, base_discrimination, base_gpcm_probs = model.__class__.__bases__[0].forward(
            model, questions, responses
        )
        
        print(f"  2. Base GPCM probabilities computed")
        print(f"     Base GPCM probs shape: {base_gpcm_probs.shape}")
        
        # Step 3: Compute CORAL probabilities using β (not τ directly)
        batch_size, seq_len = questions.shape
        coral_cum_logits = []
        for t in range(seq_len):
            theta_t = student_abilities[:, t:t+1]  # (batch_size, 1)
            beta_t = item_thresholds[:, t, :]  # Uses β, NOT τ directly
            alpha_t = discrimination_params[:, t:t+1] if discrimination_params is not None else torch.ones_like(theta_t)
            logits_t = alpha_t * (theta_t - beta_t)  # CORAL computation uses β
            coral_cum_logits.append(logits_t)
        
        coral_cum_logits = torch.stack(coral_cum_logits, dim=1)
        cum_probs = torch.sigmoid(coral_cum_logits)
        coral_probs = model._cumulative_to_categorical(cum_probs)
        
        print(f"  3. CORAL probabilities computed using β (not τ directly)")
        print(f"     CORAL probs shape: {coral_probs.shape}")
        
        # Step 4: This is where τ is USED - in the adaptive blender
        print(f"\n⚡ CRITICAL: WHERE τ IS ACTUALLY USED:")
        print(f"  τ is passed to the FullAdaptiveBlender.forward() method:")
        print(f"    - gpcm_probs: {base_gpcm_probs.shape}")
        print(f"    - coral_probs: {coral_probs.shape}")
        print(f"    - item_betas: {item_thresholds.shape} (β values)")
        print(f"    - ordinal_taus: {coral_taus.shape} (τ values) ⭐")
        
        # Step 5: Show how τ is used in blend weight calculation
        print(f"\n🧮 HOW τ IS USED IN BLEND WEIGHT CALCULATION:")
        
        # Call the blend weight calculation to see how τ is used
        if hasattr(model.threshold_blender, 'calculate_blend_weights'):
            blend_weights = model.threshold_blender.calculate_blend_weights(
                item_betas=item_thresholds,
                ordinal_taus=torch.tensor(coral_taus, device=device),
                student_abilities=student_abilities,
                discrimination_alphas=discrimination_params
            )
            print(f"  Blend weights shape: {blend_weights.shape}")
            print(f"  Blend weights sample (first sequence, first timestep): {blend_weights[0, 0, :].cpu().numpy()}")
            
            # Show the threshold geometry analysis that uses τ
            geometry = model.threshold_blender.analyze_threshold_geometry(
                item_thresholds.detach(), 
                torch.tensor(coral_taus, device=device)
            )
            
            print(f"\n📐 THRESHOLD GEOMETRY ANALYSIS (uses both β and τ):")
            print(f"  min_distance shape: {geometry['min_distance'].shape}")
            print(f"  range_divergence shape: {geometry['range_divergence'].shape}")
            print(f"  threshold_correlation shape: {geometry['threshold_correlation'].shape}")
            print(f"  distance_spread shape: {geometry['distance_spread'].shape}")
            
            print(f"\n  Sample geometry metrics (first sequence, first timestep):")
            print(f"    min_distance: {geometry['min_distance'][0, 0].item():.6f}")
            print(f"    range_divergence: {geometry['range_divergence'][0, 0].item():.6f}")
            print(f"    threshold_correlation: {geometry['threshold_correlation'][0, 0].item():.6f}")
            print(f"    distance_spread: {geometry['distance_spread'][0, 0].item():.6f}")
        
        # Step 6: Final blending
        manual_blend = (1 - blend_weights) * base_gpcm_probs + blend_weights * coral_probs
        manual_blend = manual_blend / manual_blend.sum(dim=-1, keepdim=True)
        
        # Verify this matches the model output
        blend_diff = torch.abs(final_probs - manual_blend).max().item()
        print(f"\n✅ VERIFICATION:")
        print(f"  Manual blending matches model output: {blend_diff < 1e-6}")
        print(f"  Maximum difference: {blend_diff:.10f}")
    
    print(f"\n" + "=" * 80)
    print(f"SUMMARY: HOW τ IS USED IN full_adaptive_coral_gpcm")
    print(f"=" * 80)
    print(f"1. ❌ τ is NOT used directly in GPCM probability computation")
    print(f"   GPCM uses: alpha * (theta - beta) with β values")
    print(f"")
    print(f"2. ❌ τ is NOT used directly in CORAL probability computation")
    print(f"   CORAL uses: alpha * (theta - beta) with β values (same as GPCM)")
    print(f"")
    print(f"3. ✅ τ IS used in the FullAdaptiveBlender for:")
    print(f"   - Threshold geometry analysis (compares β vs τ distributions)")
    print(f"   - Adaptive blend weight calculation")
    print(f"   - Determining how much to weight GPCM vs CORAL probabilities")
    print(f"")
    print(f"4. 🔄 COMPUTATION FLOW:")
    print(f"   β → GPCM probabilities")
    print(f"   β → CORAL probabilities (using same β, not τ)")
    print(f"   (β, τ) → FullAdaptiveBlender → blend weights")
    print(f"   blend weights → final probabilities = blend(GPCM, CORAL)")
    print(f"")
    print(f"5. 📊 ROLE OF τ:")
    print(f"   τ serves as REFERENCE ordinal thresholds for adaptive blending")
    print(f"   τ helps determine when CORAL vs GPCM should be weighted more heavily")
    print(f"   τ is used in geometric analysis but not direct probability computation")
    print(f"=" * 80)


if __name__ == "__main__":
    analyze_tau_usage()