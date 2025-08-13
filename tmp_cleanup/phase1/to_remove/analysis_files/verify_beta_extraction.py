#!/usr/bin/env python3
"""
Verify Beta Parameter Extraction by Comparing with GPCM Probability Computation
Ensures that extracted betas match the exact values used in GPCM probability calculation.
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.coral_gpcm import EnhancedCORALGPCM
from train import load_simple_data, create_data_loaders


def load_model_for_verification(model_path: str, n_questions: int, n_cats: int, device: torch.device):
    """Load model specifically for verification."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Get dimensions
    state_dict = checkpoint['model_state_dict']
    memory_size = state_dict['memory.key_memory_matrix'].shape[0]
    key_dim = state_dict['q_embed.weight'].shape[1]
    value_dim = state_dict['init_value_memory'].shape[1]
    final_fc_dim = state_dict['summary_network.0.bias'].shape[0]
    
    model = EnhancedCORALGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
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
    
    return model


def verify_beta_extraction(model, questions: torch.Tensor, responses: torch.Tensor, device: torch.device):
    """
    Verify that extracted betas match the exact values used in GPCM computation.
    
    This function runs both the beta extraction and GPCM computation simultaneously
    to verify they use the same values.
    """
    model.eval()
    with torch.no_grad():
        batch_size, seq_len = questions.shape
        
        print(f"🔍 Verifying beta extraction for batch shape: {questions.shape}")
        
        # Method 1: Get standard model outputs (this goes through full forward pass)
        student_abilities_full, item_thresholds_full, discrimination_params_full, gpcm_probs_full = model(questions, responses)
        
        # Method 2: Manual extraction following the exact computation pathway
        # Initialize memory
        model.memory.init_value_memory(batch_size, model.init_value_memory)
        
        # Create embeddings
        gpcm_embeds = model.create_embeddings(questions, responses)
        q_embeds = model.q_embed(questions)
        processed_embeds = model.process_embeddings(gpcm_embeds, q_embeds)
        
        # Extract parameters step by step
        manual_betas = []
        manual_thetas = []
        manual_alphas = []
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]
            gpcm_embed_t = processed_embeds[:, t, :]
            
            # Transform to value dimension
            value_embed_t = model.gpcm_value_embed(gpcm_embed_t)
            
            # Memory operations
            correlation_weight = model.memory.attention(q_embed_t)
            read_content = model.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = model.summary_network(summary_input)
            
            # Extract IRT parameters - THE KEY VERIFICATION POINT
            theta_t, alpha_t, betas_t = model.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)
            alpha_t = alpha_t.squeeze(1)
            betas_t = betas_t.squeeze(1)
            
            manual_betas.append(betas_t)
            manual_thetas.append(theta_t)
            manual_alphas.append(alpha_t)
            
            # Write to memory for next time step
            if t < seq_len - 1:
                model.memory.write(correlation_weight, value_embed_t)
        
        # Stack manual results
        manual_betas = torch.stack(manual_betas, dim=1)
        manual_thetas = torch.stack(manual_thetas, dim=1)
        manual_alphas = torch.stack(manual_alphas, dim=1)
        
        # Method 3: Verify GPCM probability computation uses these exact betas
        # Manually compute GPCM probabilities using the extracted betas
        manual_gpcm_probs = []
        for t in range(seq_len):
            theta_t = manual_thetas[:, t]
            alpha_t = manual_alphas[:, t]
            betas_t = manual_betas[:, t, :]
            
            # Use the exact GPCM computation from GPCMProbabilityLayer
            K = betas_t.shape[-1] + 1  # Number of categories
            
            # Compute cumulative logits
            cum_logits = torch.zeros(batch_size, K, device=theta_t.device)
            cum_logits[:, 0] = 0  # First category baseline
            
            # For k = 1, ..., K-1: sum_{h=0}^{k-1} alpha * (theta - beta_h)
            for k in range(1, K):
                cum_logits[:, k] = torch.sum(
                    alpha_t.unsqueeze(-1) * (theta_t.unsqueeze(-1) - betas_t[:, :k]), 
                    dim=-1
                )
            
            # Convert to probabilities via softmax
            probs_t = F.softmax(cum_logits, dim=-1)
            manual_gpcm_probs.append(probs_t)
        
        manual_gpcm_probs = torch.stack(manual_gpcm_probs, dim=1)
        
        # VERIFICATION: Compare all three methods
        print("\n🔍 VERIFICATION RESULTS:")
        
        # 1. Compare full forward vs manual extraction
        beta_diff_1 = torch.abs(item_thresholds_full - manual_betas).max().item()
        theta_diff_1 = torch.abs(student_abilities_full - manual_thetas).max().item()
        alpha_diff_1 = torch.abs(discrimination_params_full - manual_alphas).max().item()
        
        print(f"  📊 Full Forward vs Manual Extraction:")
        print(f"    Beta max difference: {beta_diff_1:.10f}")
        print(f"    Theta max difference: {theta_diff_1:.10f}")
        print(f"    Alpha max difference: {alpha_diff_1:.10f}")
        
        # 2. Compare GPCM computation (note: full model uses CORAL blending, so we compare the base GPCM part)
        # For GPCM comparison, we need to compute base GPCM without CORAL blending
        base_gpcm_probs = []
        for t in range(seq_len):
            # Use the exact same parameters as manual computation
            theta_t = student_abilities_full[:, t]
            alpha_t = discrimination_params_full[:, t]
            betas_t = item_thresholds_full[:, t, :]
            
            # Compute GPCM probabilities
            gpcm_prob_t = model.gpcm_layer(
                theta_t.unsqueeze(1), alpha_t.unsqueeze(1), betas_t.unsqueeze(1)
            )
            base_gpcm_probs.append(gpcm_prob_t.squeeze(1))
        
        base_gpcm_probs = torch.stack(base_gpcm_probs, dim=1)
        
        gpcm_prob_diff = torch.abs(base_gpcm_probs - manual_gpcm_probs).max().item()
        print(f"  📊 GPCM Layer vs Manual GPCM Computation:")
        print(f"    GPCM probability max difference: {gpcm_prob_diff:.10f}")
        
        # 3. Verification summary
        all_match = (beta_diff_1 < 1e-6) and (theta_diff_1 < 1e-6) and (alpha_diff_1 < 1e-6) and (gpcm_prob_diff < 1e-6)
        
        print(f"\n✅ VERIFICATION STATUS:")
        print(f"  All extractions match: {all_match}")
        print(f"  Beta extraction accuracy: {'EXACT' if beta_diff_1 < 1e-6 else 'APPROXIMATE'}")
        print(f"  GPCM computation match: {'EXACT' if gpcm_prob_diff < 1e-6 else 'APPROXIMATE'}")
        
        if all_match:
            print(f"  🎉 SUCCESS: Beta extraction uses EXACT same values as GPCM computation!")
        else:
            print(f"  ⚠️  WARNING: Some differences detected - review computation pathway")
        
        return {
            'verification_passed': all_match,
            'beta_max_diff': beta_diff_1,
            'theta_max_diff': theta_diff_1,
            'alpha_max_diff': alpha_diff_1,
            'gpcm_prob_max_diff': gpcm_prob_diff,
            'extracted_betas_sample': manual_betas[0, :5, :].cpu().numpy().tolist(),  # First 5 timesteps of first sequence
            'full_forward_betas_sample': item_thresholds_full[0, :5, :].cpu().numpy().tolist(),
            'verification_notes': 'Beta extraction follows exact GPCM computation pathway'
        }


def main():
    parser = argparse.ArgumentParser(description='Verify Beta Parameter Extraction')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    parser.add_argument('--n_samples', type=int, default=3, help='Number of batches to verify')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BETA PARAMETER EXTRACTION VERIFICATION")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📁 Model: {args.model_path}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🖥️  Device: {device}")
    print()
    
    # Load data
    train_path = f"data/{args.dataset}/{args.dataset.lower()}_train.txt"
    test_path = f"data/{args.dataset}/{args.dataset.lower()}_test.txt"
    
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    print(f"📊 Data loaded: {len(test_data)} test sequences")
    
    # Load model
    model = load_model_for_verification(args.model_path, n_questions, n_cats, device)
    print(f"✅ Model loaded for verification")
    
    # Create data loader
    _, test_loader = create_data_loaders(train_data, test_data, batch_size=8)  # Small batch for detailed verification
    
    # Run verification on multiple batches
    print(f"\n🔍 Running verification on {args.n_samples} batches...")
    
    all_results = []
    for batch_idx, (questions, responses, mask) in enumerate(test_loader):
        if batch_idx >= args.n_samples:
            break
            
        questions = questions.to(device)
        responses = responses.to(device)
        
        print(f"\n--- Batch {batch_idx + 1}/{args.n_samples} ---")
        
        batch_results = verify_beta_extraction(model, questions, responses, device)
        all_results.append(batch_results)
    
    # Summary results
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = all(r['verification_passed'] for r in all_results)
    max_beta_diff = max(r['beta_max_diff'] for r in all_results)
    max_gpcm_diff = max(r['gpcm_prob_max_diff'] for r in all_results)
    
    print(f"Overall verification: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    print(f"Maximum beta difference: {max_beta_diff:.2e}")
    print(f"Maximum GPCM probability difference: {max_gpcm_diff:.2e}")
    
    if all_passed:
        print(f"\n🎉 CONCLUSION:")
        print(f"  The beta extraction method successfully extracts the EXACT same")
        print(f"  beta parameters that are used in GPCM probability computation.")
        print(f"  This confirms the extraction method mirrors the computation pathway perfectly.")
    else:
        print(f"\n⚠️  CONCLUSION:")
        print(f"  Some verification checks failed. Review the extraction method.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())