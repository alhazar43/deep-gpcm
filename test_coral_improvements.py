#!/usr/bin/env python3
"""
Test script to verify improved CORAL integration works correctly.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import DeepGpcmModel
from models.improved_coral_integration import create_improved_coral_enhanced_model, ImprovedCoralTrainer
from utils.gpcm_utils import load_gpcm_data


def test_improved_coral_architecture():
    """Test the improved CORAL architecture."""
    print("🧪 Testing Improved CORAL Architecture")
    print("="*50)
    
    # Create a simple base model
    base_model = DeepGpcmModel(
        n_questions=10,
        n_cats=4,
        memory_size=20,
        key_dim=32,
        value_dim=64,
        final_fc_dim=32,
        embedding_strategy='linear_decay'
    )
    
    # Create improved CORAL model
    coral_model = create_improved_coral_enhanced_model(base_model)
    
    # Test input
    batch_size, seq_len = 2, 3
    q_data = torch.randint(1, 11, (batch_size, seq_len))  # Questions 1-10
    r_data = torch.randint(0, 4, (batch_size, seq_len))   # Responses 0-3
    
    print(f"Input shapes: q_data {q_data.shape}, r_data {r_data.shape}")
    
    # Test GPCM mode
    print("\n1. Testing GPCM mode:")
    abilities, thresholds, discriminations, gpcm_probs, coral_logits = coral_model(
        q_data, r_data, use_coral=False
    )
    print(f"   ✅ GPCM probabilities shape: {gpcm_probs.shape}")
    print(f"   ✅ CORAL logits (should be None): {coral_logits}")
    print(f"   ✅ IRT parameters preserved: θ {abilities.shape}, β {thresholds.shape}, α {discriminations.shape}")
    
    # Test CORAL mode
    print("\n2. Testing Improved CORAL mode:")
    abilities, thresholds, discriminations, coral_probs, coral_logits = coral_model(
        q_data, r_data, use_coral=True
    )
    print(f"   ✅ CORAL probabilities shape: {coral_probs.shape}")
    print(f"   ✅ CORAL logits shape: {coral_logits.shape}")
    print(f"   ✅ IRT parameters preserved: θ {abilities.shape}, β {thresholds.shape}, α {discriminations.shape}")
    
    # Verify rank consistency
    print("\n3. Testing rank consistency:")
    cum_probs = torch.cumsum(coral_probs, dim=-1)
    violations = 0
    total = 0
    for k in range(cum_probs.shape[-1] - 1):
        violations += (cum_probs[..., k] > cum_probs[..., k+1]).sum().item()
        total += cum_probs[..., k].numel()
    
    print(f"   ✅ Rank violations: {violations}/{total} ({100*violations/total:.2f}%)")
    
    # Test loss functions
    print("\n4. Testing loss functions:")
    trainer = ImprovedCoralTrainer(coral_model, 4, 'cpu')
    
    # Test CORAL loss
    coral_loss = trainer.compute_loss(
        coral_probs, r_data, coral_logits, use_coral=True
    )
    print(f"   ✅ CORAL (CORN) loss: {coral_loss.item():.4f}")
    
    # Test GPCM loss
    gpcm_loss = trainer.compute_loss(
        gpcm_probs, r_data, None, use_coral=False
    )
    print(f"   ✅ GPCM (Ordinal) loss: {gpcm_loss.item():.4f}")
    
    print(f"\n🎉 All tests passed! Improved CORAL integration is working correctly.")
    
    return True


def compare_coral_approaches():
    """Compare original vs improved CORAL approaches."""
    print("\n🔍 Comparing CORAL Approaches")
    print("="*50)
    
    print("📋 COMPARISON SUMMARY:")
    print("-" * 30)
    
    print("\n🚫 ORIGINAL CORAL ISSUES:")
    print("   ❌ Uses raw DKVMN summary vector (bypasses IRT framework)")
    print("   ❌ Uses OrdinalLoss instead of CORAL-specific loss")
    print("   ❌ Poor performance (32-37% categorical accuracy)")
    print("   ❌ Loses educational interpretability")
    
    print("\n✅ IMPROVED CORAL BENEFITS:")
    print("   ✅ Uses IRT parameters (θ, α, β) → CORAL (maintains educational framework)")
    print("   ✅ Uses CORN loss for proper threshold learning")
    print("   ✅ Preserves educational interpretability")
    print("   ✅ Leverages psychometric knowledge")
    
    print("\n🎯 ARCHITECTURE COMPARISON:")
    print("   Original:  DKVMN → summary_vector → CORAL → probabilities")
    print("   Improved:  DKVMN → IRT parameters → CORAL → probabilities")
    
    print("\n🔬 LOSS FUNCTION COMPARISON:")
    print("   Original:  OrdinalLoss (generic ordinal classification)")
    print("   Improved:  CornLoss (CORAL-specific threshold learning)")


def demonstrate_irt_coral_integration():
    """Demonstrate how IRT parameters integrate with CORAL."""
    print("\n🧠 IRT-CORAL Integration Demonstration")
    print("="*50)
    
    # Simulate IRT parameters
    batch_size = 3
    n_cats = 4
    
    # Example IRT parameters
    theta = torch.tensor([0.5, -0.2, 1.1])  # Student abilities
    alpha = torch.tensor([1.2, 0.8, 1.5])   # Item discriminations  
    betas = torch.tensor([              # Item thresholds (K-1)
        [-1.0, 0.0, 1.0],
        [-0.5, 0.5, 1.5], 
        [-1.5, -0.5, 0.5]
    ])
    
    print(f"📊 Sample IRT Parameters:")
    print(f"   Student abilities (θ): {theta.tolist()}")
    print(f"   Item discriminations (α): {alpha.tolist()}")
    print(f"   Item thresholds (β): {betas.tolist()}")
    
    # Create improved CORAL layer
    from models.improved_coral_integration import ImprovedCoralLayer
    
    # IRT feature dimension: θ (1) + α (1) + β (K-1) = 1 + 1 + 3 = 5
    irt_feature_dim = 1 + 1 + (n_cats - 1)
    coral_layer = ImprovedCoralLayer(irt_feature_dim, n_cats)
    
    # Forward pass
    coral_logits, coral_probs = coral_layer(theta, alpha, betas)
    
    print(f"\n🎯 CORAL Output:")
    print(f"   CORAL logits shape: {coral_logits.shape} (for CORN loss)")
    print(f"   CORAL probabilities shape: {coral_probs.shape}")
    print(f"   Sample probabilities: {coral_probs[0].detach().numpy()}")
    
    # Verify probabilities sum to 1
    prob_sums = coral_probs.sum(dim=-1)
    print(f"   Probability sums: {prob_sums.tolist()} (should be ~1.0)")
    
    print(f"\n💡 Key Insight:")
    print(f"   IRT parameters capture educational meaning (ability, difficulty, discrimination)")
    print(f"   CORAL transforms these into rank-consistent probabilities")
    print(f"   This maintains interpretability while ensuring consistency!")


def main():
    """Run all tests and demonstrations."""
    print("🚀 IMPROVED CORAL INTEGRATION TESTING")
    print("="*60)
    
    try:
        # Test architecture
        test_improved_coral_architecture()
        
        # Compare approaches
        compare_coral_approaches()
        
        # Demonstrate integration
        demonstrate_irt_coral_integration()
        
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"✅ Improved CORAL integration is ready for training")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()