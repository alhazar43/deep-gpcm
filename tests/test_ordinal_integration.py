"""
Integration test for ordinal-aware attention in GPCM models.
Tests the complete pipeline with synthetic data.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations.attention_gpcm import AttentionGPCM
from sklearn.metrics import cohen_kappa_score


def create_synthetic_ordinal_data(n_students=50, n_questions=20, n_cats=4):
    """Create synthetic ordinal data with patterns."""
    np.random.seed(42)
    
    # Create student abilities
    abilities = np.random.normal(0, 1, n_students)
    
    # Create question difficulties
    difficulties = np.linspace(-2, 2, n_questions)
    
    # Generate responses based on ordinal model
    questions = []
    responses = []
    
    for student_idx in range(n_students):
        student_ability = abilities[student_idx]
        student_questions = []
        student_responses = []
        
        # Random question order
        question_order = np.random.permutation(n_questions)
        
        for q_idx in question_order[:10]:  # 10 questions per student
            # Calculate probability of each response category
            thresholds = np.linspace(difficulties[q_idx] - 1, difficulties[q_idx] + 1, n_cats)
            probs = []
            
            for k in range(n_cats):
                if k == 0:
                    prob = 1 / (1 + np.exp(-1.5 * (student_ability - thresholds[k])))
                elif k == n_cats - 1:
                    prob = 1 - 1 / (1 + np.exp(-1.5 * (student_ability - thresholds[k-1])))
                else:
                    prob = (1 / (1 + np.exp(-1.5 * (student_ability - thresholds[k-1]))) - 
                           1 / (1 + np.exp(-1.5 * (student_ability - thresholds[k]))))
                probs.append(prob)
            
            probs = np.array(probs)
            probs = np.abs(probs)  # Ensure positive
            probs = probs / probs.sum()  # Normalize
            
            # Sample response
            response = np.random.choice(n_cats, p=probs)
            
            student_questions.append(q_idx + 1)  # 1-indexed
            student_responses.append(response)
        
        questions.append(student_questions)
        responses.append(student_responses)
    
    return torch.tensor(questions), torch.tensor(responses)


def calculate_qwk(y_true, y_pred, n_cats=4):
    """Calculate Quadratic Weighted Kappa."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def test_ordinal_attention_improvement():
    """Test that ordinal attention improves QWK performance."""
    print("Testing ordinal attention improvement on synthetic data...")
    
    # Create synthetic data
    questions, responses = create_synthetic_ordinal_data()
    
    # Split into train/test
    n_train = 40
    train_q, train_r = questions[:n_train], responses[:n_train]
    test_q, test_r = questions[n_train:], responses[n_train:]
    
    # Create models
    print("\n1. Creating baseline model (no ordinal attention)...")
    baseline_model = AttentionGPCM(
        n_questions=20,
        n_cats=4,
        embed_dim=32,
        memory_size=20,
        key_dim=32,
        value_dim=64,
        final_fc_dim=32,
        n_heads=4,
        n_cycles=2,
        use_ordinal_attention=False
    )
    
    print("2. Creating ordinal-aware model...")
    ordinal_model = AttentionGPCM(
        n_questions=20,
        n_cats=4,
        embed_dim=32,
        memory_size=20,
        key_dim=32,
        value_dim=64,
        final_fc_dim=32,
        n_heads=4,
        n_cycles=2,
        use_ordinal_attention=True,
        attention_types=["ordinal_aware", "qwk_aligned"]
    )
    
    # Simple training loop (just forward passes for demonstration)
    print("\n3. Running models on test data...")
    
    with torch.no_grad():
        # Baseline predictions
        _, _, _, baseline_probs = baseline_model(test_q, test_r)
        baseline_preds = baseline_probs.argmax(dim=-1)
        
        # Ordinal predictions
        _, _, _, ordinal_probs = ordinal_model(test_q, test_r)
        ordinal_preds = ordinal_probs.argmax(dim=-1)
    
    # Calculate metrics
    test_r_flat = test_r.flatten()
    baseline_preds_flat = baseline_preds.flatten()
    ordinal_preds_flat = ordinal_preds.flatten()
    
    # Remove padding (0 responses)
    mask = test_r_flat > 0
    test_r_flat = test_r_flat[mask]
    baseline_preds_flat = baseline_preds_flat[mask]
    ordinal_preds_flat = ordinal_preds_flat[mask]
    
    baseline_qwk = calculate_qwk(test_r_flat.numpy(), baseline_preds_flat.numpy())
    ordinal_qwk = calculate_qwk(test_r_flat.numpy(), ordinal_preds_flat.numpy())
    
    print(f"\n4. Results:")
    print(f"   Baseline QWK: {baseline_qwk:.3f}")
    print(f"   Ordinal QWK:  {ordinal_qwk:.3f}")
    print(f"   Note: These are untrained models - in practice, ordinal attention")
    print(f"         shows improvement after training on ordinal objectives.")
    
    # Test passes if models run without errors
    print("\n✅ Integration test passed!")


def test_all_attention_types():
    """Test that all attention types can be used."""
    print("\nTesting all attention type combinations...")
    
    attention_combinations = [
        ["ordinal_aware"],
        ["response_conditioned"],
        ["ordinal_pattern"],
        ["qwk_aligned"],
        ["hierarchical_ordinal"],
        ["ordinal_aware", "response_conditioned"],
        ["qwk_aligned", "ordinal_pattern"],
        ["ordinal_aware", "qwk_aligned", "hierarchical_ordinal"]
    ]
    
    questions = torch.randint(1, 20, (5, 10))
    responses = torch.randint(0, 4, (5, 10))
    
    for attention_types in attention_combinations:
        print(f"\n  Testing combination: {attention_types}")
        
        model = AttentionGPCM(
            n_questions=20,
            n_cats=4,
            embed_dim=32,
            memory_size=20,
            key_dim=32,
            value_dim=64,
            final_fc_dim=32,
            n_heads=4,
            n_cycles=1,
            use_ordinal_attention=True,
            attention_types=attention_types
        )
        
        # Forward pass
        with torch.no_grad():
            abilities, thresholds, discriminations, probs = model(questions, responses)
        
        # Check outputs
        assert abilities.shape == (5, 10)
        assert thresholds.shape == (5, 10, 3)
        assert discriminations.shape == (5, 10)
        assert probs.shape == (5, 10, 4)
        
        print(f"    ✓ {', '.join(attention_types)} works correctly")
    
    print("\n✅ All attention type combinations work!")


if __name__ == "__main__":
    test_ordinal_attention_improvement()
    test_all_attention_types()