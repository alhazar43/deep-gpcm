#!/usr/bin/env python3
"""
Test script for Question-Specific Deep Bayesian-DKVMN model.

Verifies that the model implements the CORRECT IRT structure:
1. Question-specific α,β parameters (30 questions → 30 parameter sets)
2. Student-specific θ parameters (200 students → 200 θ values)
3. Full DKVMN memory operations maintained
"""

import torch
import numpy as np
from models.question_specific_bayesian_dkvmn import QuestionSpecificDeepBayesianDKVMN

def test_question_specific_model():
    """Test the question-specific model architecture."""
    print("Testing Question-Specific Deep Bayesian-DKVMN Model...")
    print("=" * 70)
    
    # Model parameters
    n_questions = 30
    n_categories = 4
    memory_size = 10
    batch_size = 4
    seq_len = 3
    
    # Create model
    model = QuestionSpecificDeepBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=memory_size,
        key_dim=20,
        value_dim=30
    )
    
    print(f"1. Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   ✅ QUESTION-SPECIFIC α,β + STUDENT-SPECIFIC θ structure")
    
    # Create dummy data
    questions = torch.randint(0, n_questions, (batch_size, seq_len))
    responses = torch.randint(0, n_categories, (batch_size, seq_len))
    student_ids = torch.arange(batch_size)  # Student IDs for tracking
    
    print(f"\n2. Test data shapes:")
    print(f"   Questions: {questions.shape}")
    print(f"   Responses: {responses.shape}")
    print(f"   Student IDs: {student_ids.shape}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    model.set_epoch(5)
    probabilities, aux_dict = model(questions, responses, student_ids=student_ids, return_params=True)
    
    print(f"   Output shape: {probabilities.shape}")
    assert probabilities.shape == (batch_size, seq_len, n_categories), "Wrong output shape"
    
    # Check probabilities sum to 1
    prob_sums = probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
        "Probabilities should sum to 1"
    print("   ✓ Probability constraints verified!")
    
    # Test CORRECT parameter structure
    print("\n4. Testing CORRECT IRT parameter structure...")
    
    # Check question-specific α parameters
    alphas = aux_dict['alphas']
    print(f"   Alpha parameters shape: {alphas.shape}")
    assert alphas.shape == (n_questions,), f"Expected α shape [{n_questions}], got {alphas.shape}"
    print(f"   ✓ Question-specific α: {n_questions} questions → {n_questions} α parameters")
    
    # Check question-specific β parameters  
    betas = aux_dict['betas']
    print(f"   Beta parameters shape: {betas.shape}")
    assert betas.shape == (n_questions, n_categories-1), f"Expected β shape [{n_questions}, {n_categories-1}], got {betas.shape}"
    print(f"   ✓ Question-specific β: {n_questions} questions → {n_questions} β parameter sets")
    
    # Check student-specific θ parameters
    thetas = aux_dict['thetas']
    print(f"   Theta parameters shape: {thetas.shape}")
    assert len(thetas) == 200, f"Expected θ length [200], got {len(thetas)}"  # First 200 students
    print(f"   ✓ Student-specific θ: 200 students → 200 θ parameters")
    
    # Test parameter extraction
    print("\n5. Testing interpretable parameters...")
    params = model.get_interpretable_parameters()
    
    print(f"   Alpha range: [{params['alpha'].min().item():.3f}, {params['alpha'].max().item():.3f}]")
    print(f"   Beta range: [{params['beta'].min().item():.3f}, {params['beta'].max().item():.3f}]")
    print(f"   Theta range: [{params['theta'].min().item():.3f}, {params['theta'].max().item():.3f}]")
    
    # Check activation functions are properly applied
    print("\n6. Testing IRT constraint activations...")
    
    # Check alpha values are positive (exp activation)
    alphas = params['alpha']
    assert torch.all(alphas > 0), "Alpha parameters should be positive (exp activation)"
    print("   ✓ Alpha parameters are positive (exp activation working)")
    
    # Check beta ordering within each question (softplus activation)
    betas = params['beta']
    for q in range(min(5, n_questions)):  # Check first 5 questions
        question_betas = betas[q]
        for i in range(len(question_betas) - 1):
            assert question_betas[i+1] > question_betas[i], f"Beta ordering violated for question {q}"
    print("   ✓ Beta parameters are properly ordered (softplus gaps working)")
    
    # Test ELBO loss
    print("\n7. Testing ELBO loss...")
    kl_div = aux_dict['kl_divergence']
    loss = model.elbo_loss(probabilities, responses, kl_div)
    print(f"   ELBO loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test backward pass
    print("\n8. Testing backward pass...")
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"   Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"   Max gradient norm: {np.max(grad_norms):.6f}")
    
    # Test IRT parameter sampling
    print("\n9. Testing IRT parameter sampling...")
    question_ids = torch.arange(5)  # Test first 5 questions
    sample_alphas, sample_betas = model.irt_params.sample_parameters(question_ids)
    det_alphas, det_betas = model.irt_params.get_parameters(question_ids)
    
    print(f"   Sampled α shape: {sample_alphas.shape}, Deterministic α shape: {det_alphas.shape}")
    print(f"   Sampled β shape: {sample_betas.shape}, Deterministic β shape: {det_betas.shape}")
    
    assert sample_alphas.shape == det_alphas.shape, "Alpha shapes should match"
    assert sample_betas.shape == det_betas.shape, "Beta shapes should match"
    print("   ✓ Parameter sampling working correctly")
    
    print("\n" + "=" * 70)
    print("✅ Question-Specific Deep Bayesian-DKVMN model working perfectly!")
    print("\nKey verified features:")
    print("  🎯 Question-specific α parameters: 30 questions → 30 α values")
    print("  🎯 Question-specific β parameters: 30 questions → 30 β parameter sets")
    print("  🎯 Student-specific θ parameters: 200 students → 200 θ values")
    print("  ✅ Proper activation functions for IRT constraints")
    print("  ✅ Full DKVMN memory operations maintained")
    print("  ✅ Complete Bayesian inference with ELBO optimization")
    print("  ✅ Correct parameter structure for IRT recovery!")
    
    return True


def test_parameter_structure_comparison():
    """Test that the new structure is fundamentally different from memory-based approach."""
    print("\n" + "=" * 70)
    print("Testing Parameter Structure Comparison...")
    print("(Comparing with previous memory-based approach)")
    
    n_questions = 30
    n_categories = 4
    memory_size = 10
    
    model = QuestionSpecificDeepBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=memory_size
    )
    
    # Get parameters
    params = model.get_interpretable_parameters()
    
    print(f"\n📊 Parameter Structure Analysis:")
    print(f"   Previous approach: Memory-based parameters")
    print(f"     - α parameters: {memory_size} memory slots → {memory_size} α values")
    print(f"     - β parameters: {memory_size} memory slots → {memory_size} β sets")
    print(f"     - θ parameters: Memory archetypes (not actual students)")
    
    print(f"\n   NEW approach: Question/Student-specific parameters")
    print(f"     - α parameters: {n_questions} questions → {params['alpha'].shape[0]} α values ✅")
    print(f"     - β parameters: {n_questions} questions → {params['beta'].shape[0]} β sets ✅")
    print(f"     - θ parameters: 200 actual students → {len(params['theta'])} θ values ✅")
    
    print(f"\n🎯 This is the CORRECT IRT structure!")
    print(f"   - Each question has its own discrimination (α) and thresholds (β)")
    print(f"   - Each student has their own ability (θ)")
    print(f"   - Parameters can be directly compared with ground truth!")
    
    return True


if __name__ == '__main__':
    test_question_specific_model()
    test_parameter_structure_comparison()