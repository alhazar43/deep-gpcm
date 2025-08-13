#!/usr/bin/env python3
"""
Test Combined Loss Mathematical Correctness
Verify that the weighted combination logic is working correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training.losses import CombinedLoss, FocalLoss, QWKLoss, CORALLoss

def test_combined_loss_mathematics():
    """Test the mathematical correctness of CombinedLoss weighted combinations."""
    
    print("🧮 Testing Combined Loss Mathematical Correctness")
    print("=" * 60)
    
    # Test configuration
    n_cats = 4
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    torch.manual_seed(42)
    logits = torch.randn(batch_size, n_cats, device=device, requires_grad=True)
    targets = torch.randint(0, n_cats, (batch_size,), device=device)
    
    print(f"Test data: {batch_size} samples, {n_cats} categories")
    print(f"Device: {device}")
    print(f"Logits shape: {logits.shape}")
    print(f"Targets: {targets.cpu().numpy()}")
    
    # Test 1: Single component matches individual loss
    print("\n" + "=" * 60)
    print("TEST 1: Single Component Verification")
    print("=" * 60)
    
    # Test CrossEntropy only
    ce_combined = CombinedLoss(n_cats, ce_weight=1.0)
    ce_individual = nn.CrossEntropyLoss()
    
    combined_loss = ce_combined(logits, targets)
    individual_loss = ce_individual(logits, targets)
    
    print(f"CE Combined:    {combined_loss.item():.6f}")
    print(f"CE Individual:  {individual_loss.item():.6f}")
    print(f"Difference:     {abs(combined_loss.item() - individual_loss.item()):.8f}")
    
    assert torch.allclose(combined_loss, individual_loss, atol=1e-6), "CrossEntropy single component failed!"
    print("✅ CrossEntropy single component: PASS")
    
    # Test Focal only (improved design)
    focal_combined = CombinedLoss(n_cats, focal_weight=1.0, focal_alpha=1.0, focal_gamma=2.0)
    focal_individual = FocalLoss(alpha=1.0, gamma=2.0)
    
    combined_loss = focal_combined(logits, targets)
    individual_loss = focal_individual(logits, targets)
    
    print(f"\nFocal Combined:    {combined_loss.item():.6f}")
    print(f"Focal Individual:  {individual_loss.item():.6f}")
    print(f"Difference:        {abs(combined_loss.item() - individual_loss.item()):.8f}")
    
    assert torch.allclose(combined_loss, individual_loss, atol=1e-6), "Focal single component failed!"
    print("✅ Focal single component: PASS")
    
    # Test QWK only (improved design)
    qwk_combined = CombinedLoss(n_cats, qwk_weight=1.0)
    qwk_individual = QWKLoss(n_cats)
    
    combined_loss = qwk_combined(logits, targets)
    individual_loss = qwk_individual(logits, targets)
    
    print(f"\nQWK Combined:    {combined_loss.item():.6f}")
    print(f"QWK Individual:  {individual_loss.item():.6f}")
    print(f"Difference:      {abs(combined_loss.item() - individual_loss.item()):.8f}")
    
    assert torch.allclose(combined_loss, individual_loss, atol=1e-6), "QWK single component failed!"
    print("✅ QWK single component: PASS")
    
    # Test CORAL only (improved design)
    coral_combined = CombinedLoss(n_cats, coral_weight=1.0)
    coral_individual = CORALLoss(n_cats)
    
    combined_loss = coral_combined(logits, targets)
    individual_loss = coral_individual(logits, targets)
    
    print(f"\nCORAL Combined:    {combined_loss.item():.6f}")
    print(f"CORAL Individual:  {individual_loss.item():.6f}")
    print(f"Difference:        {abs(combined_loss.item() - individual_loss.item()):.8f}")
    
    assert torch.allclose(combined_loss, individual_loss, atol=1e-6), "CORAL single component failed!"
    print("✅ CORAL single component: PASS")
    
    # Test 2: Manual weighted combination vs CombinedLoss
    print("\n" + "=" * 60)
    print("TEST 2: Manual vs Combined Loss Verification")
    print("=" * 60)
    
    # Test weights
    ce_weight = 0.6
    focal_weight = 0.2
    qwk_weight = 0.2
    
    # Manual calculation
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)(logits, targets)
    qwk_loss = QWKLoss(n_cats)(logits, targets)
    
    manual_combined = ce_weight * ce_loss + focal_weight * focal_loss + qwk_weight * qwk_loss
    
    # CombinedLoss calculation
    combined_loss_fn = CombinedLoss(
        n_cats, 
        ce_weight=ce_weight, 
        focal_weight=focal_weight, 
        qwk_weight=qwk_weight,
        focal_alpha=1.0, 
        focal_gamma=2.0
    )
    combined_result = combined_loss_fn(logits, targets)
    
    print(f"Manual calculation:")
    print(f"  CE loss:     {ce_loss.item():.6f} × {ce_weight} = {(ce_weight * ce_loss).item():.6f}")
    print(f"  Focal loss:  {focal_loss.item():.6f} × {focal_weight} = {(focal_weight * focal_loss).item():.6f}")
    print(f"  QWK loss:    {qwk_loss.item():.6f} × {qwk_weight} = {(qwk_weight * qwk_loss).item():.6f}")
    print(f"  Total:       {manual_combined.item():.6f}")
    print(f"\nCombined Loss: {combined_result.item():.6f}")
    print(f"Difference:    {abs(manual_combined.item() - combined_result.item()):.8f}")
    
    assert torch.allclose(manual_combined, combined_result, atol=1e-6), "Manual vs Combined calculation failed!"
    print("✅ Manual vs Combined calculation: PASS")
    
    # Test 3: attn_gpcm configuration verification
    print("\n" + "=" * 60)
    print("TEST 3: attn_gpcm Configuration Verification")
    print("=" * 60)
    
    # This is the actual attn_gpcm configuration from factory.py
    attn_combined = CombinedLoss(
        n_cats,
        ce_weight=0.6,
        qwk_weight=0.2,
        focal_weight=0.2,
        focal_alpha=1.0,
        focal_gamma=2.0
    )
    
    # Manual calculation with same weights
    ce_loss = nn.CrossEntropyLoss()(logits, targets)
    qwk_loss = QWKLoss(n_cats)(logits, targets)
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)(logits, targets)
    
    manual_attn = 0.6 * ce_loss + 0.2 * qwk_loss + 0.2 * focal_loss
    combined_attn = attn_combined(logits, targets)
    
    print(f"attn_gpcm configuration:")
    print(f"  CE weight:     0.6")
    print(f"  QWK weight:    0.2")
    print(f"  Focal weight:  0.2")
    print(f"\nManual:        {manual_attn.item():.6f}")
    print(f"Combined:      {combined_attn.item():.6f}")
    print(f"Difference:    {abs(manual_attn.item() - combined_attn.item()):.8f}")
    
    assert torch.allclose(manual_attn, combined_attn, atol=1e-6), "attn_gpcm configuration failed!"
    print("✅ attn_gpcm configuration: PASS")
    
    # Test 4: Gradient flow verification
    print("\n" + "=" * 60)
    print("TEST 4: Gradient Flow Verification")
    print("=" * 60)
    
    # Create fresh tensors with gradients
    logits_grad = torch.randn(batch_size, n_cats, device=device, requires_grad=True)
    targets_grad = torch.randint(0, n_cats, (batch_size,), device=device)
    
    combined_loss_fn = CombinedLoss(n_cats, ce_weight=0.6, focal_weight=0.2, qwk_weight=0.2)
    loss = combined_loss_fn(logits_grad, targets_grad)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = logits_grad.grad is not None
    grad_mean = logits_grad.grad.mean().item() if has_gradients else 0.0
    grad_std = logits_grad.grad.std().item() if has_gradients else 0.0
    
    print(f"Loss value:         {loss.item():.6f}")
    print(f"Has gradients:      {has_gradients}")
    print(f"Gradient mean:      {grad_mean:.6f}")
    print(f"Gradient std:       {grad_std:.6f}")
    print(f"Gradient range:     [{logits_grad.grad.min().item():.6f}, {logits_grad.grad.max().item():.6f}]")
    
    assert has_gradients, "No gradients computed!"
    assert not torch.isnan(logits_grad.grad).any(), "NaN gradients detected!"
    assert not torch.isinf(logits_grad.grad).any(), "Infinite gradients detected!"
    print("✅ Gradient flow: PASS")
    
    # Test 5: Perfect prediction behavior
    print("\n" + "=" * 60)
    print("TEST 5: Perfect Prediction Behavior")
    print("=" * 60)
    
    # Create perfect predictions
    perfect_logits = torch.zeros(batch_size, n_cats, device=device)
    perfect_targets = torch.randint(0, n_cats, (batch_size,), device=device)
    
    # Set perfect predictions (high logit for correct class)
    for i, target in enumerate(perfect_targets):
        perfect_logits[i, target] = 10.0  # Very high confidence
    
    combined_loss_fn = CombinedLoss(n_cats, ce_weight=0.6, focal_weight=0.2, qwk_weight=0.2)
    perfect_loss = combined_loss_fn(perfect_logits, perfect_targets)
    
    print(f"Perfect prediction loss: {perfect_loss.item():.6f}")
    
    # Should be very close to 0 for perfect predictions
    assert perfect_loss.item() < 0.1, f"Perfect prediction loss too high: {perfect_loss.item()}"
    print("✅ Perfect prediction behavior: PASS")
    
    # Test 6: Zero weight verification
    print("\n" + "=" * 60)
    print("TEST 6: Zero Weight Components")
    print("=" * 60)
    
    # Test with some zero weights
    zero_weight_combined = CombinedLoss(
        n_cats,
        ce_weight=1.0,
        focal_weight=0.0,  # Should be ignored
        qwk_weight=0.0,    # Should be ignored
        coral_weight=0.0   # Should be ignored
    )
    
    # Should equal pure CrossEntropy
    ce_only = nn.CrossEntropyLoss()(logits, targets)
    zero_weight_loss = zero_weight_combined(logits, targets)
    
    print(f"CE only:           {ce_only.item():.6f}")
    print(f"Zero weights:      {zero_weight_loss.item():.6f}")
    print(f"Difference:        {abs(ce_only.item() - zero_weight_loss.item()):.8f}")
    print(f"Active components: {len(zero_weight_combined.active_components)}")
    
    assert torch.allclose(ce_only, zero_weight_loss, atol=1e-6), "Zero weight handling failed!"
    assert len(zero_weight_combined.active_components) == 1, "Should have only 1 active component!"
    print("✅ Zero weight handling: PASS")
    
    # Test 7: Safety fallback when no weights specified
    print("\n" + "=" * 60)
    print("TEST 7: Safety Fallback Behavior")
    print("=" * 60)
    
    # Test with no weights specified (should default to CrossEntropy)
    default_combined = CombinedLoss(n_cats)
    ce_fallback = nn.CrossEntropyLoss()(logits, targets)
    default_loss = default_combined(logits, targets)
    
    print(f"No weights specified:")
    print(f"CE fallback:      {ce_fallback.item():.6f}")
    print(f"Default combined: {default_loss.item():.6f}")
    print(f"Difference:       {abs(ce_fallback.item() - default_loss.item()):.8f}")
    print(f"Active components: {len(default_combined.active_components)}")
    print(f"Component name:    {default_combined.active_components[0][0] if default_combined.active_components else 'None'}")
    
    assert torch.allclose(ce_fallback, default_loss, atol=1e-6), "Safety fallback failed!"
    assert len(default_combined.active_components) == 1, "Should have exactly 1 active component!"
    assert default_combined.active_components[0][0] == 'ce', "Should default to CrossEntropy!"
    print("✅ Safety fallback behavior: PASS")
    
    print("\n" + "=" * 60)
    print("🎉 ALL COMBINED LOSS TESTS PASSED!")
    print("=" * 60)
    print("✅ Single components match individual losses")
    print("✅ Manual calculation matches CombinedLoss")
    print("✅ attn_gpcm configuration is mathematically correct")
    print("✅ Gradient flow is working properly")
    print("✅ Perfect predictions behave correctly")
    print("✅ Zero weight components are handled correctly")
    print("✅ Safety fallback to CrossEntropy works properly")
    print("\n💡 The Combined Loss implementation is mathematically sound!")

if __name__ == "__main__":
    test_combined_loss_mathematics()