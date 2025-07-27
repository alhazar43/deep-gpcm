#!/usr/bin/env python3
"""
Test Fixed Deep Integration Model
Verify the fixed model can train without NaN values.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from models.deep_integration_fixed import FixedDeepIntegrationGPCM
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader


def test_model_creation():
    """Test if the fixed model can be created without errors."""
    print("🧪 Testing model creation...")
    
    try:
        model = FixedDeepIntegrationGPCM(
            n_questions=29,
            n_cats=4,
            embed_dim=64,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully: {param_count:,} parameters")
        
        return model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_forward_pass():
    """Test if the model can perform forward pass without NaN."""
    print("\n🧪 Testing forward pass...")
    
    model = test_model_creation()
    if model is None:
        return False
    
    try:
        # Create test data
        batch_size, seq_len = 2, 10
        questions = torch.randint(1, 30, (batch_size, seq_len))
        responses = torch.randint(0, 4, (batch_size, seq_len))
        
        print(f"Input shapes: questions {questions.shape}, responses {responses.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            read_content, mastery_level, logits, probs = model(questions, responses)
        
        # Check outputs
        print(f"Output shapes:")
        print(f"  read_content: {read_content.shape}")
        print(f"  mastery_level: {mastery_level.shape}")
        print(f"  logits: {logits.shape}")
        print(f"  probs: {probs.shape}")
        
        # Check for NaN values
        if torch.isnan(read_content).any():
            print("❌ NaN detected in read_content")
            return False
        if torch.isnan(mastery_level).any():
            print("❌ NaN detected in mastery_level")
            return False
        if torch.isnan(logits).any():
            print("❌ NaN detected in logits")
            return False
        if torch.isnan(probs).any():
            print("❌ NaN detected in probs")
            return False
        
        # Check probability validity
        prob_sums = probs.sum(dim=-1)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6):
            print("❌ Probabilities don't sum to 1")
            return False
        
        print("✅ Forward pass successful - no NaN values detected")
        print(f"✅ Probabilities valid - sum to 1.0 ± 1e-6")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test if the model can perform one training step without errors."""
    print("\n🧪 Testing training step...")
    
    model = test_model_creation()
    if model is None:
        return False
    
    try:
        # Create test data
        batch_size, seq_len = 4, 15
        questions = torch.randint(1, 30, (batch_size, seq_len))
        responses = torch.randint(0, 4, (batch_size, seq_len))
        
        # Create optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        read_content, mastery_level, logits, probs = model(questions, responses)
        
        # Compute loss (flatten for CrossEntropyLoss)
        probs_flat = probs.view(-1, 4)
        responses_flat = responses.view(-1)
        loss = criterion(probs_flat, responses_flat)
        
        print(f"Loss value: {loss.item():.6f}")
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            print("❌ Loss is NaN or Inf")
            return False
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0.0
        param_count = 0
        nan_grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_count += 1
                grad_norm = param.grad.data.norm().item()
                total_grad_norm += grad_norm
                
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"❌ NaN/Inf gradient in {name}")
                    nan_grad_count += 1
        
        if nan_grad_count > 0:
            print(f"❌ {nan_grad_count} parameters have NaN/Inf gradients")
            return False
        
        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
        print(f"✅ Average gradient norm: {avg_grad_norm:.6f}")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        print("✅ Training step successful - no NaN gradients")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_data():
    """Test with real data to ensure compatibility."""
    print("\n🧪 Testing with real data...")
    
    try:
        # Load real data
        dataset_name = "synthetic_OC"
        train_path = f"data/{dataset_name}/{dataset_name.lower()}_train.txt"
        
        train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
        
        # Get number of questions
        all_questions = []
        for seq in train_questions:
            all_questions.extend(seq)
        n_questions = max(all_questions)
        
        print(f"Real data: {len(train_questions)} sequences, {n_questions} questions, {n_cats} categories")
        
        # Create model
        model = FixedDeepIntegrationGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=64
        )
        
        # Create data loader
        device = torch.device('cpu')
        train_loader = UnifiedDataLoader(train_questions, train_responses, batch_size=8, shuffle=True, device=device)
        
        # Test one batch
        model.eval()
        with torch.no_grad():
            for questions, responses, mask in train_loader:
                print(f"Real batch shapes: questions {questions.shape}, responses {responses.shape}")
                
                read_content, mastery_level, logits, probs = model(questions, responses)
                
                # Check for NaN
                if torch.isnan(probs).any():
                    print("❌ NaN detected with real data")
                    return False
                
                print("✅ Real data test successful")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 TESTING FIXED DEEP INTEGRATION MODEL")
    print("="*60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Training Step", test_training_step),
        ("Real Data", test_real_data)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Model Creation":
                result = test_func() is not None
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Fixed model is ready for training!")
        return True
    else:
        print("🚨 SOME TESTS FAILED - Model needs further fixes")
        return False


if __name__ == "__main__":
    main()