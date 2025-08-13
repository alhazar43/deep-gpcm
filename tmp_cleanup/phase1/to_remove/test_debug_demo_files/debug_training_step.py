#!/usr/bin/env python3
"""
Debug script to test the first training step of adaptive_coral_gpcm to identify
where the gradient explosion occurs in the actual training pipeline.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import create_model
import numpy as np

def load_data(dataset_name):
    """Load synthetic_OC data for testing."""
    # Simple data loader for debugging
    if dataset_name == "synthetic_OC":
        # Generate simple test data
        n_students = 100
        n_questions = 50
        n_cats = 4
        seq_len = 20
        
        questions = []
        responses = []
        
        for _ in range(n_students):
            student_seq_len = np.random.randint(10, seq_len + 1)
            q_seq = np.random.randint(0, n_questions, size=student_seq_len)
            r_seq = np.random.randint(0, n_cats, size=student_seq_len)
            
            # Pad sequences
            q_padded = np.zeros(seq_len, dtype=int)
            r_padded = np.zeros(seq_len, dtype=float)
            q_padded[:student_seq_len] = q_seq
            r_padded[:student_seq_len] = r_seq
            
            questions.append(q_padded)
            responses.append(r_padded)
        
        questions = np.array(questions)
        responses = np.array(responses)
        
        # Split train/test
        split_idx = int(0.8 * len(questions))
        train_data = (questions[:split_idx], responses[:split_idx])
        test_data = (questions[split_idx:], responses[split_idx:])
        
        return train_data, test_data, n_questions, n_cats
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in debug mode")

def test_first_training_step():
    """Test exactly what happens in the first training step."""
    print("🔍 Testing First Training Step of adaptive_coral_gpcm")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the exact same data as used in training
    print("📊 Loading synthetic_OC dataset...")
    train_data, test_data, n_questions, n_cats = load_data("synthetic_OC")
    
    print(f"✓ Data loaded: {n_questions} questions, {n_cats} categories")
    print(f"  - Train samples: {len(train_data[0])}")
    print(f"  - Test samples: {len(test_data[0])}")
    
    # Create the exact model as used in training
    print("\\n🤖 Creating adaptive_coral_gpcm model...")
    model = create_model(
        model_type="adaptive_coral_gpcm",
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=20,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy="linear_decay",
        ability_scale=1.0,
        use_discrimination=True,
        dropout_rate=0.1,
        # BGT parameters (ultra-conservative)
        range_sensitivity_init=0.01,
        distance_sensitivity_init=0.01,
        baseline_bias_init=0.0
    ).to(device)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Enable adaptive blending: {model.enable_adaptive_blending}")
    print(f"  - Enable threshold coupling: {model.enable_threshold_coupling}")
    
    # Set up optimizer (same as training)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Get a single batch from training data
    questions_np, responses_np = train_data
    batch_size = 32  # Smaller batch for debugging
    
    # Take first batch
    questions_batch = torch.tensor(questions_np[:batch_size], dtype=torch.long).to(device)
    responses_batch = torch.tensor(responses_np[:batch_size], dtype=torch.float).to(device)
    
    print(f"\\n📦 Batch prepared:")
    print(f"  - questions: {questions_batch.shape}, range [{questions_batch.min()}, {questions_batch.max()}]")
    print(f"  - responses: {responses_batch.shape}, range [{responses_batch.min():.3f}, {responses_batch.max():.3f}]")
    
    # Check for invalid data
    if torch.isnan(questions_batch).any() or torch.isnan(responses_batch).any():
        print("🚨 ERROR: NaN detected in input data!")
        return False
    
    model.train()
    
    try:
        print("\\n🔄 Testing forward pass...")
        
        # Forward pass (this is where the issue might occur)
        student_abilities, item_thresholds, discrimination_params, predictions = model(
            questions_batch, responses_batch
        )
        
        print(f"✓ Forward pass completed")
        print(f"  - student_abilities: {student_abilities.shape}, range [{student_abilities.min():.6f}, {student_abilities.max():.6f}]")
        print(f"  - item_thresholds: {item_thresholds.shape}, range [{item_thresholds.min():.6f}, {item_thresholds.max():.6f}]")
        print(f"  - predictions: {predictions.shape}, range [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        # Check for numerical issues in outputs
        outputs_to_check = [
            (student_abilities, "student_abilities"),
            (item_thresholds, "item_thresholds"), 
            (predictions, "predictions")
        ]
        
        if discrimination_params is not None:
            outputs_to_check.append((discrimination_params, "discrimination_params"))
        
        for tensor, name in outputs_to_check:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"🚨 ERROR: NaN/Inf detected in {name}!")
                return False
        
        print("\\n🧪 Testing loss computation...")
        
        # Convert responses to categorical (ordinal → categorical mapping)
        targets = responses_batch.long()  # Convert to integer categories
        
        # Create mask (all valid for this test)
        mask = torch.ones_like(targets).bool()
        
        # Compute cross-entropy loss (same as training)
        criterion = nn.CrossEntropyLoss()
        
        predictions_flat = predictions.view(-1, n_cats)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        valid_predictions = predictions_flat[mask_flat]
        valid_targets = targets_flat[mask_flat]
        
        # Check that targets are in valid range
        if (valid_targets < 0).any() or (valid_targets >= n_cats).any():
            print(f"🚨 ERROR: Invalid target values! Range: [{valid_targets.min()}, {valid_targets.max()}], expected [0, {n_cats-1}]")
            return False
        
        loss = criterion(valid_predictions, valid_targets)
        
        print(f"✓ Loss computed: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("🚨 ERROR: Loss is NaN/Inf!")
            return False
        
        print("\\n🔄 Testing backward pass...")
        
        optimizer.zero_grad()
        loss.backward()
        
        # Analyze gradients in detail  
        print(f"\\n📊 Detailed Gradient Analysis:")
        
        total_grad_norm = 0
        param_count = 0
        large_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1
                
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
                
                if grad_norm > 10:
                    large_grads.append((name, grad_norm))
        
        avg_grad_norm = total_grad_norm / max(param_count, 1)
        
        print(f"\\n📊 Gradient Summary:")
        print(f"  - Total gradient norm: {total_grad_norm:.3f}")
        print(f"  - Average gradient norm: {avg_grad_norm:.3f}")
        print(f"  - Parameters with gradients: {param_count}")
        
        if large_grads:
            print(f"\\n⚠️  Large gradients ({len(large_grads)}):")
            for name, grad_norm in sorted(large_grads, key=lambda x: x[1], reverse=True):
                print(f"    {name}: {grad_norm:.3f}")
        
        # Determine gradient stability
        if total_grad_norm > 10000 or avg_grad_norm > 1000 or any(grad_norm > 10000 for _, grad_norm in large_grads):
            print("\\n🚨 GRADIENT EXPLOSION DETECTED!")
            print("This explains why training fails with infinite loss.")
            return False
        elif avg_grad_norm > 10:
            print("\\n⚠️  Large gradients detected but manageable")
            return True
        else:
            print("\\n✅ Gradients are stable!")
            return True
            
    except Exception as e:
        print(f"❌ Error during training step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_first_training_step()
    
    if success:
        print("\\n✅ FIRST TRAINING STEP TEST PASSED")
        print("The model can complete one training step successfully.")
    else:
        print("\\n❌ FIRST TRAINING STEP TEST FAILED")
        print("Issue identified in the first training step.")