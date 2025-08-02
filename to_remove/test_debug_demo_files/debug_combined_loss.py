#!/usr/bin/env python3
"""
Debug script to test the combined loss function that's causing gradient explosion
in real training of adaptive_coral_gpcm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import create_model
from training.ordinal_losses import CombinedOrdinalLoss
import numpy as np

def load_simple_data(train_path, test_path):
    """Load data exactly like train.py."""
    def read_data(file_path):
        sequences = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if i + 2 >= len(lines):
                    break
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(int, lines[i+2].strip().split(',')))
                
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                sequences.append((questions, responses))
                i += 3
        return sequences
    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    all_questions = []
    all_responses = []
    
    for questions, responses in train_data + test_data:
        all_questions.extend(questions)
        all_responses.extend(responses)
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats

def test_combined_loss_gradient_explosion():
    """Test the combined loss function that's causing gradient explosion."""
    print("🔍 Testing Combined Loss Function (CORAL Weight = 0.4)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load real data
    train_path = "data/synthetic_OC/synthetic_oc_train.txt"
    test_path = "data/synthetic_OC/synthetic_oc_test.txt"
    
    try:
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
        print(f"📊 Data loaded: {len(train_data)} train, {len(test_data)} test")
        print(f"Questions: {n_questions}, Categories: {n_cats}")
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        return False
    
    # Create adaptive_coral_gpcm model
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
        dropout_rate=0.1
    ).to(device)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create the EXACT same combined loss as train.py (the suspect!)
    criterion = CombinedOrdinalLoss(
        n_cats=n_cats,
        ce_weight=0.4,      # Same as --ce_weight 0.4
        qwk_weight=0.2,     # Same as --qwk_weight 0.2
        emd_weight=0.0,     # Same as --emd_weight 0.0
        coral_weight=0.4    # Same as --coral_weight 0.4 (SUSPECT!)
    )
    
    print(f"✓ Combined loss created with weights:")
    print(f"  - CE weight: {criterion.ce_weight}")
    print(f"  - QWK weight: {criterion.qwk_weight}")
    print(f"  - EMD weight: {criterion.emd_weight}")
    print(f"  - CORAL weight: {criterion.coral_weight} (SUSPECT)")
    
    # Get a real batch
    sample_data = train_data[:32]  # First 32 sequences
    
    questions_batch = []
    responses_batch = []
    masks_batch = []
    
    max_len = max(len(seq[0]) for seq in sample_data)
    
    for questions, responses in sample_data:
        padded_questions = questions + [0] * (max_len - len(questions))
        padded_responses = responses + [0] * (max_len - len(responses))
        mask = [1] * len(questions) + [0] * (max_len - len(questions))
        
        questions_batch.append(padded_questions)
        responses_batch.append(padded_responses)
        masks_batch.append(mask)
    
    questions_tensor = torch.tensor(questions_batch, dtype=torch.long).to(device)
    responses_tensor = torch.tensor(responses_batch, dtype=torch.float32).to(device)
    masks_tensor = torch.tensor(masks_batch, dtype=torch.float32).to(device)
    
    print(f"\\n📦 Batch prepared:")
    print(f"  - questions: {questions_tensor.shape}")
    print(f"  - responses: {responses_tensor.shape}")
    print(f"  - masks: {masks_tensor.shape}")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        print(f"\\n🔄 Forward pass...")
        
        # Forward pass
        student_abilities, item_thresholds, discrimination_params, predictions = model(
            questions_tensor, responses_tensor
        )
        
        print(f"✓ Forward pass completed")
        print(f"  - predictions: {predictions.shape}, range [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        # Check for NaN/Inf in predictions
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print(f"🚨 ERROR: NaN/Inf in predictions!")
            return False
        
        # Get CORAL info (this is the suspect!)
        coral_info = None
        if hasattr(model, '_last_coral_logits') and model._last_coral_logits is not None:
            coral_info = {
                'cumulative_logits': model._last_coral_logits,
                'model_type': 'adaptive_coral_gpcm'
            }
            print(f"  - CORAL info available: {model._last_coral_logits.shape}")
        else:
            print(f"  - No CORAL info available")
        
        print(f"\\n🧪 Testing combined loss computation...")
        
        targets = responses_tensor.long()
        mask = masks_tensor.bool()
        
        # Test 1: Simple CE loss only (should be stable)
        print(f"\\n1️⃣ Testing simple CE loss (no CORAL)...")
        simple_criterion = nn.CrossEntropyLoss()
        predictions_flat = predictions.view(-1, n_cats)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)
        
        valid_predictions = predictions_flat[mask_flat]
        valid_targets = targets_flat[mask_flat]
        
        simple_loss = simple_criterion(valid_predictions, valid_targets)
        print(f"✓ Simple CE loss: {simple_loss.item():.6f}")
        
        # Test backward pass with simple loss
        optimizer.zero_grad()
        simple_loss.backward(retain_graph=True)
        
        simple_total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"✓ Simple CE gradients: {simple_total_grad_norm:.3f} (should be stable)")
        
        # Test 2: Combined loss WITHOUT CORAL (coral_weight=0)
        print(f"\\n2️⃣ Testing combined loss without CORAL...")
        
        # Need fresh forward pass for clean computation graph
        print(f"  🔄 Fresh forward pass for clean graph...")
        student_abilities2, item_thresholds2, discrimination_params2, predictions2 = model(
            questions_tensor, responses_tensor
        )
        
        safe_criterion = CombinedOrdinalLoss(
            n_cats=n_cats,
            ce_weight=0.4,
            qwk_weight=0.2,
            emd_weight=0.0,
            coral_weight=0.0  # NO CORAL
        )
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Get fresh CORAL info
        coral_info2 = None
        if hasattr(model, '_last_coral_logits') and model._last_coral_logits is not None:
            coral_info2 = {
                'cumulative_logits': model._last_coral_logits,
                'model_type': 'adaptive_coral_gpcm'
            }
        
        loss_dict = safe_criterion(predictions2, targets, mask, coral_info=coral_info2)
        safe_loss = loss_dict['total_loss']
        print(f"✓ Combined loss (no CORAL): {safe_loss.item():.6f}")
        
        safe_loss.backward()
        
        safe_total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        print(f"✓ Combined (no CORAL) gradients: {safe_total_grad_norm:.3f}")
        
        # Test 3: Combined loss WITH CORAL (coral_weight=0.4) - THE SUSPECT!
        print(f"\\n3️⃣ Testing combined loss WITH CORAL (coral_weight=0.4)...")
        
        # Need fresh forward pass for clean computation graph
        print(f"  🔄 Fresh forward pass for CORAL test...")
        student_abilities3, item_thresholds3, discrimination_params3, predictions3 = model(
            questions_tensor, responses_tensor
        )
        
        # Get fresh CORAL info with correct key names
        coral_info3 = None
        if hasattr(model, '_last_coral_logits') and model._last_coral_logits is not None:
            coral_info3 = {
                'logits': model._last_coral_logits,  # Key should be 'logits', not 'cumulative_logits'
                'model_type': 'adaptive_coral_gpcm'
            }
            print(f"  - CORAL logits: {model._last_coral_logits.shape}")
        
        # Reset gradients
        optimizer.zero_grad()
        
        loss_dict = criterion(predictions3, targets, mask, coral_info=coral_info3)
        combined_loss = loss_dict['total_loss']
        
        print(f"✓ Combined loss (with CORAL): {combined_loss.item():.6f}")
        print(f"  - CE component: {loss_dict.get('ce_loss', 'N/A')}")
        print(f"  - QWK component: {loss_dict.get('qwk_loss', 'N/A')}")
        print(f"  - CORAL component: {loss_dict.get('coral_loss', 'N/A')}")
        
        if torch.isnan(combined_loss) or torch.isinf(combined_loss):
            print(f"🚨 ERROR: Combined loss is NaN/Inf!")
            return False
        
        print(f"  🔄 Backward pass with CORAL loss...")
        combined_loss.backward()
        
        # Analyze gradients
        total_grad_norm = 0
        large_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                
                if grad_norm > 1000:
                    large_grads.append((name, grad_norm))
        
        print(f"✓ Combined (with CORAL) gradients: {total_grad_norm:.3f}")
        
        if large_grads:
            print(f"🚨 LARGE GRADIENTS DETECTED ({len(large_grads)}):")
            for name, grad_norm in sorted(large_grads, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {name}: {grad_norm:.3f}")
        
        if total_grad_norm > 10000:
            print(f"🚨 GRADIENT EXPLOSION CONFIRMED!")
            print(f"The CORAL loss component is causing gradient explosion.")
            return False
        else:
            print(f"✅ Gradients are stable even with CORAL loss")
            return True
        
    except Exception as e:
        print(f"❌ Error during combined loss test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_combined_loss_gradient_explosion()
    
    if success:
        print(f"\\n✅ COMBINED LOSS TEST PASSED")
        print(f"Combined loss with CORAL works correctly.")
    else:
        print(f"\\n❌ COMBINED LOSS TEST FAILED")
        print(f"CORAL loss component causes gradient explosion.")