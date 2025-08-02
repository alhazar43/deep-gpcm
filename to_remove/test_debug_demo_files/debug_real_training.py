#!/usr/bin/env python3
"""
Debug script to reproduce the exact gradient explosion conditions from train.py
using the real synthetic_OC dataset and cross-validation setup.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.factory import create_model
import numpy as np

def load_simple_data(train_path, test_path):
    """Exact same data loading as train.py."""
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
                
                # Ensure lengths match
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                sequences.append((questions, responses))
                i += 3
        return sequences
    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    # Get dataset stats
    all_questions = []
    all_responses = []
    
    for questions, responses in train_data + test_data:
        all_questions.extend(questions)
        all_responses.extend(responses)
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats

def collate_fn(batch):
    """Collate function for data loading (from train.py)."""
    batch_questions = []
    batch_responses = []
    batch_masks = []
    
    max_len = max(len(seq[0]) for seq in batch)
    
    for questions, responses in batch:
        # Pad sequences
        padded_questions = questions + [0] * (max_len - len(questions))
        padded_responses = responses + [0] * (max_len - len(responses))
        mask = [1] * len(questions) + [0] * (max_len - len(questions))
        
        batch_questions.append(padded_questions)
        batch_responses.append(padded_responses)
        batch_masks.append(mask)
    
    return (torch.tensor(batch_questions, dtype=torch.long),
            torch.tensor(batch_responses, dtype=torch.float32),
            torch.tensor(batch_masks, dtype=torch.float32))

def test_real_training_conditions():
    """Test with the exact same conditions as real training."""
    print("🔍 Testing Real Training Conditions (synthetic_OC)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load exact same data as train.py
    train_path = "data/synthetic_OC/synthetic_oc_train.txt"
    test_path = "data/synthetic_OC/synthetic_oc_test.txt"
    
    try:
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
        print(f"📊 Data loaded: {len(train_data)} train, {len(test_data)} test")
        print(f"Questions: {n_questions}, Categories: {n_cats}")
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        return False
    
    # Analyze data ranges
    all_questions = []
    all_responses = []
    seq_lengths = []
    
    for questions, responses in train_data:
        all_questions.extend(questions)
        all_responses.extend(responses)
        seq_lengths.append(len(questions))
    
    print(f"📈 Data Analysis:")
    print(f"  - Question range: [{min(all_questions)}, {max(all_questions)}]")
    print(f"  - Response range: [{min(all_responses)}, {max(all_responses)}]")
    print(f"  - Sequence lengths: [{min(seq_lengths)}, {max(seq_lengths)}]")
    print(f"  - Avg sequence length: {np.mean(seq_lengths):.1f}")
    
    # Create exact same model as train.py
    print(f"\\n🤖 Creating adaptive_coral_gpcm model...")
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
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Enable adaptive blending: {model.enable_adaptive_blending}")
    print(f"  - BGT parameters:")
    print(f"    Range sensitivity: {model.threshold_blender.range_sensitivity.item():.3f}")
    print(f"    Distance sensitivity: {model.threshold_blender.distance_sensitivity.item():.3f}")
    print(f"    Baseline bias: {model.threshold_blender.baseline_bias.item():.3f}")
    
    # Create data loader (exact same as train.py)
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor([[0]], dtype=torch.long),  # Dummy, will be replaced by collate_fn
        torch.tensor([[0]], dtype=torch.float32)
    )
    
    # Use actual data with custom dataset
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = CustomDataset(train_data)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32,  # Same as train.py
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Create optimizer (same as train.py)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    try:
        print(f"\\n🔄 Testing with real data batches...")
        
        # Get first few batches (same as train.py)
        batch_count = 0
        for batch_idx, (questions, responses, masks) in enumerate(train_loader):
            if batch_count >= 3:  # Test first 3 batches like train.py
                break
                
            batch_count += 1
            
            questions = questions.to(device)
            responses = responses.to(device)
            masks = masks.to(device).bool()
            
            print(f"\\n📦 Batch {batch_idx + 1}:")
            print(f"  - questions: {questions.shape}, range [{questions.min()}, {questions.max()}]")
            print(f"  - responses: {responses.shape}, range [{responses.min():.3f}, {responses.max():.3f}]")
            print(f"  - masks: {masks.sum().item()}/{masks.numel()} valid positions")
            
            # Check for invalid data
            if torch.isnan(questions).any() or torch.isnan(responses).any():
                print("🚨 ERROR: NaN detected in input data!")
                return False
            
            # Forward pass
            print(f"  🔄 Forward pass...")
            student_abilities, item_thresholds, discrimination_params, predictions = model(
                questions, responses
            )
            
            print(f"  ✓ Forward pass completed")
            print(f"    - student_abilities: range [{student_abilities.min():.6f}, {student_abilities.max():.6f}]")
            print(f"    - item_thresholds: range [{item_thresholds.min():.6f}, {item_thresholds.max():.6f}]")
            print(f"    - predictions: range [{predictions.min():.6f}, {predictions.max():.6f}]")
            
            # Check for NaN/Inf in outputs
            outputs_to_check = [
                (student_abilities, "student_abilities"),
                (item_thresholds, "item_thresholds"),
                (predictions, "predictions")
            ]
            
            if discrimination_params is not None:
                outputs_to_check.append((discrimination_params, "discrimination_params"))
            
            for tensor, name in outputs_to_check:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"  🚨 ERROR: NaN/Inf detected in {name}!")
                    return False
            
            # Loss computation (exact same as train.py)
            print(f"  🧪 Loss computation...")
            
            targets = responses.long()
            
            # Apply mask
            predictions_flat = predictions.view(-1, n_cats)
            targets_flat = targets.view(-1)
            masks_flat = masks.view(-1)
            
            valid_predictions = predictions_flat[masks_flat]
            valid_targets = targets_flat[masks_flat]
            
            if len(valid_targets) == 0:
                print(f"  ⚠️  No valid targets in batch!")
                continue
            
            # Check target validity
            if (valid_targets < 0).any() or (valid_targets >= n_cats).any():
                print(f"  🚨 ERROR: Invalid targets! Range: [{valid_targets.min()}, {valid_targets.max()}]")
                return False
            
            loss = criterion(valid_predictions, valid_targets)
            
            print(f"  ✓ Loss computed: {loss.item():.6f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  🚨 ERROR: Loss is NaN/Inf!")
                return False
            
            # Backward pass
            print(f"  🔄 Backward pass...")
            
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            total_grad_norm = 0
            large_grads = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    
                    if grad_norm > 1000:
                        large_grads.append((name, grad_norm))
            
            print(f"  📊 Gradient analysis:")
            print(f"    - Total gradient norm: {total_grad_norm:.3f}")
            
            if large_grads:
                print(f"    🚨 LARGE GRADIENTS ({len(large_grads)}):")
                for name, grad_norm in sorted(large_grads, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"      {name}: {grad_norm:.3f}")
                
                if total_grad_norm > 10000:
                    print(f"  🚨 GRADIENT EXPLOSION DETECTED!")
                    print(f"  This explains why training fails.")
                    return False
            else:
                print(f"    ✅ Gradients are stable")
        
        print(f"\\n✅ All batches processed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during real training test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_training_conditions()
    
    if success:
        print(f"\\n✅ REAL TRAINING CONDITIONS TEST PASSED")
        print(f"The model works with real data - issue may be in training loop.")
    else:
        print(f"\\n❌ REAL TRAINING CONDITIONS TEST FAILED")
        print(f"Gradient explosion reproduced with real data conditions.")