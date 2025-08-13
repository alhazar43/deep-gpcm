#!/usr/bin/env python3
"""
Debug script to examine the data scale in the synthetic_OC dataset.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from train.py
def load_simple_data(train_path, test_path):
    """Load data in the simple format used by train.py"""
    def parse_file(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                questions = [int(x) for x in parts[::2]]
                responses = [int(x) for x in parts[1::2]]
                data.append({'questions': questions, 'responses': responses})
        return data
    
    train_data = parse_file(train_path)
    test_data = parse_file(test_path)
    
    # Calculate n_questions and n_cats
    all_questions = []
    all_responses = []
    
    for data_list in [train_data, test_data]:
        for student in data_list:
            all_questions.extend(student['questions'])
            all_responses.extend(student['responses'])
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats

def load_data(dataset):
    """Load data for the given dataset."""
    train_path = f"data/{dataset}/{dataset.lower()}_train.txt"
    test_path = f"data/{dataset}/{dataset.lower()}_test.txt"
    return load_simple_data(train_path, test_path)

def debug_data_scale():
    """Debug the data scale and ranges."""
    print("🔍 Debugging Synthetic OC Dataset Scale")
    print("=" * 60)
    
    # Load the actual training data
    try:
        train_data, test_data, n_questions, n_cats = load_data('synthetic_OC')
        print(f"✓ Data loaded successfully")
        print(f"  - n_questions: {n_questions}")
        print(f"  - n_cats: {n_cats}")
        print(f"  - train_data length: {len(train_data)}")
        print(f"  - test_data length: {len(test_data)}")
        
        # Check data ranges
        all_questions = []
        all_responses = []
        all_seq_lengths = []
        
        for student_data in train_data:
            questions = student_data['questions']
            responses = student_data['responses']
            
            all_questions.extend(questions)
            all_responses.extend(responses)
            all_seq_lengths.append(len(questions))
        
        print(f"\n📊 Data Statistics:")
        print(f"  - Question range: [{min(all_questions)}, {max(all_questions)}]") 
        print(f"  - Response range: [{min(all_responses)}, {max(all_responses)}]")
        print(f"  - Sequence length range: [{min(all_seq_lengths)}, {max(all_seq_lengths)}]")
        print(f"  - Average sequence length: {sum(all_seq_lengths)/len(all_seq_lengths):.1f}")
        print(f"  - Total interactions: {len(all_questions)}")
        
        # Check if responses are in valid range
        invalid_responses = [r for r in all_responses if r < 0 or r >= n_cats]
        if invalid_responses:
            print(f"🚨 WARNING: {len(invalid_responses)} invalid responses found!")
            print(f"  - Invalid values: {set(invalid_responses)}")
        else:
            print(f"✅ All responses are in valid range [0, {n_cats-1}]")
            
        # Check if questions are in valid range
        invalid_questions = [q for q in all_questions if q < 0 or q >= n_questions]
        if invalid_questions:
            print(f"🚨 WARNING: {len(invalid_questions)} invalid questions found!")
            print(f"  - Invalid values: {set(invalid_questions)}")
        else:
            print(f"✅ All questions are in valid range [0, {n_questions-1}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_with_real_data():
    """Test the adaptive model with real data sample."""
    print("\n🔍 Testing Adaptive Model with Real Data Sample")
    print("=" * 60)
    
    from models.factory import create_model
    from torch.utils.data import DataLoader
# Import dataset utilities
class ResponseDataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_data_loaders(train_data, test_data, batch_size=64):
    from torch.utils.data import DataLoader
    
    def pad_sequence_batch(batch):
        questions_list = [student['questions'] for student in batch]
        responses_list = [student['responses'] for student in batch]
        
        max_len = max(len(q) for q in questions_list)
        
        padded_questions = []
        padded_responses = []
        masks = []
        
        for questions, responses in zip(questions_list, responses_list):
            seq_len = len(questions)
            
            pad_questions = questions + [0] * (max_len - seq_len)
            pad_responses = responses + [0] * (max_len - seq_len)
            mask = [1] * seq_len + [0] * (max_len - seq_len)
            
            padded_questions.append(pad_questions)
            padded_responses.append(pad_responses)
            masks.append(mask)
        
        return (torch.tensor(padded_questions, dtype=torch.long),
                torch.tensor(padded_responses, dtype=torch.long),
                torch.tensor(masks, dtype=torch.float))
    
    train_dataset = ResponseDataset(train_data)
    test_dataset = ResponseDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_batch)
    
    return train_loader, test_loader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load data
        train_data, test_data, n_questions, n_cats = load_data('synthetic_OC')
        
        # Take a small sample
        small_train_data = train_data[:16]  # Just 16 students
        small_test_data = test_data[:4]     # Just 4 students for testing
        
        train_loader, test_loader = create_data_loaders(small_train_data, small_test_data, batch_size=4)
        
        # Create model
        model = create_model('adaptive_coral_gpcm', n_questions=n_questions, n_cats=n_cats)
        model = model.to(device)
        
        print(f"✓ Model created for real data scale")
        print(f"  - n_questions: {n_questions}")
        print(f"  - n_cats: {n_cats}")
        
        # Test one batch
        for questions, responses, mask in train_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            print(f"✓ Batch loaded:")
            print(f"  - questions shape: {questions.shape}")
            print(f"  - responses shape: {responses.shape}")
            print(f"  - mask shape: {mask.shape}")
            print(f"  - questions range: [{questions.min().item()}, {questions.max().item()}]")
            print(f"  - responses range: [{responses.min().item()}, {responses.max().item()}]")
            
            # Enable gradients
            for param in model.parameters():
                param.requires_grad_(True)
            
            # Forward pass
            student_abilities, item_thresholds, discrimination_params, final_probs = model(questions, responses)
            
            print(f"✓ Forward pass completed")
            print(f"  - student_abilities range: [{student_abilities.min():.6f}, {student_abilities.max():.6f}]")
            print(f"  - item_thresholds range: [{item_thresholds.min():.6f}, {item_thresholds.max():.6f}]")
            print(f"  - final_probs range: [{final_probs.min():.6f}, {final_probs.max():.6f}]")
            
            # Check for NaN/Inf
            if torch.isnan(final_probs).any() or torch.isinf(final_probs).any():
                print("🚨 WARNING: NaN/Inf detected in outputs!")
                return False
            
            # Compute loss (Cross-entropy)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Flatten and apply mask
            probs_flat = final_probs.view(-1, final_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1).bool()
            
            valid_probs = probs_flat[mask_flat]
            valid_responses = responses_flat[mask_flat]
            
            if len(valid_responses) == 0:
                print("🚨 WARNING: No valid responses in batch!")
                continue
            
            loss = criterion(valid_probs, valid_responses)
            print(f"✓ Loss computed: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            total_grad_norm = 0
            param_count = 0
            large_grad_params = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    param_count += 1
                    
                    if grad_norm > 100:
                        large_grad_params.append((name, grad_norm))
            
            avg_grad_norm = total_grad_norm / max(param_count, 1)
            
            print(f"✓ Gradient computation completed")
            print(f"  - Total gradient norm: {total_grad_norm:.3f}")
            print(f"  - Average gradient norm: {avg_grad_norm:.3f}")
            
            if large_grad_params:
                print(f"🚨 Large gradient parameters ({len(large_grad_params)}):")
                for name, grad_norm in sorted(large_grad_params, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {name}: {grad_norm:.3f}")
            
            if avg_grad_norm > 1000:
                print("🚨 GRADIENT EXPLOSION DETECTED!")
                return False
            else:
                print("✅ Gradients are stable with real data!")
                return True
            
            # Test only first batch
            break
            
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        return False

if __name__ == "__main__":
    success1 = debug_data_scale()
    success2 = test_with_real_data()
    
    if success1 and success2:
        print("\n✅ DEBUG PASSED: Data scale is appropriate and model is stable")
    else:
        print("\n❌ DEBUG FAILED: Issues detected with data or model stability")