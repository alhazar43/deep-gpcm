#!/usr/bin/env python3
"""
Investigate what data the evaluation script actually processes.
"""

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import numpy as np
from utils.data_utils import UnifiedDataLoader

def load_simple_data(train_path, test_path):
    """Simple data loading function matching evaluate.py."""
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
    return train_data, test_data

def investigate_evaluation_setup():
    """Investigate the exact evaluation setup that produced 63,296 samples."""
    print("=== INVESTIGATING EVALUATION DATA ===")
    
    dataset = "synthetic_OC"
    data_dir = f"data/{dataset}"
    
    train_path = f"{data_dir}/synthetic_oc_train.txt"
    test_path = f"{data_dir}/synthetic_oc_test.txt"
    
    print(f"Dataset: {dataset}")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")
    
    # Load data exactly as evaluate.py does
    train_data, test_data = load_simple_data(train_path, test_path)
    
    print(f"\nRaw data loaded:")
    print(f"  Train sequences: {len(train_data)}")
    print(f"  Test sequences: {len(test_data)}")
    
    # Extract questions and responses
    test_questions = [seq[0] for seq in test_data]
    test_responses = [seq[1] for seq in test_data]
    
    print(f"\nProcessing test data through UnifiedDataLoader...")
    
    # Create test loader (batch_size=32, default in evaluate.py)
    test_loader = UnifiedDataLoader(
        test_questions, test_responses, 
        batch_size=32, shuffle=False, device=torch.device('cpu')
    )
    
    print(f"Test loader created with {len(test_loader)} batches")
    
    # Process exactly like evaluate.py does in lines 214-225
    all_targets = []
    
    for batch_idx, (questions, responses, mask) in enumerate(test_loader):
        # This is the key: responses.view(-1) flattens ALL responses including padding
        responses_flat = responses.view(-1)
        all_targets.append(responses_flat.cpu())
    
    # Combine all (including padding)
    all_targets_with_padding = torch.cat(all_targets, dim=0)
    
    print(f"\nWith padding: {len(all_targets_with_padding)} samples")
    
    # Now filter out padding (this is what should happen but might not be)
    all_targets_no_padding = []
    
    for batch_idx, (questions, responses, mask) in enumerate(test_loader):
        responses_flat = responses.view(-1)
        mask_flat = mask.view(-1)
        
        # Only keep non-padded responses
        valid_responses = responses_flat[mask_flat]
        all_targets_no_padding.extend(valid_responses.tolist())
    
    print(f"Without padding: {len(all_targets_no_padding)} samples")
    
    # Check if there's any data combination happening
    print(f"\n=== HYPOTHESIS TESTING ===")
    print(f"Standard test data: {len(all_targets_no_padding)} samples")
    print(f"Confusion matrix shows: 63,296 samples")
    print(f"Difference: {63296 - len(all_targets_no_padding)} samples")
    
    # Could it be train + test?
    train_questions = [seq[0] for seq in train_data]
    train_responses = [seq[1] for seq in train_data]
    
    train_loader = UnifiedDataLoader(
        train_questions, train_responses,
        batch_size=32, shuffle=False, device=torch.device('cpu')
    )
    
    all_train_responses = []
    for batch_idx, (questions, responses, mask) in enumerate(train_loader):
        responses_flat = responses.view(-1)
        mask_flat = mask.view(-1)
        valid_responses = responses_flat[mask_flat]
        all_train_responses.extend(valid_responses.tolist())
    
    combined_total = len(all_train_responses) + len(all_targets_no_padding)
    print(f"\nTrain responses: {len(all_train_responses)}")
    print(f"Train + Test total: {combined_total}")
    print(f"Matches confusion matrix: {combined_total == 63296}")
    
    if combined_total == 63296:
        print("✅ FOUND IT: Models were evaluated on COMBINED train+test data!")
        
        # Check category distribution of combined data
        all_combined = all_train_responses + all_targets_no_padding
        print(f"\n=== COMBINED DATA CATEGORY DISTRIBUTION ===")
        categories = np.arange(4)
        for cat in categories:
            count = (np.array(all_combined) == cat).sum()
            percentage = (count / len(all_combined)) * 100
            print(f"Cat {cat}: {count:,} responses ({percentage:.1f}%)")

if __name__ == "__main__":
    investigate_evaluation_setup()