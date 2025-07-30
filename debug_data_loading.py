#!/usr/bin/env python3
"""
Debug script to analyze data loading discrepancy.
"""

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import numpy as np
from utils.data_utils import UnifiedDataLoader

def load_simple_data(train_path, test_path):
    """Simple data loading function."""
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

def analyze_data_processing():
    """Analyze how data is processed during loading."""
    print("=== DATA LOADING ANALYSIS ===")
    
    # Load raw data
    train_path = "data/synthetic_OC/synthetic_oc_train.txt"
    test_path = "data/synthetic_OC/synthetic_oc_test.txt"
    
    print(f"Loading data from {test_path}...")
    train_data, test_data = load_simple_data(train_path, test_path)
    
    print(f"Raw test sequences: {len(test_data)}")
    
    # Analyze raw sequences
    raw_responses = []
    for questions, responses in test_data:
        raw_responses.extend(responses)
    
    print(f"Raw total responses: {len(raw_responses)}")
    
    # Category breakdown of raw data
    print("\n=== RAW DATA CATEGORY DISTRIBUTION ===")
    categories = np.arange(4)
    for cat in categories:
        count = (np.array(raw_responses) == cat).sum()
        percentage = (count / len(raw_responses)) * 100
        print(f"Cat {cat}: {count:,} responses ({percentage:.1f}%)")
    
    # Now process through data loader
    print("\n=== PROCESSED DATA ANALYSIS ===")
    test_questions = [seq[0] for seq in test_data]
    test_responses = [seq[1] for seq in test_data]
    
    # Create data loader (same as evaluation)
    test_loader = UnifiedDataLoader(
        test_questions, test_responses, 
        batch_size=32, shuffle=False, device=torch.device('cpu')
    )
    
    print(f"Data loader batches: {len(test_loader)}")
    
    # Process through loader
    all_processed_responses = []
    total_samples = 0
    
    for batch_idx, (questions, responses, mask) in enumerate(test_loader):
        # Flatten responses like in evaluation
        responses_flat = responses.view(-1)
        mask_flat = mask.view(-1)
        
        # Only keep valid (non-padded) responses
        valid_responses = responses_flat[mask_flat]
        all_processed_responses.extend(valid_responses.tolist())
        total_samples += valid_responses.size(0)
        
        if batch_idx < 3:  # Show first few batches
            print(f"Batch {batch_idx}: {questions.shape}, valid responses: {valid_responses.size(0)}")
    
    print(f"\nProcessed total responses: {total_samples}")
    print(f"Matches confusion matrix total (63,296): {total_samples == 63296}")
    
    # Category breakdown of processed data
    print("\n=== PROCESSED DATA CATEGORY DISTRIBUTION ===")
    all_processed_responses = np.array(all_processed_responses)
    for cat in categories:
        count = (all_processed_responses == cat).sum()
        percentage = (count / len(all_processed_responses)) * 100
        print(f"Cat {cat}: {count:,} responses ({percentage:.1f}%)")
    
    print(f"\n=== COMPARISON ===")
    print(f"Raw responses: {len(raw_responses):,}")
    print(f"Processed responses: {len(all_processed_responses):,}")
    print(f"Difference: {len(all_processed_responses) - len(raw_responses):,}")
    
    if len(all_processed_responses) != len(raw_responses):
        print("❌ Data processing is modifying the total number of responses!")
        print("This suggests padding/masking or sequence truncation is happening.")

if __name__ == "__main__":
    analyze_data_processing()