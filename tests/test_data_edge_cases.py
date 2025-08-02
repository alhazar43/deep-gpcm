#!/usr/bin/env python3
"""
Create comprehensive test dataset with edge cases for unified prediction system.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List

def create_edge_case_data(n_cats: int = 4) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, str]]:
    """Create test cases covering various edge cases.
    
    Returns:
        Dictionary mapping test case name to (probabilities, targets, description)
    """
    test_cases = {}
    
    # 1. Normal case - well-separated probabilities
    test_cases['normal'] = (
        torch.tensor([
            [0.7, 0.2, 0.05, 0.05],
            [0.1, 0.6, 0.2, 0.1],
            [0.05, 0.15, 0.6, 0.2],
            [0.05, 0.05, 0.1, 0.8]
        ]),
        torch.tensor([0, 1, 2, 3]),
        "Normal case with clear predictions"
    )
    
    # 2. Uniform probabilities
    uniform_prob = 1.0 / n_cats
    test_cases['uniform'] = (
        torch.tensor([[uniform_prob] * n_cats] * 4),
        torch.tensor([0, 1, 2, 3]),
        "Uniform probabilities - maximum uncertainty"
    )
    
    # 3. Near-ties
    test_cases['near_ties'] = (
        torch.tensor([
            [0.35, 0.34, 0.16, 0.15],
            [0.24, 0.26, 0.25, 0.25],
            [0.249, 0.251, 0.250, 0.250],
            [0.2500001, 0.2499999, 0.2500001, 0.2499999]
        ]),
        torch.tensor([0, 1, 1, 0]),
        "Near-tie probabilities"
    )
    
    # 4. Extreme confidence
    test_cases['extreme_confidence'] = (
        torch.tensor([
            [0.999, 0.0003, 0.0003, 0.0004],
            [0.0001, 0.9997, 0.0001, 0.0001],
            [1e-6, 1e-6, 0.999998, 0],
            [1e-8, 1e-8, 1e-8, 1.0 - 3e-8]
        ]),
        torch.tensor([0, 1, 2, 3]),
        "Extreme confidence predictions"
    )
    
    # 5. Ordinal structure - smooth transitions
    test_cases['ordinal_smooth'] = (
        torch.tensor([
            [0.4, 0.3, 0.2, 0.1],     # Peak at 0
            [0.2, 0.4, 0.3, 0.1],     # Peak at 1
            [0.1, 0.3, 0.4, 0.2],     # Peak at 2
            [0.1, 0.2, 0.3, 0.4]      # Peak at 3
        ]),
        torch.tensor([0, 1, 2, 3]),
        "Smooth ordinal transitions"
    )
    
    # 6. Bimodal distributions
    test_cases['bimodal'] = (
        torch.tensor([
            [0.45, 0.05, 0.05, 0.45],  # Bimodal at extremes
            [0.05, 0.45, 0.45, 0.05],  # Bimodal at middle
            [0.40, 0.10, 0.40, 0.10],  # Bimodal alternating
            [0.33, 0.34, 0.0, 0.33]    # Trimodal-ish
        ]),
        torch.tensor([0, 1, 2, 3]),
        "Bimodal distributions"
    )
    
    # 7. Edge categories only
    test_cases['edge_only'] = (
        torch.tensor([
            [0.9, 0.1, 0.0, 0.0],      # Only low categories
            [0.8, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.8],      # Only high categories
            [0.0, 0.0, 0.1, 0.9]
        ]),
        torch.tensor([0, 0, 3, 3]),
        "Edge categories only"
    )
    
    # 8. Numerical precision edge cases
    test_cases['numerical_precision'] = (
        torch.tensor([
            [1.0 - 1e-7, 1e-7/3, 1e-7/3, 1e-7/3],
            [1e-15, 1.0 - 3e-15, 1e-15, 1e-15],
            [0.25 + 1e-10, 0.25 - 1e-10, 0.25, 0.25],
            [0.333333333, 0.333333334, 0.333333333, 0]
        ]),
        torch.tensor([0, 1, 0, 1]),
        "Numerical precision edge cases"
    )
    
    # 9. Zero probability cases (need handling)
    test_cases['zero_probs'] = (
        torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        torch.tensor([0, 1, 2, 3]),
        "Perfect predictions with zero probabilities"
    )
    
    # 10. Mixed confidence levels
    test_cases['mixed_confidence'] = (
        torch.tensor([
            [0.9, 0.05, 0.03, 0.02],    # High confidence
            [0.3, 0.3, 0.2, 0.2],       # Low confidence
            [0.6, 0.3, 0.07, 0.03],     # Medium confidence
            [0.25, 0.25, 0.25, 0.25]    # No confidence
        ]),
        torch.tensor([0, 1, 0, 2]),
        "Mixed confidence levels"
    )
    
    return test_cases

def create_cumulative_test_cases(n_cats: int = 4) -> Dict[str, Tuple[torch.Tensor, List[float], str]]:
    """Create test cases for cumulative probability conversion.
    
    Returns:
        Dictionary mapping test case name to (categorical_probs, thresholds, description)
    """
    test_cases = {}
    
    # Standard thresholds
    standard_thresholds = [0.25, 0.5, 0.75]
    
    # Skewed thresholds (favoring lower categories)
    skewed_low = [0.1, 0.3, 0.6]
    
    # Skewed thresholds (favoring higher categories)
    skewed_high = [0.4, 0.7, 0.9]
    
    # Extreme thresholds
    extreme = [0.05, 0.5, 0.95]
    
    # Create categorical probability examples
    cat_probs = create_edge_case_data(n_cats)
    
    test_cases['standard_thresholds'] = (
        cat_probs['normal'][0],
        standard_thresholds,
        "Standard evenly-spaced thresholds"
    )
    
    test_cases['skewed_low_thresholds'] = (
        cat_probs['ordinal_smooth'][0],
        skewed_low,
        "Thresholds favoring lower categories"
    )
    
    test_cases['skewed_high_thresholds'] = (
        cat_probs['ordinal_smooth'][0],
        skewed_high,
        "Thresholds favoring higher categories"
    )
    
    test_cases['extreme_thresholds'] = (
        cat_probs['bimodal'][0],
        extreme,
        "Extreme threshold spacing"
    )
    
    return test_cases

def create_sequence_test_cases(n_cats: int = 4, seq_len: int = 10) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """Create sequence-level test cases with masks.
    
    Returns:
        Dictionary mapping test case name to (probs, targets, mask, description)
    """
    test_cases = {}
    
    # 1. Full sequence (no padding)
    probs = torch.rand(1, seq_len, n_cats)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    targets = torch.randint(0, n_cats, (1, seq_len))
    mask = torch.ones(1, seq_len, dtype=torch.bool)
    test_cases['full_sequence'] = (probs, targets, mask, "Full sequence without padding")
    
    # 2. Partial sequence (with padding)
    valid_len = 6
    probs = torch.rand(1, seq_len, n_cats)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    probs[0, valid_len:] = 0  # Zero out padded positions
    targets = torch.randint(0, n_cats, (1, seq_len))
    targets[0, valid_len:] = 0
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[0, :valid_len] = True
    test_cases['partial_sequence'] = (probs, targets, mask, "Partial sequence with padding")
    
    # 3. Single element sequence
    probs = torch.zeros(1, seq_len, n_cats)
    probs[0, 0] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    targets = torch.zeros(1, seq_len, dtype=torch.long)
    targets[0, 0] = 0
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[0, 0] = True
    test_cases['single_element'] = (probs, targets, mask, "Single element sequence")
    
    # 4. Empty sequence (all padding)
    probs = torch.zeros(1, seq_len, n_cats)
    targets = torch.zeros(1, seq_len, dtype=torch.long)
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    test_cases['empty_sequence'] = (probs, targets, mask, "Empty sequence (all padding)")
    
    return test_cases

def create_batch_test_cases(n_cats: int = 4, batch_size: int = 4, seq_len: int = 5) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]:
    """Create batch-level test cases.
    
    Returns:
        Dictionary mapping test case name to (probs, targets, mask, description)
    """
    test_cases = {}
    
    # 1. Mixed lengths in batch
    probs = torch.rand(batch_size, seq_len, n_cats)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    targets = torch.randint(0, n_cats, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Different lengths for each sequence
    lengths = [5, 3, 4, 1]
    for i, length in enumerate(lengths):
        mask[i, length:] = False
        probs[i, length:] = 0
        targets[i, length:] = 0
    test_cases['mixed_lengths'] = (probs, targets, mask, "Batch with mixed sequence lengths")
    
    # 2. Large batch stress test
    large_batch = 128
    probs = torch.rand(large_batch, seq_len, n_cats)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    targets = torch.randint(0, n_cats, (large_batch, seq_len))
    mask = torch.ones(large_batch, seq_len, dtype=torch.bool)
    test_cases['large_batch'] = (probs, targets, mask, "Large batch stress test")
    
    return test_cases

def validate_test_case(probs: torch.Tensor, targets: torch.Tensor, 
                      mask: torch.Tensor = None, name: str = "unnamed") -> bool:
    """Validate a test case for correctness."""
    # Check probability constraints
    prob_sums = probs.sum(dim=-1)
    if not torch.allclose(prob_sums[mask] if mask is not None else prob_sums, 
                         torch.ones_like(prob_sums[mask] if mask is not None else prob_sums), 
                         atol=1e-6):
        print(f"Test case '{name}' failed: probabilities don't sum to 1")
        return False
    
    # Check probability range
    if (probs < -1e-7).any() or (probs > 1 + 1e-7).any():
        print(f"Test case '{name}' failed: probabilities out of [0,1] range")
        return False
    
    # Check target range
    n_cats = probs.shape[-1]
    if (targets < 0).any() or (targets >= n_cats).any():
        if mask is None or (targets[mask] < 0).any() or (targets[mask] >= n_cats).any():
            print(f"Test case '{name}' failed: targets out of range")
            return False
    
    return True

def save_test_cases(filename: str = "test_edge_cases.pt"):
    """Save all test cases to a file."""
    all_cases = {
        'single_predictions': create_edge_case_data(),
        'cumulative': create_cumulative_test_cases(),
        'sequences': create_sequence_test_cases(),
        'batches': create_batch_test_cases()
    }
    
    # Validate all test cases
    print("Validating test cases...")
    for category, cases in all_cases.items():
        print(f"\n{category}:")
        for name, data in cases.items():
            if category == 'cumulative':
                probs, thresholds, desc = data
                valid = validate_test_case(probs, torch.zeros(probs.shape[0], dtype=torch.long), name=name)
            else:
                probs, targets, *rest = data
                mask = rest[0] if len(rest) > 1 else None
                valid = validate_test_case(probs, targets, mask, name=name)
            print(f"  {name}: {'✓' if valid else '✗'}")
    
    torch.save(all_cases, filename)
    print(f"\nTest cases saved to {filename}")

if __name__ == "__main__":
    save_test_cases("tests/data/unified_prediction_test_cases.pt")