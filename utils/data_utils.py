"""
Data Utilities for Deep-GPCM
Unified data loading and processing functions.
"""

import torch
import numpy as np
from typing import List, Tuple, Iterator, Dict, Optional
import re


def parse_dataset_name(dataset_name: str) -> Dict[str, Optional[int]]:
    """
    Parse dataset name to extract configuration parameters.
    
    Supports formats:
    - New format: synthetic_<students>_<max_questions>_<categories>
      e.g., synthetic_4000_200_2, synthetic_4000_200_3, synthetic_4000_200_5
    - Legacy format: synthetic_OC, synthetic_PC
    
    Args:
        dataset_name: Dataset name string
        
    Returns:
        Dictionary with:
        - n_students: Number of students (or None for legacy)
        - max_questions: Maximum questions (or None for legacy)
        - n_cats: Number of categories
        - format: 'new' or 'legacy'
    """
    # Try new format first: synthetic_<students>_<max_questions>_<categories>
    new_format_match = re.match(r'^synthetic_(\d+)_(\d+)_(\d+)$', dataset_name)
    if new_format_match:
        return {
            'n_students': int(new_format_match.group(1)),
            'max_questions': int(new_format_match.group(2)),
            'n_cats': int(new_format_match.group(3)),
            'format': 'new'
        }
    
    # Check legacy format: synthetic_OC or synthetic_PC
    if dataset_name == 'synthetic_OC':
        return {
            'n_students': None,
            'max_questions': None,
            'n_cats': 4,  # OC = Ordinal Categories = 4
            'format': 'legacy'
        }
    elif dataset_name == 'synthetic_PC':
        return {
            'n_students': None,
            'max_questions': None,
            'n_cats': 3,  # PC = Partial Credit = 3
            'format': 'legacy'
        }
    
    # Not a recognized synthetic dataset format
    return {
        'n_students': None,
        'max_questions': None,
        'n_cats': None,
        'format': 'unknown'
    }


def is_synthetic_dataset(dataset_name: str) -> bool:
    """Check if dataset is synthetic based on name."""
    return dataset_name.startswith('synthetic_')


def get_dataset_params(dataset_name: str) -> Dict[str, Optional[int]]:
    """
    Get dataset parameters with backward compatibility.
    
    Args:
        dataset_name: Dataset name
        
    Returns:
        Dictionary with dataset parameters including n_cats
    """
    if is_synthetic_dataset(dataset_name):
        params = parse_dataset_name(dataset_name)
        if params['format'] != 'unknown':
            return params
    
    # Default parameters for non-synthetic or unrecognized datasets
    return {
        'n_students': None,
        'max_questions': None,
        'n_cats': 4,  # Default to 4 categories
        'format': 'default'
    }


class UnifiedDataLoader:
    """
    Unified data loader for Deep-GPCM models.
    
    Handles sequence data with proper batching and device management.
    """
    
    def __init__(self, questions: List[List[int]], responses: List[List[int]], 
                 batch_size: int = 32, shuffle: bool = True, device: torch.device = None):
        self.questions = questions
        self.responses = responses
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or torch.device('cpu')
        
        # Validate data
        assert len(questions) == len(responses), "Questions and responses must have same length"
        
        self.n_samples = len(questions)
        self.indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Extract batch data
            batch_questions = [self.questions[i] for i in batch_indices]
            batch_responses = [self.responses[i] for i in batch_indices]
            
            # Convert to tensors
            q_batch, r_batch, mask_batch = self._collate_batch(batch_questions, batch_responses)
            
            yield q_batch, r_batch, mask_batch
    
    def _collate_batch(self, questions: List[List[int]], responses: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate a batch with proper padding and masking."""
        batch_size = len(questions)
        max_len = max(len(seq) for seq in questions)
        
        # Create padded tensors
        q_batch = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        r_batch = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        
        # Fill tensors
        for i, (q_seq, r_seq) in enumerate(zip(questions, responses)):
            seq_len = len(q_seq)
            q_batch[i, :seq_len] = torch.tensor(q_seq, dtype=torch.long, device=self.device)
            r_batch[i, :seq_len] = torch.tensor(r_seq, dtype=torch.long, device=self.device)
            mask_batch[i, :seq_len] = True
        
        return q_batch, r_batch, mask_batch


def validate_data_format(questions: List[List[int]], responses: List[List[int]], 
                        n_cats: int) -> Tuple[bool, str]:
    """
    Validate data format and consistency.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check basic structure
        if len(questions) != len(responses):
            return False, "Questions and responses must have same number of sequences"
        
        if len(questions) == 0:
            return False, "No data provided"
        
        # Check sequence lengths
        for i, (q_seq, r_seq) in enumerate(zip(questions, responses)):
            if len(q_seq) != len(r_seq):
                return False, f"Sequence {i}: Questions and responses must have same length"
            
            if len(q_seq) == 0:
                return False, f"Sequence {i}: Empty sequence not allowed"
        
        # Check value ranges
        for i, r_seq in enumerate(responses):
            for j, response in enumerate(r_seq):
                if response < 0 or response >= n_cats:
                    return False, f"Sequence {i}, position {j}: Response {response} out of range [0, {n_cats})"
        
        return True, "Data format is valid"
        
    except Exception as e:
        return False, f"Data validation error: {e}"


def get_data_statistics(questions: List[List[int]], responses: List[List[int]], 
                       n_cats: int) -> dict:
    """Get comprehensive data statistics."""
    
    # Basic statistics
    n_sequences = len(questions)
    seq_lengths = [len(seq) for seq in questions]
    
    # Question statistics
    all_questions = [q for seq in questions for q in seq]
    unique_questions = set(all_questions)
    
    # Response statistics
    all_responses = [r for seq in responses for r in seq]
    response_counts = np.bincount(all_responses, minlength=n_cats)
    
    stats = {
        'n_sequences': n_sequences,
        'total_interactions': len(all_responses),
        'sequence_lengths': {
            'min': min(seq_lengths),
            'max': max(seq_lengths),
            'mean': np.mean(seq_lengths),
            'std': np.std(seq_lengths)
        },
        'questions': {
            'total_unique': len(unique_questions),
            'min_id': min(all_questions),
            'max_id': max(all_questions)
        },
        'responses': {
            'n_categories': n_cats,
            'distribution': response_counts.tolist(),
            'proportions': (response_counts / len(all_responses)).tolist()
        }
    }
    
    return stats


def print_data_summary(questions: List[List[int]], responses: List[List[int]], 
                      n_cats: int, dataset_name: str = "Dataset"):
    """Print a formatted data summary."""
    
    stats = get_data_statistics(questions, responses, n_cats)
    
    print(f"\n📊 {dataset_name.upper()} DATA SUMMARY")
    print("-" * 50)
    print(f"Sequences: {stats['n_sequences']:,}")
    print(f"Total Interactions: {stats['total_interactions']:,}")
    print(f"Questions: {stats['questions']['total_unique']} unique (ID {stats['questions']['min_id']}-{stats['questions']['max_id']})")
    print(f"Categories: {stats['responses']['n_categories']}")
    
    print(f"\nSequence Lengths:")
    print(f"  Min: {stats['sequence_lengths']['min']}")
    print(f"  Max: {stats['sequence_lengths']['max']}")
    print(f"  Mean: {stats['sequence_lengths']['mean']:.1f} ± {stats['sequence_lengths']['std']:.1f}")
    
    print(f"\nResponse Distribution:")
    for i, (count, prop) in enumerate(zip(stats['responses']['distribution'], 
                                         stats['responses']['proportions'])):
        print(f"  Category {i}: {count:,} ({prop:.1%})")


def create_synthetic_data(n_sequences: int = 100, n_questions: int = 50, 
                         n_cats: int = 4, min_len: int = 5, max_len: int = 20,
                         seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create synthetic data for testing purposes.
    
    Args:
        n_sequences: Number of sequences to generate
        n_questions: Number of unique questions
        n_cats: Number of response categories
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (questions, responses)
    """
    np.random.seed(seed)
    
    questions = []
    responses = []
    
    for _ in range(n_sequences):
        seq_len = np.random.randint(min_len, max_len + 1)
        
        # Generate question sequence
        q_seq = np.random.choice(n_questions, size=seq_len, replace=True).tolist()
        
        # Generate response sequence (with some correlation to questions)
        r_seq = []
        for q in q_seq:
            # Simple model: questions have different difficulty levels
            difficulty = (q % n_cats) / n_cats  # 0 to 1
            # Higher difficulty -> lower probability of high scores
            probs = np.array([0.4 - 0.3*difficulty, 0.3, 0.2, 0.1 + 0.3*difficulty])
            probs = probs / probs.sum()  # Normalize
            
            response = np.random.choice(n_cats, p=probs)
            r_seq.append(response)
        
        questions.append(q_seq)
        responses.append(r_seq)
    
    return questions, responses


# Testing functionality (consolidated from debug files)
def test_data_loader():
    """Test the unified data loader."""
    print("🧪 Testing UnifiedDataLoader...")
    
    # Create test data
    questions, responses = create_synthetic_data(n_sequences=10, n_questions=5, n_cats=4)
    
    # Test data loader
    loader = UnifiedDataLoader(questions, responses, batch_size=3, shuffle=False)
    
    print(f"Data loader length: {len(loader)}")
    
    for i, (q_batch, r_batch, mask_batch) in enumerate(loader):
        print(f"Batch {i+1}: Questions {q_batch.shape}, Responses {r_batch.shape}, Mask {mask_batch.shape}")
        if i >= 2:  # Test first few batches
            break
    
    print("✅ Data loader test passed!")


def test_data_validation():
    """Test data validation functionality."""
    print("🧪 Testing data validation...")
    
    # Valid data
    questions = [[1, 2, 3], [4, 5]]
    responses = [[0, 1, 2], [1, 0]]
    is_valid, msg = validate_data_format(questions, responses, 4)
    assert is_valid, f"Valid data should pass validation: {msg}"
    
    # Invalid data - mismatched lengths
    questions = [[1, 2, 3], [4, 5]]
    responses = [[0, 1], [1, 0]]
    is_valid, msg = validate_data_format(questions, responses, 4)
    assert not is_valid, "Mismatched lengths should fail validation"
    
    # Invalid data - out of range responses
    questions = [[1, 2]]
    responses = [[0, 5]]  # 5 is out of range for n_cats=4
    is_valid, msg = validate_data_format(questions, responses, 4)
    assert not is_valid, "Out of range responses should fail validation"
    
    print("✅ Data validation test passed!")


if __name__ == "__main__":
    # Run tests
    test_data_loader()
    test_data_validation()
    
    # Create and display sample data
    questions, responses = create_synthetic_data(n_sequences=5, n_questions=10, n_cats=4)
    print_data_summary(questions, responses, 4, "Test")