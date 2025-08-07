"""
Data Loading Utilities for Deep-GPCM Pipeline
Extracted from train.py to eliminate dependencies and provide consistent data loading across all components.
"""

import torch
import torch.utils.data as data_utils
from pathlib import Path
from typing import List, Tuple, Union


def load_simple_data(train_path: Union[str, Path], test_path: Union[str, Path]) -> Tuple[List, List, int, int]:
    """
    Simple data loading function for Deep-GPCM text format.
    
    Args:
        train_path: Path to training data file
        test_path: Path to test data file
        
    Returns:
        Tuple of (train_data, test_data, n_questions, n_cats)
        where train_data and test_data are lists of (questions, responses) tuples
    """
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
    
    # Find number of questions and categories
    all_questions = []
    all_responses = []
    for q, r in train_data + test_data:
        all_questions.extend(q)
        all_responses.extend(r)
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats


def pad_sequence_batch(batch):
    """Collate function for padding sequences."""
    questions_batch, responses_batch = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(seq) for seq in questions_batch)
    
    # Pad sequences
    questions_padded = []
    responses_padded = []
    masks = []
    
    for q, r in zip(questions_batch, responses_batch):
        q_len = len(q)
        # Pad questions and responses
        q_pad = q + [0] * (max_len - q_len)
        r_pad = r + [0] * (max_len - q_len)
        mask = [True] * q_len + [False] * (max_len - q_len)
        
        questions_padded.append(q_pad)
        responses_padded.append(r_pad)
        masks.append(mask)
    
    return (torch.tensor(questions_padded), 
            torch.tensor(responses_padded), 
            torch.tensor(masks, dtype=torch.bool))


def create_data_loaders(train_data: List, test_data: List, batch_size: int = 32) -> Tuple[data_utils.DataLoader, data_utils.DataLoader]:
    """
    Create data loaders from raw sequence data.
    
    Args:
        train_data: List of (questions, responses) tuples
        test_data: List of (questions, responses) tuples
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    class SequenceDataset(data_utils.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = SequenceDataset(train_data)
    test_dataset = SequenceDataset(test_data)
    
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=pad_sequence_batch
    )
    test_loader = data_utils.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=pad_sequence_batch
    )
    
    return train_loader, test_loader


def get_data_file_paths(dataset: str, data_dir: Union[str, Path] = None) -> Tuple[Path, Path]:
    """
    Get train and test file paths for a dataset using consistent naming logic.
    
    Args:
        dataset: Dataset name (e.g., 'synthetic_OC', 'synthetic_4000_200_2')
        data_dir: Base data directory (defaults to 'data/')
        
    Returns:
        Tuple of (train_path, test_path)
    """
    if data_dir is None:
        data_dir = Path('data')
    else:
        data_dir = Path(data_dir)
    
    data_dir = data_dir / dataset
    
    # Use same logic as train_optimized.py for file path detection
    if dataset.startswith('synthetic_') and '_' in dataset[10:]:
        # New format: synthetic_4000_200_2
        train_path = data_dir / f'{dataset}_train.txt'
        test_path = data_dir / f'{dataset}_test.txt'
    else:
        # Legacy format: synthetic_OC -> synthetic_oc_train.txt
        train_path = data_dir / f'{dataset.lower()}_train.txt'
        test_path = data_dir / f'{dataset.lower()}_test.txt'
    
    return train_path, test_path


def load_dataset(dataset: str, data_dir: Union[str, Path] = None, batch_size: int = 32) -> Tuple[data_utils.DataLoader, data_utils.DataLoader, int, int]:
    """
    Complete dataset loading function that handles file path detection and data loading.
    
    Args:
        dataset: Dataset name
        data_dir: Base data directory (defaults to 'data/')
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, test_loader, n_questions, n_cats)
    """
    train_path, test_path = get_data_file_paths(dataset, data_dir)
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size)
    
    return train_loader, test_loader, n_questions, n_cats