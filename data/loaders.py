"""
Data loading and management for Deep-GPCM models.
Unified interface for different dataset formats.
"""

import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import parse_dataset_name, validate_data_format, get_data_statistics


class GPCMDataset(Dataset):
    """PyTorch Dataset for GPCM data with proper batching."""
    
    def __init__(self, questions: List[List[int]], responses: List[List[int]], device: str = 'cpu'):
        self.questions = questions
        self.responses = responses
        self.device = device
        
        # Validate data
        assert len(questions) == len(responses), "Questions and responses must have same length"
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'questions': torch.tensor(self.questions[idx], dtype=torch.long),
            'responses': torch.tensor(self.responses[idx], dtype=torch.long),
            'student_id': idx
        }


def collate_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    
    # Extract sequences
    questions = [item['questions'] for item in batch]
    responses = [item['responses'] for item in batch]
    student_ids = [item['student_id'] for item in batch]
    
    # Get dimensions
    batch_size = len(batch)
    max_len = max(len(seq) for seq in questions)
    
    # Create padded tensors
    q_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    r_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # Fill tensors
    for i, (q_seq, r_seq) in enumerate(zip(questions, responses)):
        seq_len = len(q_seq)
        q_padded[i, :seq_len] = q_seq
        r_padded[i, :seq_len] = r_seq
        mask[i, :seq_len] = True
    
    return {
        'questions': q_padded,
        'responses': r_padded,
        'mask': mask,
        'student_ids': torch.tensor(student_ids, dtype=torch.long),
        'seq_lens': torch.tensor([len(seq) for seq in questions], dtype=torch.long)
    }


class DataLoaderManager:
    """Unified data loader manager for Deep-GPCM datasets."""
    
    def __init__(self, dataset_name: str, batch_size: int = 32, device: str = 'cpu'):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device
        
        # Dataset metadata
        self.dataset_info = parse_dataset_name(dataset_name)
        self.data_path = Path(f"data/{dataset_name}")
        
        # Will be set after loading data
        self.n_questions = None
        self.n_cats = None
        self.dataset = None
        
    def load_data(self, split_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Load and split dataset into train/test loaders.
        
        Args:
            split_ratio: Fraction of data to use for training
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        
        # Load raw data
        questions, responses = self._load_raw_data()
        
        # Validate data
        is_valid, error_msg = validate_data_format(questions, responses, self.n_cats)
        if not is_valid:
            raise ValueError(f"Data validation failed: {error_msg}")
        
        # Get data statistics
        stats = get_data_statistics(questions, responses, self.n_cats)
        self.n_questions = stats['questions']['max_id'] + 1  # 0-indexed
        
        print(f"📊 Loaded {self.dataset_name}: {len(questions)} sequences, "
              f"{self.n_questions} questions, {self.n_cats} categories")
        
        # Split data
        n_train = int(len(questions) * split_ratio)
        
        train_questions = questions[:n_train]
        train_responses = responses[:n_train]
        test_questions = questions[n_train:]
        test_responses = responses[n_train:]
        
        # Create datasets
        train_dataset = GPCMDataset(train_questions, train_responses, self.device)
        test_dataset = GPCMDataset(test_questions, test_responses, self.device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_sequences,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
            drop_last=False
        )
        
        return train_loader, test_loader
    
    def _load_raw_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Load raw data from dataset directory."""
        
        # Check if dataset directory exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_path}")
        
        # Try different data file formats
        data_files = [
            self.data_path / "data.pkl",
            self.data_path / "sequences.pkl", 
            self.data_path / "train_data.pkl",
            self.data_path / "synthetic_data.pkl"
        ]
        
        for data_file in data_files:
            if data_file.exists():
                return self._load_pickle_data(data_file)
        
        # Try JSON format
        json_files = [
            self.data_path / "data.json",
            self.data_path / "sequences.json"
        ]
        
        for json_file in json_files:
            if json_file.exists():
                return self._load_json_data(json_file)
        
        # Try numpy format
        npz_files = [
            self.data_path / "data.npz",
            self.data_path / "sequences.npz"
        ]
        
        for npz_file in npz_files:
            if npz_file.exists():
                return self._load_npz_data(npz_file)
        
        # If no data files found, try to generate synthetic data
        if self.dataset_name.startswith('synthetic_'):
            print(f"⚠️  No data files found for {self.dataset_name}, generating synthetic data...")
            return self._generate_synthetic_data()
        
        raise FileNotFoundError(f"No data files found in {self.data_path}")
    
    def _load_pickle_data(self, file_path: Path) -> Tuple[List[List[int]], List[List[int]]]:
        """Load data from pickle file."""
        print(f"📥 Loading data from {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different pickle formats
        if isinstance(data, dict):
            if 'questions' in data and 'responses' in data:
                questions = data['questions']
                responses = data['responses']
                if 'n_cats' in data:
                    self.n_cats = data['n_cats']
            elif 'train_questions' in data and 'train_responses' in data:
                # Combine train and test if available
                questions = data['train_questions']
                responses = data['train_responses']
                if 'test_questions' in data:
                    questions.extend(data['test_questions'])
                    responses.extend(data['test_responses'])
                if 'n_cats' in data:
                    self.n_cats = data['n_cats']
            else:
                raise ValueError("Unknown pickle data format")
        elif isinstance(data, tuple) and len(data) == 2:
            questions, responses = data
        else:
            raise ValueError("Unknown pickle data format")
        
        # Set n_cats if not already set
        if self.n_cats is None:
            self.n_cats = self.dataset_info.get('n_cats', 4)
        
        return questions, responses
    
    def _load_json_data(self, file_path: Path) -> Tuple[List[List[int]], List[List[int]]]:
        """Load data from JSON file."""
        print(f"📥 Loading data from {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        questions = data['questions']
        responses = data['responses']
        self.n_cats = data.get('n_cats', self.dataset_info.get('n_cats', 4))
        
        return questions, responses
    
    def _load_npz_data(self, file_path: Path) -> Tuple[List[List[int]], List[List[int]]]:
        """Load data from NumPy npz file."""
        print(f"📥 Loading data from {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        
        questions = data['questions'].tolist()
        responses = data['responses'].tolist()
        self.n_cats = int(data.get('n_cats', self.dataset_info.get('n_cats', 4)))
        
        return questions, responses
    
    def _generate_synthetic_data(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate synthetic data based on dataset name."""
        from utils.data_utils import create_synthetic_data
        
        # Parse dataset parameters
        params = self.dataset_info
        
        if params['format'] == 'new':
            n_sequences = params['n_students']
            n_questions = params['max_questions'] 
            self.n_cats = params['n_cats']
        else:
            # Legacy format defaults
            n_sequences = 1000
            n_questions = 50
            self.n_cats = params['n_cats']
        
        print(f"🔄 Generating synthetic data: {n_sequences} sequences, "
              f"{n_questions} questions, {self.n_cats} categories")
        
        questions, responses = create_synthetic_data(
            n_sequences=n_sequences,
            n_questions=n_questions,
            n_cats=self.n_cats,
            min_len=5,
            max_len=20,
            seed=42
        )
        
        # Save generated data for future use
        save_path = self.data_path / "synthetic_data.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump({
                'questions': questions,
                'responses': responses,
                'n_cats': self.n_cats,
                'generated': True
            }, f)
        
        print(f"💾 Saved generated data to {save_path}")
        
        return questions, responses
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'name': self.dataset_name,
            'n_questions': self.n_questions,
            'n_cats': self.n_cats,
            'batch_size': self.batch_size,
            'device': self.device,
            'data_path': str(self.data_path),
            'format_info': self.dataset_info
        }


class MultiDatasetManager:
    """Manager for handling multiple datasets in batch operations."""
    
    def __init__(self, dataset_names: List[str], batch_size: int = 32, device: str = 'cpu'):
        self.dataset_names = dataset_names
        self.batch_size = batch_size
        self.device = device
        self.managers = {}
        
    def load_all_datasets(self) -> Dict[str, Tuple[DataLoader, DataLoader]]:
        """Load all datasets and return dictionary of loaders."""
        dataset_loaders = {}
        
        for dataset_name in self.dataset_names:
            print(f"\n📂 Loading dataset: {dataset_name}")
            
            try:
                manager = DataLoaderManager(dataset_name, self.batch_size, self.device)
                train_loader, test_loader = manager.load_data()
                
                dataset_loaders[dataset_name] = (train_loader, test_loader)
                self.managers[dataset_name] = manager
                
                print(f"✅ Successfully loaded {dataset_name}")
                
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")
                continue
        
        return dataset_loaders
    
    def get_dataset_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information for all loaded datasets."""
        summary = {}
        
        for dataset_name, manager in self.managers.items():
            summary[dataset_name] = manager.get_dataset_info()
        
        return summary