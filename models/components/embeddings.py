"""Embedding strategies for knowledge tracing models."""

from abc import ABC, abstractmethod
import torch
from typing import Tuple


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""
    
    @abstractmethod
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """Embed question-response pairs.
        
        Args:
            q_data: One-hot question vectors, shape (batch_size, seq_len, n_questions)
            r_data: Response categories, shape (batch_size, seq_len)
            n_questions: Number of questions
            n_cats: Number of response categories
            
        Returns:
            Embedded vectors, shape (batch_size, seq_len, embed_dim)
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output embedding dimension."""
        pass


class OrderedEmbedding(EmbeddingStrategy):
    """Ordered embedding for partial credit responses."""
    
    def __init__(self, n_questions: int, n_cats: int):
        self.n_questions = n_questions
        self.n_cats = n_cats
    
    @property
    def output_dim(self) -> int:
        return 2 * self.n_questions
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """Ordered embedding with correctness and score components."""
        # Correctness component
        correctness_indicator = (r_data > 0).float().unsqueeze(-1)
        correctness_component = q_data * correctness_indicator
        
        # Score component
        normalized_response = r_data.float().unsqueeze(-1) / (n_cats - 1)
        score_component = q_data * normalized_response
        
        # Combine components
        embedded = torch.cat([correctness_component, score_component], dim=-1)
        return embedded


class UnorderedEmbedding(EmbeddingStrategy):
    """Unordered embedding for categorical responses."""
    
    def __init__(self, n_questions: int, n_cats: int):
        self.n_questions = n_questions
        self.n_cats = n_cats
    
    @property
    def output_dim(self) -> int:
        return self.n_cats * self.n_questions
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """One-hot style embedding for each category."""
        embedded_list = []
        for k in range(n_cats):
            indicator = (r_data == k).float().unsqueeze(-1)
            category_embedding = q_data * indicator
            embedded_list.append(category_embedding)
        
        embedded = torch.cat(embedded_list, dim=-1)
        return embedded


class LinearDecayEmbedding(EmbeddingStrategy):
    """Linear decay embedding with triangular weights."""
    
    def __init__(self, n_questions: int, n_cats: int):
        self.n_questions = n_questions
        self.n_cats = n_cats
    
    @property
    def output_dim(self) -> int:
        return self.n_cats * self.n_questions
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """Linear decay embedding: x_t^(k) = max(0, 1 - |k-r_t|/(K-1)) * q_t"""
        batch_size, seq_len = r_data.shape
        device = r_data.device
        
        # Create category indices k = 0, 1, ..., K-1
        k_indices = torch.arange(n_cats, device=device).float()
        
        # Expand for broadcasting
        r_expanded = r_data.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        k_expanded = k_indices.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        
        # Compute |k - r_t| / (K-1)
        distance = torch.abs(k_expanded - r_expanded) / (n_cats - 1)
        
        # Compute triangular weights: max(0, 1 - distance)
        weights = torch.clamp(1.0 - distance, min=0.0)  # (batch_size, seq_len, K)
        
        # Apply weights to question vectors for each category
        weighted_q = weights.unsqueeze(-1) * q_data.unsqueeze(2)  # (batch_size, seq_len, K, Q)
        
        # Flatten to (batch_size, seq_len, K*Q)
        embedded = weighted_q.view(batch_size, seq_len, n_cats * n_questions)
        return embedded


class AdjacentWeightedEmbedding(EmbeddingStrategy):
    """Adjacent weighted embedding for ordinal responses."""
    
    def __init__(self, n_questions: int, n_cats: int):
        self.n_questions = n_questions
        self.n_cats = n_cats
    
    @property
    def output_dim(self) -> int:
        return self.n_cats * self.n_questions
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """Adjacent weighted embedding: 1.0 for exact, 0.5 for adjacent, 0.0 for others."""
        embedded_list = []
        for k in range(n_cats):
            distances = torch.abs(r_data.float() - k)
            
            # Weight scheme
            weights = torch.zeros_like(distances)
            weights[distances == 0] = 1.0  # Exact match
            weights[distances == 1] = 0.5  # Adjacent
            
            weights = weights.unsqueeze(-1)
            category_embedding = q_data * weights
            embedded_list.append(category_embedding)
        
        embedded = torch.cat(embedded_list, dim=-1)
        return embedded


# Factory function for embedding strategies
def create_embedding_strategy(strategy_name: str, n_questions: int, n_cats: int) -> EmbeddingStrategy:
    """Factory function to create embedding strategies."""
    strategies = {
        'ordered': OrderedEmbedding,
        'unordered': UnorderedEmbedding, 
        'linear_decay': LinearDecayEmbedding,
        'adjacent_weighted': AdjacentWeightedEmbedding
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown embedding strategy: {strategy_name}")
    
    return strategies[strategy_name](n_questions, n_cats)