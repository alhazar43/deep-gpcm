import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any


class BaseKnowledgeTracingModel(nn.Module, ABC):
    """Abstract base class for all knowledge tracing models."""
    
    def __init__(self):
        super().__init__()
        self.model_name = "base"
        self.n_questions = None
        self.n_cats = None
    
    @abstractmethod
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass through the model.
        
        Args:
            questions: Question IDs, shape (batch_size, seq_len)
            responses: Response categories, shape (batch_size, seq_len)
            
        Returns:
            Model outputs (probabilities, IRT parameters, etc.)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.model_name,
            "type": self.__class__.__name__,
            "parameters": sum(p.numel() for p in self.parameters()),
            "n_questions": self.n_questions,
            "n_cats": self.n_cats
        }
    
    def extract_irt_parameters(self, *args, **kwargs):
        """Extract IRT parameters if supported."""
        return None

