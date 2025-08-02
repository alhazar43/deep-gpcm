"""Memory network implementations for knowledge tracing models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MemoryNetwork(nn.Module, ABC):
    """Abstract base class for memory networks."""
    
    @abstractmethod
    def forward(self, query_embedded: torch.Tensor, value_embedded: torch.Tensor, 
                value_memory_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory network forward pass."""
        pass


class MemoryHeadGroup(nn.Module):
    """DKVMN memory head for read/write operations."""
    
    def __init__(self, memory_size: int, memory_state_dim: int, is_write: bool = False):
        super().__init__()
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        
        if self.is_write:
            self.erase_linear = nn.Linear(memory_state_dim, memory_state_dim, bias=True)
            self.add_linear = nn.Linear(memory_state_dim, memory_state_dim, bias=True)
            
            # Weight init
            nn.init.kaiming_normal_(self.erase_linear.weight)
            nn.init.kaiming_normal_(self.add_linear.weight)
            nn.init.constant_(self.erase_linear.bias, 0)
            nn.init.constant_(self.add_linear.bias, 0)
    
    def correlation_weight(self, embedded_query_vector: torch.Tensor, 
                          key_memory_matrix: torch.Tensor) -> torch.Tensor:
        """Compute query-key correlation weights."""
        similarity_scores = torch.matmul(embedded_query_vector, key_memory_matrix.t())
        correlation_weight = F.softmax(similarity_scores, dim=1)
        return correlation_weight
    
    def read(self, value_memory_matrix: torch.Tensor, 
             correlation_weight: torch.Tensor) -> torch.Tensor:
        """Read from memory."""
        read_content = torch.matmul(correlation_weight.unsqueeze(1), value_memory_matrix)
        return read_content.squeeze(1)
    
    def write(self, value_memory_matrix: torch.Tensor, embedded_content_vector: torch.Tensor,
              correlation_weight: torch.Tensor) -> torch.Tensor:
        """Write to memory with erase-add."""
        assert self.is_write, "This head group is not configured for writing"
        
        # Erase/add signals
        erase_signal = torch.sigmoid(self.erase_linear(embedded_content_vector))
        add_signal = torch.tanh(self.add_linear(embedded_content_vector))
        
        # Reshape for broadcasting
        erase_signal_expanded = erase_signal.unsqueeze(1)
        add_signal_expanded = add_signal.unsqueeze(1)
        correlation_weight_expanded = correlation_weight.unsqueeze(2)
        
        # Erase/add operations
        erase_mul = torch.bmm(correlation_weight_expanded, erase_signal_expanded)
        add_mul = torch.bmm(correlation_weight_expanded, add_signal_expanded)
        
        # Memory update
        new_value_memory_matrix = value_memory_matrix * (1 - erase_mul) + add_mul
        return new_value_memory_matrix


class DKVMN(MemoryNetwork):
    """Dynamic Key-Value Memory Network."""
    
    def __init__(self, memory_size: int, key_dim: int, value_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Key memory
        self.key_memory_matrix = nn.Parameter(torch.randn(memory_size, key_dim))
        nn.init.kaiming_normal_(self.key_memory_matrix)
        
        # Read/write heads
        self.read_head = MemoryHeadGroup(memory_size, value_dim, is_write=False)
        self.write_head = MemoryHeadGroup(memory_size, value_dim, is_write=True)
        
        # Query transformation
        self.query_key_linear = nn.Linear(key_dim, key_dim, bias=True)
        nn.init.kaiming_normal_(self.query_key_linear.weight)
        nn.init.constant_(self.query_key_linear.bias, 0)
        
        # Value memory (batch-initialized)
        self.value_memory_matrix = None
    
    def init_value_memory(self, batch_size: int, init_value_memory: Optional[torch.Tensor] = None):
        """Initialize value memory."""
        if init_value_memory is not None:
            if init_value_memory.dim() == 2:
                # Expand batch dim
                self.value_memory_matrix = init_value_memory.unsqueeze(0).expand(
                    batch_size, self.memory_size, self.value_dim
                ).contiguous()
            else:
                self.value_memory_matrix = init_value_memory.clone()
        else:
            # Zero init
            device = next(self.parameters()).device
            self.value_memory_matrix = torch.zeros(
                batch_size, self.memory_size, self.value_dim, device=device
            )
    
    def attention(self, embedded_query_vector: torch.Tensor) -> torch.Tensor:
        """Compute attention weights."""
        query_key = torch.tanh(self.query_key_linear(embedded_query_vector))
        correlation_weight = self.read_head.correlation_weight(query_key, self.key_memory_matrix)
        return correlation_weight
    
    def read(self, correlation_weight: torch.Tensor) -> torch.Tensor:
        """Read from memory."""
        read_content = self.read_head.read(self.value_memory_matrix, correlation_weight)
        return read_content
    
    def write(self, correlation_weight: torch.Tensor, embedded_content_vector: torch.Tensor) -> torch.Tensor:
        """Write to memory."""
        new_value_memory = self.write_head.write(
            self.value_memory_matrix, embedded_content_vector, correlation_weight
        )
        self.value_memory_matrix = new_value_memory
        return new_value_memory
    
    def forward(self, query_embedded: torch.Tensor, value_embedded: torch.Tensor, 
                value_memory_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DKVMN forward pass."""
        # Query transform
        query_key = torch.tanh(self.query_key_linear(query_embedded))
        
        # Attention weights
        correlation_weight = self.read_head.correlation_weight(query_key, self.key_memory_matrix)
        
        # Memory read
        read_content = self.read_head.read(value_memory_matrix, correlation_weight)
        
        # Memory write
        updated_memory = self.write_head.write(value_memory_matrix, value_embedded, correlation_weight)
        
        return read_content, updated_memory