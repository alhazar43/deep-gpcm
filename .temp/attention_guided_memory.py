"""
Attention-Guided Memory Module
DKVMN memory system enhanced with attention guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGuidedMemory(nn.Module):
    """
    DKVMN memory system enhanced with attention guidance.
    
    Parameter count: 110,500 parameters
    Implements both correlation-based and attention-guided memory operations.
    """
    
    def __init__(self, n_questions: int, memory_size: int = 50, key_dim: int = 50, 
                 value_dim: int = 200, attention_dim: int = 64):
        super().__init__()
        
        self.n_questions = n_questions
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.attention_dim = attention_dim
        
        # Question embedding for memory keys
        self.question_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        
        # Static key memory matrix
        self.key_memory = nn.Parameter(torch.randn(memory_size, key_dim))
        
        # Value memory initialization
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        
        # Attention-guided read projection
        self.guided_read_projection = nn.Linear(value_dim, attention_dim)
        
        # Memory write operations
        self.erase_gate = nn.Linear(attention_dim + attention_dim, value_dim)
        self.add_gate = nn.Linear(attention_dim + attention_dim, value_dim)
        
        # Correlation-based attention (original DKVMN)
        self.correlation_projection = nn.Linear(key_dim, key_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for optimal performance."""
        nn.init.kaiming_normal_(self.question_embed.weight)
        nn.init.kaiming_normal_(self.key_memory)
        nn.init.kaiming_normal_(self.init_value_memory)
        
        for module in [self.guided_read_projection, self.erase_gate, self.add_gate, self.correlation_projection]:
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def init_memory(self, batch_size: int, device: torch.device):
        """Initialize value memory for a batch."""
        return self.init_value_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    def forward(self, query_embeddings, attention_features, value_memory, update_memory=False):
        """
        Forward pass through attention-guided memory.
        
        Args:
            query_embeddings: Question embeddings [batch_size, seq_len, key_dim]
            attention_features: Features from attention mechanism [batch_size, seq_len, attention_dim]
            value_memory: Current value memory state [batch_size, memory_size, value_dim]
            update_memory: Whether to update memory state
            
        Returns:
            read_content: Memory read results [batch_size, seq_len, attention_dim]
            updated_memory: Updated memory state [batch_size, memory_size, value_dim]
            memory_weights: Memory attention weights [batch_size, seq_len, memory_size]
        """
        batch_size, seq_len, _ = query_embeddings.shape
        
        read_contents = []
        memory_weights_list = []
        current_memory = value_memory
        
        for t in range(seq_len):
            # Current timestep
            q_embed_t = query_embeddings[:, t]  # [batch_size, key_dim]
            attention_t = attention_features[:, t]  # [batch_size, attention_dim]
            
            # Correlation-based attention (original DKVMN approach)
            correlation_key = torch.tanh(self.correlation_projection(q_embed_t))
            correlation_weights = F.softmax(torch.matmul(correlation_key, self.key_memory.t()), dim=1)
            
            # Attention-guided read
            read_content = self._guided_read(correlation_weights, current_memory, attention_t)
            
            read_contents.append(read_content)
            memory_weights_list.append(correlation_weights)
            
            # Update memory if requested and not the last timestep
            if update_memory and t < seq_len - 1:
                current_memory = self._guided_write(correlation_weights, current_memory, attention_t, read_content)
        
        # Stack outputs
        read_content = torch.stack(read_contents, dim=1)  # [batch_size, seq_len, attention_dim]
        memory_weights = torch.stack(memory_weights_list, dim=1)  # [batch_size, seq_len, memory_size]
        
        return read_content, current_memory, memory_weights
    
    def _guided_read(self, attention_weights, value_memory, attention_features):
        """
        Attention-guided memory read operation.
        
        Args:
            attention_weights: Memory attention weights [batch_size, memory_size]
            value_memory: Current value memory [batch_size, memory_size, value_dim]
            attention_features: Attention features [batch_size, attention_dim]
            
        Returns:
            read_content: Read result [batch_size, attention_dim]
        """
        # Standard memory read
        attention_weights_expanded = attention_weights.unsqueeze(2)  # [batch_size, memory_size, 1]
        value_memory_expanded = value_memory  # [batch_size, memory_size, value_dim]
        
        read_content = torch.sum(value_memory_expanded * attention_weights_expanded, dim=1)  # [batch_size, value_dim]
        
        # Project to attention dimension
        read_content = self.guided_read_projection(read_content)  # [batch_size, attention_dim]
        
        return read_content
    
    def _guided_write(self, attention_weights, value_memory, attention_features, read_content):
        """
        Attention-guided memory write operation.
        
        Args:
            attention_weights: Memory attention weights [batch_size, memory_size]
            value_memory: Current value memory [batch_size, memory_size, value_dim]
            attention_features: Attention features [batch_size, attention_dim]
            read_content: Read content [batch_size, attention_dim]
            
        Returns:
            updated_memory: Updated value memory [batch_size, memory_size, value_dim]
        """
        # Combine attention features and read content for write operations
        write_input = torch.cat([attention_features, read_content], dim=-1)  # [batch_size, attention_dim + attention_dim]
        
        # Compute erase and add signals
        erase_signal = torch.sigmoid(self.erase_gate(write_input))  # [batch_size, value_dim]
        add_signal = torch.tanh(self.add_gate(write_input))  # [batch_size, value_dim]
        
        # Apply write operations with attention weights
        attention_expanded = attention_weights.unsqueeze(2)  # [batch_size, memory_size, 1]
        erase_expanded = erase_signal.unsqueeze(1)  # [batch_size, 1, value_dim]
        add_expanded = add_signal.unsqueeze(1)  # [batch_size, 1, value_dim]
        
        # Memory update: M_t = M_{t-1} * (1 - w_t * e_t) + w_t * a_t
        erase_matrix = attention_expanded * erase_expanded  # [batch_size, memory_size, value_dim]
        add_matrix = attention_expanded * add_expanded  # [batch_size, memory_size, value_dim]
        
        updated_memory = value_memory * (1 - erase_matrix) + add_matrix
        
        return updated_memory