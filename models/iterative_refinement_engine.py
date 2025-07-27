"""
Iterative Refinement Engine
Core engine for memory-attention co-evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .memory_aware_attention import MemoryAwareAKTAttention
    from .attention_guided_memory import AttentionGuidedMemory
except ImportError:
    from memory_aware_attention import MemoryAwareAKTAttention
    from attention_guided_memory import AttentionGuidedMemory


class IterativeRefinementEngine(nn.Module):
    """
    Core engine for memory-attention co-evolution.
    
    Parameter count: 223,723 parameters
    Implements progressive refinement through multiple cycles.
    """
    
    def __init__(self, n_questions: int, embed_dim: int = 64, memory_size: int = 50, 
                 key_dim: int = 50, value_dim: int = 200, n_cycles: int = 2):
        super().__init__()
        
        self.n_questions = n_questions
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.n_cycles = n_cycles
        
        # Memory-aware attention module
        self.memory_attention = MemoryAwareAKTAttention(
            embed_dim=embed_dim,
            n_heads=4,
            dropout=0.1
        )
        
        # Attention-guided memory module
        self.attention_memory = AttentionGuidedMemory(
            n_questions=n_questions,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            attention_dim=embed_dim
        )
        
        # Cross-cycle enhancement layers
        self.cycle_enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(n_cycles)
        ])
        
        # Query embedding projection for memory operations
        self.query_projection = nn.Linear(embed_dim, key_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for optimal performance."""
        for enhancement in self.cycle_enhancement:
            for module in enhancement:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
        
        nn.init.kaiming_normal_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
    
    def forward(self, memory_embed, attention_embed, initial_memory_state=None):
        """
        Forward pass through iterative refinement cycles.
        
        Args:
            memory_embed: Memory embeddings [batch_size, seq_len, embed_dim]
            attention_embed: Attention embeddings [batch_size, seq_len, embed_dim]
            initial_memory_state: Optional initial memory state
            
        Returns:
            final_features: Final refined features [batch_size, seq_len, embed_dim]
            memory_weights: Final memory weights [batch_size, seq_len, memory_size]
            attention_weights: Final attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, embed_dim = attention_embed.shape
        device = attention_embed.device
        
        # Initialize memory state
        if initial_memory_state is None:
            memory_state = self.attention_memory.init_memory(batch_size, device)
        else:
            memory_state = initial_memory_state
        
        # Initialize states
        attention_state = attention_embed
        current_features = attention_embed
        
        # Project attention embeddings to query embeddings for memory operations
        query_embeddings = self.query_projection(attention_embed)  # [batch_size, seq_len, key_dim]
        
        # Iterative refinement cycles
        for cycle in range(self.n_cycles):
            # Memory-aware attention: enhance attention with memory context
            enhanced_attention, attention_weights = self.memory_attention(
                attention_state, 
                memory_state=current_features
            )
            
            # Attention-guided memory: update memory with attention guidance
            memory_read, updated_memory, memory_weights = self.attention_memory(
                query_embeddings, 
                enhanced_attention, 
                memory_state, 
                update_memory=True
            )
            
            # Cross-cycle enhancement
            cycle_features = self.cycle_enhancement[cycle](enhanced_attention)
            
            # Combine enhanced attention with memory read
            combined_features = enhanced_attention + memory_read + cycle_features
            
            # Update states for next cycle
            attention_state = combined_features
            memory_state = updated_memory
            current_features = combined_features
        
        return current_features, memory_weights, attention_weights