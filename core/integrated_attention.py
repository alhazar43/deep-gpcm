"""
Truly Integrated Attention-DKVMN-GPCM Model

This implementation deeply integrates attention with memory operations,
allowing attention to see and be influenced by current memory state at each timestep.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .model import DeepGPCM
from .layers import AttentionRefinementModule


class IntegratedAttentionGPCM(DeepGPCM):
    """
    Deep integration of attention with DKVMN memory operations.
    
    Key differences from AttentionGPCM:
    1. Attention refinement happens INSIDE the sequential loop
    2. Attention can see current memory state at each timestep
    3. Memory updates incorporate attention patterns
    4. True co-evolution of attention and memory
    """
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64,
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 2.0,
                 dropout_rate: float = 0.1):
        
        # Initialize base model
        super().__init__(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            embedding_strategy=embedding_strategy,
            ability_scale=ability_scale,
            use_discrimination=True,
            dropout_rate=dropout_rate
        )
        
        self.model_name = "integrated_attention_gpcm"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        
        # Import here to avoid circular imports
        from .layers import EmbeddingProjection
        
        # Embedding projection to fixed dimension
        self.embedding_projection = EmbeddingProjection(
            input_dim=self.embedding.output_dim,
            output_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        
        # Override the value embedding to work with projected dimension
        self.gpcm_value_embed = nn.Linear(embed_dim, value_dim)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
        
        # Memory-aware attention module
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=value_dim,  # Works on memory dimension
            num_heads=n_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Embedding refinement based on memory
        self.memory_embedding_fusion = nn.Sequential(
            nn.Linear(embed_dim + value_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Refinement gate
        self.refinement_gate = nn.Sequential(
            nn.Linear(embed_dim + value_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Enhanced value embedding that considers refined features
        self.enhanced_value_embed = nn.Sequential(
            nn.Linear(embed_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, value_dim)
        )
        
        # Memory update gate
        self.memory_update_gate = nn.Sequential(
            nn.Linear(value_dim * 2, value_dim),
            nn.Sigmoid()
        )
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with deep attention-memory integration.
        
        Args:
            questions: Question IDs, shape (batch_size, seq_len)
            responses: Response categories, shape (batch_size, seq_len)
            
        Returns:
            tuple: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
        """
        batch_size, seq_len = questions.shape
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Create base embeddings
        base_embeds = self.create_embeddings(questions, responses)  # (batch_size, seq_len, raw_embed_dim)
        gpcm_embeds = self.embedding_projection(base_embeds)  # (batch_size, seq_len, embed_dim)
        q_embeds = self.q_embed(questions)  # (batch_size, seq_len, key_dim)
        
        # Sequential processing with integrated attention
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        gpcm_probs = []
        
        # Keep track of memory states for attention
        memory_states = []
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]  # (batch_size, key_dim)
            gpcm_embed_t = gpcm_embeds[:, t, :]  # (batch_size, embed_dim)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)  # (batch_size, value_dim)
            
            # Deep integration: Refine embedding based on current memory state
            if t > 0:
                # Use previous memory states for attention context
                memory_context = torch.stack(memory_states, dim=1)  # (batch_size, t, value_dim)
                
                # Apply memory-aware attention to current read
                read_content_seq = read_content.unsqueeze(1)  # (batch_size, 1, value_dim)
                attended_memory, _ = self.memory_attention(
                    read_content_seq, memory_context, memory_context
                )
                attended_memory = attended_memory.squeeze(1)  # (batch_size, value_dim)
                
                # Fuse attended memory with embedding
                combined = torch.cat([gpcm_embed_t, attended_memory], dim=-1)
                refined_embed = self.memory_embedding_fusion(combined)
                
                # Gated update
                gate = self.refinement_gate(torch.cat([gpcm_embed_t, read_content], dim=-1))
                gpcm_embed_t = gpcm_embed_t + gate * (refined_embed - gpcm_embed_t)
            
            # Transform refined embedding to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Create summary vector with enhanced features
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            
            # Extract IRT parameters
            theta_t, alpha_t, betas_t = self.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)  # (batch_size,)
            alpha_t = alpha_t.squeeze(1)  # (batch_size,)
            betas_t = betas_t.squeeze(1)  # (batch_size, K-1)
            
            # GPCM probability calculation
            gpcm_prob_t = self.gpcm_layer(
                theta_t.unsqueeze(1), alpha_t.unsqueeze(1), betas_t.unsqueeze(1)
            )
            gpcm_prob_t = gpcm_prob_t.squeeze(1)  # (batch_size, K)
            
            # Store outputs
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            gpcm_probs.append(gpcm_prob_t)
            
            # Enhanced memory write with attention information
            if t < seq_len - 1:
                # Combine original value with attention-enhanced value
                gate_value = self.memory_update_gate(
                    torch.cat([value_embed_t, read_content], dim=-1)
                )
                enhanced_value = gate_value * value_embed_t + (1 - gate_value) * read_content
                
                self.memory.write(correlation_weight, enhanced_value)
                memory_states.append(enhanced_value)
            else:
                memory_states.append(read_content)
        
        # Stack outputs
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        gpcm_probs = torch.stack(gpcm_probs, dim=1)  # (batch_size, seq_len, K)
        
        return student_abilities, item_thresholds, discrimination_params, gpcm_probs