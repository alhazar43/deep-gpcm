"""Attention-enhanced DKVMN-GPCM model with iterative refinement using modular components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from core.base_model import BaseKnowledgeTracingModel
from core.memory_networks import DKVMN
from core.embeddings import create_embedding_strategy
from core.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from .model_factory import register_model


@register_model("attention_dkvmn_gpcm")
class AttentionDKVMNGPCM(BaseKnowledgeTracingModel):
    """Attention-enhanced DKVMN with GPCM and iterative refinement using modular architecture."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 2.0,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.model_name = "attention_dkvmn_gpcm"
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.embedding_strategy = embedding_strategy
        self.dropout_rate = dropout_rate
        
        # Ability scaling parameter
        self.ability_scale = ability_scale
        
        # Create embedding strategy
        self.embedding = create_embedding_strategy(embedding_strategy, n_questions, n_cats)
        
        # Base embeddings
        self.question_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        
        # Embedding transformation to fixed dimension
        self.gpcm_embed = nn.Linear(self.embedding.output_dim, embed_dim)
        self.gpcm_value_embed = nn.Linear(embed_dim, value_dim)
        
        # DKVMN memory network
        self.memory = DKVMN(memory_size, key_dim, value_dim)
        
        # Iterative refinement components
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(n_cycles)
        ])
        
        # Feature fusion for each cycle
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(n_cycles)
        ])
        
        # Cycle normalization
        self.cycle_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_cycles)
        ])
        
        # Refinement gates (to control update magnitude)
        self.refinement_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            ) for _ in range(n_cycles)
        ])
        
        # Summary network
        summary_input_dim = value_dim + key_dim
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_fc_dim, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # IRT parameter extraction using modular component
        self.irt_extractor = IRTParameterExtractor(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            ability_scale=ability_scale,
            use_discrimination=True,
            dropout_rate=dropout_rate,
            question_dim=key_dim
        )
        
        # GPCM probability computation
        self.gpcm_layer = GPCMProbabilityLayer()
        
        # Initialize value memory parameter
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_value_memory)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for numerical stability."""
        # Question embeddings
        nn.init.kaiming_normal_(self.question_embed.weight)
        
        # GPCM embeddings
        nn.init.kaiming_normal_(self.gpcm_embed.weight)
        nn.init.constant_(self.gpcm_embed.bias, 0)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
        
        # Summary network
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        # Refinement components
        for fusion in self.fusion_layers:
            for module in fusion:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
        
        for gate in self.refinement_gates:
            for module in gate:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def _create_gpcm_embedding(self, q_data, r_data):
        """Create GPCM embeddings using embedding strategy."""
        batch_size, seq_len = q_data.shape
        
        # Convert question IDs to one-hot vectors
        q_one_hot = F.one_hot(q_data, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Use embedding strategy
        gpcm_embeds = self.embedding.embed(
            q_one_hot, r_data, self.n_questions, self.n_cats
        )  # (batch_size, seq_len, embed_dim)
        
        # Transform to fixed embedding dimension
        if gpcm_embeds.shape[-1] != self.embed_dim:
            gpcm_embeds = self.gpcm_embed(gpcm_embeds)
        
        return gpcm_embeds
    
    def _iterative_refinement(self, gpcm_embeds, q_embeds):
        """Apply iterative refinement using multi-head attention."""
        batch_size, seq_len, embed_dim = gpcm_embeds.shape
        
        # Start with initial embeddings
        current_embed = gpcm_embeds
        
        for cycle in range(self.n_cycles):
            # Multi-head self-attention
            attn_output, _ = self.attention_layers[cycle](
                current_embed, current_embed, current_embed
            )
            
            # Feature fusion
            fused_input = torch.cat([current_embed, attn_output], dim=-1)
            fused_output = self.fusion_layers[cycle](fused_input)
            
            # Apply refinement gate
            gate = self.refinement_gates[cycle](current_embed)
            refined_output = gate * fused_output + (1 - gate) * current_embed
            
            # Update current embedding
            current_embed = refined_output
            
            # Cycle normalization
            current_embed = self.cycle_norms[cycle](current_embed)
        
        return current_embed
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through attention-enhanced DKVMN-GPCM model.
        
        Args:
            questions: Question IDs, shape (batch_size, seq_len)
            responses: Response categories, shape (batch_size, seq_len)
            
        Returns:
            tuple: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Question embeddings
        q_embeds = self.question_embed(questions)  # (batch_size, seq_len, key_dim)
        
        # GPCM embeddings
        gpcm_embeds = self._create_gpcm_embedding(questions, responses)  # (batch_size, seq_len, embed_dim)
        
        # Apply iterative refinement
        enhanced_features = self._iterative_refinement(gpcm_embeds, q_embeds)
        
        # Sequential processing
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        gpcm_probs = []
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]  # (batch_size, key_dim)
            gpcm_embed_t = enhanced_features[:, t, :]  # (batch_size, embed_dim)
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            
            # Extract IRT parameters using modular component
            theta_t, alpha_t, betas_t = self.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)  # (batch_size,)
            alpha_t = alpha_t.squeeze(1)  # (batch_size,)
            betas_t = betas_t.squeeze(1)  # (batch_size, K-1)
            
            # GPCM probability calculation using modular component
            gpcm_prob_t = self.gpcm_layer(
                theta_t.unsqueeze(1), alpha_t.unsqueeze(1), betas_t.unsqueeze(1)
            )
            gpcm_prob_t = gpcm_prob_t.squeeze(1)  # (batch_size, K)
            
            # Store outputs
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            gpcm_probs.append(gpcm_prob_t)
            
            # Write to memory for next time step
            if t < seq_len - 1:
                self.memory.write(correlation_weight, value_embed_t)
        
        # Stack outputs
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        gpcm_probs = torch.stack(gpcm_probs, dim=1)  # (batch_size, seq_len, K)
        
        return student_abilities, item_thresholds, discrimination_params, gpcm_probs
    
    def get_model_info(self):
        """Get model information."""
        return {
            "name": self.model_name,
            "type": "attention_dkvmn_gpcm", 
            "parameters": sum(p.numel() for p in self.parameters()),
            "architecture": "Attention-DKVMN-GPCM",
            "features": ["Dynamic Memory", "Multi-Head Attention", "Iterative Refinement", "Polytomous Support"]
        }