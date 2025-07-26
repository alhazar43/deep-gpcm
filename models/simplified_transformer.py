#!/usr/bin/env python3
"""
Simplified Phase 2.1: Direct Transformer Integration into DKVMN Memory
Instead of complex residual connections, integrate transformer directly into memory operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimplifiedTransformerGPCM(nn.Module):
    """
    Simplified transformer integration that replaces DKVMN memory with transformer-enhanced memory.
    More direct integration for better learning.
    """
    
    def __init__(self, base_gpcm_model, d_model=128, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.base_model = base_gpcm_model
        self.d_model = d_model
        
        # Copy key parameters from base model
        self.n_questions = base_gpcm_model.n_questions
        self.n_cats = base_gpcm_model.n_cats
        self.embedding_strategy = base_gpcm_model.embedding_strategy
        
        # Use base model embedding layers
        self.q_embed = base_gpcm_model.q_embed
        self.gpcm_value_embed = base_gpcm_model.gpcm_value_embed
        
        # Transformer for sequence modeling
        self.input_projection = nn.Linear(self.q_embed.embedding_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Enhanced summary network using transformer features
        self.summary_network = nn.Sequential(
            nn.Linear(d_model, base_gpcm_model.final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Use base model IRT parameter networks
        self.student_ability_network = base_gpcm_model.student_ability_network
        self.question_threshold_network = base_gpcm_model.question_threshold_network
        self.discrimination_network = base_gpcm_model.discrimination_network
        
        # Copy other parameters
        self.ability_scale = base_gpcm_model.ability_scale
        
    def _get_embeddings(self, q_data, r_data):
        """Get embeddings using base model strategy."""
        batch_size, seq_len = q_data.shape
        
        # Convert question IDs to one-hot vectors
        q_one_hot = F.one_hot(q_data, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Apply embedding strategy from base model
        if self.embedding_strategy == 'linear_decay':
            from .model import linear_decay_embedding
            embedded = linear_decay_embedding(q_one_hot, r_data, self.n_questions, self.n_cats)
        elif self.embedding_strategy == 'ordered':
            from .model import ordered_embedding
            embedded = ordered_embedding(q_one_hot, r_data, self.n_questions, self.n_cats)
        elif self.embedding_strategy == 'unordered':
            from .model import unordered_embedding
            embedded = unordered_embedding(q_one_hot, r_data, self.n_questions, self.n_cats)
        elif self.embedding_strategy == 'adjacent_weighted':
            from .model import adjacent_weighted_embedding
            embedded = adjacent_weighted_embedding(q_one_hot, r_data, self.n_questions, self.n_cats)
        else:
            raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")
        
        return embedded
    
    def forward(self, q_data, r_data):
        """
        Simplified forward pass using transformer for sequence modeling.
        """
        batch_size, seq_len = q_data.shape
        
        # Get question embeddings for each timestep
        q_embeddings = []
        for t in range(seq_len):
            q_embed_t = self.q_embed(q_data[:, t])  # (batch_size, key_dim)
            q_embeddings.append(q_embed_t)
        
        q_embeddings = torch.stack(q_embeddings, dim=1)  # (batch_size, seq_len, key_dim)
        
        # Project to transformer dimension
        projected_embeddings = self.input_projection(q_embeddings)  # (batch_size, seq_len, d_model)
        
        # Apply transformer for sequence modeling
        transformer_output = self.transformer(projected_embeddings)  # (batch_size, seq_len, d_model)
        
        # Generate predictions for each timestep
        theta_list = []
        alpha_list = []
        beta_list = []
        probs_list = []
        
        for t in range(seq_len):
            # Get transformer features for this timestep
            transformer_features = transformer_output[:, t, :]  # (batch_size, d_model)
            
            # Generate summary vector using transformer features
            summary_vector = self.summary_network(transformer_features)  # (batch_size, final_fc_dim)
            
            # Generate IRT parameters
            theta_t = self.student_ability_network(summary_vector).squeeze(-1) * self.ability_scale
            
            # Question embedding for this timestep
            q_embed_t = self.q_embed(q_data[:, t])
            beta_t = self.question_threshold_network(q_embed_t)
            
            # Discrimination parameter
            discrim_input = torch.cat([summary_vector, q_embed_t], dim=-1)
            alpha_t = self.discrimination_network(discrim_input).squeeze(-1)
            
            # GPCM probability calculation
            theta_expanded = theta_t.unsqueeze(1)  # (batch_size, 1)
            alpha_expanded = alpha_t.unsqueeze(1)  # (batch_size, 1)
            beta_expanded = beta_t.unsqueeze(1)  # (batch_size, 1, K-1)
            
            probs_t = self.base_model.gpcm_probability(theta_expanded, alpha_expanded, beta_expanded)
            probs_t = probs_t.squeeze(1)  # (batch_size, K)
            
            theta_list.append(theta_t)
            alpha_list.append(alpha_t)
            beta_list.append(beta_t)
            probs_list.append(probs_t)
        
        # Stack outputs
        theta = torch.stack(theta_list, dim=1)  # (batch_size, seq_len)
        alpha = torch.stack(alpha_list, dim=1)  # (batch_size, seq_len)
        beta = torch.stack(beta_list, dim=1)  # (batch_size, seq_len, K-1)
        probs = torch.stack(probs_list, dim=1)  # (batch_size, seq_len, K)
        
        return theta, alpha, beta, probs


def create_simplified_transformer_gpcm(base_model, d_model=128, nhead=4, num_layers=1, dropout=0.1):
    """
    Factory function to create SimplifiedTransformerGPCM model.
    
    Args:
        base_model: Base DeepGpcmModel instance
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        
    Returns:
        SimplifiedTransformerGPCM model instance
    """
    return SimplifiedTransformerGPCM(
        base_gpcm_model=base_model,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )