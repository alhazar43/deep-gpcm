"""
Deep-GPCM model extending Deep-IRT with GPCM (Generalized Partial Credit Model) support.
Implements Strategy 3 (Linear Decay) embedding from paper reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .memory import DKVMN


def ordered_embedding(q_data, r_data, n_questions, n_cats):
    """
    Strategy 1: Ordered Embedding - R^(2Q)
    Most intuitive for partial credit models.
    x_t = [q_t * I(r_t > 0), q_t * r_t / (K-1)]
    
    Args:
        q_data: Question one-hot vectors, shape (batch_size, seq_len, n_questions)
        r_data: Response categories, shape (batch_size, seq_len) with values 0 to K-1
        n_questions: Number of questions
        n_cats: Number of categories (K)
        
    Returns:
        embedded: Shape (batch_size, seq_len, 2*Q) - Ordered embedding
    """
    batch_size, seq_len = r_data.shape
    device = r_data.device
    
    # Component 1: q_t * I(r_t > 0) - Binary correctness
    correctness_indicator = (r_data > 0).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
    correctness_component = q_data * correctness_indicator  # (batch_size, seq_len, Q)
    
    # Component 2: q_t * r_t / (K-1) - Normalized score
    normalized_response = r_data.float().unsqueeze(-1) / (n_cats - 1)  # (batch_size, seq_len, 1)
    score_component = q_data * normalized_response  # (batch_size, seq_len, Q)
    
    # Concatenate components
    embedded = torch.cat([correctness_component, score_component], dim=-1)  # (batch_size, seq_len, 2*Q)
    
    return embedded


def unordered_embedding(q_data, r_data, n_questions, n_cats):
    """
    Strategy 2: Unordered Embedding - R^(KQ)  
    For MCQ-style responses where categories are not ordered.
    x_t^(k) = q_t * I(r_t = k) for k = 0, 1, ..., K-1
    
    Args:
        q_data: Question one-hot vectors, shape (batch_size, seq_len, n_questions)  
        r_data: Response categories, shape (batch_size, seq_len) with values 0 to K-1
        n_questions: Number of questions
        n_cats: Number of categories (K)
        
    Returns:
        embedded: Shape (batch_size, seq_len, K*Q) - Unordered embedding
    """
    batch_size, seq_len = r_data.shape
    device = r_data.device
    
    # Create category indices k = 0, 1, ..., K-1
    k_indices = torch.arange(n_cats, device=device).long()  # Shape: (K,)
    
    # Expand for broadcasting
    r_expanded = r_data.unsqueeze(-1)  # (batch_size, seq_len, 1)
    k_expanded = k_indices.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
    
    # Create one-hot indicators: I(r_t = k)
    indicators = (r_expanded == k_expanded).float()  # (batch_size, seq_len, K)
    
    # Apply indicators to question vectors for each category
    # q_data: (batch_size, seq_len, Q), indicators: (batch_size, seq_len, K)
    # Result: (batch_size, seq_len, K, Q)
    weighted_q = indicators.unsqueeze(-1) * q_data.unsqueeze(2)  # (batch_size, seq_len, K, Q)
    
    # Flatten to (batch_size, seq_len, K*Q)
    embedded = weighted_q.view(batch_size, seq_len, -1)
    
    return embedded


def linear_decay_embedding(q_data, r_data, n_questions, n_cats):
    """
    Strategy 3: Linear Decay Embedding - R^(KQ)
    x_t^(k) = max(0, 1 - |k-r_t|/(K-1)) * q_t
    
    Args:
        q_data: Question one-hot vectors, shape (batch_size, seq_len, n_questions)
        r_data: Response categories, shape (batch_size, seq_len) with values 0 to K-1
        n_questions: Number of questions
        n_cats: Number of categories (K)
        
    Returns:
        embedded: Shape (batch_size, seq_len, K*Q) - Linear decay embedding
    """
    batch_size, seq_len = r_data.shape
    device = r_data.device
    
    # Create category indices k = 0, 1, ..., K-1
    k_indices = torch.arange(n_cats, device=device).float()  # Shape: (K,)
    
    # Expand for broadcasting
    r_expanded = r_data.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
    k_expanded = k_indices.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
    
    # Compute |k - r_t| / (K-1)
    distance = torch.abs(k_expanded - r_expanded) / (n_cats - 1)  # (batch_size, seq_len, K)
    
    # Compute triangular weights: max(0, 1 - distance)
    weights = torch.clamp(1.0 - distance, min=0.0)  # (batch_size, seq_len, K)
    
    # Apply weights to question vectors for each category
    # q_data: (batch_size, seq_len, Q), weights: (batch_size, seq_len, K)
    # Result: (batch_size, seq_len, K, Q)
    weighted_q = weights.unsqueeze(-1) * q_data.unsqueeze(2)  # (batch_size, seq_len, K, Q)
    
    # Flatten to (batch_size, seq_len, K*Q)
    embedded = weighted_q.view(batch_size, seq_len, n_cats * n_questions)
    
    return embedded


class DeepGpcmModel(nn.Module):
    """
    Deep-GPCM model extending Deep-IRT for multi-category responses.
    Uses DKVMN memory with GPCM probability prediction.
    """
    
    def __init__(self, n_questions, n_cats=4, memory_size=50, key_dim=50, 
                 value_dim=200, final_fc_dim=50, dropout_rate=0.0,
                 ability_scale=3.0, use_discrimination=True, embedding_strategy='linear_decay'):
        """
        Initialize Deep-GPCM model.
        
        Args:
            n_questions: Number of unique questions
            n_cats: Number of response categories (K)
            memory_size: Size of memory matrix
            key_dim: Dimension of key embeddings
            value_dim: Dimension of value embeddings  
            final_fc_dim: Hidden dimension for prediction networks
            dropout_rate: Dropout rate
            ability_scale: IRT ability scaling factor
            use_discrimination: Whether to use discrimination parameters
            embedding_strategy: Which embedding strategy to use ('ordered', 'unordered', 'linear_decay')
        """
        super(DeepGpcmModel, self).__init__()
        
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.dropout_rate = dropout_rate
        self.embedding_strategy = embedding_strategy
        
        # Determine embedding dimension based on strategy
        if embedding_strategy == 'ordered':
            gpcm_embed_dim = 2 * n_questions  # Strategy 1: R^(2Q)
        elif embedding_strategy in ['unordered', 'linear_decay']:
            gpcm_embed_dim = n_cats * n_questions  # Strategy 2 & 3: R^(KQ)
        else:
            raise ValueError(f"Unknown embedding strategy: {embedding_strategy}")
        
        # Embedding layers - following deep-2pl pattern
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        
        # Value embedding for K-category responses
        self.gpcm_value_embed = nn.Linear(gpcm_embed_dim, value_dim)
        
        # Initialize DKVMN memory
        init_key_memory = torch.randn(memory_size, key_dim)
        nn.init.kaiming_normal_(init_key_memory)
        
        self.memory = DKVMN(
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            init_key_memory=init_key_memory
        )
        
        # GPCM prediction networks
        summary_input_dim = value_dim + key_dim
        
        # Summary vector network
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, final_fc_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # Student ability network (theta)
        self.student_ability_network = nn.Linear(final_fc_dim, 1)
        
        # Question difficulty thresholds (beta) - K-1 thresholds per question
        self.question_threshold_network = nn.Sequential(
            nn.Linear(key_dim, n_cats - 1),
            nn.Tanh()
        )
        
        # Discrimination parameter (alpha) - from summary + question embedding
        discrimination_input_dim = final_fc_dim + key_dim
        self.discrimination_network = nn.Sequential(
            nn.Linear(discrimination_input_dim, 1),
            nn.Softplus()  # Ensures positive discrimination
        )
        
        # Initialize value memory parameter
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_value_memory)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding layers
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
        
        # Initialize prediction networks
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        nn.init.kaiming_normal_(self.student_ability_network.weight)
        nn.init.constant_(self.student_ability_network.bias, 0)
        
        for module in self.question_threshold_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        
        for module in self.discrimination_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def gpcm_probability(self, theta, alpha, betas):
        """
        Calculate GPCM response probabilities using cumulative logits.
        
        Args:
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)  
            betas: Difficulty thresholds, shape (batch_size, seq_len, K-1)
            
        Returns:
            probs: GPCM probabilities, shape (batch_size, seq_len, K)
        """
        batch_size, seq_len = theta.shape
        K = betas.shape[-1] + 1  # Number of categories
        
        # Compute cumulative logits
        cum_logits = torch.zeros(batch_size, seq_len, K, device=theta.device)
        cum_logits[:, :, 0] = 0  # First category baseline
        
        # For k = 1, ..., K-1: sum_{h=0}^{k-1} alpha * (theta - beta_h)
        for k in range(1, K):
            cum_logits[:, :, k] = torch.sum(
                alpha.unsqueeze(-1) * (theta.unsqueeze(-1) - betas[:, :, :k]), 
                dim=-1
            )
        
        # Convert to probabilities via softmax
        probs = F.softmax(cum_logits, dim=-1)
        return probs
    
    def forward(self, q_data, r_data, target_mask=None):
        """
        Forward pass through Deep-GPCM model.
        
        Args:
            q_data: Question IDs, shape (batch_size, seq_len)
            r_data: Response categories, shape (batch_size, seq_len) with values 0 to K-1
            target_mask: Optional mask for valid positions
            
        Returns:
            tuple: (predictions, student_abilities, item_thresholds, discrimination_params, gpcm_probs)
        """
        batch_size, seq_len = q_data.shape
        device = q_data.device
        
        # Convert question IDs to one-hot vectors
        q_one_hot = F.one_hot(q_data, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        predictions = []
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        gpcm_probs = []
        
        for t in range(seq_len):
            # Current question and response
            q_t = q_data[:, t]  # (batch_size,)
            r_t = r_data[:, t]  # (batch_size,)
            
            # Get question embedding for memory key
            q_embed_t = self.q_embed(q_t)  # (batch_size, key_dim)
            
            # Create GPCM embedding based on strategy
            q_one_hot_t = q_one_hot[:, t:t+1, :]  # (batch_size, 1, Q)
            r_t_unsqueezed = r_t.unsqueeze(1)  # (batch_size, 1)
            
            if self.embedding_strategy == 'ordered':
                gpcm_embed_t = ordered_embedding(
                    q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats
                )  # (batch_size, 1, 2*Q)
            elif self.embedding_strategy == 'unordered':
                gpcm_embed_t = unordered_embedding(
                    q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats
                )  # (batch_size, 1, K*Q)
            elif self.embedding_strategy == 'linear_decay':
                gpcm_embed_t = linear_decay_embedding(
                    q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats
                )  # (batch_size, 1, K*Q)
            else:
                raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")
            
            gpcm_embed_t = gpcm_embed_t.squeeze(1)  # (batch_size, embed_dim)
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            
            # GPCM parameter prediction
            theta_t = self.student_ability_network(summary_vector).squeeze(-1) * self.ability_scale
            betas_t = self.question_threshold_network(q_embed_t)  # (batch_size, K-1)
            
            # Discrimination parameter
            discrim_input = torch.cat([summary_vector, q_embed_t], dim=-1)
            alpha_t = self.discrimination_network(discrim_input).squeeze(-1)
            
            # GPCM probability calculation
            theta_expanded = theta_t.unsqueeze(1)  # (batch_size, 1)
            alpha_expanded = alpha_t.unsqueeze(1)  # (batch_size, 1)
            betas_expanded = betas_t.unsqueeze(1)  # (batch_size, 1, K-1)
            
            gpcm_prob_t = self.gpcm_probability(theta_expanded, alpha_expanded, betas_expanded)
            gpcm_prob_t = gpcm_prob_t.squeeze(1)  # (batch_size, K)
            
            # Prediction is the category with highest probability
            pred_t = torch.argmax(gpcm_prob_t, dim=-1).float()
            
            # Store outputs
            predictions.append(pred_t)
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            gpcm_probs.append(gpcm_prob_t)
            
            # Write to memory for next time step
            if t < seq_len - 1:  # Don't write after last step
                self.memory.write(correlation_weight, value_embed_t)
        
        # Stack outputs
        predictions = torch.stack(predictions, dim=1)  # (batch_size, seq_len)
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        gpcm_probs = torch.stack(gpcm_probs, dim=1)  # (batch_size, seq_len, K)
        
        return predictions, student_abilities, item_thresholds, discrimination_params, gpcm_probs