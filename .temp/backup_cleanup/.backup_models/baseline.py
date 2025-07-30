"""
Baseline Deep-GPCM Implementation
Consolidated implementation containing DKVMN memory network and Deep-GPCM model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


# ============================================================================
# DKVMN Memory Network Implementation
# ============================================================================

class MemoryHeadGroup(nn.Module):
    """Memory head group for DKVMN, handles read/write operations on memory matrix."""
    
    def __init__(self, memory_size, memory_state_dim, is_write=False):
        super(MemoryHeadGroup, self).__init__()
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        
        if self.is_write:
            self.erase_linear = nn.Linear(memory_state_dim, memory_state_dim, bias=True)
            self.add_linear = nn.Linear(memory_state_dim, memory_state_dim, bias=True)
            
            # Initialize weights
            nn.init.kaiming_normal_(self.erase_linear.weight)
            nn.init.kaiming_normal_(self.add_linear.weight)
            nn.init.constant_(self.erase_linear.bias, 0)
            nn.init.constant_(self.add_linear.bias, 0)
    
    def correlation_weight(self, embedded_query_vector, key_memory_matrix):
        """Calculate correlation weight between query and key memory."""
        similarity_scores = torch.matmul(embedded_query_vector, key_memory_matrix.t())
        correlation_weight = F.softmax(similarity_scores, dim=1)
        return correlation_weight
    
    def read(self, value_memory_matrix, correlation_weight):
        """Read from value memory using correlation weights."""
        read_content = torch.matmul(correlation_weight.unsqueeze(1), value_memory_matrix)
        return read_content.squeeze(1)
    
    def write(self, value_memory_matrix, embedded_content_vector, correlation_weight):
        """Write to value memory using erase-add mechanism (matching old working interface)."""
        assert self.is_write, "This head group is not configured for writing"
        
        # Generate erase and add signals
        erase_signal = torch.sigmoid(self.erase_linear(embedded_content_vector))
        add_signal = torch.tanh(self.add_linear(embedded_content_vector))
        
        # Reshape for broadcasting - matching reference implementations
        erase_signal_expanded = erase_signal.unsqueeze(1)  # (batch_size, 1, value_dim)
        add_signal_expanded = add_signal.unsqueeze(1)      # (batch_size, 1, value_dim)
        correlation_weight_expanded = correlation_weight.unsqueeze(2)  # (batch_size, memory_size, 1)
        
        # Compute erase and add operations - matching reference style
        erase_mul = torch.bmm(correlation_weight_expanded, erase_signal_expanded)  # (batch_size, memory_size, value_dim)
        add_mul = torch.bmm(correlation_weight_expanded, add_signal_expanded)    # (batch_size, memory_size, value_dim)
        
        # Update memory: erase then add
        new_value_memory_matrix = value_memory_matrix * (1 - erase_mul) + add_mul
        
        return new_value_memory_matrix


class DKVMN(nn.Module):
    """Dynamic Key-Value Memory Network implementation."""
    
    def __init__(self, memory_size, key_dim, value_dim):
        super(DKVMN, self).__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Memory matrices
        self.key_memory_matrix = nn.Parameter(torch.randn(memory_size, key_dim))
        nn.init.kaiming_normal_(self.key_memory_matrix)
        
        # Memory head groups
        self.read_head = MemoryHeadGroup(memory_size, value_dim, is_write=False)
        self.write_head = MemoryHeadGroup(memory_size, value_dim, is_write=True)
        
        # Query key network
        self.query_key_linear = nn.Linear(key_dim, key_dim, bias=True)
        nn.init.kaiming_normal_(self.query_key_linear.weight)
        nn.init.constant_(self.query_key_linear.bias, 0)
        
        # Value memory will be initialized per batch (matching old working model)
        self.value_memory_matrix = None
    
    def init_value_memory(self, batch_size, init_value_memory=None):
        """Initialize value memory for a batch (matching old working interface)."""
        if init_value_memory is not None:
            if init_value_memory.dim() == 2:
                # Expand to batch dimension
                self.value_memory_matrix = init_value_memory.unsqueeze(0).expand(
                    batch_size, self.memory_size, self.value_dim
                ).contiguous()
            else:
                self.value_memory_matrix = init_value_memory.clone()
        else:
            # Initialize with zeros
            device = next(self.parameters()).device
            self.value_memory_matrix = torch.zeros(
                batch_size, self.memory_size, self.value_dim, device=device
            )
    
    def attention(self, embedded_query_vector):
        """Compute attention weights for memory access (matching old working interface)."""
        # Transform query for key lookup  
        query_key = torch.tanh(self.query_key_linear(embedded_query_vector))
        correlation_weight = self.read_head.correlation_weight(query_key, self.key_memory_matrix)
        return correlation_weight
    
    def read(self, correlation_weight):
        """Read from value memory (matching old working interface)."""
        read_content = self.read_head.read(self.value_memory_matrix, correlation_weight)
        return read_content
    
    def write(self, correlation_weight, embedded_content_vector):
        """Write to value memory (matching old working interface)."""
        new_value_memory = self.write_head.write(
            self.value_memory_matrix, embedded_content_vector, correlation_weight
        )
        self.value_memory_matrix = new_value_memory
        return new_value_memory
    
    def forward(self, query_embedded, value_embedded, value_memory_matrix):
        """
        Forward pass through DKVMN (backward compatibility).
        
        Args:
            query_embedded: Query vectors (batch_size, key_dim)
            value_embedded: Value vectors (batch_size, value_dim)
            value_memory_matrix: Current memory state (batch_size, memory_size, value_dim)
            
        Returns:
            read_content: Read content from memory (batch_size, value_dim)
            updated_memory: Updated memory matrix (batch_size, memory_size, value_dim)
        """
        # Transform query for key lookup
        query_key = torch.tanh(self.query_key_linear(query_embedded))
        
        # Compute correlation weights
        correlation_weight = self.read_head.correlation_weight(query_key, self.key_memory_matrix)
        
        # Read from memory
        read_content = self.read_head.read(value_memory_matrix, correlation_weight)
        
        # Write to memory
        updated_memory = self.write_head.write(value_memory_matrix, value_embedded, correlation_weight)
        
        return read_content, updated_memory


# ============================================================================
# Embedding Strategies
# ============================================================================

def ordered_embedding(q_data, r_data, n_questions, n_cats):
    """Strategy 1: Ordered Embedding - Most intuitive for partial credit models."""
    batch_size, seq_len = r_data.shape
    
    # Binary correctness component
    correctness_indicator = (r_data > 0).float().unsqueeze(-1)
    correctness_component = q_data * correctness_indicator
    
    # Normalized score component
    normalized_response = r_data.float().unsqueeze(-1) / (n_cats - 1)
    score_component = q_data * normalized_response
    
    # Concatenate components
    embedded = torch.cat([correctness_component, score_component], dim=-1)
    return embedded


def unordered_embedding(q_data, r_data, n_questions, n_cats):
    """Strategy 2: Unordered Embedding - For MCQ-style responses."""
    batch_size, seq_len = r_data.shape
    device = r_data.device
    
    embedded_list = []
    for k in range(n_cats):
        indicator = (r_data == k).float().unsqueeze(-1)
        category_embedding = q_data * indicator
        embedded_list.append(category_embedding)
    
    embedded = torch.cat(embedded_list, dim=-1)
    return embedded


def linear_decay_embedding(q_data, r_data, n_questions, n_cats):
    """
    Strategy 3: Linear Decay Embedding - R^(KQ)
    x_t^(k) = max(0, 1 - |k-r_t|/(K-1)) * q_t
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




def adjacent_weighted_embedding(q_data, r_data, n_questions, n_cats):
    """Strategy 4: Adjacent Weighted Embedding - Focus on actual + adjacent categories."""
    batch_size, seq_len = r_data.shape
    device = r_data.device
    
    embedded_list = []
    for k in range(n_cats):
        distances = torch.abs(r_data.float() - k)
        
        # Weight scheme: 1.0 for exact match, 0.5 for adjacent, 0.0 for others
        weights = torch.zeros_like(distances)
        weights[distances == 0] = 1.0  # Exact match
        weights[distances == 1] = 0.5  # Adjacent
        
        weights = weights.unsqueeze(-1)
        category_embedding = q_data * weights
        embedded_list.append(category_embedding)
    
    embedded = torch.cat(embedded_list, dim=-1)
    return embedded


# ============================================================================
# Deep-GPCM Model
# ============================================================================

class DeepGpcmModel(nn.Module):
    """
    Deep Generalized Partial Credit Model with DKVMN memory network.
    Uses proper GPCM with IRT parameters (theta, beta, alpha).
    """
    
    def __init__(self, n_questions, n_cats=4, memory_size=50, key_dim=50, value_dim=200,
                 final_fc_dim=50, embedding_strategy="linear_decay", prediction_method="cumulative",
                 ability_scale=3.0, use_discrimination=True, dropout_rate=0.0):
        super(DeepGpcmModel, self).__init__()
        
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.embedding_strategy = embedding_strategy
        self.prediction_method = prediction_method
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.dropout_rate = dropout_rate
        
        # Determine embedding dimensions
        if embedding_strategy == "ordered":
            gpcm_embed_dim = 2 * n_questions  # 2Q
        else:
            gpcm_embed_dim = n_cats * n_questions  # KQ
        
        # Embedding layers - following deep-2pl pattern
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        
        # Value embedding for K-category responses
        self.gpcm_value_embed = nn.Linear(gpcm_embed_dim, value_dim)
        
        # Initialize DKVMN memory with proper interface
        self.memory = DKVMN(memory_size, key_dim, value_dim)
        
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
        
        # Initialize weights
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
            tuple: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
        """
        batch_size, seq_len = q_data.shape
        device = q_data.device
        
        # Convert question IDs to one-hot vectors
        q_one_hot = F.one_hot(q_data, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Initialize memory using old working interface
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
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
                )
            elif self.embedding_strategy == 'adjacent_weighted':
                gpcm_embed_t = adjacent_weighted_embedding(
                    q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats
                )
            else:
                raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")
            
            gpcm_embed_t = gpcm_embed_t.squeeze(1)  # (batch_size, embed_dim)
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Memory operations using old working interface
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
            
            # Store outputs
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            gpcm_probs.append(gpcm_prob_t)
            
            # Write to memory for next time step (matching old working model)
            if t < seq_len - 1:  # Don't write after last step
                self.memory.write(correlation_weight, value_embed_t)
        
        # Stack outputs
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        gpcm_probs = torch.stack(gpcm_probs, dim=1)  # (batch_size, seq_len, K)
        
        return student_abilities, item_thresholds, discrimination_params, gpcm_probs


# ============================================================================
# Factory Functions
# ============================================================================

class BaselineGPCM(nn.Module):
    """Wrapper class for baseline GPCM model with standard interface."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, 
                 memory_size: int = 50, key_dim: int = 50, 
                 value_dim: int = 200, final_fc_dim: int = 50,
                 embedding_strategy: str = "linear_decay",
                 prediction_method: str = "cumulative"):
        super().__init__()
        
        self.model_name = "baseline"
        self.n_questions = n_questions
        self.n_cats = n_cats
        
        # Core Deep-GPCM model
        self.gpcm_model = DeepGpcmModel(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            embedding_strategy=embedding_strategy,
            prediction_method=prediction_method
        )
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor):
        """Forward pass through baseline model."""
        # Call the GPCM model with proper interface
        return self.gpcm_model(questions, responses)
    
    def get_model_info(self):
        """Get model information."""
        return {
            "name": self.model_name,
            "type": "baseline",
            "parameters": sum(p.numel() for p in self.parameters()),
            "architecture": "DKVMN-GPCM",
            "features": ["Dynamic Memory", "Optimal Embedding", "Polytomous Support"]
        }


def create_baseline_gpcm(**kwargs):
    """Factory function to create baseline GPCM model."""
    return BaselineGPCM(**kwargs)