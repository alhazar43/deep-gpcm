"""CORAL-GPCM Fixed implementation with combined beta and tau parameters."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from ..base.base_model import BaseKnowledgeTracingModel
from ..components.memory_networks import DKVMN
from ..components.embeddings import create_embedding_strategy
from ..components.irt_layers import IRTParameterExtractor
from .coral_gpcm_proper import CoralTauHead


class CORALGPCMFixed(BaseKnowledgeTracingModel):
    """CORAL-GPCM Fixed with combined beta and tau parameters.
    
    Key features:
    - For K categories, K-1 beta and K-1 tau parameters
    - Computes cum_logits = sum(alpha(theta - (beta + tau)))
    - No beta ordering constraints
    - Direct GPCM probability computation
    """
    
    def __init__(self,
                 n_questions: int,
                 n_cats: int = 4,
                 memory_size: int = 50,
                 key_dim: int = 50,
                 value_dim: int = 200,
                 final_fc_dim: int = 50,
                 embedding_strategy: str = "linear_decay",
                 ability_scale: float = 1.0,
                 use_discrimination: bool = True,
                 dropout_rate: float = 0.0,
                 coral_dropout: float = 0.1):
        """Initialize CORAL-GPCM Fixed model."""
        super().__init__()
        
        self.model_name = "coral_gpcm_fixed"
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        
        # Memory network (shared foundation)
        self.memory = DKVMN(
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim
        )
        
        # Embeddings
        self.embedding = create_embedding_strategy(embedding_strategy, n_questions, n_cats)
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        self.gpcm_value_embed = nn.Linear(self.embedding.output_dim, value_dim)
        
        # Summary network (shared between branches)
        self.summary_fc = nn.Linear(key_dim + value_dim, final_fc_dim)
        
        # IRT Branch: Extract α, θ, β parameters
        self.irt_extractor = IRTParameterExtractor(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            ability_scale=ability_scale,
            use_discrimination=use_discrimination,
            dropout_rate=dropout_rate
        )
        
        # CORAL Branch: Extract ordered τ thresholds
        self.coral_tau_head = CoralTauHead(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            dropout_rate=coral_dropout
        )
        
        # Initialize value memory parameter
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_value_memory)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.zeros_(self.gpcm_value_embed.bias)
        nn.init.kaiming_normal_(self.summary_fc.weight)
        nn.init.zeros_(self.summary_fc.bias)
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with combined beta + tau calculation.
        
        Args:
            questions: Question IDs, shape (batch_size, seq_len)
            responses: Student responses, shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
            - student_abilities: Student ability estimates
            - item_thresholds: Item threshold parameters (beta)
            - discrimination_params: Discrimination parameters (alpha)
            - probs: Final probability predictions
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Storage for outputs
        student_abilities = []
        discrimination_params = []
        item_thresholds = []
        summaries = []
        
        # Process each timestep
        for t in range(seq_len):
            q = questions[:, t]  # Current question
            r = responses[:, t]  # Current response
            
            # Get embeddings
            key = self.q_embed(q)
            
            # Create one-hot question vectors for embedding
            q_one_hot = F.one_hot(q, num_classes=self.n_questions + 1).float()
            # Remove padding
            q_one_hot = q_one_hot[:, 1:]  # Remove first column (padding)
            
            # Create response embedding using the strategy
            response_embed = self.embedding.embed(q_one_hot.unsqueeze(1), r.unsqueeze(1), 
                                                 self.n_questions, self.n_cats)
            response_embed = response_embed.squeeze(1)  # Remove seq dimension
            
            value_input = self.gpcm_value_embed(response_embed)
            
            # Memory operations
            correlation_weight = self.memory.attention(key)
            read_value = self.memory.read(correlation_weight)
            
            # Combine key and value for summary
            combined = torch.cat([key, read_value], dim=1)
            summary = F.elu(self.summary_fc(combined))
            
            # Extract IRT parameters
            ability, discrimination, threshold = self.irt_extractor(summary)
            
            # Store outputs
            student_abilities.append(ability)
            discrimination_params.append(discrimination)
            item_thresholds.append(threshold)
            summaries.append(summary)
            
            # Memory write
            self.memory.write(correlation_weight, value_input)
        
        # Stack temporal outputs
        student_abilities = torch.stack(student_abilities, dim=1)
        discrimination_params = torch.stack(discrimination_params, dim=1)
        item_thresholds = torch.stack(item_thresholds, dim=1)
        summaries = torch.stack(summaries, dim=1)
        
        # Extract CORAL tau thresholds (global, ordered)
        coral_tau = self.coral_tau_head(summaries)  # Shape: (n_cats-1,)
        
        # Expand tau to match beta dimensions
        tau_expanded = coral_tau.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Combine beta and tau
        combined_thresholds = item_thresholds + tau_expanded  # Shape: (batch_size, seq_len, n_cats-1)
        
        # Compute GPCM probabilities with combined thresholds
        probs = self._compute_gpcm_probs(student_abilities, discrimination_params, combined_thresholds)
        
        return student_abilities, item_thresholds, discrimination_params, probs
    
    def _compute_gpcm_probs(self, theta: torch.Tensor, alpha: torch.Tensor, 
                            combined_thresholds: torch.Tensor) -> torch.Tensor:
        """Compute GPCM probabilities with combined thresholds.
        
        Args:
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)
            combined_thresholds: Combined beta + tau, shape (batch_size, seq_len, K-1)
            
        Returns:
            probs: GPCM probabilities, shape (batch_size, seq_len, K)
        """
        batch_size, seq_len = theta.shape
        K = self.n_cats
        
        # Compute cumulative logits
        cum_logits = torch.zeros(batch_size, seq_len, K, device=theta.device)
        cum_logits[:, :, 0] = 0  # First category baseline
        
        # For k = 1, ..., K-1: sum_{h=0}^{k-1} alpha * (theta - combined_threshold_h)
        for k in range(1, K):
            cum_logits[:, :, k] = torch.sum(
                alpha.unsqueeze(-1) * (theta.unsqueeze(-1) - combined_thresholds[:, :, :k]), 
                dim=-1
            )
        
        # Convert to probabilities via softmax
        probs = F.softmax(cum_logits, dim=-1)
        return probs
    
    def get_model_info(self) -> Dict[str, str]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "architecture": "CORAL-GPCM Fixed with combined thresholds",
            "formula": "cum_logits = sum(alpha * (theta - (beta + tau)))",
            "parameters": f"K-1 beta (item-specific) + K-1 tau (global ordered)"
        }