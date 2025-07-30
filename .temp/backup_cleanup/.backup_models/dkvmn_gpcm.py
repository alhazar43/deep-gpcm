"""DKVMN-GPCM model implementation using modular components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from core.base_model import BaseKnowledgeTracingModel
from core.memory_networks import DKVMN
from core.embeddings import create_embedding_strategy
from core.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from .model_factory import register_model


@register_model("dkvmn_gpcm")
class DKVMNGPCM(BaseKnowledgeTracingModel):
    """DKVMN with GPCM for polytomous responses."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, memory_size: int = 50, 
                 key_dim: int = 50, value_dim: int = 200, final_fc_dim: int = 50,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 3.0,
                 use_discrimination: bool = True, dropout_rate: float = 0.0):
        super().__init__()
        
        self.model_name = "dkvmn_gpcm"
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.final_fc_dim = final_fc_dim
        self.embedding_strategy = embedding_strategy
        self.ability_scale = ability_scale
        self.use_discrimination = use_discrimination
        self.dropout_rate = dropout_rate
        
        # Create embedding strategy
        self.embedding = create_embedding_strategy(embedding_strategy, n_questions, n_cats)
        
        # Embedding layers
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        self.gpcm_value_embed = nn.Linear(self.embedding.output_dim, value_dim)
        
        # DKVMN memory network
        self.memory = DKVMN(memory_size, key_dim, value_dim)
        
        # Summary network
        summary_input_dim = value_dim + key_dim
        self.summary_network = nn.Sequential(
            nn.Linear(summary_input_dim, final_fc_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        
        # IRT parameter extraction using modular component
        self.irt_extractor = IRTParameterExtractor(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            ability_scale=ability_scale,
            use_discrimination=use_discrimination,
            dropout_rate=dropout_rate,
            question_dim=key_dim
        )
        
        # GPCM probability computation
        self.gpcm_layer = GPCMProbabilityLayer()
        
        # Initialize value memory parameter
        self.init_value_memory = nn.Parameter(torch.randn(memory_size, value_dim))
        nn.init.kaiming_normal_(self.init_value_memory)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Embedding layers
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
        
        # Summary network
        for module in self.summary_network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through DKVMN-GPCM model.
        
        Args:
            questions: Question IDs, shape (batch_size, seq_len)
            responses: Response categories, shape (batch_size, seq_len)
            
        Returns:
            tuple: (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Convert question IDs to one-hot vectors
        q_one_hot = F.one_hot(questions, num_classes=self.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        gpcm_probs = []
        
        for t in range(seq_len):
            # Current question and response
            q_t = questions[:, t]  # (batch_size,)
            r_t = responses[:, t]  # (batch_size,)
            
            # Get question embedding for memory key
            q_embed_t = self.q_embed(q_t)  # (batch_size, key_dim)
            
            # Create GPCM embedding using strategy
            q_one_hot_t = q_one_hot[:, t:t+1, :]  # (batch_size, 1, Q)
            r_t_unsqueezed = r_t.unsqueeze(1)  # (batch_size, 1)
            
            # Use embedding strategy
            gpcm_embed_t = self.embedding.embed(
                q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats
            )  # (batch_size, 1, embed_dim)
            gpcm_embed_t = gpcm_embed_t.squeeze(1)  # (batch_size, embed_dim)
            
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