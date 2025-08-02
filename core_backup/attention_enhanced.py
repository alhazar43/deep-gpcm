"""Enhanced AttentionGPCM with learnable parameters from old AKVMN implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .model import AttentionGPCM
from .embeddings import EmbeddingStrategy


class LearnableLinearDecayEmbedding(nn.Module, EmbeddingStrategy):
    """Linear decay embedding with learnable weights (from old AKVMN)."""
    
    def __init__(self, n_questions: int, n_cats: int, embed_dim: int = 64):
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        
        # Learnable decay weights (like old AKVMN)
        self.decay_weights = nn.Parameter(torch.ones(n_cats))
        self.gpcm_embed = nn.Linear(n_cats, embed_dim)
        
        # Initialize exactly like old working version
        nn.init.ones_(self.decay_weights)
        nn.init.kaiming_normal_(self.gpcm_embed.weight)
        nn.init.zeros_(self.gpcm_embed.bias)
    
    @property
    def output_dim(self) -> int:
        return self.embed_dim
    
    def embed(self, q_data: torch.Tensor, r_data: torch.Tensor, 
              n_questions: int, n_cats: int) -> torch.Tensor:
        """Embed with learnable decay weights."""
        batch_size, seq_len = r_data.shape
        device = r_data.device
        
        # Convert responses to one-hot
        r_onehot = F.one_hot(r_data, num_classes=n_cats).float()
        
        # Apply learnable decay weights with softmax
        decay_weights = F.softmax(self.decay_weights, dim=0)
        
        # Weight the one-hot responses
        weighted_responses = r_onehot * decay_weights.unsqueeze(0).unsqueeze(0)
        
        # Apply linear transformation
        gpcm_embed = self.gpcm_embed(weighted_responses)
        
        return gpcm_embed


class EnhancedAttentionGPCM(AttentionGPCM):
    """AttentionGPCM with learnable parameters restored from old AKVMN."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 2.0,
                 dropout_rate: float = 0.1):
        
        # Initialize base AttentionGPCM
        super().__init__(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=embed_dim,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            n_heads=n_heads,
            n_cycles=n_cycles,
            embedding_strategy=embedding_strategy,
            ability_scale=1.0,  # Use 1.0 here, will apply learnable scale in IRT extractor
            dropout_rate=dropout_rate
        )
        
        # Override model name to match evaluation expectations
        self.model_name = "enhanced_akvmn_gpcm"
        
        # Make ability_scale learnable (like old AKVMN)
        self.learnable_ability_scale = nn.Parameter(torch.tensor(ability_scale))
        
        # Replace the IRT parameter extractor to use learnable scale
        from .layers import IRTParameterExtractor
        self.irt_extractor = IRTParameterExtractor(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            ability_scale=1.0,  # Will be multiplied by learnable scale
            use_discrimination=True,
            dropout_rate=dropout_rate,
            question_dim=key_dim
        )
        
        # Replace embedding with learnable version if using linear_decay
        if embedding_strategy == "linear_decay":
            self.learnable_embedding = LearnableLinearDecayEmbedding(
                n_questions, n_cats, embed_dim
            )
            # Update embedding projection input dimension
            from .layers import EmbeddingProjection
            self.embedding_projection = EmbeddingProjection(
                input_dim=embed_dim,  # Already the right size
                output_dim=embed_dim,
                dropout_rate=dropout_rate
            )
    
    
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create embeddings with learnable components."""
        batch_size, seq_len = questions.shape
        
        if self.embedding_strategy == "linear_decay" and hasattr(self, 'learnable_embedding'):
            # Use learnable linear decay embedding
            # Get question one-hot vectors
            q_one_hot = F.one_hot(questions, num_classes=self.n_questions + 1).float()
            q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
            
            # Get embeddings with learnable weights
            gpcm_embeds = self.learnable_embedding.embed(
                q_one_hot, responses, self.n_questions, self.n_cats
            )
            
            # No need for projection since it's already embed_dim
            return gpcm_embeds
        else:
            # Fall back to parent implementation
            return super().create_embeddings(questions, responses)
    
    def extract_irt_params(self, features: torch.Tensor, question_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract IRT parameters with learnable ability scale."""
        # Extract base parameters
        student_ability, item_thresholds, discrimination_param = self.irt_extractor(features, question_embeds)
        
        # Apply learnable ability scale
        student_ability = student_ability * self.learnable_ability_scale
        
        return student_ability, item_thresholds, discrimination_param
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using parent's logic but with our IRT extraction."""
        # Use parent's forward method which will call our overridden extract_irt_params
        return super().forward(questions, responses)
    
    def get_model_info(self):
        """Get model information."""
        info = super().get_model_info()
        info['learnable_ability_scale'] = self.learnable_ability_scale.item()
        info['has_learnable_embedding'] = hasattr(self, 'learnable_embedding')
        return info