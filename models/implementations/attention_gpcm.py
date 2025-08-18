import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from ..base.base_model import BaseKnowledgeTracingModel
from .deep_gpcm import DeepGPCM
from ..components.memory_networks import DKVMN
from ..components.embeddings import create_embedding_strategy, LinearDecayEmbedding, EmbeddingStrategy
from ..components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from ..components.attention_layers import AttentionRefinementModule, EmbeddingProjection


class AttentionGPCM(DeepGPCM):
    """Attention-enhanced Deep GPCM model that builds on the base model."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 1.0,
                 dropout_rate: float = 0.1, use_ordinal_attention: bool = False,
                 attention_types: Optional[List[str]] = None):
        
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
        
        self.model_name = "attention_dkvmn_gpcm"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.use_ordinal_attention = use_ordinal_attention
        self.attention_types = attention_types
        
        # Import here to avoid circular imports
        from ..components.attention_layers import AttentionRefinementModule, EmbeddingProjection
        
        # Embedding projection to fixed dimension
        self.embedding_projection = EmbeddingProjection(
            input_dim=self.embedding.output_dim,
            output_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        
        # Attention refinement module - use ordinal-aware if requested
        if use_ordinal_attention:
            from ..components.ordinal_attention_integration import OrdinalAwareAttentionRefinement
            self.attention_refinement = OrdinalAwareAttentionRefinement(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_cycles=n_cycles,
                dropout_rate=dropout_rate,
                n_cats=n_cats,
                attention_types=attention_types,
                use_legacy=False
            )
        else:
            self.attention_refinement = AttentionRefinementModule(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_cycles=n_cycles,
                dropout_rate=dropout_rate
            )
        
        # Override the value embedding to work with the new embed_dim
        self.gpcm_value_embed = nn.Linear(embed_dim, value_dim)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
    
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create and project GPCM embeddings to fixed dimension."""
        # Get base embeddings
        base_embeds = super().create_embeddings(questions, responses)
        
        # Project to fixed embedding dimension
        projected_embeds = self.embedding_projection(base_embeds)
        
        return projected_embeds
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor,
                          responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention refinement to embeddings."""
        # Apply iterative attention refinement
        if self.use_ordinal_attention and responses is not None:
            # Pass responses for ordinal-aware attention
            refined_embeds = self.attention_refinement(gpcm_embeds, responses)
        else:
            refined_embeds = self.attention_refinement(gpcm_embeds)
        
        return refined_embeds
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with response-aware attention."""
        # Store responses for process_embeddings
        self._current_responses = responses
        
        # Call parent forward
        return super().forward(questions, responses)

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
                 embedding_strategy: str = "linear_decay", ability_scale: float = 1.0,
                 dropout_rate: float = 0.1, use_learnable_embedding: bool = True):
        
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
        
        # Only make ability_scale learnable when using learnable embeddings
        if embedding_strategy == "linear_decay" and use_learnable_embedding:
            self.learnable_ability_scale = nn.Parameter(torch.tensor(ability_scale))
            self.use_learnable_scale = True
        else:
            self.use_learnable_scale = False
            self.fixed_ability_scale = ability_scale
        
        # Replace the IRT parameter extractor to use learnable scale (stable tanh for attention)
        from ..components.irt_layers import IRTParameterExtractor
        self.irt_extractor = IRTParameterExtractor(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            ability_scale=1.0,  # Will be multiplied by learnable scale
            use_discrimination=True,
            dropout_rate=dropout_rate,
            question_dim=key_dim,
            use_research_beta=True  # Use research-based approach for better beta recovery
        )
        
        # Replace embedding with learnable version if using linear_decay and enabled
        if embedding_strategy == "linear_decay" and use_learnable_embedding:
            self.learnable_embedding = LearnableLinearDecayEmbedding(
                n_questions, n_cats, embed_dim
            )
            # Update embedding projection input dimension
            from ..components.attention_layers import EmbeddingProjection
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
            # For linear embeddings, create embeddings directly at embed_dim like learnable path
            # Get question one-hot vectors
            q_one_hot = F.one_hot(questions, num_classes=self.n_questions + 1).float()
            q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
            
            batch_size, seq_len = questions.shape
            embeddings = []
            
            for t in range(seq_len):
                q_one_hot_t = q_one_hot[:, t:t+1, :]  # (batch_size, 1, Q)
                r_t_unsqueezed = responses[:, t].unsqueeze(1)  # (batch_size, 1)
                
                # Use standard embedding strategy to get base embedding
                base_embed_t = self.embedding.embed(
                    q_one_hot_t, r_t_unsqueezed, self.n_questions, self.n_cats
                )  # (batch_size, 1, 800)
                base_embed_t = base_embed_t.squeeze(1)  # (batch_size, 800)
                
                # Project to target dimension directly (skip the AttentionGPCM projection layer)
                projected_embed_t = self.embedding_projection.projection(base_embed_t)  # (batch_size, 64)
                embeddings.append(projected_embed_t)
            
            # Stack embeddings
            gpcm_embeds = torch.stack(embeddings, dim=1)  # (batch_size, seq_len, 64)
            return gpcm_embeds
    
    def extract_irt_params(self, features: torch.Tensor, question_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract IRT parameters with conditional ability scale."""
        # Extract base parameters
        student_ability, item_thresholds, discrimination_param = self.irt_extractor(features, question_embeds)
        
        # Apply appropriate ability scale
        if self.use_learnable_scale:
            student_ability = student_ability * self.learnable_ability_scale
        else:
            student_ability = student_ability * self.fixed_ability_scale
        
        return student_ability, item_thresholds, discrimination_param
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass using parent's logic but with our IRT extraction."""
        # Use parent's forward method which will call our overridden extract_irt_params
        return super().forward(questions, responses)
    
    def get_model_info(self):
        """Get model information."""
        info = super().get_model_info()
        if self.use_learnable_scale:
            info['learnable_ability_scale'] = self.learnable_ability_scale.item()
        else:
            info['fixed_ability_scale'] = self.fixed_ability_scale
        info['use_learnable_scale'] = self.use_learnable_scale
        info['has_learnable_embedding'] = hasattr(self, 'learnable_embedding')
        return info

