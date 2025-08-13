import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List

from ..base.base_model import BaseKnowledgeTracingModel
from .attention_gpcm import AttentionGPCM
from ..components.memory_networks import DKVMN
from ..components.embeddings import create_embedding_strategy, LinearDecayEmbedding, EmbeddingStrategy
from ..components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from ..components.attention_layers import AttentionRefinementModule, EmbeddingProjection


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence modeling."""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Position-encoded embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        return x


class TemporalFeatureExtractor(nn.Module):
    """Extract temporal features from question/response sequences."""
    
    def __init__(self, window_size: int = 3, feature_dim: int = 8):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Linear layer to project temporal features
        self.temporal_projection = nn.Linear(window_size * 2, feature_dim)  # 2 features per timestep
        nn.init.kaiming_normal_(self.temporal_projection.weight)
        nn.init.zeros_(self.temporal_projection.bias)
    
    def extract_features(self, questions: torch.Tensor, responses: torch.Tensor, 
                        timestep: int) -> torch.Tensor:
        """Extract temporal features for a specific timestep.
        
        Args:
            questions: Question sequence [batch_size, seq_len]
            responses: Response sequence [batch_size, seq_len]  
            timestep: Current timestep index
            
        Returns:
            Temporal features [batch_size, feature_dim]
        """
        batch_size = questions.shape[0]
        device = questions.device
        
        # Define window boundaries
        start_idx = max(0, timestep - self.window_size + 1)
        end_idx = timestep + 1
        
        # Extract windowed features
        features = []
        
        for t in range(start_idx, end_idx):
            if t >= 0 and t < questions.shape[1]:
                # Time gap from current timestep
                time_gap = float(timestep - t)
                # Response correctness (normalized)
                correctness = responses[:, t].float() / 4.0  # Assume 0-4 scale
            else:
                # Padding for out-of-bounds
                time_gap = float(self.window_size)
                correctness = torch.zeros(batch_size, device=device)
            
            # Stack features
            features.append(torch.full((batch_size,), time_gap, device=device))
            features.append(correctness)
        
        # Pad if necessary to maintain consistent size
        while len(features) < self.window_size * 2:
            features.extend([
                torch.full((batch_size,), float(self.window_size), device=device),
                torch.zeros(batch_size, device=device)
            ])
        
        # Stack and project
        temporal_input = torch.stack(features, dim=1)  # [batch_size, window_size * 2]
        temporal_features = self.temporal_projection(temporal_input)  # [batch_size, feature_dim]
        
        return temporal_features


class FeatureFusionLayer(nn.Module):
    """Fuse attention-refined embeddings with temporal features."""
    
    def __init__(self, embed_dim: int, temporal_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Gate for controlling temporal influence
        self.temporal_gate = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
        
        for module in self.temporal_gate:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, attention_embed: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """Fuse attention embeddings with temporal features.
        
        Args:
            attention_embed: Attention-refined embeddings [batch_size, embed_dim]
            temporal_features: Temporal features [batch_size, temporal_dim]
            
        Returns:
            Fused features [batch_size, embed_dim]
        """
        # Concatenate features
        combined = torch.cat([attention_embed, temporal_features], dim=-1)
        
        # Compute fusion
        fused = self.fusion(combined)
        
        # Apply gating
        gate = self.temporal_gate(combined)
        
        # Gated combination: gate controls temporal influence
        output = gate * fused + (1 - gate) * attention_embed
        
        return output


class TemporalAttentionGPCM(AttentionGPCM):
    """Attention GPCM enhanced with positional encoding and temporal features.
    
    This model extends AttentionGPCM by adding:
    1. Positional encoding for global sequence awareness
    2. Temporal feature extraction for local context
    3. Feature fusion for combining attention and temporal information
    
    The architecture maintains DKVMN's sequential processing constraint while
    enhancing representation learning through temporal modeling.
    """
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 1.0,
                 dropout_rate: float = 0.1, max_seq_len: int = 1000,
                 temporal_window: int = 3, temporal_dim: int = 8):
        
        # Initialize base attention model
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
            ability_scale=ability_scale,
            dropout_rate=dropout_rate
        )
        
        # Override model name
        self.model_name = "temporal_attention_gpcm"
        
        # Store temporal parameters
        self.max_seq_len = max_seq_len
        self.temporal_window = temporal_window
        self.temporal_dim = temporal_dim
        
        # Add positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        
        # Add temporal feature extractor
        self.temporal_extractor = TemporalFeatureExtractor(
            window_size=temporal_window,
            feature_dim=temporal_dim
        )
        
        # Add feature fusion layer  
        self.feature_fusion = FeatureFusionLayer(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
            dropout_rate=dropout_rate
        )
        
        # Override value embedding to work with fused features
        self.gpcm_value_embed = nn.Linear(embed_dim, value_dim)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
    
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create embeddings with positional encoding."""
        # Get base embeddings (projected to embed_dim)
        base_embeds = super().create_embeddings(questions, responses)  # [B, T, embed_dim]
        
        # Add positional encoding
        position_encoded = self.positional_encoding(base_embeds)  # [B, T, embed_dim]
        
        return position_encoded
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor,
                          responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention refinement to embeddings with temporal features."""
        batch_size, seq_len = gpcm_embeds.shape[:2]
        
        # Apply standard attention refinement first
        if self.use_ordinal_attention and responses is not None:
            refined_embeds = self.attention_refinement(gpcm_embeds, responses)
        else:
            refined_embeds = self.attention_refinement(gpcm_embeds)
        
        # Add temporal features to each timestep
        enhanced_embeds = []
        for t in range(seq_len):
            # Extract temporal features for current timestep
            if responses is not None and hasattr(self, '_current_questions'):
                temporal_features_t = self.temporal_extractor.extract_features(
                    self._current_questions, responses, t
                )
            else:
                # Fallback: create zero temporal features
                temporal_features_t = torch.zeros(batch_size, self.temporal_dim, device=gpcm_embeds.device)
            
            # Fuse attention-refined embedding with temporal features
            enhanced_embed_t = self.feature_fusion(
                refined_embeds[:, t, :],  # [B, embed_dim]
                temporal_features_t       # [B, temporal_dim] 
            )  # [B, embed_dim]
            enhanced_embeds.append(enhanced_embed_t)
        
        # Stack enhanced embeddings
        enhanced_embeds = torch.stack(enhanced_embeds, dim=1)  # [B, T, embed_dim]
        
        return enhanced_embeds
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with temporal enhancement.
        
        Follows the exact same pipeline as parent:
        Embedding + PositionalEncoding → Attention + TemporalFeatures → DKVMN → IRT → GPCM
        """
        # Store questions and responses for process_embeddings
        self._current_questions = questions
        self._current_responses = responses
        
        # Call parent forward - this handles the entire pipeline correctly
        return super().forward(questions, responses)
    
    def get_model_info(self) -> Dict:
        """Get model information including temporal enhancements."""
        base_info = super().get_model_info() if hasattr(super(), 'get_model_info') else {}
        
        temporal_info = {
            'model_type': 'TemporalAttentionGPCM',
            'has_positional_encoding': True,
            'has_temporal_features': True,
            'temporal_window': self.temporal_window,
            'temporal_dim': self.temporal_dim,
            'max_seq_len': self.max_seq_len,
            'enhancements': ['positional_encoding', 'temporal_features', 'feature_fusion']
        }
        
        return {**base_info, **temporal_info}