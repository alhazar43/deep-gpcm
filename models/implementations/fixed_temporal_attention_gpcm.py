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


class AdaptivePositionalEncoding(nn.Module):
    """Learnable positional encoding that adapts to educational sequences."""
    
    def __init__(self, embed_dim: int, max_seq_len: int = 1000, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # Learnable positional embeddings (much smaller scale than sinusoidal)
        self.positional_embeddings = nn.Parameter(
            torch.zeros(max_seq_len, embed_dim)
        )
        
        # Scale parameter to control positional influence
        self.position_scale = nn.Parameter(torch.tensor(0.1))
        
        # Initialize with small random values
        nn.init.normal_(self.positional_embeddings, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add scaled learnable positional encoding.
        
        Args:
            x: Input embeddings [batch_size, seq_len, embed_dim]
            
        Returns:
            Position-encoded embeddings [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Get positional embeddings for this sequence length
        positions = self.positional_embeddings[:seq_len, :].unsqueeze(0)
        positions = positions.expand(batch_size, -1, -1)
        
        # Apply learnable scaling
        scaled_positions = positions * torch.sigmoid(self.position_scale)
        
        # Add to input with dropout
        return self.dropout(x + scaled_positions)


class StabilizedTemporalFeatureExtractor(nn.Module):
    """Improved temporal feature extractor with better initialization and stability."""
    
    def __init__(self, window_size: int = 3, feature_dim: int = 8):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Improved temporal projection with batch norm
        self.temporal_projection = nn.Sequential(
            nn.Linear(window_size * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Better initialization
        for module in self.temporal_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)  # Smaller gain
                nn.init.zeros_(module.bias)
    
    def extract_features(self, questions: torch.Tensor, responses: torch.Tensor, 
                        timestep: int) -> torch.Tensor:
        """Extract stabilized temporal features."""
        batch_size = questions.shape[0]
        device = questions.device
        
        # Define window boundaries
        start_idx = max(0, timestep - self.window_size + 1)
        end_idx = timestep + 1
        
        # Extract windowed features with normalization
        features = []
        
        for t in range(start_idx, end_idx):
            if t >= 0 and t < questions.shape[1]:
                # Normalized time gap (0 to 1 range)
                time_gap = float(timestep - t) / self.window_size
                # Normalized correctness
                correctness = responses[:, t].float() / 4.0  # Assume 0-4 scale
            else:
                # Padding values
                time_gap = 1.0  # Max normalized gap
                correctness = torch.zeros(batch_size, device=device)
            
            features.append(torch.full((batch_size,), time_gap, device=device))
            features.append(correctness)
        
        # Pad to consistent size
        while len(features) < self.window_size * 2:
            features.extend([
                torch.ones(batch_size, device=device),  # Max gap
                torch.zeros(batch_size, device=device)  # Zero correctness
            ])
        
        # Stack and project
        temporal_input = torch.stack(features, dim=1)
        temporal_features = self.temporal_projection(temporal_input)
        
        return temporal_features


class StabilizedFeatureFusionLayer(nn.Module):
    """Improved feature fusion with better gradient flow."""
    
    def __init__(self, embed_dim: int, temporal_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Improved fusion with residual connection
        self.temporal_transform = nn.Sequential(
            nn.Linear(temporal_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # Learnable fusion weight (starts conservative)
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
        
        # Layer norm for final output
        self.output_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Conservative initialization."""
        for module in self.temporal_transform:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(self, attention_embed: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """Stabilized feature fusion with residual connection."""
        
        # Transform temporal features to embedding space
        temporal_transformed = self.temporal_transform(temporal_features)
        
        # Learnable weighted fusion with residual connection
        fusion_strength = torch.sigmoid(self.fusion_weight)
        fused = attention_embed + fusion_strength * temporal_transformed
        
        # Layer normalization
        return self.output_norm(fused)


class FixedTemporalAttentionGPCM(AttentionGPCM):
    """Fixed Temporal Attention GPCM with stabilized components.
    
    Key fixes:
    1. Replaced sinusoidal with adaptive learnable positional encoding
    2. Improved temporal feature extraction with batch norm and better init
    3. Stabilized feature fusion with residual connections
    4. Conservative initialization and scaling parameters
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
        self.model_name = "fixed_temporal_attention_gpcm"
        
        # Store parameters
        self.max_seq_len = max_seq_len
        self.temporal_window = temporal_window
        self.temporal_dim = temporal_dim
        
        # Replace with adaptive positional encoding
        self.positional_encoding = AdaptivePositionalEncoding(
            embed_dim, max_seq_len, dropout_rate
        )
        
        # Stabilized temporal feature extractor
        self.temporal_extractor = StabilizedTemporalFeatureExtractor(
            window_size=temporal_window,
            feature_dim=temporal_dim
        )
        
        # Stabilized feature fusion layer
        self.feature_fusion = StabilizedFeatureFusionLayer(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
            dropout_rate=dropout_rate
        )
        
        # Keep the same value embedding
        self.gpcm_value_embed = nn.Linear(embed_dim, value_dim)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
    
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create embeddings with adaptive positional encoding."""
        # Get base embeddings from AttentionGPCM (which projects to embed_dim)
        base_embeds = super().create_embeddings(questions, responses)
        
        # Add adaptive positional encoding
        position_encoded = self.positional_encoding(base_embeds)
        
        return position_encoded
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor,
                          responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention refinement with stabilized temporal features."""
        batch_size, seq_len = gpcm_embeds.shape[:2]
        
        # Apply standard attention refinement first
        if self.use_ordinal_attention and responses is not None:
            refined_embeds = self.attention_refinement(gpcm_embeds, responses)
        else:
            refined_embeds = self.attention_refinement(gpcm_embeds)
        
        # Add stabilized temporal features
        enhanced_embeds = []
        for t in range(seq_len):
            # Extract temporal features
            if responses is not None and hasattr(self, '_current_questions'):
                temporal_features_t = self.temporal_extractor.extract_features(
                    self._current_questions, responses, t
                )
            else:
                # Fallback: create zero temporal features
                temporal_features_t = torch.zeros(batch_size, self.temporal_dim, device=gpcm_embeds.device)
            
            # Stabilized feature fusion
            enhanced_embed_t = self.feature_fusion(
                refined_embeds[:, t, :],
                temporal_features_t
            )
            enhanced_embeds.append(enhanced_embed_t)
        
        # Stack enhanced embeddings
        enhanced_embeds = torch.stack(enhanced_embeds, dim=1)
        return enhanced_embeds
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with stabilized temporal enhancement."""
        # Store questions and responses for process_embeddings
        self._current_questions = questions
        self._current_responses = responses
        
        # Call parent forward (AttentionGPCM, which calls DeepGPCM)
        return super().forward(questions, responses)
    
    def get_model_info(self) -> Dict:
        """Get model information including fixes applied."""
        base_info = super().get_model_info() if hasattr(super(), 'get_model_info') else {}
        
        fix_info = {
            'model_type': 'FixedTemporalAttentionGPCM',
            'fixes_applied': [
                'adaptive_learnable_positional_encoding',
                'stabilized_temporal_extraction', 
                'improved_feature_fusion',
                'conservative_initialization',
                'residual_connections',
                'batch_normalization'
            ],
            'has_positional_encoding': True,
            'has_temporal_features': True,
            'temporal_window': self.temporal_window,
            'temporal_dim': self.temporal_dim,
            'max_seq_len': self.max_seq_len,
            'position_scale': self.positional_encoding.position_scale.item(),
            'fusion_weight': self.feature_fusion.fusion_weight.item()
        }
        
        return {**base_info, **fix_info}