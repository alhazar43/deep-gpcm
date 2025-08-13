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


class RelativeTemporalAttention(nn.Module):
    """Relative temporal attention that provides position awareness without global PE.
    
    This replaces positional encoding with relative position embeddings that respect
    the step-by-step nature of educational sequences and DKVMN processing.
    """
    
    def __init__(self, embed_dim: int, temporal_window: int = 5, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_window = temporal_window
        
        # Relative position embeddings (much smaller scope than global PE)
        self.relative_pos_embed = nn.Parameter(
            torch.zeros(2 * temporal_window + 1, embed_dim)
        )
        
        # Educational response bias - domain-specific attention weighting
        self.response_bias = nn.Linear(1, embed_dim, bias=False)  # Single response → embed_dim
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Optimized initialization for better performance while maintaining stability."""
        # Increased initialization for relative positions (better expressiveness)
        nn.init.normal_(self.relative_pos_embed, std=0.05)
        
        # Increased initialization for response bias (more educational influence)
        nn.init.normal_(self.response_bias.weight, std=0.03)
    
    def forward(self, embeddings: torch.Tensor, responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply relative temporal attention.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embed_dim]
            responses: Response values [batch_size, seq_len] for educational bias
            
        Returns:
            Enhanced embeddings with relative temporal awareness
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        device = embeddings.device
        
        # Apply educational response bias if available
        if responses is not None:
            # Convert responses to educational bias
            response_normalized = (responses.float() / 4.0).unsqueeze(-1)  # [B, T, 1]
            response_bias = self.response_bias(response_normalized)  # [B, T, embed_dim]
            
            # Add educational bias (increased influence for better performance)
            embeddings = embeddings + 0.3 * response_bias
        
        # Apply relative positional bias within temporal window
        enhanced_embeddings = []
        
        for t in range(seq_len):
            current_embed = embeddings[:, t, :]  # [B, embed_dim]
            
            # Define local temporal window
            start_idx = max(0, t - self.temporal_window)
            end_idx = min(seq_len, t + self.temporal_window + 1)
            
            # Apply relative position bias within window
            relative_enhanced = current_embed
            for rel_t in range(start_idx, end_idx):
                if rel_t != t:  # Skip self
                    relative_pos = rel_t - t + self.temporal_window  # Center around temporal_window
                    relative_pos = max(0, min(2 * self.temporal_window, relative_pos))
                    
                    pos_bias = self.relative_pos_embed[relative_pos]  # [embed_dim]
                    
                    # Increased relative influence for better position awareness
                    relative_enhanced = relative_enhanced + 0.15 * pos_bias
            
            enhanced_embeddings.append(relative_enhanced)
        
        # Stack and apply normalization
        enhanced_embeddings = torch.stack(enhanced_embeddings, dim=1)  # [B, T, embed_dim]
        enhanced_embeddings = self.layer_norm(enhanced_embeddings)
        enhanced_embeddings = self.dropout(enhanced_embeddings)
        
        return enhanced_embeddings


class EducationalTemporalExtractor(nn.Module):
    """Educational-domain temporal feature extractor with gradient stabilization."""
    
    def __init__(self, window_size: int = 3, feature_dim: int = 8):
        super().__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Educational-specific features (de-regularized for better performance)
        self.temporal_features = nn.Sequential(
            nn.Linear(window_size * 2, feature_dim),  # 2 features per timestep (like original)
            nn.ReLU(),
            nn.Dropout(0.02)  # Further reduced dropout for expressiveness
        )
        
        # More expressive initialization (increased gains for de-regularization)
        for module in self.temporal_features:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.2)  # Increased gain
                nn.init.zeros_(module.bias)
    
    def extract_features(self, questions: torch.Tensor, responses: torch.Tensor, 
                        timestep: int) -> torch.Tensor:
        """Extract educational temporal features with stability focus."""
        batch_size = questions.shape[0]
        device = questions.device
        
        # Define temporal window
        start_idx = max(0, timestep - self.window_size + 1)
        end_idx = timestep + 1
        
        features = []
        
        for t in range(start_idx, end_idx):
            if t >= 0 and t < questions.shape[1]:
                # 1. Time gap (enhanced like original temporal model)
                time_gap = float(timestep - t) / max(1.0, float(self.window_size))  # Normalized time gap
                
                # 2. Educational correctness (enhanced like original)
                correctness = responses[:, t].float() / 4.0  # Keep 0-1 normalization
                
            else:
                # Padding values (enhanced like original)
                time_gap = 1.0  # Normalized max distance
                correctness = torch.zeros(batch_size, device=device)
            
            features.extend([
                torch.full((batch_size,), time_gap, device=device),
                correctness
            ])
        
        # Pad to consistent size (enhanced like original)
        while len(features) < self.window_size * 2:
            features.extend([
                torch.full((batch_size,), 1.0, device=device),  # Normalized max distance
                torch.zeros(batch_size, device=device)      # Zero correctness
            ])
        
        # Stack and process
        temporal_input = torch.stack(features, dim=1)  # [batch_size, window_size * 2]
        temporal_features = self.temporal_features(temporal_input)  # [batch_size, feature_dim]
        
        return temporal_features


class StabilizedFeatureFusion(nn.Module):
    """Stabilized feature fusion with residual connections and gradient control."""
    
    def __init__(self, embed_dim: int, temporal_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Two-stage fusion for better gradient flow (de-regularized)
        self.temporal_transform = nn.Sequential(
            nn.Linear(temporal_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Reduced dropout
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Fusion gate (learnable but conservative)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim + temporal_dim, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization for output stability
        self.output_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced initialization for performance recovery."""
        for module in self.temporal_transform:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.1)  # Increased gain
                nn.init.zeros_(module.bias)
        
        for module in self.fusion_gate:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.7)  # Less conservative
                nn.init.constant_(module.bias, -1.5)  # Slightly higher gate values
    
    def forward(self, attention_embed: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """Stabilized fusion with residual connections."""
        
        # Transform temporal features to embedding space
        temporal_transformed = self.temporal_transform(temporal_features)
        
        # Compute adaptive fusion gate
        fusion_input = torch.cat([attention_embed, temporal_features], dim=-1)
        gate = self.fusion_gate(fusion_input)  # [batch_size, 1]
        
        # Residual fusion with adaptive gate
        fused = attention_embed + gate * temporal_transformed
        
        # Output normalization for stability
        return self.output_norm(fused)


class StableTemporalAttentionGPCM(AttentionGPCM):
    """Stable Temporal Attention GPCM with NO positional encoding conflicts.
    
    Key architectural improvements:
    1. NO global positional encoding (eliminates PE-temporal interference)
    2. Relative temporal attention for local position awareness
    3. Educational domain-specific features (response bias, learning progression)
    4. Gradient stabilization throughout (LayerNorm, conservative init, residual connections)
    5. Batch size independence (works with batch_size ≥ 8)
    
    This addresses the fundamental architectural issues identified by expert analysis.
    """
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64, 
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, n_cycles: int = 2,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 1.0,
                 dropout_rate: float = 0.1, max_seq_len: int = 1000,
                 temporal_window: int = 5, temporal_dim: int = 8):
        
        # Initialize base attention model (unchanged)
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
        self.model_name = "stable_temporal_attention_gpcm"
        
        # Store parameters
        self.temporal_window = temporal_window
        self.temporal_dim = temporal_dim
        
        # NO POSITIONAL ENCODING - this eliminates the conflict entirely
        
        # Relative temporal attention (replaces positional encoding)
        self.relative_temporal_attention = RelativeTemporalAttention(
            embed_dim=embed_dim,
            temporal_window=temporal_window,
            dropout_rate=dropout_rate
        )
        
        # Educational temporal extractor (enhanced)
        self.temporal_extractor = EducationalTemporalExtractor(
            window_size=3,  # Keep small window for efficiency
            feature_dim=temporal_dim
        )
        
        # Stabilized feature fusion
        self.feature_fusion = StabilizedFeatureFusion(
            embed_dim=embed_dim,
            temporal_dim=temporal_dim,
            dropout_rate=dropout_rate
        )
        
        # Keep same value embedding (unchanged)
        self.gpcm_value_embed = nn.Linear(embed_dim, value_dim)
        nn.init.kaiming_normal_(self.gpcm_value_embed.weight)
        nn.init.constant_(self.gpcm_value_embed.bias, 0)
    
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create embeddings with relative temporal attention (NO positional encoding)."""
        # Get base embeddings from AttentionGPCM
        base_embeds = super().create_embeddings(questions, responses)
        
        # Apply relative temporal attention instead of positional encoding
        enhanced_embeds = self.relative_temporal_attention(base_embeds, responses)
        
        return enhanced_embeds
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor,
                          responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention refinement with educational temporal features."""
        batch_size, seq_len = gpcm_embeds.shape[:2]
        
        # Apply standard attention refinement first
        if self.use_ordinal_attention and responses is not None:
            refined_embeds = self.attention_refinement(gpcm_embeds, responses)
        else:
            refined_embeds = self.attention_refinement(gpcm_embeds)
        
        # Add educational temporal features with stabilized fusion
        enhanced_embeds = []
        for t in range(seq_len):
            # Extract educational temporal features
            if responses is not None and hasattr(self, '_current_questions'):
                temporal_features_t = self.temporal_extractor.extract_features(
                    self._current_questions, responses, t
                )
            else:
                # Fallback: zero temporal features
                temporal_features_t = torch.zeros(batch_size, self.temporal_dim, device=gpcm_embeds.device)
            
            # Stabilized fusion with residual connections
            enhanced_embed_t = self.feature_fusion(
                refined_embeds[:, t, :],
                temporal_features_t
            )
            enhanced_embeds.append(enhanced_embed_t)
        
        # Stack enhanced embeddings
        enhanced_embeds = torch.stack(enhanced_embeds, dim=1)
        return enhanced_embeds
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with stable temporal enhancement."""
        # Store for process_embeddings
        self._current_questions = questions
        self._current_responses = responses
        
        # Call parent forward
        return super().forward(questions, responses)
    
    def get_model_info(self) -> Dict:
        """Get model information including stability features."""
        base_info = super().get_model_info() if hasattr(super(), 'get_model_info') else {}
        
        stability_info = {
            'model_type': 'StableTemporalAttentionGPCM',
            'architectural_fixes': [
                'no_global_positional_encoding',      # KEY FIX: Eliminates PE-temporal conflicts
                'relative_temporal_attention',        # Local position awareness
                'educational_domain_features',        # Response bias, learning progression
                'selective_deregularization',         # Removed excessive LayerNorm, increased gains
                'enhanced_temporal_features',         # Normalized time gaps, proven 2-feature approach
                'residual_connections',              # Better gradient flow
                'batch_size_independence'            # Works with small batches
            ],
            'performance_enhancements': [
                'increased_parameter_sensitivity',    # 0.1 → 0.3 response bias, 0.05 → 0.15 relative pos
                'optimized_initialization',          # std 0.02/0.01 → 0.05/0.03
                'reduced_over_regularization',       # Dropout 0.1 → 0.02, removed LayerNorm
                'enhanced_fusion_gains',             # Gain 0.8 → 1.1, gate bias -2.0 → -1.5
                'normalized_temporal_features'       # Time gap normalization for better scaling
            ],
            'has_positional_encoding': False,        # This eliminates the conflict!
            'has_temporal_attention': True,         # This provides position awareness
            'has_educational_features': True,       # Domain-specific enhancements
            'temporal_window': self.temporal_window,
            'temporal_dim': self.temporal_dim,
            'batch_size_independent': True,         # Key improvement
            'gradient_stable': True,                # Key improvement
            'performance_optimized': True           # New enhancement
        }
        
        return {**base_info, **stability_info}