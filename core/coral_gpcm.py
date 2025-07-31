"""CORAL-enhanced Deep GPCM model that extends the base architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .model import DeepGPCM
from .coral_layer import CORALLayer


class CORALDeepGPCM(DeepGPCM):
    """CORAL-enhanced Deep GPCM model with ordinal regression improvements.
    
    This model extends the base DeepGPCM by replacing the standard GPCM probability
    layer with a CORAL (COnsistent RAnk Logits) layer that enforces ordinal
    constraints. It maintains full compatibility with the existing pipeline.
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
                 # CORAL-specific parameters
                 coral_hidden_dim: Optional[int] = None,
                 use_coral_thresholds: bool = True,
                 coral_dropout: float = 0.1):
        """Initialize CORAL-enhanced Deep GPCM.
        
        Args:
            All base DeepGPCM parameters plus:
            coral_hidden_dim: Hidden dimension for CORAL shared layer (default: final_fc_dim)
            use_coral_thresholds: Whether to use learnable ordinal thresholds
            coral_dropout: Dropout rate for CORAL layer
        """
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
            use_discrimination=use_discrimination,
            dropout_rate=dropout_rate
        )
        
        # Update model name
        self.model_name = "coral_deepgpcm"
        
        # Store CORAL parameters
        self.use_coral = True
        coral_hidden = coral_hidden_dim or final_fc_dim
        
        # Replace GPCM layer with CORAL layer
        self.coral_layer = CORALLayer(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            use_bias=True,
            dropout_rate=coral_dropout,
            shared_hidden_dim=coral_hidden
        )
        self.coral_layer.use_thresholds = use_coral_thresholds
        
        # Keep original GPCM layer for IRT parameter compatibility
        # This allows us to still extract and return IRT parameters
        self.use_hybrid_mode = True
        
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through CORAL-enhanced DKVMN-GPCM model.
        
        This method maintains the same interface as the base model, returning
        IRT parameters for compatibility while using CORAL for predictions.
        
        Args:
            questions: Question IDs, shape (batch_size, seq_len)
            responses: Response categories, shape (batch_size, seq_len)
            
        Returns:
            tuple: (student_abilities, item_thresholds, discrimination_params, coral_probs)
                - student_abilities: θ parameters extracted from memory
                - item_thresholds: β parameters (from CORAL thresholds if available)
                - discrimination_params: α parameters
                - coral_probs: CORAL ordinal probabilities
        """
        batch_size, seq_len = questions.shape
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Create embeddings
        gpcm_embeds = self.create_embeddings(questions, responses)
        q_embeds = self.q_embed(questions)
        
        # Process embeddings
        processed_embeds = self.process_embeddings(gpcm_embeds, q_embeds)
        
        # Sequential processing
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        coral_probs = []
        summary_vectors = []
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]
            gpcm_embed_t = processed_embeds[:, t, :]
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            summary_vectors.append(summary_vector)
            
            # Extract IRT parameters (for compatibility and analysis)
            theta_t, alpha_t, betas_t = self.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)
            alpha_t = alpha_t.squeeze(1) if alpha_t is not None else None
            betas_t = betas_t.squeeze(1)
            
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            if alpha_t is not None:
                discrimination_params.append(alpha_t)
            
            # Memory update
            self.memory.write(correlation_weight, value_embed_t)
        
        # Stack temporal sequences
        student_abilities = torch.stack(student_abilities, dim=1)
        item_thresholds = torch.stack(item_thresholds, dim=1)
        discrimination_params = torch.stack(discrimination_params, dim=1) if discrimination_params else None
        
        # Stack summary vectors for CORAL
        summary_features = torch.stack(summary_vectors, dim=1)  # (batch_size, seq_len, final_fc_dim)
        
        # Apply CORAL layer for ordinal predictions
        coral_probs, coral_info = self.coral_layer(summary_features)
        
        # Store CORAL info for loss computation
        self.last_coral_info = coral_info
        
        # If using hybrid mode, we can optionally blend CORAL thresholds with IRT betas
        if self.use_hybrid_mode and self.coral_layer.use_thresholds:
            # Use CORAL thresholds as refined beta parameters
            coral_thresholds = coral_info['thresholds']
            if coral_thresholds is not None:
                # Broadcast thresholds to match sequence length
                refined_thresholds = coral_thresholds.unsqueeze(0).unsqueeze(0).expand(
                    batch_size, seq_len, -1
                )
                # Optionally blend with IRT thresholds
                item_thresholds = 0.5 * item_thresholds + 0.5 * refined_thresholds
        
        # Return in same format as base model for compatibility
        return student_abilities, item_thresholds, discrimination_params, coral_probs
    
    def get_coral_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get CORAL-specific information from last forward pass.
        
        Returns:
            Dictionary with CORAL logits, cumulative probabilities, etc.
        """
        return getattr(self, 'last_coral_info', None)


class HybridCORALGPCM(DeepGPCM):
    """Hybrid model that combines CORAL structure with GPCM IRT parameters.
    
    This model uses CORAL's rank-consistent structure while maintaining
    explicit IRT parameter interpretation through a hybrid approach.
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
                 # Hybrid-specific parameters
                 use_coral_structure: bool = True,
                 blend_weight: float = 0.5):
        """Initialize Hybrid CORAL-GPCM model.
        
        Args:
            All base DeepGPCM parameters plus:
            use_coral_structure: Whether to use CORAL cumulative structure
            blend_weight: Weight for blending CORAL and GPCM predictions (0=GPCM, 1=CORAL)
        """
        super().__init__(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            embedding_strategy=embedding_strategy,
            ability_scale=ability_scale,
            use_discrimination=use_discrimination,
            dropout_rate=dropout_rate
        )
        
        self.model_name = "hybrid_coral_gpcm"
        self.use_coral_structure = use_coral_structure
        self.blend_weight = blend_weight
        
        # Additional projection for CORAL-style cumulative logits
        if use_coral_structure:
            self.coral_projection = nn.Linear(final_fc_dim, n_cats - 1)
            nn.init.kaiming_normal_(self.coral_projection.weight)
            nn.init.zeros_(self.coral_projection.bias)
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with hybrid CORAL-GPCM predictions."""
        # Get base model outputs
        student_abilities, item_thresholds, discrimination_params, gpcm_probs = super().forward(
            questions, responses
        )
        
        if not self.use_coral_structure:
            return student_abilities, item_thresholds, discrimination_params, gpcm_probs
        
        # Apply CORAL structure using IRT parameters
        batch_size, seq_len = questions.shape
        
        # Compute CORAL-style cumulative logits using IRT parameters
        # For each threshold k: logit_k = alpha * (theta - beta_k)
        coral_logits = []
        
        for t in range(seq_len):
            theta_t = student_abilities[:, t:t+1]  # (batch_size, 1)
            beta_t = item_thresholds[:, t, :]  # (batch_size, n_cats-1)
            
            if discrimination_params is not None:
                alpha_t = discrimination_params[:, t:t+1]  # (batch_size, 1)
            else:
                alpha_t = torch.ones_like(theta_t)
            
            # Compute cumulative logits
            logits_t = alpha_t * (theta_t - beta_t)  # (batch_size, n_cats-1)
            coral_logits.append(logits_t)
        
        coral_logits = torch.stack(coral_logits, dim=1)  # (batch_size, seq_len, n_cats-1)
        
        # Convert to probabilities using CORAL structure
        cum_probs = torch.sigmoid(coral_logits)
        
        # Convert cumulative to categorical
        coral_probs = self._cumulative_to_categorical(cum_probs)
        
        # Blend CORAL and GPCM predictions
        final_probs = (1 - self.blend_weight) * gpcm_probs + self.blend_weight * coral_probs
        
        # Renormalize
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        
        return student_abilities, item_thresholds, discrimination_params, final_probs
    
    def _cumulative_to_categorical(self, cum_probs: torch.Tensor) -> torch.Tensor:
        """Convert cumulative probabilities to categorical."""
        # P(Y = 0) = 1 - P(Y > 0)
        p0 = 1 - cum_probs[..., 0:1]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., K-2
        pk = cum_probs[..., :-1] - cum_probs[..., 1:]
        
        # P(Y = K-1) = P(Y > K-2)
        pK = cum_probs[..., -1:]
        
        # Concatenate all probabilities
        probs = torch.cat([p0, pk, pK], dim=-1)
        
        # Ensure non-negative and normalized
        probs = torch.clamp(probs, min=1e-7)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        return probs