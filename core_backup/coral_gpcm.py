"""CORAL-enhanced Deep GPCM models with clean, consolidated architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .model import DeepGPCM


@dataclass
class ThresholdCouplingConfig:
    """Simple configuration for threshold coupling."""
    enabled: bool = False
    coupling_type: str = "linear"
    gpcm_weight: float = 0.7
    coral_weight: float = 0.3


class HybridCORALGPCM(DeepGPCM):
    """Hybrid model that combines CORAL structure with GPCM IRT parameters.
    
    This model uses CORAL's rank-consistent structure while maintaining
    explicit IRT parameter interpretation through DIRECT mathematical integration.
    High-performance architecture with IRT-direct CORAL computation.
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
        """Initialize Hybrid CORAL-GPCM model with direct IRT-CORAL integration.
        
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
        
        # NO separate CORAL layer - use direct IRT-CORAL mathematical integration
    
    def _cumulative_to_categorical(self, cum_probs: torch.Tensor) -> torch.Tensor:
        """Convert cumulative probabilities to categorical probabilities."""
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
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with DIRECT IRT-CORAL mathematical integration.
        
        This uses the high-performance architecture with direct alpha*(theta-beta)
        computation for CORAL structure, eliminating separate CORAL layer overhead.
        """
        # Get base model outputs first
        student_abilities, item_thresholds, discrimination_params, gpcm_probs = super().forward(
            questions, responses
        )
        
        if not self.use_coral_structure:
            return student_abilities, item_thresholds, discrimination_params, gpcm_probs
        
        # Apply CORAL structure using DIRECT IRT-CORAL mathematical integration
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
            
            # DIRECT IRT-CORAL integration: alpha * (theta - beta)
            logits_t = alpha_t * (theta_t - beta_t)  # (batch_size, n_cats-1)
            coral_logits.append(logits_t)
        
        coral_logits = torch.stack(coral_logits, dim=1)  # (batch_size, seq_len, n_cats-1)
        
        # Convert to probabilities using CORAL structure
        cum_probs = torch.sigmoid(coral_logits)
        
        # Convert cumulative to categorical
        coral_probs = self._cumulative_to_categorical(cum_probs)
        
        # Store CORAL logits for potential loss computation
        self._last_coral_logits = coral_logits
        
        # Blend CORAL and GPCM predictions
        final_probs = (1 - self.blend_weight) * gpcm_probs + self.blend_weight * coral_probs
        
        # Renormalize
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        
        return student_abilities, item_thresholds, discrimination_params, final_probs
    
    def get_coral_info(self) -> Optional[Dict[str, Any]]:
        """Get CORAL-specific information from last forward pass."""
        if not hasattr(self, '_last_coral_logits'):
            return None
        
        return {
            "logits": self._last_coral_logits,
            "integration_type": "direct_irt_coral"
        }


class EnhancedCORALGPCM(HybridCORALGPCM):
    """Enhanced CORAL-GPCM with adaptive threshold-distance-based blending.
    
    This model implements sophisticated integration between GPCM β thresholds and 
    CORAL τ thresholds through adaptive coupling mechanisms and threshold-distance-based
    dynamic blending to address middle category prediction imbalance.
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
                 # Enhanced adaptive coupling parameters
                 enable_threshold_coupling: bool = False,  # Disable to prevent interaction with adaptive blending
                 coupling_type: str = "linear",
                 gpcm_weight: float = 0.7,
                 coral_weight: float = 0.3,
                 # Adaptive blending configuration
                 enable_adaptive_blending: bool = False,
                 blend_weight: float = 0.5,  # Fallback for fixed blending
                 range_sensitivity_init: float = 0.01,  # Very conservative init for stability
                 distance_sensitivity_init: float = 0.01,  # Very conservative init for stability
                 baseline_bias_init: float = 0.0,
                 **kwargs):
        """Initialize Enhanced CORAL-GPCM with optional adaptive blending.
        
        Args:
            All base parameters plus:
            enable_threshold_coupling: Whether to enable adaptive threshold coupling
            coupling_type: Type of threshold coupling ('linear')
            gpcm_weight: Weight for GPCM thresholds in adaptive coupling
            coral_weight: Weight for CORAL thresholds in adaptive coupling
            enable_adaptive_blending: Enable threshold-distance-based adaptive blending
            blend_weight: Fallback weight for fixed blending when adaptive is disabled
            range_sensitivity_init: Initial range sensitivity for adaptive blending
            distance_sensitivity_init: Initial distance sensitivity for adaptive blending  
            baseline_bias_init: Initial baseline bias for adaptive blending
        """
        # Filter out kwargs that HybridCORALGPCM doesn't accept
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['use_coral_structure', 'blend_weight']}
        
        # Override blend_weight with the adaptive blending fallback weight
        base_kwargs['blend_weight'] = blend_weight
        
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
            dropout_rate=dropout_rate,
            **base_kwargs
        )
        
        self.model_name = "enhanced_coral_gpcm"
        self.enable_threshold_coupling = enable_threshold_coupling
        self.coupling_type = coupling_type
        self.enable_adaptive_blending = enable_adaptive_blending
        
        # Model versioning for backward compatibility
        if enable_adaptive_blending:
            self.model_version = "v2.0_adaptive"
            self.model_name = "enhanced_coral_gpcm_adaptive"
        else:
            self.model_version = "v1.0_standard" 
            # Keep original model_name for backward compatibility
        
        # Import and initialize proper threshold coupling system
        if enable_threshold_coupling:
            from .threshold_coupling import LinearThresholdCoupler
            self.threshold_coupler = LinearThresholdCoupler(
                n_thresholds=n_cats - 1,
                gpcm_weight=gpcm_weight,
                coral_weight=coral_weight
            )
        else:
            self.threshold_coupler = None
        
        # Dedicated CORAL layer for proper ordinal structure
        from .coral_layer import CORALLayer
        self.coral_layer = CORALLayer(
            input_dim=final_fc_dim,
            n_cats=n_cats,
            use_bias=True,
            dropout_rate=0.1,
            shared_hidden_dim=final_fc_dim
        )
        
        # coral_projection for compatibility (points to CORAL layer rank_classifier)
        self.coral_projection = self.coral_layer.rank_classifier
        
        # Initialize adaptive blending system
        if enable_adaptive_blending:
            # Check if we should use full or minimal blender
            use_full_blender = kwargs.get('use_full_blender', False)
            
            if use_full_blender:
                from .full_adaptive_blender import FullAdaptiveBlender
                self.threshold_blender = FullAdaptiveBlender(
                    n_categories=n_cats,
                    range_sensitivity_init=range_sensitivity_init,
                    distance_sensitivity_init=distance_sensitivity_init,
                    baseline_bias_init=baseline_bias_init,
                    use_bgt_transforms=kwargs.get('use_bgt_transforms', True),
                    gradient_clipping=kwargs.get('gradient_clipping', 1.0),
                    parameter_bounds=kwargs.get('parameter_bounds', True)
                )
            else:
                from .minimal_adaptive_blender import MinimalAdaptiveBlender
                self.threshold_blender = MinimalAdaptiveBlender(
                    n_categories=n_cats,
                    base_sensitivity=0.1,
                    distance_threshold=1.0
                )
        else:
            self.threshold_blender = None
    
    def _apply_adaptive_blending(self, 
                                 gpcm_probs: torch.Tensor,
                                 coral_probs: torch.Tensor,
                                 blend_weights: torch.Tensor) -> torch.Tensor:
        """Apply category-specific adaptive blending with robust normalization.
        
        Args:
            gpcm_probs: GPCM probabilities, shape (batch_size, seq_len, n_categories)
            coral_probs: CORAL probabilities, shape (batch_size, seq_len, n_categories)
            blend_weights: Adaptive weights, shape (batch_size, seq_len, n_categories)
            
        Returns:
            final_probs: Adaptively blended probabilities, shape (batch_size, seq_len, n_categories)
        """
        # Element-wise adaptive blending: (1 - w_i) * P_gpcm,i + w_i * P_coral,i
        final_probs = (1 - blend_weights) * gpcm_probs + blend_weights * coral_probs
        
        # Robust probability normalization with numerical stability
        prob_sums = final_probs.sum(dim=-1, keepdim=True)
        final_probs = final_probs / torch.clamp(prob_sums, min=1e-7)
        
        # Ensure probabilities are in valid range [0, 1]
        final_probs = torch.clamp(final_probs, min=1e-7, max=1.0 - 1e-7)
        
        # Final renormalization to ensure exact sum = 1
        final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
        
        return final_probs
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced forward pass with adaptive threshold coupling between GPCM and CORAL."""
        # Get base HybridCORALGPCM forward pass first
        student_abilities, item_thresholds, discrimination_params, base_probs = super().forward(
            questions, responses
        )
        
        # Apply adaptive threshold coupling if enabled
        if self.enable_threshold_coupling and self.threshold_coupler is not None:
            # Get CORAL thresholds from dedicated CORAL layer  
            coral_thresholds = self.coral_projection.bias  # τ thresholds (CORAL)
            
            # Apply adaptive coupling between GPCM β and CORAL τ thresholds
            coupled_thresholds = self.threshold_coupler.couple(
                gpcm_thresholds=item_thresholds,
                coral_thresholds=coral_thresholds,
                student_ability=student_abilities,
                item_discrimination=discrimination_params
            )
            
            # Recompute probabilities with adaptively coupled thresholds
            batch_size, seq_len = questions.shape
            
            # Use CORAL structure with coupled thresholds for enhanced integration
            coral_cum_logits = []
            for t in range(seq_len):
                theta_t = student_abilities[:, t:t+1]  # (batch_size, 1)
                beta_coupled_t = coupled_thresholds[:, t, :]  # (batch_size, n_cats-1)
                
                if discrimination_params is not None:
                    alpha_t = discrimination_params[:, t:t+1]  # (batch_size, 1)
                else:
                    alpha_t = torch.ones_like(theta_t)
                
                # CORAL structure with adaptively coupled thresholds
                logits_t = alpha_t * (theta_t - beta_coupled_t)
                coral_cum_logits.append(logits_t)
            
            coral_cum_logits = torch.stack(coral_cum_logits, dim=1)  # (batch_size, seq_len, n_cats-1)
            
            # Store CORAL logits for loss computation
            self._last_coral_logits = coral_cum_logits
            
            # Convert to categorical probabilities via CORAL structure
            cum_probs = torch.sigmoid(coral_cum_logits)
            coral_probs_coupled = self._cumulative_to_categorical(cum_probs)
            
            # Recompute GPCM with coupled thresholds for consistency
            gpcm_probs_coupled = self.gpcm_layer(student_abilities, discrimination_params, coupled_thresholds)
            
            # ===== MAIN ADAPTIVE BLENDING LOGIC (REPLACES LINE 285) =====
            if self.enable_adaptive_blending and self.threshold_blender is not None:
                try:
                    # Apply adaptive blending (minimal or full depending on blender type)
                    if hasattr(self.threshold_blender, 'calculate_blend_weights'):
                        # Full adaptive blender - needs student abilities and discrimination
                        final_probs = self.threshold_blender(
                            gpcm_probs=gpcm_probs_coupled,
                            coral_probs=coral_probs_coupled,
                            item_betas=coupled_thresholds,
                            ordinal_taus=self.coral_projection.bias,
                            student_abilities=student_abilities,
                            discrimination_alphas=discrimination_params
                        )
                    else:
                        # Minimal adaptive blender - simpler interface
                        final_probs = self.threshold_blender(
                            gpcm_probs=gpcm_probs_coupled,
                            coral_probs=coral_probs_coupled,
                            item_betas=coupled_thresholds,  # Will be detached inside blender
                            ordinal_taus=self.coral_projection.bias
                        )
                    
                except Exception as e:
                    # Graceful fallback to fixed blending on any adaptive blending failure
                    print(f"Warning: Adaptive blending failed: {e}. Using fixed blending.")
                    final_probs = (1 - self.blend_weight) * gpcm_probs_coupled + self.blend_weight * coral_probs_coupled
                    final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
            else:
                # Fixed blending fallback (original line 285 behavior)
                final_probs = (1 - self.blend_weight) * gpcm_probs_coupled + self.blend_weight * coral_probs_coupled
                final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)
            
            # Return with coupled thresholds to show the adaptive integration effect
            return student_abilities, coupled_thresholds, discrimination_params, final_probs
        
        else:
            # No adaptive coupling - return base HybridCORALGPCM results
            # Still need to store some CORAL logits for loss computation if using CORAL loss
            if hasattr(self, 'use_coral_structure') and self.use_coral_structure:
                # Generate simple CORAL logits from base model outputs for loss compatibility
                batch_size, seq_len = questions.shape
                simple_coral_logits = []
                for t in range(seq_len):
                    theta_t = student_abilities[:, t:t+1]  # (batch_size, 1)
                    beta_t = item_thresholds[:, t, :]  # (batch_size, n_cats-1)
                    if discrimination_params is not None:
                        alpha_t = discrimination_params[:, t:t+1]  # (batch_size, 1)
                    else:
                        alpha_t = torch.ones_like(theta_t)
                    logits_t = alpha_t * (theta_t - beta_t)
                    simple_coral_logits.append(logits_t)
                self._last_coral_logits = torch.stack(simple_coral_logits, dim=1)
            
            # Apply adaptive blending even without threshold coupling
            if self.enable_adaptive_blending and self.threshold_blender is not None:
                try:
                    # Need to compute CORAL probabilities for blending
                    batch_size, seq_len = questions.shape
                    coral_cum_logits = []
                    for t in range(seq_len):
                        theta_t = student_abilities[:, t:t+1]  # (batch_size, 1)
                        beta_t = item_thresholds[:, t, :]  # (batch_size, n_cats-1)
                        if discrimination_params is not None:
                            alpha_t = discrimination_params[:, t:t+1]  # (batch_size, 1)
                        else:
                            alpha_t = torch.ones_like(theta_t)
                        logits_t = alpha_t * (theta_t - beta_t)
                        coral_cum_logits.append(logits_t)
                    
                    coral_cum_logits = torch.stack(coral_cum_logits, dim=1)
                    self._last_coral_logits = coral_cum_logits
                    
                    cum_probs = torch.sigmoid(coral_cum_logits)
                    coral_probs = self._cumulative_to_categorical(cum_probs)
                    
                    # Apply adaptive blending using same interface as coupled branch
                    if hasattr(self.threshold_blender, 'calculate_blend_weights'):
                        # Full adaptive blender - needs student abilities and discrimination
                        final_probs = self.threshold_blender(
                            gpcm_probs=base_probs,
                            coral_probs=coral_probs,
                            item_betas=item_thresholds,
                            ordinal_taus=self.coral_projection.bias,
                            student_abilities=student_abilities,
                            discrimination_alphas=discrimination_params
                        )
                    else:
                        # Minimal adaptive blender - simpler interface  
                        final_probs = self.threshold_blender(
                            gpcm_probs=base_probs,
                            coral_probs=coral_probs,
                            item_betas=item_thresholds,  # Will be detached inside blender
                            ordinal_taus=self.coral_projection.bias
                        )
                    
                    return student_abilities, item_thresholds, discrimination_params, final_probs
                    
                except Exception as e:
                    # Graceful fallback to base model on any adaptive blending failure
                    print(f"Warning: Adaptive blending failed: {e}. Using base model.")
            
            return student_abilities, item_thresholds, discrimination_params, base_probs
    
    def get_coupling_info(self) -> Dict[str, Any]:
        """Get enhanced adaptive coupling information."""
        if not self.enable_threshold_coupling or self.threshold_coupler is None:
            return {"coupling_enabled": False}
        
        coupling_weights = self.threshold_coupler.get_coupling_info()
        return {
            "coupling_enabled": True,
            "coupling_type": self.coupling_type,
            "model_type": "enhanced_adaptive",
            "integration_type": "adaptive_threshold_coupling",
            "current_weights": coupling_weights,
            "description": "Sophisticated GPCM β and CORAL τ threshold integration"
        }
    
    def get_adaptive_blending_info(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive adaptive blending analysis information."""
        if not self.enable_adaptive_blending or self.threshold_blender is None:
            return {"adaptive_blending_enabled": False}
        
        blender_info = self.threshold_blender.get_analysis_info()
        if blender_info is None:
            return {"adaptive_blending_enabled": True, "analysis_available": False}
        
        return {
            "adaptive_blending_enabled": True,
            "analysis_available": True,
            "threshold_geometry": blender_info['geometry_metrics'],
            "blend_weights": blender_info['blend_weights'],
            "learnable_parameters": blender_info['learnable_params'],
            "model_type": "enhanced_coral_gpcm_adaptive"
        }
    
    def get_coral_info(self) -> Optional[Dict[str, Any]]:
        """Get CORAL-specific information for loss computation."""
        if not hasattr(self, '_last_coral_logits'):
            return None
        
        return {
            "logits": self._last_coral_logits,
            "integration_type": "adaptive_threshold_coupling"
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for existing trained models."""
        # Always filter out metadata to avoid conflicts
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('_model_metadata')}
        
        # Filter out adaptive blending parameters if they don't exist in the loaded model
        if not self.enable_adaptive_blending:
            # Remove any adaptive blending parameters from state_dict to avoid errors
            filtered_state_dict = {k: v for k, v in filtered_state_dict.items() 
                                 if not k.startswith('threshold_blender.')}
            
            original_len = len([k for k in state_dict.keys() if not k.startswith('_model_metadata')])
            if len(filtered_state_dict) != original_len:
                print(f"Info: Filtered out {original_len - len(filtered_state_dict)} adaptive blending parameters for backward compatibility")
            
            return super().load_state_dict(filtered_state_dict, strict=strict)
        else:
            # For adaptive models, try to load all parameters
            try:
                return super().load_state_dict(filtered_state_dict, strict=strict)
            except RuntimeError as e:
                if "threshold_blender" in str(e) and not strict:
                    # Initialize threshold_blender parameters if missing
                    print("Warning: Initializing missing adaptive blending parameters")
                    return super().load_state_dict(filtered_state_dict, strict=False)
                else:
                    raise e
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict with version information for compatibility tracking."""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add version metadata for compatibility checking (as regular dict entries, not tensors)
        if hasattr(self, 'model_version'):
            # Store as metadata in the returned dict (not in the actual state_dict)
            if destination is None:
                state_dict['_model_metadata'] = {
                    'model_version': self.model_version,
                    'adaptive_blending_enabled': self.enable_adaptive_blending,
                    'model_name': self.model_name
                }
        
        return state_dict