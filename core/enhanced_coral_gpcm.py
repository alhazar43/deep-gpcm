"""
Enhanced CORAL-GPCM model with deep mathematical integration of threshold parameters.

This module implements the research insights for mathematically coupling
GPCM β thresholds with CORAL ordinal thresholds through various integration
mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .coral_gpcm import CORALDeepGPCM
from .threshold_coupling import (
    HierarchicalThresholdCoupler,
    AttentionThresholdCoupler, 
    NeuralThresholdCoupler,
    AdaptiveThresholdCoupler
)


class MathematicallyIntegratedCORALGPCM(CORALDeepGPCM):
    """
    Enhanced CORAL-GPCM model with deep mathematical integration of thresholds.
    
    This model replaces the simple blending approach (line 171 in coral_gpcm.py)
    with sophisticated mathematical coupling mechanisms that leverage the 
    complementary nature of GPCM and CORAL K-1 threshold parameters.
    
    Key innovations:
    1. Hierarchical threshold coupling (GPCM base + CORAL refinement)
    2. Cross-threshold attention mechanisms  
    3. Adaptive weighting based on student ability
    4. Multiple coupling strategies with automatic selection
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
                 # Enhanced coupling parameters
                 coupling_mode: str = 'hierarchical',
                 enable_adaptive_weighting: bool = True,
                 enable_threshold_analysis: bool = False,
                 **kwargs):
        """
        Initialize mathematically integrated CORAL-GPCM model.
        
        Args:
            All base parameters plus:
            coupling_mode: Type of threshold coupling ('hierarchical', 'attention', 'neural', 'adaptive')
            enable_adaptive_weighting: Whether to use adaptive weighting based on θ and α
            enable_threshold_analysis: Whether to perform threshold relationship analysis
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
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        # Update model name
        self.model_name = f"enhanced_coral_gpcm_{coupling_mode}"
        
        # Store coupling parameters
        self.coupling_mode = coupling_mode
        self.enable_adaptive_weighting = enable_adaptive_weighting
        self.enable_threshold_analysis = enable_threshold_analysis
        
        # Initialize threshold coupling mechanism
        self._init_coupling_mechanism()
        
        # Threshold analysis (for research)
        if enable_threshold_analysis:
            self.threshold_analyzer = ThresholdAnalyzer(n_cats - 1)
    
    def _init_coupling_mechanism(self):
        """Initialize the appropriate threshold coupling mechanism"""
        n_thresholds = self.n_cats - 1
        
        if self.coupling_mode == 'hierarchical':
            self.threshold_coupler = HierarchicalThresholdCoupler(
                n_thresholds=n_thresholds,
                enable_adaptive_weighting=self.enable_adaptive_weighting
            )
        elif self.coupling_mode == 'attention':
            self.threshold_coupler = AttentionThresholdCoupler(
                n_thresholds=n_thresholds,
                n_heads=2
            )
        elif self.coupling_mode == 'neural':
            self.threshold_coupler = NeuralThresholdCoupler(
                n_thresholds=n_thresholds,
                hidden_dim=n_thresholds * 3
            )
        elif self.coupling_mode == 'adaptive':
            self.threshold_coupler = AdaptiveThresholdCoupler(
                n_thresholds=n_thresholds
            )
        else:
            raise ValueError(f"Unknown coupling mode: {self.coupling_mode}")
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with mathematical threshold integration.
        
        This method enhances the base CORALDeepGPCM forward pass by replacing
        the simple threshold blending with sophisticated mathematical coupling.
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
        summary_features = torch.stack(summary_vectors, dim=1)
        
        # Apply CORAL layer for ordinal predictions
        coral_probs, coral_info = self.coral_layer(summary_features)
        
        # Store CORAL info for loss computation
        self.last_coral_info = coral_info
        
        # ===== MATHEMATICAL THRESHOLD INTEGRATION =====
        # This replaces the simple blending in the base model
        if self.use_hybrid_mode and self.coral_layer.use_thresholds:
            coral_thresholds = coral_info['thresholds']
            if coral_thresholds is not None:
                # Apply sophisticated mathematical coupling
                unified_thresholds, coupling_info = self.threshold_coupler(
                    gpcm_betas=item_thresholds,           # GPCM β parameters
                    coral_taus=coral_thresholds,          # CORAL τ parameters
                    theta=student_abilities,              # θ for adaptive weighting
                    alpha=discrimination_params           # α for context
                )
                
                # Store coupling information for analysis
                self.last_coupling_info = coupling_info
                
                # Use unified thresholds
                item_thresholds = unified_thresholds
                
                # Optional threshold analysis (for research)
                if self.enable_threshold_analysis and self.training:
                    self._analyze_thresholds(coupling_info)
        
        return student_abilities, item_thresholds, discrimination_params, coral_probs
    
    def _analyze_thresholds(self, coupling_info: Dict[str, torch.Tensor]):
        """Analyze threshold relationships during training (research mode)"""
        if hasattr(self, 'threshold_analyzer'):
            with torch.no_grad():
                analysis = self.threshold_analyzer.analyze_coupling(coupling_info)
                self.last_threshold_analysis = analysis
    
    def get_coupling_info(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get threshold coupling information from last forward pass"""
        return getattr(self, 'last_coupling_info', None)
    
    def get_threshold_analysis(self) -> Optional[Dict[str, Any]]:
        """Get threshold analysis from last forward pass (research mode)"""
        return getattr(self, 'last_threshold_analysis', None)
    
    def get_integration_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive integration diagnostics"""
        diagnostics = {
            'model_name': self.model_name,
            'coupling_mode': self.coupling_mode,
            'enable_adaptive_weighting': self.enable_adaptive_weighting,
            'coral_info': self.get_coral_info(),
            'coupling_info': self.get_coupling_info(),
        }
        
        if self.enable_threshold_analysis:
            diagnostics['threshold_analysis'] = self.get_threshold_analysis()
        
        return diagnostics


class ThresholdAnalyzer(nn.Module):
    """Research module for analyzing threshold relationships and coupling effectiveness"""
    
    def __init__(self, n_thresholds: int):
        super().__init__()
        self.n_thresholds = n_thresholds
        
        # Running statistics for threshold analysis
        self.register_buffer('gpcm_mean', torch.zeros(n_thresholds))
        self.register_buffer('coral_mean', torch.zeros(n_thresholds))
        self.register_buffer('unified_mean', torch.zeros(n_thresholds))
        self.register_buffer('update_count', torch.tensor(0))
    
    def analyze_coupling(self, coupling_info: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze threshold coupling effectiveness"""
        analysis = {}
        
        # Extract threshold information
        if 'gpcm_contribution' in coupling_info:
            gpcm_thresholds = coupling_info['gpcm_contribution']
            analysis['gpcm_std'] = gpcm_thresholds.std().item()
            analysis['gpcm_mean'] = gpcm_thresholds.mean().item()
        
        if 'coral_contribution' in coupling_info:
            coral_contribution = coupling_info['coral_contribution']
            analysis['coral_contribution_std'] = coral_contribution.std().item()
            analysis['coral_contribution_mean'] = coral_contribution.mean().item()
        
        if 'adaptive_weights' in coupling_info:
            weights = coupling_info['adaptive_weights']
            analysis['weight_entropy'] = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean().item()
            analysis['weight_balance'] = (weights - 0.5).abs().mean().item()
        
        # Update running statistics
        self._update_running_stats(coupling_info)
        
        return analysis
    
    def _update_running_stats(self, coupling_info: Dict[str, torch.Tensor]):
        """Update running statistics for threshold analysis"""
        self.update_count += 1
        alpha = 1.0 / self.update_count.float()
        
        if 'gpcm_contribution' in coupling_info:
            batch_mean = coupling_info['gpcm_contribution'].mean(dim=(0, 1))
            self.gpcm_mean = (1 - alpha) * self.gpcm_mean + alpha * batch_mean
        
        if 'coral_taus_expanded' in coupling_info:
            batch_mean = coupling_info['coral_taus_expanded'].mean(dim=(0, 1))
            self.coral_mean = (1 - alpha) * self.coral_mean + alpha * batch_mean


def test_enhanced_coral_gpcm():
    """Test the enhanced CORAL-GPCM model with different coupling modes"""
    
    print("Testing Enhanced CORAL-GPCM Models")
    print("=" * 60)
    
    # Test parameters
    n_questions = 100
    n_cats = 4
    batch_size, seq_len = 4, 8
    
    # Sample data
    questions = torch.randint(1, n_questions + 1, (batch_size, seq_len))
    responses = torch.randint(0, n_cats, (batch_size, seq_len))
    
    # Test each coupling mode
    coupling_modes = ['hierarchical', 'attention', 'neural', 'adaptive']
    
    for mode in coupling_modes:
        print(f"\nTesting {mode.capitalize()} Coupling:")
        print("-" * 40)
        
        # Create model
        model = MathematicallyIntegratedCORALGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            coupling_mode=mode,
            enable_adaptive_weighting=True,
            enable_threshold_analysis=True
        )
        
        # Forward pass
        theta, beta, alpha, probs = model(questions, responses)
        
        print(f"  Model: {model.model_name}")
        print(f"  Output shapes: θ={theta.shape}, β={beta.shape}, α={alpha.shape}, P={probs.shape}")
        
        # Get integration diagnostics
        diagnostics = model.get_integration_diagnostics()
        coupling_info = diagnostics['coupling_info']
        
        if coupling_info:
            print(f"  Coupling info keys: {list(coupling_info.keys())[:3]}...")
            
            # Show adaptive weighting if available
            if 'adaptive_weights' in coupling_info:
                weights = coupling_info['adaptive_weights']
                weight_mean = weights.mean().item()
                weight_std = weights.std().item() 
                print(f"  Adaptive weights: μ={weight_mean:.3f}, σ={weight_std:.3f}")
        
        # Validate outputs
        assert theta.shape == (batch_size, seq_len), f"Theta shape mismatch in {mode}"
        assert beta.shape == (batch_size, seq_len, n_cats - 1), f"Beta shape mismatch in {mode}"
        assert probs.shape == (batch_size, seq_len, n_cats), f"Probs shape mismatch in {mode}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))), f"Probs don't sum to 1 in {mode}"
        
        print(f"  ✓ All checks passed")
    
    print("\n" + "=" * 60)
    print("All enhanced models tested successfully!")


def compare_coupling_effectiveness():
    """Compare the effectiveness of different coupling mechanisms"""
    
    print("\nCoupling Effectiveness Comparison")
    print("=" * 60)
    
    # Create models with different coupling modes
    n_questions, n_cats = 50, 4
    batch_size, seq_len = 16, 12
    
    models = {}
    for mode in ['hierarchical', 'attention', 'neural', 'adaptive']:
        models[mode] = MathematicallyIntegratedCORALGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            coupling_mode=mode,
            enable_threshold_analysis=True
        )
    
    # Generate test data
    questions = torch.randint(1, n_questions + 1, (batch_size, seq_len))
    responses = torch.randint(0, n_cats, (batch_size, seq_len))
    
    # Compare outputs
    results = {}
    for mode, model in models.items():
        model.eval()
        with torch.no_grad():
            theta, beta, alpha, probs = model(questions, responses)  
            
            results[mode] = {
                'theta_std': theta.std().item(),
                'beta_std': beta.std().item(),
                'prob_entropy': -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item(),
                'beta_range': (beta.max() - beta.min()).item()
            }
    
    # Display comparison
    print("\nMetric Comparison:")
    print(f"{'Mode':<12} {'θ_std':<8} {'β_std':<8} {'H(P)':<8} {'β_range':<8}")
    print("-" * 45)
    
    for mode, metrics in results.items():
        print(f"{mode:<12} {metrics['theta_std']:<8.3f} {metrics['beta_std']:<8.3f} "
              f"{metrics['prob_entropy']:<8.3f} {metrics['beta_range']:<8.3f}")


if __name__ == "__main__":
    test_enhanced_coral_gpcm()
    compare_coupling_effectiveness()