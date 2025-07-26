"""
Phase 2.4: Bayesian Enhancement for Deep-GPCM
Uncertainty quantification with variational inference and confidence scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from typing import Tuple, Dict, Optional


class VariationalLinear(nn.Module):
    """
    Variational linear layer with learnable mean and variance parameters.
    Implements Bayes by Backprop for uncertainty quantification.
    """
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5.0)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1 - 5.0)
        
        # Prior distribution
        self.prior = dist.Normal(0, prior_std)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty sampling.
        
        Returns:
            output: Layer output
            kl_div: KL divergence for regularization
        """
        # Sample weights from variational posterior
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_dist = dist.Normal(self.weight_mu, weight_sigma)
        weight = weight_dist.rsample()
        
        # Sample bias from variational posterior
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_dist = dist.Normal(self.bias_mu, bias_sigma)
        bias = bias_dist.rsample()
        
        # Compute output
        output = F.linear(x, weight, bias)
        
        # Compute KL divergence
        kl_weight = dist.kl_divergence(weight_dist, self.prior).sum()
        kl_bias = dist.kl_divergence(bias_dist, self.prior).sum()
        kl_div = kl_weight + kl_bias
        
        return output, kl_div


class UncertaintyEstimator(nn.Module):
    """
    Estimates prediction uncertainty using ensemble and dropout approaches.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output uncertainty score [0, 1]
        )
        
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            x: Input features
            n_samples: Number of MC samples
            
        Returns:
            mean_uncertainty: Mean uncertainty estimate
            std_uncertainty: Standard deviation of uncertainty
        """
        self.train()  # Enable dropout for uncertainty estimation
        
        uncertainties = []
        for _ in range(n_samples):
            uncertainty = self.network(x)
            uncertainties.append(uncertainty)
        
        uncertainties = torch.stack(uncertainties, dim=0)
        mean_uncertainty = uncertainties.mean(dim=0)
        std_uncertainty = uncertainties.std(dim=0)
        
        return mean_uncertainty, std_uncertainty


class BayesianKnowledgeState(nn.Module):
    """
    Bayesian knowledge state tracker with uncertainty quantification.
    Maintains probabilistic beliefs about student knowledge.
    """
    
    def __init__(self, n_concepts: int, state_dim: int = 32):
        super().__init__()
        self.n_concepts = n_concepts
        self.state_dim = state_dim
        
        # Knowledge state evolution networks
        self.state_mu_network = nn.Sequential(
            nn.Linear(state_dim + 2, state_dim * 2),  # +2 for difficulty and response
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim)
        )
        
        self.state_sigma_network = nn.Sequential(
            nn.Linear(state_dim + 2, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim),
            nn.Softplus()  # Ensure positive variance
        )
        
        # Concept-specific knowledge mapping
        self.concept_mapping = nn.Linear(state_dim, n_concepts)
        
    def forward(self, prev_state: torch.Tensor, difficulty: torch.Tensor, 
                response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update knowledge state based on response and difficulty.
        
        Args:
            prev_state: Previous knowledge state (batch_size, state_dim)
            difficulty: Question difficulty (batch_size, 1)
            response: Student response (batch_size, 1)
            
        Returns:
            new_mu: Updated state mean
            new_sigma: Updated state variance
            concept_knowledge: Concept-specific knowledge probabilities
        """
        # Combine inputs
        state_input = torch.cat([prev_state, difficulty, response], dim=-1)
        
        # Update state distribution
        new_mu = self.state_mu_network(state_input)
        new_sigma = self.state_sigma_network(state_input)
        
        # Map to concept knowledge
        concept_knowledge = torch.sigmoid(self.concept_mapping(new_mu))
        
        return new_mu, new_sigma, concept_knowledge


class BayesianGPCM(nn.Module):
    """
    Bayesian-enhanced GPCM model with uncertainty quantification.
    Phase 2.4: Integrates variational inference and confidence estimation.
    """
    
    def __init__(self, base_gpcm_model, n_concepts: int = 20, 
                 state_dim: int = 32, n_mc_samples: int = 10, 
                 kl_weight: float = 0.01):
        super().__init__()
        self.base_model = base_gpcm_model
        self.n_concepts = n_concepts
        self.state_dim = state_dim
        self.n_mc_samples = n_mc_samples
        self.kl_weight = kl_weight
        
        # Bayesian components
        self.knowledge_state = BayesianKnowledgeState(n_concepts, state_dim)
        self.uncertainty_estimator = UncertaintyEstimator(
            base_gpcm_model.final_fc_dim + state_dim + 1  # +1 for concept knowledge
        )
        
        # Variational enhancement layers
        enhanced_dim = base_gpcm_model.final_fc_dim + state_dim + n_concepts
        self.variational_predictor = VariationalLinear(
            enhanced_dim, base_gpcm_model.final_fc_dim
        )
        
        # Knowledge state initialization
        self.initial_state_mu = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.initial_state_sigma = nn.Parameter(torch.ones(state_dim) * 0.5)
        
        # Preserve original networks
        self.student_ability_network = base_gpcm_model.student_ability_network
        self.question_threshold_network = base_gpcm_model.question_threshold_network
        self.discrimination_network = base_gpcm_model.discrimination_network
        
        # Bayesian integration weight
        self.bayesian_weight = nn.Parameter(torch.tensor(0.3))
        
    def _get_embeddings(self, q_data, r_data):
        """Get embeddings using base model strategy."""
        batch_size, seq_len = q_data.shape
        
        # Convert question IDs to one-hot vectors
        q_one_hot = F.one_hot(q_data, num_classes=self.base_model.n_questions + 1).float()
        q_one_hot = q_one_hot[:, :, 1:]  # Remove padding dimension
        
        # Apply embedding strategy
        if self.base_model.embedding_strategy == 'linear_decay':
            from .model import linear_decay_embedding
            embedded = linear_decay_embedding(q_one_hot, r_data, self.base_model.n_questions, self.base_model.n_cats)
        elif self.base_model.embedding_strategy == 'ordered':
            from .model import ordered_embedding
            embedded = ordered_embedding(q_one_hot, r_data, self.base_model.n_questions, self.base_model.n_cats)
        elif self.base_model.embedding_strategy == 'unordered':
            from .model import unordered_embedding
            embedded = unordered_embedding(q_one_hot, r_data, self.base_model.n_questions, self.base_model.n_cats)
        elif self.base_model.embedding_strategy == 'adjacent_weighted':
            from .model import adjacent_weighted_embedding
            embedded = adjacent_weighted_embedding(q_one_hot, r_data, self.base_model.n_questions, self.base_model.n_cats)
        else:
            raise ValueError(f"Unknown embedding strategy: {self.base_model.embedding_strategy}")
        
        return embedded
    
    def forward(self, q_data, r_data, return_uncertainty: bool = True):
        """
        Enhanced forward pass with Bayesian uncertainty quantification.
        
        Args:
            q_data: Question indices (batch_size, seq_len)
            r_data: Response data (batch_size, seq_len)
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            theta: Student ability parameters
            alpha: Item discrimination parameters
            beta: Item threshold parameters
            probs: Response probabilities
            uncertainty_info: Dict with uncertainty estimates (if requested)
        """
        batch_size, seq_len = q_data.shape
        device = q_data.device
        
        # Get embeddings
        embedded = self._get_embeddings(q_data, r_data)
        
        # Initialize memory and knowledge state
        self.base_model.memory.init_value_memory(batch_size, self.base_model.init_value_memory)
        
        # Initialize Bayesian knowledge state
        current_state_mu = self.initial_state_mu.unsqueeze(0).expand(batch_size, -1)
        current_state_sigma = self.initial_state_sigma.unsqueeze(0).expand(batch_size, -1)
        
        # Collect outputs
        theta_list = []
        alpha_list = []
        beta_list = []
        uncertainty_list = []
        confidence_list = []
        kl_divergences = []
        
        # Track previous ability for memory conditioning
        prev_ability = torch.zeros(batch_size, 1, device=device)
        
        for t in range(seq_len):
            # Current timestep embeddings
            q_embed_t = self.base_model.q_embed(q_data[:, t])
            r_embed_t = embedded[:, t]
            value_embed_t = self.base_model.gpcm_value_embed(r_embed_t)
            
            # Memory operations - use fixed attention signature
            correlation_weight = self.base_model.memory.attention(q_embed_t)
            read_content = self.base_model.memory.read(correlation_weight)
            
            # Base summary computation
            base_summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            base_summary = self.base_model.summary_network(base_summary_input)
            
            # Bayesian knowledge state update
            if t > 0:
                difficulty = torch.randn(batch_size, 1, device=device) * 0.5  # Placeholder difficulty
                response_norm = (r_data[:, t-1].float() / (self.base_model.n_cats - 1)).unsqueeze(-1)
                
                current_state_mu, current_state_sigma, concept_knowledge = self.knowledge_state(
                    current_state_mu, difficulty, response_norm
                )
            else:
                concept_knowledge = torch.sigmoid(
                    self.knowledge_state.concept_mapping(current_state_mu)
                )
            
            # Enhanced features with Bayesian components
            avg_concept_knowledge = concept_knowledge.mean(dim=-1, keepdim=True)
            enhanced_features = torch.cat([
                base_summary, current_state_mu, concept_knowledge
            ], dim=-1)
            
            # Variational prediction with uncertainty
            bayesian_summary, kl_div = self.variational_predictor(enhanced_features)
            kl_divergences.append(kl_div)
            
            # Weighted combination
            final_summary = (self.bayesian_weight * bayesian_summary + 
                           (1 - self.bayesian_weight) * base_summary)
            
            # IRT parameter prediction
            theta = self.student_ability_network(final_summary).squeeze(-1) * self.base_model.ability_scale
            beta = self.question_threshold_network(q_embed_t)
            
            discrim_input = torch.cat([final_summary, q_embed_t], dim=-1)
            alpha = self.discrimination_network(discrim_input).squeeze(-1)
            
            # Uncertainty estimation
            if return_uncertainty:
                uncertainty_input = torch.cat([
                    final_summary, current_state_mu, avg_concept_knowledge
                ], dim=-1)
                
                uncertainty_mean, uncertainty_std = self.uncertainty_estimator(
                    uncertainty_input, self.n_mc_samples
                )
                
                # Confidence score (inverse of uncertainty)
                confidence = 1.0 - uncertainty_mean
                
                uncertainty_list.append(uncertainty_mean)
                confidence_list.append(confidence)
            
            theta_list.append(theta)
            alpha_list.append(alpha)
            beta_list.append(beta)
            
            # Update memory
            if t < seq_len - 1:
                self.base_model.memory.write(correlation_weight, value_embed_t)
            
            # Update previous ability
            prev_ability = theta.unsqueeze(1)
        
        # Stack outputs
        theta = torch.stack(theta_list, dim=1)
        alpha = torch.stack(alpha_list, dim=1)
        beta = torch.stack(beta_list, dim=1)
        
        # Generate GPCM probabilities
        probs = self._gpcm_probability_softmax(theta, alpha, beta)
        
        # Prepare uncertainty information
        uncertainty_info = {}
        if return_uncertainty:
            uncertainty_info = {
                'prediction_uncertainty': torch.stack(uncertainty_list, dim=1),
                'prediction_confidence': torch.stack(confidence_list, dim=1),
                'kl_divergence': torch.stack(kl_divergences),
                'total_kl': sum(kl_divergences),
                'knowledge_state_mu': current_state_mu,
                'knowledge_state_sigma': current_state_sigma,
                'concept_knowledge': concept_knowledge
            }
        
        if return_uncertainty:
            return theta, alpha, beta, probs, uncertainty_info
        else:
            return theta, alpha, beta, probs
    
    def _gpcm_probability_softmax(self, theta, alpha, betas):
        """
        Calculate GPCM response probabilities using softmax.
        
        Args:
            theta: Student abilities, shape (batch_size, seq_len)
            alpha: Discrimination parameters, shape (batch_size, seq_len)
            betas: Difficulty thresholds, shape (batch_size, seq_len, K-1)
            
        Returns:
            probs: GPCM probabilities, shape (batch_size, seq_len, K)
        """
        batch_size, seq_len = theta.shape
        K = betas.shape[-1] + 1  # Number of categories
        
        # Compute cumulative logits
        cum_logits = torch.zeros(batch_size, seq_len, K, device=theta.device)
        cum_logits[:, :, 0] = 0  # First category baseline
        
        # For k = 1, ..., K-1: sum_{h=0}^{k-1} alpha * (theta - beta_h)
        for k in range(1, K):
            cum_logits[:, :, k] = torch.sum(
                alpha.unsqueeze(-1) * (theta.unsqueeze(-1) - betas[:, :, :k]),
                dim=-1
            )
        
        # Convert to probabilities via softmax
        probs = F.softmax(cum_logits, dim=-1)
        return probs
    
    def compute_uncertainty_loss(self, uncertainty_info: Dict) -> torch.Tensor:
        """
        Compute additional loss terms for uncertainty quantification.
        
        Args:
            uncertainty_info: Dictionary with uncertainty estimates
            
        Returns:
            uncertainty_loss: Additional loss for uncertainty training
        """
        kl_loss = uncertainty_info['total_kl'] * self.kl_weight
        
        # Regularization on uncertainty estimates
        uncertainty_reg = uncertainty_info['prediction_uncertainty'].mean() * 0.01
        
        return kl_loss + uncertainty_reg
    
    def predict_with_uncertainty(self, q_data, r_data, n_samples: int = 50):
        """
        Make predictions with comprehensive uncertainty quantification.
        
        Args:
            q_data: Question indices
            r_data: Response data
            n_samples: Number of Monte Carlo samples
            
        Returns:
            predictions: Dict with mean predictions and uncertainty bounds
        """
        self.train()  # Enable dropout/variational sampling
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                _, _, _, probs, uncertainty_info = self.forward(q_data, r_data)
                predictions.append(probs)
                uncertainties.append(uncertainty_info['prediction_uncertainty'])
        
        # Aggregate predictions
        predictions = torch.stack(predictions, dim=0)
        uncertainties = torch.stack(uncertainties, dim=0)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        mean_uncertainty = uncertainties.mean(dim=0)
        
        # Confidence intervals
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'mean_prediction': mean_pred,
            'prediction_std': std_pred,
            'confidence_lower': lower_bound,
            'confidence_upper': upper_bound,
            'epistemic_uncertainty': mean_uncertainty,
            'aleatoric_uncertainty': std_pred,
            'total_uncertainty': torch.sqrt(mean_uncertainty**2 + std_pred**2)
        }


def create_bayesian_gpcm(base_model, n_concepts=20, state_dim=32, 
                        n_mc_samples=10, kl_weight=0.01):
    """
    Factory function to create BayesianGPCM model.
    
    Args:
        base_model: Base DeepGpcmModel instance
        n_concepts: Number of knowledge concepts
        state_dim: Dimension of knowledge state
        n_mc_samples: Number of Monte Carlo samples
        kl_weight: Weight for KL divergence regularization
        
    Returns:
        BayesianGPCM model instance
    """
    return BayesianGPCM(
        base_gpcm_model=base_model,
        n_concepts=n_concepts,
        state_dim=state_dim,
        n_mc_samples=n_mc_samples,
        kl_weight=kl_weight
    )