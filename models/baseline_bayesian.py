#!/usr/bin/env python3
"""
Variational Bayesian GPCM Model

This module implements a Variational Bayesian approach to the Generalized Partial Credit Model (GPCM)
that incorporates proper prior distributions for IRT parameters and uses variational inference
instead of simple MAP or regularization approaches.

Key Features:
- Variational distributions for θ (student ability), α (discrimination), and β (thresholds)
- ELBO optimization with proper KL divergences
- Reparameterization trick for gradient propagation
- Prior distributions matching synthetic data generation:
  - θ ~ N(0, 1)
  - α ~ LogNormal(0, 0.3)
  - β ~ Ordered Normal with base difficulty N(0, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional


class VariationalDistribution(nn.Module):
    """Base class for variational distributions with reparameterization."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def rsample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample using reparameterization trick."""
        raise NotImplementedError
    
    def kl_divergence(self, prior) -> torch.Tensor:
        """Compute KL divergence with prior."""
        raise NotImplementedError


class NormalVariational(VariationalDistribution):
    """Normal variational distribution with learnable mean and log variance."""
    
    def __init__(self, dim: int, prior_mean: float = 0.0, prior_std: float = 1.0):
        super().__init__(dim)
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_var = nn.Parameter(torch.zeros(dim))
        self.prior = Normal(prior_mean, prior_std)
    
    def rsample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample using reparameterization trick."""
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn(n_samples, self.dim).to(self.mean.device)
        return self.mean + eps * std
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence with prior N(prior_mean, prior_std)."""
        std = torch.exp(0.5 * self.log_var)
        q = Normal(self.mean, std)
        # Create prior distribution with same shape
        prior = Normal(
            torch.full_like(self.mean, self.prior.mean),
            torch.full_like(std, self.prior.stddev)
        )
        return kl_divergence(q, prior).sum()


class LogNormalVariational(VariationalDistribution):
    """Log-normal variational distribution for positive parameters."""
    
    def __init__(self, dim: int, prior_mean: float = 0.0, prior_std: float = 0.3):
        super().__init__(dim)
        # Parameters are in log-space
        self.log_mean = nn.Parameter(torch.zeros(dim))
        self.log_var = nn.Parameter(torch.zeros(dim) - 1.0)  # Start with smaller variance
        self.prior_mean = prior_mean
        self.prior_std = prior_std
    
    def rsample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample using reparameterization trick."""
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn(n_samples, self.dim).to(self.log_mean.device)
        log_samples = self.log_mean + eps * std
        return torch.exp(log_samples)  # Transform to log-normal
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence with prior LogNormal(prior_mean, prior_std)."""
        # For log-normal, we compute KL in log-space (normal distributions)
        std = torch.exp(0.5 * self.log_var)
        q = Normal(self.log_mean, std)
        prior = Normal(
            torch.full_like(self.log_mean, self.prior_mean),
            torch.full_like(std, self.prior_std)
        )
        return kl_divergence(q, prior).sum()


class OrderedNormalVariational(VariationalDistribution):
    """Ordered normal variational distribution for threshold parameters."""
    
    def __init__(self, n_questions: int, n_thresholds: int):
        super().__init__(n_questions)
        self.n_questions = n_questions
        self.n_thresholds = n_thresholds
        
        # Base difficulty per question
        self.base_mean = nn.Parameter(torch.zeros(n_questions))
        self.base_log_var = nn.Parameter(torch.zeros(n_questions))
        
        # Threshold offsets (positive to ensure ordering)
        self.threshold_offsets = nn.Parameter(torch.ones(n_questions, n_thresholds - 1) * 0.5)
        self.threshold_log_var = nn.Parameter(torch.zeros(n_questions, n_thresholds))
    
    def rsample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample ordered thresholds using reparameterization trick."""
        # Sample base difficulties
        base_std = torch.exp(0.5 * self.base_log_var)
        base_eps = torch.randn(n_samples, self.n_questions).to(self.base_mean.device)
        base = self.base_mean + base_eps * base_std
        
        # Build ordered thresholds
        thresholds = torch.zeros(n_samples, self.n_questions, self.n_thresholds).to(self.base_mean.device)
        
        # First threshold is just the base
        thresholds[:, :, 0] = base
        
        # Add cumulative offsets for remaining thresholds
        for i in range(1, self.n_thresholds):
            # Add positive offset to ensure ordering
            offset = F.softplus(self.threshold_offsets[:, i-1])
            thresholds[:, :, i] = thresholds[:, :, i-1] + offset.unsqueeze(0)
        
        # Add noise to each threshold
        thresh_std = torch.exp(0.5 * self.threshold_log_var)
        thresh_eps = torch.randn(n_samples, self.n_questions, self.n_thresholds).to(self.base_mean.device)
        
        result = thresholds + thresh_eps * thresh_std.unsqueeze(0)
        
        # Ensure ordering is maintained after adding noise
        result, _ = torch.sort(result, dim=-1)
        
        return result
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence with ordered normal prior."""
        # KL for base difficulties
        base_std = torch.exp(0.5 * self.base_log_var)
        base_q = Normal(self.base_mean, base_std)
        base_prior = Normal(torch.zeros_like(self.base_mean), torch.ones_like(base_std))
        kl = kl_divergence(base_q, base_prior).sum()
        
        # KL for threshold variances
        thresh_std = torch.exp(0.5 * self.threshold_log_var)
        thresh_q = Normal(torch.zeros_like(thresh_std), thresh_std)
        thresh_prior = Normal(torch.zeros_like(thresh_std), torch.ones_like(thresh_std) * 0.5)
        kl += kl_divergence(thresh_q, thresh_prior).sum()
        
        return kl


class VariationalBayesianGPCM(nn.Module):
    """
    Variational Bayesian GPCM model with proper prior incorporation.
    
    This model uses variational inference to learn posterior distributions over IRT parameters
    while incorporating prior knowledge about their distributions.
    """
    
    def __init__(self, n_students: int, n_questions: int, n_categories: int = 4,
                 embed_dim: int = 64, memory_size: int = 20, memory_key_dim: int = 50,
                 memory_value_dim: int = 200, kl_weight: float = 1.0):
        super().__init__()
        
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.kl_weight = kl_weight
        
        # Variational distributions for IRT parameters
        self.theta_dist = NormalVariational(n_students, prior_mean=0.0, prior_std=1.0)
        self.alpha_dist = LogNormalVariational(n_questions, prior_mean=0.0, prior_std=0.3)
        self.beta_dist = OrderedNormalVariational(n_questions, n_categories - 1)
        
        # DKVMN components (same as baseline)
        self.embed_dim = embed_dim
        self.q_embed = nn.Embedding(n_questions + 1, embed_dim)
        self.qa_embed = nn.Linear(2, embed_dim)
        
        # Memory networks
        self.memory_size = memory_size
        self.memory_key_dim = memory_key_dim
        self.memory_value_dim = memory_value_dim
        
        self.key_memory = nn.Parameter(torch.randn(memory_size, memory_key_dim))
        self.value_memory = nn.Parameter(torch.randn(memory_size, memory_value_dim))
        
        # Transformation layers
        self.q_to_key = nn.Linear(embed_dim, memory_key_dim)
        self.qa_to_value = nn.Linear(embed_dim, memory_value_dim)
        self.erase_layer = nn.Linear(memory_value_dim, memory_value_dim)
        self.add_layer = nn.Linear(memory_value_dim, memory_value_dim)
        
        # Output layers
        self.output_layer = nn.Linear(memory_value_dim + embed_dim, 256)
        self.final_layer = nn.Linear(256, 1)  # Output single ability value
        
        # Initialize parameters
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor,
                student_ids: Optional[torch.Tensor] = None,
                return_params: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with variational sampling.
        
        Args:
            questions: Question IDs [batch_size, seq_len]
            responses: Student responses [batch_size, seq_len]
            student_ids: Student IDs for ability lookup [batch_size]
            return_params: Whether to return sampled IRT parameters
            
        Returns:
            probabilities: GPCM probabilities [batch_size, seq_len, n_categories]
            aux_dict: Dictionary containing KL divergence and optionally sampled parameters
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Sample IRT parameters
        theta = self.theta_dist.rsample(1).squeeze(0)  # [n_students]
        alpha = self.alpha_dist.rsample(1).squeeze(0)  # [n_questions]
        beta = self.beta_dist.rsample(1).squeeze(0)    # [n_questions, n_categories-1]
        
        # Get student abilities (handle missing student_ids)
        if student_ids is not None:
            student_abilities = theta[student_ids]  # [batch_size]
        else:
            # Use learned abilities from memory network
            student_abilities = self._compute_abilities_from_memory(questions, responses)
        
        # Question embeddings
        q_embed = self.q_embed(questions)  # [batch_size, seq_len, embed_dim]
        
        # Response embeddings (linear decay strategy)
        qa_embed = self._compute_response_embedding(questions, responses)
        
        # Memory network updates
        memory_values = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        all_read_values = []
        for t in range(seq_len):
            # Read from memory
            q_key = self.q_to_key(q_embed[:, t])  # [batch_size, memory_key_dim]
            correlation = F.softmax(torch.matmul(q_key, self.key_memory.t()), dim=1)
            read_value = torch.matmul(correlation.unsqueeze(1), memory_values).squeeze(1)
            all_read_values.append(read_value)
            
            # Write to memory
            qa_value = self.qa_to_value(qa_embed[:, t])
            erase = torch.sigmoid(self.erase_layer(qa_value))
            add = torch.tanh(self.add_layer(qa_value))
            memory_values = memory_values * (1 - correlation.unsqueeze(2) * erase.unsqueeze(1))
            memory_values = memory_values + correlation.unsqueeze(2) * add.unsqueeze(1)
        
        # Stack read values
        read_values = torch.stack(all_read_values, dim=1)  # [batch_size, seq_len, memory_value_dim]
        
        # Combine with question embeddings
        combined = torch.cat([read_values, q_embed], dim=2)
        
        # Predict abilities per time step
        hidden = F.relu(self.output_layer(combined))
        predicted_abilities = self.final_layer(hidden).squeeze(-1)  # [batch_size, seq_len]
        
        # Compute GPCM probabilities
        probabilities = []
        for t in range(seq_len):
            q_ids = questions[:, t]
            
            # Get parameters for current questions
            q_alpha = alpha[q_ids]  # [batch_size]
            q_beta = beta[q_ids]    # [batch_size, n_categories-1]
            
            # Use predicted abilities
            q_theta = predicted_abilities[:, t]  # [batch_size]
            
            # Compute GPCM probabilities
            probs = self._gpcm_probability(q_theta, q_alpha, q_beta)
            probabilities.append(probs)
        
        probabilities = torch.stack(probabilities, dim=1)  # [batch_size, seq_len, n_categories]
        
        # Compute KL divergence
        kl_div = (self.theta_dist.kl_divergence() + 
                  self.alpha_dist.kl_divergence() + 
                  self.beta_dist.kl_divergence())
        
        aux_dict = {'kl_divergence': kl_div * self.kl_weight}
        
        if return_params:
            aux_dict['theta'] = theta
            aux_dict['alpha'] = alpha
            aux_dict['beta'] = beta
            aux_dict['predicted_abilities'] = predicted_abilities
        
        return probabilities, aux_dict
    
    def _compute_abilities_from_memory(self, questions: torch.Tensor, 
                                     responses: torch.Tensor) -> torch.Tensor:
        """Compute initial student abilities from memory network."""
        batch_size = questions.shape[0]
        
        # Use first question-response pair to initialize
        q_embed = self.q_embed(questions[:, 0])
        qa_embed = self._compute_response_embedding(questions[:, :1], responses[:, :1])[:, 0]
        
        # Read from memory
        q_key = self.q_to_key(q_embed)
        correlation = F.softmax(torch.matmul(q_key, self.key_memory.t()), dim=1)
        memory_values = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1)
        read_value = torch.matmul(correlation.unsqueeze(1), memory_values).squeeze(1)
        
        # Combine and predict
        combined = torch.cat([read_value, q_embed], dim=1)
        hidden = F.relu(self.output_layer(combined))
        abilities = self.final_layer(hidden).squeeze(-1)
        
        return abilities
    
    def _compute_response_embedding(self, questions: torch.Tensor, 
                                  responses: torch.Tensor) -> torch.Tensor:
        """Compute response embeddings using linear decay strategy."""
        batch_size, seq_len = questions.shape
        embeddings = []
        
        for t in range(seq_len):
            # Linear decay weights around actual response
            weights = torch.zeros(batch_size, self.n_categories).to(questions.device)
            for b in range(batch_size):
                r = int(responses[b, t].item())
                for k in range(self.n_categories):
                    distance = abs(k - r)
                    weights[b, k] = max(0, 1 - distance / (self.n_categories - 1))
            
            # Create 2D input [correctness, score]
            correctness = (responses[:, t] == self.n_categories - 1).float()
            score = responses[:, t] / (self.n_categories - 1)
            qa_input = torch.stack([correctness, score], dim=1)
            
            # Apply embedding
            embed = self.qa_embed(qa_input)
            embeddings.append(embed)
        
        return torch.stack(embeddings, dim=1)
    
    def _gpcm_probability(self, theta: torch.Tensor, alpha: torch.Tensor, 
                         beta: torch.Tensor) -> torch.Tensor:
        """
        Compute GPCM probabilities.
        
        Args:
            theta: Student abilities [batch_size]
            alpha: Discrimination parameters [batch_size]
            beta: Threshold parameters [batch_size, n_categories-1]
            
        Returns:
            probabilities: [batch_size, n_categories]
        """
        batch_size = theta.shape[0]
        K = self.n_categories
        
        # Compute cumulative logits
        cum_logits = torch.zeros(batch_size, K).to(theta.device)
        
        for k in range(1, K):
            # Sum of alpha * (theta - beta_h) for h < k
            cum_sum = 0
            for h in range(k):
                cum_sum += alpha * (theta - beta[:, h])
            cum_logits[:, k] = cum_sum
        
        # Compute probabilities
        exp_logits = torch.exp(cum_logits - cum_logits.max(dim=1, keepdim=True)[0])
        probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        
        return probabilities
    
    def elbo_loss(self, probabilities: torch.Tensor, targets: torch.Tensor,
                  kl_divergence: torch.Tensor) -> torch.Tensor:
        """
        Compute ELBO loss = -log likelihood + KL divergence.
        
        Args:
            probabilities: Predicted probabilities [batch_size, seq_len, n_categories]
            targets: True responses [batch_size, seq_len]
            kl_divergence: KL divergence term
            
        Returns:
            loss: Scalar loss value
        """
        # Negative log likelihood
        targets_long = targets.long()
        log_probs = torch.log(probabilities + 1e-8)
        nll = F.nll_loss(
            log_probs.reshape(-1, self.n_categories),
            targets_long.reshape(-1),
            reduction='mean'
        )
        
        # ELBO = -NLL + KL
        batch_size = probabilities.shape[0]
        elbo = nll + kl_divergence / batch_size
        
        return elbo
    
    def sample_prior(self) -> Dict[str, torch.Tensor]:
        """Sample from prior distributions for comparison."""
        with torch.no_grad():
            # Sample from priors
            theta_prior = torch.randn(self.n_students) * 1.0  # N(0, 1)
            alpha_prior = torch.exp(torch.randn(self.n_questions) * 0.3)  # LogNormal(0, 0.3)
            
            # Sample ordered betas
            beta_prior = []
            for q in range(self.n_questions):
                base_diff = torch.randn(1) * 1.0
                thresh = torch.sort(torch.randn(self.n_categories - 1) * 0.5 + base_diff)[0]
                beta_prior.append(thresh)
            beta_prior = torch.stack(beta_prior)
            
            return {
                'theta': theta_prior,
                'alpha': alpha_prior,
                'beta': beta_prior
            }
    
    def get_posterior_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get statistics of the learned posterior distributions."""
        with torch.no_grad():
            # Sample multiple times to estimate statistics
            n_samples = 100
            theta_samples = []
            alpha_samples = []
            beta_samples = []
            
            for _ in range(n_samples):
                theta_samples.append(self.theta_dist.rsample(1).squeeze(0))
                alpha_samples.append(self.alpha_dist.rsample(1).squeeze(0))
                beta_samples.append(self.beta_dist.rsample(1).squeeze(0))
            
            theta_samples = torch.stack(theta_samples)
            alpha_samples = torch.stack(alpha_samples)
            beta_samples = torch.stack(beta_samples)
            
            return {
                'theta': {
                    'mean': theta_samples.mean(dim=0),
                    'std': theta_samples.std(dim=0),
                    'samples': theta_samples
                },
                'alpha': {
                    'mean': alpha_samples.mean(dim=0),
                    'std': alpha_samples.std(dim=0),
                    'samples': alpha_samples
                },
                'beta': {
                    'mean': beta_samples.mean(dim=0),
                    'std': beta_samples.std(dim=0),
                    'samples': beta_samples
                }
            }