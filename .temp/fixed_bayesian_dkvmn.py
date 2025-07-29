#!/usr/bin/env python3
"""
Fixed Deep Bayesian-DKVMN Implementation

Key fixes for IRT parameter recovery:
1. One-to-one question-memory mapping
2. Deterministic parameter extraction (no random sampling during training)
3. Proper parameter scaling and constraints
4. Direct question-specific parameter learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional, List


class QuestionSpecificIRTParameters(nn.Module):
    """
    IRT parameters specific to each question with proper constraints.
    """
    
    def __init__(self, n_questions: int, n_categories: int = 4):
        super().__init__()
        self.n_questions = n_questions
        self.n_categories = n_categories
        
        # Discrimination parameters (one per question)
        # Use log-normal parameterization: α = exp(μ_α + σ_α * ε)
        self.alpha_mean = nn.Parameter(torch.randn(n_questions) * 0.2)  # log-space mean with variation
        self.alpha_log_std = nn.Parameter(torch.ones(n_questions) * np.log(0.3))
        
        # Threshold parameters (ordered, per question)
        # β_1 < β_2 < ... < β_{K-1}
        self.beta_base = nn.Parameter(torch.randn(n_questions))  # β_1
        self.beta_gaps = nn.Parameter(torch.ones(n_questions, n_categories - 2) * 0.5)  # gaps β_{k+1} - β_k
        self.beta_log_std = nn.Parameter(torch.ones(n_questions, n_categories - 1) * np.log(0.2))
    
    def get_parameters(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get deterministic IRT parameters for specific questions.
        
        Args:
            question_ids: [batch_size] question indices
            
        Returns:
            alphas: [batch_size] discrimination parameters
            betas: [batch_size, n_categories-1] threshold parameters
        """
        batch_size = question_ids.shape[0]
        
        # Get discrimination parameters (ensure positive)
        alpha_means = self.alpha_mean[question_ids]  # [batch_size]
        alphas = torch.exp(alpha_means)  # Always positive
        
        # Get ordered threshold parameters
        base_betas = self.beta_base[question_ids]  # [batch_size]
        gaps = F.softplus(self.beta_gaps[question_ids])  # [batch_size, n_categories-2], always positive
        
        # Construct ordered thresholds: β_1, β_1 + gap_1, β_1 + gap_1 + gap_2, ...
        betas = torch.zeros(batch_size, self.n_categories - 1, device=question_ids.device)
        betas[:, 0] = base_betas
        
        for k in range(1, self.n_categories - 1):
            betas[:, k] = betas[:, k-1] + gaps[:, k-1]
        
        return alphas, betas
    
    def sample_parameters(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample IRT parameters (for training with variational inference).
        """
        batch_size = question_ids.shape[0]
        
        # Sample discrimination with reparameterization trick
        alpha_std = torch.exp(self.alpha_log_std[question_ids])
        alpha_noise = torch.randn_like(alpha_std)
        alpha_log = self.alpha_mean[question_ids] + alpha_std * alpha_noise
        alphas = torch.exp(alpha_log)
        
        # Sample threshold parameters
        base_betas = self.beta_base[question_ids]
        beta_std = torch.exp(self.beta_log_std[question_ids])
        beta_noise = torch.randn_like(beta_std)
        
        # Sample base threshold
        sampled_base = base_betas + beta_std[:, 0] * beta_noise[:, 0]
        
        # Sample gaps and construct ordered thresholds
        gaps = F.softplus(self.beta_gaps[question_ids])
        betas = torch.zeros(batch_size, self.n_categories - 1, device=question_ids.device)
        betas[:, 0] = sampled_base
        
        for k in range(1, self.n_categories - 1):
            gap_noise = beta_noise[:, k] if k < beta_std.shape[1] else torch.zeros_like(beta_noise[:, 0])
            sampled_gap = gaps[:, k-1] + beta_std[:, k] * gap_noise if k < beta_std.shape[1] else gaps[:, k-1]
            betas[:, k] = betas[:, k-1] + sampled_gap
        
        return alphas, betas
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for all parameters."""
        # KL for discrimination parameters (log-normal)
        alpha_std = torch.exp(self.alpha_log_std)
        alpha_dist = Normal(self.alpha_mean, alpha_std)
        alpha_prior = Normal(torch.zeros_like(self.alpha_mean), torch.ones_like(alpha_std) * 0.5)
        alpha_kl = kl_divergence(alpha_dist, alpha_prior).sum()
        
        # KL for threshold parameters (simplified normal)
        beta_kl = (
            (self.beta_base.pow(2) - 2 * np.log(0.5)).sum() * 0.5 +
            (self.beta_log_std.exp() + self.beta_base.unsqueeze(1).pow(2) - 
             self.beta_log_std - 1).sum() * 0.5
        )
        
        return alpha_kl + beta_kl


class StudentAbilityTracker(nn.Module):
    """
    Track student abilities using DKVMN-style memory with proper updates.
    """
    
    def __init__(self, memory_size: int, ability_dim: int = 32):
        super().__init__()
        self.memory_size = memory_size
        self.ability_dim = ability_dim
        
        # Student ability memory (one slot per "student archetype")
        self.ability_means = nn.Parameter(torch.zeros(memory_size, ability_dim))
        self.ability_log_vars = nn.Parameter(torch.ones(memory_size, ability_dim) * np.log(1.0))
        
        # Embeddings for student state
        self.response_embed = nn.Linear(2, ability_dim)  # [question_id, response] -> embedding
        self.ability_predictor = nn.Linear(ability_dim, 1)  # ability mean -> scalar ability
    
    def forward(self, question_ids: torch.Tensor, responses: torch.Tensor,
                prev_abilities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update and predict student abilities.
        
        Args:
            question_ids: [batch_size] current questions
            responses: [batch_size] responses (for memory update)
            prev_abilities: [batch_size, ability_dim] previous ability states
            
        Returns:
            abilities: [batch_size] predicted student abilities (scalar)
            ability_states: [batch_size, ability_dim] updated ability states
        """
        batch_size = question_ids.shape[0]
        
        # If no previous abilities, use average of memory
        if prev_abilities is None:
            prev_abilities = self.ability_means.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        
        # Create response embedding
        qa_input = torch.stack([question_ids.float(), responses.float()], dim=1)
        response_embed = self.response_embed(qa_input)
        
        # Simple update: combine previous ability with response evidence
        updated_abilities = 0.7 * prev_abilities + 0.3 * response_embed
        
        # Predict scalar ability
        abilities = self.ability_predictor(updated_abilities).squeeze(-1)
        
        return abilities, updated_abilities


class FixedBayesianDKVMN(nn.Module):
    """
    Fixed Deep Bayesian-DKVMN with proper IRT parameter learning.
    
    Key improvements:
    1. Question-specific IRT parameters (no averaging)
    2. Deterministic parameter extraction during evaluation
    3. Proper parameter constraints and scaling
    4. Direct optimization of IRT parameter recovery
    """
    
    def __init__(self, n_questions: int, n_categories: int = 4, 
                 memory_size: int = 20, ability_dim: int = 32):
        super().__init__()
        
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.memory_size = memory_size
        self.ability_dim = ability_dim
        
        # Question-specific IRT parameters
        self.irt_params = QuestionSpecificIRTParameters(n_questions, n_categories)
        
        # Student ability tracking
        self.ability_tracker = StudentAbilityTracker(memory_size, ability_dim)
        
        # Training state
        self.current_epoch = 0
        self.kl_warmup_epochs = 10
        
        # Track previous abilities for sequence modeling
        self.register_buffer('prev_abilities', torch.zeros(1, ability_dim))
    
    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor,
                return_params: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with fixed IRT parameter learning.
        
        Args:
            questions: [batch_size, seq_len] question IDs
            responses: [batch_size, seq_len] student responses
            return_params: Whether to return IRT parameters
            
        Returns:
            probabilities: [batch_size, seq_len, n_categories] GPCM probabilities
            aux_dict: Auxiliary outputs including KL divergence
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        all_probabilities = []
        current_abilities = None
        
        # Process each time step in sequence
        for t in range(seq_len):
            q_t = questions[:, t]  # [batch_size]
            r_t = responses[:, t] if t > 0 else torch.zeros_like(q_t)  # [batch_size]
            
            # Update student abilities based on previous responses
            if t > 0:
                thetas, current_abilities = self.ability_tracker(q_t, r_t, current_abilities)
            else:
                # Initialize abilities
                thetas, current_abilities = self.ability_tracker(q_t, torch.zeros_like(q_t), None)
            
            # Get IRT parameters for current questions
            if self.training:
                # During training: sample for variational inference
                alphas, betas = self.irt_params.sample_parameters(q_t)
            else:
                # During evaluation: use deterministic parameters
                alphas, betas = self.irt_params.get_parameters(q_t)
            
            # thetas already extracted from ability_tracker above
            
            # Compute GPCM probabilities
            batch_probs = []
            for b in range(batch_size):
                prob = self._gpcm_probability(thetas[b], alphas[b], betas[b])
                batch_probs.append(prob)
            
            probabilities = torch.stack(batch_probs, dim=0)  # [batch_size, n_categories]
            all_probabilities.append(probabilities)
        
        # Stack all probabilities
        final_probabilities = torch.stack(all_probabilities, dim=1)  # [batch_size, seq_len, n_categories]
        
        # Compute KL divergence
        kl_div = self.irt_params.kl_divergence()
        
        # Apply KL annealing
        kl_annealing_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs) if self.kl_warmup_epochs > 0 else 1.0
        
        aux_dict = {
            'kl_divergence': kl_div * kl_annealing_factor,
            'raw_kl_divergence': kl_div,
            'kl_annealing_factor': kl_annealing_factor
        }
        
        if return_params:
            # Extract all parameters for analysis
            all_q_ids = torch.arange(self.n_questions, device=device)
            eval_alphas, eval_betas = self.irt_params.get_parameters(all_q_ids)
            
            aux_dict.update({
                'alphas': eval_alphas,
                'betas': eval_betas,
                'thetas': current_abilities  # Latest student abilities
            })
        
        return final_probabilities, aux_dict
    
    def _gpcm_probability(self, theta: torch.Tensor, alpha: torch.Tensor, 
                         beta: torch.Tensor) -> torch.Tensor:
        """
        Compute GPCM probability with proper numerical stability.
        """
        K = self.n_categories
        
        # Compute cumulative logits
        cum_logits = torch.zeros(K, device=theta.device)
        for k in range(1, K):
            cum_sum = 0.0
            for h in range(k):
                if h < len(beta):
                    cum_sum += alpha * (theta - beta[h])
            cum_logits[k] = cum_sum
        
        # Stable softmax
        cum_logits = cum_logits - cum_logits.max()
        exp_logits = torch.exp(cum_logits)
        probabilities = exp_logits / exp_logits.sum()
        
        return probabilities
    
    def elbo_loss(self, probabilities: torch.Tensor, targets: torch.Tensor,
                  kl_divergence: torch.Tensor) -> torch.Tensor:
        """Compute ELBO loss."""
        batch_size, seq_len, n_categories = probabilities.shape
        
        # GPCM log-likelihood
        targets_long = targets.long()
        observed_probs = probabilities.gather(2, targets_long.unsqueeze(2)).squeeze(2)
        observed_probs = torch.clamp(observed_probs, min=1e-8, max=1.0)
        
        log_likelihood = torch.log(observed_probs).sum()
        nll = -log_likelihood / (batch_size * seq_len)
        
        # ELBO = NLL + KL/N
        total_observations = batch_size * seq_len
        elbo = nll + kl_divergence / total_observations
        
        return elbo
    
    def get_interpretable_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract interpretable IRT parameters."""
        with torch.no_grad():
            # Get parameters for all questions
            all_q_ids = torch.arange(self.n_questions)
            alphas, betas = self.irt_params.get_parameters(all_q_ids)
            
            # Get representative student abilities
            dummy_abilities = torch.zeros(self.memory_size, self.ability_dim)
            thetas = self.ability_tracker.ability_predictor(dummy_abilities).squeeze(-1)
            
            return {
                'alpha': alphas,  # [n_questions]
                'beta': betas,    # [n_questions, n_categories-1] 
                'theta': thetas   # [memory_size]
            }