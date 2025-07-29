#!/usr/bin/env python3
"""
No-Activation Deep Bayesian-DKVMN Implementation

This version removes activation functions from IRT parameter computation while
maintaining ALL the deep integration functionality:
1. Full DKVMN memory operations (read/write with attention)
2. Complete IRT computation (GPCM probability)
3. Bayesian inference with ELBO optimization
4. NO activations for alpha, beta, theta parameters

Key change: Remove torch.exp(), F.softplus(), and activation layers while
keeping the complete deep Bayesian-DKVMN architecture intact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional, List


class NoActivationBayesianMemoryKey(nn.Module):
    """
    Bayesian memory key with NO activation functions for IRT parameters.
    
    Maintains all Bayesian inference functionality but removes:
    - torch.exp() for alpha
    - F.softplus() for beta gaps
    - All parameter activations
    """
    
    def __init__(self, key_dim: int, n_categories: int = 4):
        super().__init__()
        self.key_dim = key_dim
        self.n_categories = n_categories
        
        # Question embedding (unchanged)
        self.embedding = nn.Parameter(torch.randn(key_dim))
        
        # Discrimination parameter (NO activation - direct learning)
        self.alpha_mean = nn.Parameter(torch.randn(1) * 0.5 + 1.0)  # Initialize around 1.0
        self.alpha_log_var = nn.Parameter(torch.ones(1) * np.log(0.3))
        
        # Threshold parameters (NO activation - direct learning with ordering constraint)
        self.beta_base = nn.Parameter(torch.randn(1) * 0.5)
        # Use raw gaps (no softplus) but ensure ordering in get_parameters
        self.beta_gaps = nn.Parameter(torch.ones(n_categories - 2) * 0.5)
        self.beta_log_var = nn.Parameter(torch.ones(n_categories - 1) * np.log(0.1))
    
    def sample_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample α and β parameters WITHOUT activations."""
        # Sample discrimination (NO torch.exp - direct sampling)
        alpha_std = torch.exp(0.5 * self.alpha_log_var)
        alpha = self.alpha_mean + torch.randn_like(self.alpha_mean) * alpha_std
        
        # Sample ordered thresholds (NO F.softplus - direct with ordering)
        beta_base = self.beta_base + torch.randn_like(self.beta_base) * 0.1
        
        # Use absolute value to ensure positive gaps (instead of softplus)
        beta_gaps_positive = torch.abs(self.beta_gaps)
        beta = torch.zeros(self.n_categories - 1)
        beta[0] = beta_base
        for i in range(1, self.n_categories - 1):
            beta[i] = beta[i-1] + beta_gaps_positive[i-1]
        
        return alpha.squeeze(), beta
    
    def get_deterministic_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get deterministic parameters WITHOUT activations (for evaluation)."""
        # Direct alpha (no exp)
        alpha = self.alpha_mean
        
        # Direct beta with ordering (no softplus)
        beta_gaps_positive = torch.abs(self.beta_gaps)
        beta = torch.zeros(self.n_categories - 1)
        beta[0] = self.beta_base
        for i in range(1, self.n_categories - 1):
            beta[i] = beta[i-1] + beta_gaps_positive[i-1]
        
        return alpha.squeeze(), beta
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for this key's parameters."""
        # KL for discrimination (normal distribution now, not log-normal)
        alpha_std = torch.exp(0.5 * self.alpha_log_var)
        alpha_dist = Normal(self.alpha_mean, alpha_std)
        alpha_prior = Normal(torch.ones_like(self.alpha_mean), torch.ones_like(alpha_std) * 0.5)
        alpha_kl = kl_divergence(alpha_dist, alpha_prior).sum()
        
        # KL for thresholds (simplified)
        beta_kl = (self.beta_log_var.exp() + self.beta_base.pow(2) - self.beta_log_var - 1).sum() * 0.5
        
        return alpha_kl + beta_kl


class NoActivationBayesianMemoryValue(nn.Module):
    """
    Bayesian memory value with NO activation for student abilities.
    
    Maintains full Bayesian update functionality but removes activation layers.
    """
    
    def __init__(self, value_dim: int):
        super().__init__()
        self.value_dim = value_dim
        
        # Ability belief parameters (NO activation layers)
        self.theta_mean = nn.Parameter(torch.zeros(value_dim))  # Direct theta values
        self.theta_log_var = nn.Parameter(torch.ones(value_dim) * np.log(1.0))
        
        # Update tracking
        self.update_count = nn.Parameter(torch.zeros(value_dim), requires_grad=False)
    
    def get_belief_distribution(self) -> Normal:
        """Get current belief distribution over student ability."""
        theta_std = torch.exp(0.5 * self.theta_log_var)
        return Normal(self.theta_mean, theta_std)
    
    def get_deterministic_ability(self) -> torch.Tensor:
        """Get deterministic ability (mean of distribution) WITHOUT activation."""
        return self.theta_mean  # Direct return, no activation
    
    def bayesian_update(self, evidence_mean: torch.Tensor, evidence_precision: torch.Tensor,
                       update_weight: torch.Tensor):
        """Perform Bayesian update of ability beliefs."""
        with torch.no_grad():
            # Current beliefs
            prior_precision = torch.exp(-self.theta_log_var)
            prior_mean = self.theta_mean
            
            # Bayesian update formulas
            posterior_precision = prior_precision + evidence_precision * update_weight
            posterior_mean = (prior_precision * prior_mean + 
                            evidence_precision * evidence_mean * update_weight) / posterior_precision
            
            # Update parameters
            self.theta_mean.data = posterior_mean
            self.theta_log_var.data = -torch.log(posterior_precision)
            self.update_count.data += update_weight


class NoActivationDeepBayesianDKVMN(nn.Module):
    """
    Deep Bayesian-DKVMN with NO activations for IRT parameters.
    
    Maintains ALL functionality:
    1. Full DKVMN memory operations (read/write with attention)
    2. Complete IRT computation (GPCM probability)
    3. Bayesian inference with ELBO optimization
    4. Deep integration of memory + IRT
    
    Key change: Removes ALL activation functions from parameter computation.
    """
    
    def __init__(self, n_questions: int, n_categories: int = 4, memory_size: int = 20,
                 key_dim: int = 50, value_dim: int = 100, embed_dim: int = 64):
        super().__init__()
        
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.embed_dim = embed_dim
        
        # Question embeddings
        self.q_embed = nn.Embedding(n_questions + 1, embed_dim)
        self.qa_embed = nn.Linear(2, embed_dim)  # [question_id, response] -> embedding
        
        # Bayesian Memory Network (NO activations)
        self.memory_keys = nn.ModuleList([
            NoActivationBayesianMemoryKey(key_dim, n_categories) for _ in range(memory_size)
        ])
        self.memory_values = nn.ModuleList([
            NoActivationBayesianMemoryValue(value_dim) for _ in range(memory_size)
        ])
        
        # Attention and transformation layers
        self.q_to_key = nn.Linear(embed_dim, key_dim)
        self.qa_to_evidence = nn.Linear(embed_dim, value_dim)
        
        # Student ability prediction (NO activation layer)
        # Instead of Linear layer, use direct weighted combination
        self.ability_weights = nn.Parameter(torch.ones(value_dim) / value_dim)
        
        # Current epoch for KL annealing
        self.current_epoch = 0
        self.kl_warmup_epochs = 20
        
    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch
    
    def compute_attention(self, query_embed: torch.Tensor, 
                         question_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights using BOTH embedding similarity AND IRT parameter similarity.
        Maintains full DKVMN functionality.
        """
        batch_size = query_embed.shape[0]
        attention_logits = torch.zeros(batch_size, self.memory_size, device=query_embed.device)
        
        # Sample IRT parameters from all memory keys for similarity computation
        memory_alphas = []
        memory_betas = []
        for memory_key in self.memory_keys:
            if self.training:
                alpha, beta = memory_key.sample_parameters()
            else:
                alpha, beta = memory_key.get_deterministic_parameters()
            memory_alphas.append(alpha)
            memory_betas.append(beta)
        
        for i, memory_key in enumerate(self.memory_keys):
            # === DKVMN EMBEDDING SIMILARITY ===
            embed_sim = torch.matmul(query_embed, memory_key.embedding)  # [batch_size]
            
            # === IRT PARAMETER SIMILARITY (NO activations) ===
            alpha_key = memory_alphas[i]
            beta_key = memory_betas[i]
            
            # Direct discrimination similarity (no exp activation)
            alpha_sim = torch.ones_like(embed_sim) * alpha_key.item()
            
            # Direct difficulty similarity (no exp activation)
            beta_mean = beta_key.mean().item()
            difficulty_sim = torch.ones_like(embed_sim) * (-0.5 * beta_mean**2)
            
            # === COMBINED ATTENTION LOGITS ===
            combined_logits = (
                embed_sim +                    # DKVMN embedding similarity
                0.3 * alpha_sim +             # IRT discrimination similarity (no activation)
                0.2 * difficulty_sim          # IRT difficulty similarity (no activation)
            )
            
            attention_logits[:, i] = combined_logits
        
        return F.softmax(attention_logits, dim=1)
    
    def bayesian_read(self, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Bayesian read operation from memory.
        Maintains full DKVMN read functionality.
        """
        batch_size = attention_weights.shape[0]
        
        # Weighted combination of ability beliefs
        ability_means = []
        ability_vars = []
        
        for memory_value in self.memory_values:
            belief_dist = memory_value.get_belief_distribution()
            ability_means.append(belief_dist.mean)
            ability_vars.append(belief_dist.variance)
        
        ability_means = torch.stack(ability_means, dim=0)  # [memory_size, value_dim]
        ability_vars = torch.stack(ability_vars, dim=0)    # [memory_size, value_dim]
        
        # Weighted average (Bayesian combination)
        weights = attention_weights.unsqueeze(2)  # [batch_size, memory_size, 1]
        
        combined_mean = torch.sum(weights * ability_means.unsqueeze(0), dim=1)  # [batch_size, value_dim]
        combined_var = torch.sum(weights * ability_vars.unsqueeze(0), dim=1)    # [batch_size, value_dim]
        
        return combined_mean, combined_var
    
    def bayesian_write(self, attention_weights: torch.Tensor, 
                      evidence_embed: torch.Tensor, responses: torch.Tensor):
        """
        Perform full DKVMN Bayesian write operation to memory.
        Maintains complete DKVMN write functionality.
        """
        batch_size = attention_weights.shape[0]
        
        # Convert responses to ability evidence using IRT principles
        response_normalized = responses.float() / (self.n_categories - 1)  # [0, 1]
        
        # IRT-based ability evidence (NO activation - direct logit)
        ability_evidence = torch.logit(torch.clamp(response_normalized, 0.01, 0.99))  # Real scale
        ability_evidence_expanded = ability_evidence.unsqueeze(1).expand(-1, self.value_dim)
        
        # Evidence precision based on response patterns
        response_extremeness = torch.abs(response_normalized - 0.5) * 2  # [0, 1]
        evidence_precision = 0.5 + response_extremeness  # [0.5, 1.5]
        evidence_precision_expanded = evidence_precision.unsqueeze(1).expand(-1, self.value_dim)
        
        # Update each memory slot based on DKVMN attention distribution
        for i, memory_value in enumerate(self.memory_values):
            slot_attention = attention_weights[:, i]  # [batch_size]
            
            if slot_attention.sum() > 1e-6:
                attention_normalized = slot_attention / (slot_attention.sum() + 1e-8)
                
                # Weighted evidence using DKVMN attention + response evidence
                combined_evidence = (
                    evidence_embed * slot_attention.unsqueeze(1) +           # DKVMN evidence
                    ability_evidence_expanded * slot_attention.unsqueeze(1)  # IRT ability evidence
                ) / 2.0
                
                # Aggregate across batch dimension with attention weighting
                aggregated_evidence = torch.sum(
                    combined_evidence * attention_normalized.unsqueeze(1), 
                    dim=0
                )
                
                aggregated_precision = torch.sum(
                    evidence_precision_expanded * attention_normalized.unsqueeze(1),
                    dim=0
                )
                
                total_update_weight = slot_attention.mean()
                
                # Bayesian memory update
                memory_value.bayesian_update(
                    aggregated_evidence,      # Evidence mean
                    aggregated_precision,     # Evidence precision
                    total_update_weight       # Update confidence
                )
    
    def sample_memory_parameters(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Sample IRT parameters from all memory keys."""
        alphas = []
        betas = []
        
        for memory_key in self.memory_keys:
            if self.training:
                alpha, beta = memory_key.sample_parameters()
            else:
                alpha, beta = memory_key.get_deterministic_parameters()
            alphas.append(alpha)
            betas.append(beta)
        
        return alphas, betas
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor,
                return_params: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with NO activation functions but FULL integration.
        
        Maintains:
        1. Complete DKVMN memory operations
        2. Full IRT computation with GPCM
        3. Bayesian inference and ELBO
        4. Deep integration of all systems
        
        Removes: ALL activation functions from parameter computation
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Question embeddings
        q_embed = self.q_embed(questions)  # [batch_size, seq_len, embed_dim]
        
        # Initialize probabilities list and memory tracking
        all_probabilities = []
        total_kl = 0.0
        memory_usage_tracking = []
        
        # Process sequence with full DKVMN + IRT integration
        for t in range(seq_len):
            q_t = questions[:, t]  # [batch_size]
            r_t = responses[:, t] if t > 0 else torch.zeros_like(questions[:, t])  # [batch_size]
            
            # Question-response embedding for DKVMN operations
            qa_input = torch.stack([q_t.float(), r_t.float()], dim=1)  # [batch_size, 2]
            qa_embed = self.qa_embed(qa_input)  # [batch_size, embed_dim]
            
            # Transform to key space for memory attention
            query_key = self.q_to_key(q_embed[:, t])  # [batch_size, key_dim]
            
            # === FULL DKVMN MEMORY READ OPERATION ===
            attention_weights = self.compute_attention(query_key, q_t)
            ability_mean, ability_var = self.bayesian_read(attention_weights)
            
            # Extract student abilities from memory (NO activation layer)
            # Direct weighted combination instead of Linear layer
            predicted_abilities = torch.sum(ability_mean * self.ability_weights, dim=1)  # [batch_size]
            
            # === COMPLETE IRT PARAMETER COMPUTATION ===
            memory_alphas, memory_betas = self.sample_memory_parameters()
            
            # Compute attention-weighted IRT parameters (combines DKVMN attention with IRT)
            batch_probs = []
            current_memory_usage = []
            
            for b in range(batch_size):
                attention_dist = attention_weights[b, :]  # [memory_size]
                
                # Compute weighted IRT parameters using DKVMN attention weights
                weighted_alpha = torch.sum(attention_dist * torch.stack(memory_alphas))
                weighted_beta = torch.sum(attention_dist.unsqueeze(1) * torch.stack(memory_betas), dim=0)
                
                # Get student ability from DKVMN memory read (NO activation)
                theta = predicted_abilities[b]
                
                # === FULL GPCM COMPUTATION ===
                prob = self._gpcm_probability(theta, weighted_alpha, weighted_beta)
                batch_probs.append(prob)
                
                # Track memory usage for analysis
                current_memory_usage.append({
                    'attention_weights': attention_dist.detach().cpu(),
                    'theta': theta.detach().cpu(),
                    'alpha': weighted_alpha.detach().cpu(),
                    'beta': weighted_beta.detach().cpu()
                })
            
            probabilities = torch.stack(batch_probs, dim=0)  # [batch_size, n_categories]
            all_probabilities.append(probabilities)
            memory_usage_tracking.append(current_memory_usage)
            
            # === FULL DKVMN MEMORY WRITE OPERATION ===
            if t > 0:  # Only update after seeing responses
                evidence_embed = self.qa_to_evidence(qa_embed)
                self.bayesian_write(attention_weights, evidence_embed, r_t)
        
        # Stack all probabilities across time steps
        final_probabilities = torch.stack(all_probabilities, dim=1)  # [batch_size, seq_len, n_categories]
        
        # === BAYESIAN REGULARIZATION ===
        for memory_key in self.memory_keys:
            total_kl += memory_key.kl_divergence()
        
        # Apply KL annealing for stable training
        kl_annealing_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs) if self.kl_warmup_epochs > 0 else 1.0
        
        # Prepare auxiliary outputs
        aux_dict = {
            'kl_divergence': total_kl * kl_annealing_factor,
            'raw_kl_divergence': total_kl,
            'kl_annealing_factor': kl_annealing_factor,
            'memory_usage': memory_usage_tracking
        }
        
        if return_params:
            memory_alphas, memory_betas = self.sample_memory_parameters()
            aux_dict['memory_alphas'] = memory_alphas
            aux_dict['memory_betas'] = memory_betas
            aux_dict['ability_beliefs'] = [mv.get_belief_distribution() for mv in self.memory_values]
            aux_dict['attention_patterns'] = [usage[0]['attention_weights'] for usage in memory_usage_tracking]
        
        return final_probabilities, aux_dict
    
    def _gpcm_probability(self, theta: torch.Tensor, alpha: torch.Tensor, 
                         beta: torch.Tensor) -> torch.Tensor:
        """
        Compute GPCM probability with proper numerical stability.
        """
        K = self.n_categories
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
            # Sample parameters from each memory key
            alphas = []
            betas = []
            
            for memory_key in self.memory_keys:
                alpha, beta = memory_key.get_deterministic_parameters()
                alphas.append(alpha)
                betas.append(beta)
            
            # Student ability beliefs from memory values
            thetas = []
            for memory_value in self.memory_values:
                # Get deterministic ability (NO activation)
                theta = memory_value.get_deterministic_ability()
                # Take mean across dimensions to get scalar ability
                thetas.append(theta.mean())
            
            return {
                'alpha': torch.stack(alphas),  # [memory_size]
                'beta': torch.stack(betas),    # [memory_size, n_categories-1]
                'theta': torch.stack(thetas),  # [memory_size]
                'memory_keys': [mk.embedding for mk in self.memory_keys]
            }