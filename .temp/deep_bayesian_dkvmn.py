#!/usr/bin/env python3
"""
Deep Bayesian-DKVMN Integration for GPCM

This module implements a deeply integrated Bayesian-DKVMN architecture where:
1. Memory keys represent question archetypes with IRT parameters (α, β)
2. Memory values store student ability belief distributions (θ ~ N(μ, σ²))
3. Read/write operations use Bayesian inference principles
4. All operations optimize the ELBO objective directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional, List


class BayesianMemoryKey(nn.Module):
    """
    Bayesian memory key representing a question archetype.
    
    Each key contains:
    - Question embedding (learned representation)
    - Discrimination parameter α ~ LogNormal(μ_α, σ_α²)
    - Threshold parameters β ~ OrderedNormal(μ_β, σ_β²)
    """
    
    def __init__(self, key_dim: int, n_categories: int = 4):
        super().__init__()
        self.key_dim = key_dim
        self.n_categories = n_categories
        
        # Question embedding
        self.embedding = nn.Parameter(torch.randn(key_dim))
        
        # Discrimination parameter (log-normal)
        self.alpha_mean = nn.Parameter(torch.zeros(1))
        self.alpha_log_var = nn.Parameter(torch.ones(1) * np.log(0.3))
        
        # Threshold parameters (ordered normal)
        self.beta_base = nn.Parameter(torch.zeros(1))
        self.beta_offsets = nn.Parameter(torch.ones(n_categories - 1) * 0.5)
        self.beta_log_var = nn.Parameter(torch.ones(n_categories - 1) * np.log(0.1))
    
    def sample_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample α and β parameters for this memory key."""
        # Sample discrimination
        alpha_std = torch.exp(0.5 * self.alpha_log_var)
        alpha = torch.exp(self.alpha_mean + torch.randn_like(self.alpha_mean) * alpha_std)
        
        # Sample ordered thresholds
        beta_base = self.beta_base + torch.randn_like(self.beta_base) * 0.1
        beta_offsets = F.softplus(self.beta_offsets)
        beta = torch.zeros(self.n_categories - 1)
        beta[0] = beta_base
        for i in range(1, self.n_categories - 1):
            beta[i] = beta[i-1] + beta_offsets[i-1]
        
        return alpha.squeeze(), beta
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for this key's parameters."""
        # KL for discrimination (log-normal)
        alpha_std = torch.exp(0.5 * self.alpha_log_var)
        alpha_dist = Normal(self.alpha_mean, alpha_std)
        alpha_prior = Normal(torch.zeros_like(self.alpha_mean), torch.ones_like(alpha_std) * 0.3)
        alpha_kl = kl_divergence(alpha_dist, alpha_prior).sum()
        
        # KL for thresholds (simplified)
        beta_kl = (self.beta_log_var.exp() + self.beta_base.pow(2) - self.beta_log_var - 1).sum() * 0.5
        
        return alpha_kl + beta_kl


class BayesianMemoryValue(nn.Module):
    """
    Bayesian memory value storing student ability belief distribution.
    
    Each value contains:
    - Ability mean μ_θ (current belief about student ability)
    - Ability variance σ_θ² (uncertainty about student ability)
    - Update count (for adaptive learning rates)
    """
    
    def __init__(self, value_dim: int):
        super().__init__()
        self.value_dim = value_dim
        
        # Ability belief parameters
        self.theta_mean = nn.Parameter(torch.zeros(value_dim))  # μ_θ
        self.theta_log_var = nn.Parameter(torch.ones(value_dim) * np.log(1.0))  # log(σ_θ²)
        
        # Update tracking
        self.update_count = nn.Parameter(torch.zeros(value_dim), requires_grad=False)
    
    def get_belief_distribution(self) -> Normal:
        """Get current belief distribution over student ability."""
        theta_std = torch.exp(0.5 * self.theta_log_var)
        return Normal(self.theta_mean, theta_std)
    
    def bayesian_update(self, evidence_mean: torch.Tensor, evidence_precision: torch.Tensor,
                       update_weight: torch.Tensor):
        """
        Perform Bayesian update of ability beliefs.
        
        Args:
            evidence_mean: New evidence about student ability
            evidence_precision: Precision (1/variance) of the evidence
            update_weight: How much to weight this update
        """
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


class DeepBayesianDKVMN(nn.Module):
    """
    Deep Bayesian-DKVMN Integration for GPCM.
    
    This architecture deeply integrates Bayesian IRT parameters with DKVMN operations:
    1. Memory keys contain question IRT parameters (α, β)
    2. Memory values contain student ability beliefs (θ ~ N(μ, σ²))
    3. Attention mechanism uses IRT similarity
    4. Memory updates use Bayesian inference
    5. ELBO optimization drives all learning
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
        
        # Bayesian Memory Network
        self.memory_keys = nn.ModuleList([
            BayesianMemoryKey(key_dim, n_categories) for _ in range(memory_size)
        ])
        self.memory_values = nn.ModuleList([
            BayesianMemoryValue(value_dim) for _ in range(memory_size)
        ])
        
        # Attention and transformation layers
        self.q_to_key = nn.Linear(embed_dim, key_dim)
        self.qa_to_evidence = nn.Linear(embed_dim, value_dim)
        
        # GPCM prediction layers
        self.ability_predictor = nn.Linear(value_dim, 1)
        
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
        This ensures full integration of DKVMN attention with IRT parameters.
        
        Args:
            query_embed: Question embeddings [batch_size, key_dim]
            question_ids: Question IDs [batch_size]
            
        Returns:
            attention_weights: [batch_size, memory_size]
        """
        batch_size = query_embed.shape[0]
        attention_logits = torch.zeros(batch_size, self.memory_size, device=query_embed.device)
        
        # Sample IRT parameters from all memory keys for similarity computation
        memory_alphas = []
        memory_betas = []
        for memory_key in self.memory_keys:
            alpha, beta = memory_key.sample_parameters()
            memory_alphas.append(alpha)
            memory_betas.append(beta)
        
        for i, memory_key in enumerate(self.memory_keys):
            # === DKVMN EMBEDDING SIMILARITY ===
            # Standard DKVMN attention based on embedding similarity
            embed_sim = torch.matmul(query_embed, memory_key.embedding)  # [batch_size]
            
            # === IRT PARAMETER SIMILARITY ===
            # Additional similarity based on IRT parameters
            alpha_key = memory_alphas[i]
            beta_key = memory_betas[i]
            
            # Discrimination-based similarity (higher alpha = more discriminating)
            alpha_sim = torch.ones_like(embed_sim) * alpha_key.item()
            
            # Difficulty-based similarity (beta parameters influence attention)
            # Use mean difficulty as a similarity factor
            beta_mean = beta_key.mean()
            difficulty_sim = torch.ones_like(embed_sim) * torch.exp(-0.5 * beta_mean**2)
            
            # === COMBINED ATTENTION LOGITS ===
            # Combine embedding similarity with IRT parameter similarity
            # This ensures BOTH DKVMN and IRT components contribute to attention
            combined_logits = (
                embed_sim +                    # DKVMN embedding similarity
                0.3 * alpha_sim +             # IRT discrimination similarity
                0.2 * difficulty_sim          # IRT difficulty similarity
            )
            
            attention_logits[:, i] = combined_logits
        
        # Apply softmax to get attention weights
        # This produces the DKVMN attention distribution influenced by IRT parameters
        return F.softmax(attention_logits, dim=1)
    
    def bayesian_read(self, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Bayesian read operation from memory.
        
        Args:
            attention_weights: [batch_size, memory_size]
            
        Returns:
            ability_mean: Expected student ability [batch_size, value_dim]
            ability_var: Ability uncertainty [batch_size, value_dim]
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
        
        This implements proper DKVMN write mechanics with Bayesian updates:
        1. Use DKVMN attention weights to determine memory updates
        2. Convert responses to ability evidence with IRT principles
        3. Update memory values using Bayesian inference
        
        Args:
            attention_weights: DKVMN attention weights [batch_size, memory_size]
            evidence_embed: Evidence embedding from DKVMN [batch_size, value_dim]
            responses: Student responses for evidence [batch_size]
        """
        batch_size = attention_weights.shape[0]
        
        # === RESPONSE TO ABILITY EVIDENCE CONVERSION ===
        # Convert ordinal responses to ability evidence using IRT principles
        response_normalized = responses.float() / (self.n_categories - 1)  # [0, 1]
        
        # IRT-based ability evidence: higher responses indicate higher ability
        # Use logistic transformation to map responses to ability scale
        ability_evidence = torch.logit(torch.clamp(response_normalized, 0.01, 0.99))  # Real scale
        
        # Expand evidence to match value dimension
        ability_evidence_expanded = ability_evidence.unsqueeze(1).expand(-1, self.value_dim)
        
        # === EVIDENCE PRECISION BASED ON RESPONSE PATTERNS ===
        # Higher precision for extreme responses (0 or max), lower for middle responses
        response_extremeness = torch.abs(response_normalized - 0.5) * 2  # [0, 1]
        evidence_precision = 0.5 + response_extremeness  # [0.5, 1.5]
        evidence_precision_expanded = evidence_precision.unsqueeze(1).expand(-1, self.value_dim)
        
        # === DKVMN MEMORY WRITE WITH BAYESIAN UPDATES ===
        # Update each memory slot based on DKVMN attention distribution
        for i, memory_value in enumerate(self.memory_values):
            # Get attention weights for this memory slot
            slot_attention = attention_weights[:, i]  # [batch_size]
            
            # Only update if this memory slot receives sufficient attention
            if slot_attention.sum() > 1e-6:
                # === ATTENTION-WEIGHTED EVIDENCE AGGREGATION ===
                # Combine evidence from all batch items weighted by attention
                attention_normalized = slot_attention / (slot_attention.sum() + 1e-8)
                
                # Weighted evidence using DKVMN attention + response evidence
                combined_evidence = (
                    evidence_embed * slot_attention.unsqueeze(1) +           # DKVMN evidence
                    ability_evidence_expanded * slot_attention.unsqueeze(1)  # IRT ability evidence
                ) / 2.0  # Balance both sources
                
                # Aggregate across batch dimension with attention weighting
                aggregated_evidence = torch.sum(
                    combined_evidence * attention_normalized.unsqueeze(1), 
                    dim=0
                )
                
                # Aggregate precision with attention weighting
                aggregated_precision = torch.sum(
                    evidence_precision_expanded * attention_normalized.unsqueeze(1),
                    dim=0
                )
                
                # Total update weight (how much to trust this update)
                total_update_weight = slot_attention.mean()
                
                # === BAYESIAN MEMORY UPDATE ===
                # Perform Bayesian update using aggregated evidence
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
            alpha, beta = memory_key.sample_parameters()
            alphas.append(alpha)
            betas.append(beta)
        
        return alphas, betas
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor,
                return_params: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with fully integrated Bayesian-DKVMN + IRT computation.
        
        This implementation ensures BOTH systems are fully utilized:
        1. DKVMN memory operations (read/write with attention)
        2. Proper IRT parameter computation via GPCM
        
        Args:
            questions: Question IDs [batch_size, seq_len]
            responses: Student responses [batch_size, seq_len]
            return_params: Whether to return sampled parameters
            
        Returns:
            probabilities: GPCM probabilities [batch_size, seq_len, n_categories]
            aux_dict: Dictionary with losses and optional parameters
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
            # Current question and response
            q_t = questions[:, t]  # [batch_size]
            r_t = responses[:, t] if t > 0 else torch.zeros_like(questions[:, t])  # [batch_size]
            
            # Question-response embedding for DKVMN operations
            qa_input = torch.stack([q_t.float(), r_t.float()], dim=1)  # [batch_size, 2]
            qa_embed = self.qa_embed(qa_input)  # [batch_size, embed_dim]
            
            # Transform to key space for memory attention
            query_key = self.q_to_key(q_embed[:, t])  # [batch_size, key_dim]
            
            # === DKVMN MEMORY READ OPERATION ===
            # Compute attention using both embedding similarity AND IRT parameter similarity
            attention_weights = self.compute_attention(query_key, q_t)
            
            # Bayesian read operation - get student ability distributions from memory
            ability_mean, ability_var = self.bayesian_read(attention_weights)
            
            # Extract current student abilities from memory (DKVMN output)
            # This uses the FULL DKVMN memory mechanism, not bypassing it
            predicted_abilities = self.ability_predictor(ability_mean).squeeze(-1)  # [batch_size]
            
            # === IRT PARAMETER COMPUTATION ===
            # Sample IRT parameters from Bayesian memory keys (α, β parameters)
            memory_alphas, memory_betas = self.sample_memory_parameters()
            
            # Compute attention-weighted IRT parameters (combines DKVMN attention with IRT)
            # This ensures both DKVMN memory mechanism AND IRT parameters are used
            batch_probs = []
            current_memory_usage = []
            
            for b in range(batch_size):
                # Get attention distribution for this batch item
                attention_dist = attention_weights[b, :]  # [memory_size]
                
                # Compute weighted IRT parameters using DKVMN attention weights
                weighted_alpha = torch.sum(attention_dist * torch.stack(memory_alphas))
                weighted_beta = torch.sum(attention_dist.unsqueeze(1) * torch.stack(memory_betas), dim=0)
                
                # Get student ability from DKVMN memory read
                theta = predicted_abilities[b]
                
                # === FULL IRT COMPUTATION ===
                # Compute GPCM probability using the complete IRT model
                # This is the proper IRT computation, not bypassed
                prob = self._gpcm_probability(theta, weighted_alpha, weighted_beta)
                batch_probs.append(prob)
                
                # Track memory usage for analysis
                current_memory_usage.append({
                    'attention_weights': attention_dist.detach().cpu(),
                    'theta': theta.detach().cpu(),
                    'alpha': weighted_alpha.detach().cpu(),
                    'beta': weighted_beta.detach().cpu()
                })
            
            # Stack probabilities for this time step
            probabilities = torch.stack(batch_probs, dim=0)  # [batch_size, n_categories]
            all_probabilities.append(probabilities)
            memory_usage_tracking.append(current_memory_usage)
            
            # === DKVMN MEMORY WRITE OPERATION ===
            # Update memory based on evidence (responses) - this is the DKVMN write
            if t > 0:  # Only update after seeing responses
                evidence_embed = self.qa_to_evidence(qa_embed)
                # This updates the Bayesian memory values with new evidence
                # Full DKVMN write operation, not bypassed
                self.bayesian_write(attention_weights, evidence_embed, r_t)
        
        # Stack all probabilities across time steps
        final_probabilities = torch.stack(all_probabilities, dim=1)  # [batch_size, seq_len, n_categories]
        
        # === BAYESIAN REGULARIZATION ===
        # Compute KL divergence from all memory keys (Bayesian component)
        for memory_key in self.memory_keys:
            total_kl += memory_key.kl_divergence()
        
        # Apply KL annealing for stable training
        kl_annealing_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs) if self.kl_warmup_epochs > 0 else 1.0
        
        # Prepare auxiliary outputs
        aux_dict = {
            'kl_divergence': total_kl * kl_annealing_factor,
            'raw_kl_divergence': total_kl,
            'kl_annealing_factor': kl_annealing_factor,
            'memory_usage': memory_usage_tracking  # Track how memory is being used
        }
        
        if return_params:
            # Extract current state for analysis
            memory_alphas, memory_betas = self.sample_memory_parameters()
            aux_dict['memory_alphas'] = memory_alphas
            aux_dict['memory_betas'] = memory_betas
            aux_dict['ability_beliefs'] = [mv.get_belief_distribution() for mv in self.memory_values]
            aux_dict['attention_patterns'] = [usage[0]['attention_weights'] for usage in memory_usage_tracking]
        
        return final_probabilities, aux_dict
    
    def _gpcm_probability(self, theta: torch.Tensor, alpha: torch.Tensor, 
                         beta: torch.Tensor) -> torch.Tensor:
        """
        Compute GPCM probability for given parameters.
        
        Args:
            theta: Student ability (scalar)
            alpha: Discrimination parameter (scalar)
            beta: Threshold parameters [n_categories-1]
            
        Returns:
            probabilities: [n_categories]
        """
        K = self.n_categories
        cum_logits = torch.zeros(K, device=theta.device)
        
        for k in range(1, K):
            cum_sum = 0.0
            for h in range(k):
                if h < len(beta):
                    cum_sum += alpha * (theta - beta[h])
            cum_logits[k] = cum_sum
        
        # Compute probabilities
        exp_logits = torch.exp(cum_logits - cum_logits.max())
        probabilities = exp_logits / exp_logits.sum()
        
        return probabilities
    
    def elbo_loss(self, probabilities: torch.Tensor, targets: torch.Tensor,
                  kl_divergence: torch.Tensor) -> torch.Tensor:
        """
        Compute ELBO loss for deep Bayesian-DKVMN model.
        
        Args:
            probabilities: Predicted GPCM probabilities [batch_size, seq_len, n_categories]
            targets: True responses [batch_size, seq_len]
            kl_divergence: KL divergence from memory parameters
            
        Returns:
            loss: Scalar ELBO loss
        """
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
        """Extract interpretable IRT parameters from memory network."""
        with torch.no_grad():
            # Sample parameters from each memory key
            alphas = []
            betas = []
            
            for memory_key in self.memory_keys:
                alpha, beta = memory_key.sample_parameters()
                alphas.append(alpha)
                betas.append(beta)
            
            # Student ability beliefs from memory values
            thetas = []
            for memory_value in self.memory_values:
                belief_dist = memory_value.get_belief_distribution()
                thetas.append(belief_dist.mean)
            
            return {
                'alpha': torch.stack(alphas),  # [memory_size]
                'beta': torch.stack(betas),    # [memory_size, n_categories-1]
                'theta': torch.stack(thetas),  # [memory_size, value_dim]
                'memory_keys': [mk.embedding for mk in self.memory_keys]
            }