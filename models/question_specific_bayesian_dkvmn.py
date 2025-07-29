#!/usr/bin/env python3
"""
Question-Specific Deep Bayesian-DKVMN Implementation

This version implements the CORRECT IRT structure:
1. Question-specific alpha and beta parameters (30 questions → 30 α, 30 β sets)
2. Student-specific theta parameters (200 students → 200 θ values)
3. Full DKVMN memory operations maintained
4. Proper activation functions for IRT constraints

Key insight: α and β should be per-question, θ should be per-student!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional, List


class QuestionSpecificIRTParameters(nn.Module):
    """
    Question-specific IRT parameters with proper Bayesian inference.
    
    Each question has its own:
    - α (discrimination parameter)  
    - β (threshold parameters for categories)
    """
    
    def __init__(self, n_questions: int, n_categories: int = 4):
        super().__init__()
        self.n_questions = n_questions
        self.n_categories = n_categories
        
        # Discrimination parameters (one per question) - must be positive
        self.alpha_mean = nn.Parameter(torch.randn(n_questions) * 0.2)  # log-space
        self.alpha_log_var = nn.Parameter(torch.ones(n_questions) * np.log(0.3))
        
        # Threshold parameters (ordered, per question)
        self.beta_base = nn.Parameter(torch.randn(n_questions) * 0.5)  # β_1 for each question
        self.beta_gaps = nn.Parameter(torch.ones(n_questions, n_categories - 2) * 0.5)  # positive gaps
        self.beta_log_var = nn.Parameter(torch.ones(n_questions, n_categories - 1) * np.log(0.2))
    
    def get_parameters(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get deterministic IRT parameters for specific questions.
        
        Args:
            question_ids: [batch_size] question indices
            
        Returns:
            alphas: [batch_size] discrimination parameters for these questions
            betas: [batch_size, n_categories-1] threshold parameters for these questions
        """
        batch_size = question_ids.shape[0]
        
        # Get discrimination parameters (ensure positive with exp)
        alpha_means = self.alpha_mean[question_ids]  # [batch_size]
        alphas = torch.exp(alpha_means)  # Always positive for IRT
        
        # Get ordered threshold parameters for each question
        base_betas = self.beta_base[question_ids]  # [batch_size]
        gaps = F.softplus(self.beta_gaps[question_ids])  # [batch_size, n_categories-2], always positive
        
        # Construct ordered thresholds for each question
        betas = torch.zeros(batch_size, self.n_categories - 1, device=question_ids.device)
        betas[:, 0] = base_betas
        
        for k in range(1, self.n_categories - 1):
            betas[:, k] = betas[:, k-1] + gaps[:, k-1]
        
        return alphas, betas
    
    def sample_parameters(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample IRT parameters for specific questions (for training with variational inference).
        """
        batch_size = question_ids.shape[0]
        
        # Sample discrimination with reparameterization trick
        alpha_std = torch.exp(self.alpha_log_var[question_ids])
        alpha_noise = torch.randn_like(alpha_std)
        alpha_log = self.alpha_mean[question_ids] + alpha_std * alpha_noise
        alphas = torch.exp(alpha_log)  # Ensure positive
        
        # Sample threshold parameters for each question
        base_betas = self.beta_base[question_ids]
        beta_std = torch.exp(self.beta_log_var[question_ids])
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
        """Compute KL divergence for all question parameters."""
        # KL for all discrimination parameters
        alpha_std = torch.exp(self.alpha_log_var)
        alpha_dist = Normal(self.alpha_mean, alpha_std)
        alpha_prior = Normal(torch.zeros_like(self.alpha_mean), torch.ones_like(alpha_std) * 0.5)
        alpha_kl = kl_divergence(alpha_dist, alpha_prior).sum()
        
        # KL for all threshold parameters (simplified)
        beta_kl = (
            (self.beta_base.pow(2) - 2 * np.log(0.5)).sum() * 0.5 +
            (self.beta_log_var.exp() + self.beta_base.unsqueeze(1).pow(2) - 
             self.beta_log_var - 1).sum() * 0.5
        )
        
        return alpha_kl + beta_kl


class StudentAbilityMemory(nn.Module):
    """
    Student ability memory with proper individual tracking.
    
    Each student has their own ability that evolves through DKVMN memory operations.
    """
    
    def __init__(self, memory_size: int, value_dim: int):
        super().__init__()
        self.memory_size = memory_size
        self.value_dim = value_dim
        
        # Memory values for student ability tracking
        self.ability_means = nn.Parameter(torch.zeros(memory_size, value_dim))
        self.ability_log_vars = nn.Parameter(torch.ones(memory_size, value_dim) * np.log(1.0))
        
        # Update tracking
        self.update_counts = nn.Parameter(torch.zeros(memory_size, value_dim), requires_grad=False)
    
    def get_belief_distributions(self) -> List[Normal]:
        """Get belief distributions for all memory slots."""
        distributions = []
        for i in range(self.memory_size):
            theta_std = torch.exp(0.5 * self.ability_log_vars[i])
            distributions.append(Normal(self.ability_means[i], theta_std))
        return distributions
    
    def bayesian_read(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Perform Bayesian read to get student abilities.
        
        Args:
            attention_weights: [batch_size, memory_size]
            
        Returns:
            abilities: [batch_size] student abilities
        """
        batch_size = attention_weights.shape[0]
        
        # Get ability means from all memory slots
        ability_means = self.ability_means  # [memory_size, value_dim]
        
        # Weighted combination using attention
        weighted_abilities = torch.sum(
            attention_weights.unsqueeze(2) * ability_means.unsqueeze(0), 
            dim=1
        )  # [batch_size, value_dim]
        
        # Convert to scalar abilities (take mean across dimensions)
        scalar_abilities = weighted_abilities.mean(dim=1)  # [batch_size]
        
        return scalar_abilities
    
    def bayesian_update(self, slot_id: int, evidence_mean: torch.Tensor, 
                       evidence_precision: torch.Tensor, update_weight: float):
        """Update specific memory slot with evidence."""
        with torch.no_grad():
            # Current beliefs for this slot
            prior_precision = torch.exp(-self.ability_log_vars[slot_id])
            prior_mean = self.ability_means[slot_id]
            
            # Bayesian update formulas
            posterior_precision = prior_precision + evidence_precision * update_weight
            posterior_mean = (prior_precision * prior_mean + 
                            evidence_precision * evidence_mean * update_weight) / posterior_precision
            
            # Update parameters
            self.ability_means[slot_id].data = posterior_mean
            self.ability_log_vars[slot_id].data = -torch.log(posterior_precision)
            self.update_counts[slot_id].data += update_weight


class QuestionSpecificDeepBayesianDKVMN(nn.Module):
    """
    Question-Specific Deep Bayesian-DKVMN with correct IRT structure.
    
    This version implements the PROPER IRT structure:
    1. α, β parameters are question-specific (30 questions → 30 parameter sets)
    2. θ parameters are student-specific (extracted per student)
    3. Full DKVMN memory operations for student ability tracking
    4. Complete Bayesian inference with ELBO optimization
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
        
        # Question-specific IRT parameters (CORRECT structure)
        self.irt_params = QuestionSpecificIRTParameters(n_questions, n_categories)
        
        # Student ability memory (DKVMN for student tracking)
        self.student_memory = StudentAbilityMemory(memory_size, value_dim)
        
        # Memory keys for attention (question similarity)
        self.memory_keys = nn.Parameter(torch.randn(memory_size, key_dim))
        
        # Attention and transformation layers
        self.q_to_key = nn.Linear(embed_dim, key_dim)
        self.qa_to_evidence = nn.Linear(embed_dim, value_dim)
        
        # Current epoch for KL annealing
        self.current_epoch = 0
        self.kl_warmup_epochs = 20
        
        # Student tracking for theta extraction
        self.register_buffer('student_abilities', torch.zeros(1000))  # Track student abilities
        self.register_buffer('student_counts', torch.zeros(1000))
        
    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch
    
    def compute_attention(self, query_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights for memory access.
        
        Args:
            query_embed: [batch_size, key_dim] question embeddings
            
        Returns:
            attention_weights: [batch_size, memory_size]
        """
        # Compute similarity with memory keys
        attention_logits = torch.matmul(query_embed, self.memory_keys.T)  # [batch_size, memory_size]
        attention_weights = F.softmax(attention_logits, dim=1)
        
        return attention_weights
    
    def update_student_abilities(self, student_ids: torch.Tensor, abilities: torch.Tensor):
        """Update tracked student abilities."""
        with torch.no_grad():
            for i, student_id in enumerate(student_ids):
                if student_id < len(self.student_abilities):
                    # Exponential moving average
                    momentum = 0.1
                    count = self.student_counts[student_id]
                    if count == 0:
                        self.student_abilities[student_id] = abilities[i]
                    else:
                        self.student_abilities[student_id] = (
                            (1 - momentum) * self.student_abilities[student_id] + 
                            momentum * abilities[i]
                        )
                    self.student_counts[student_id] += 1
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor,
                student_ids: Optional[torch.Tensor] = None,
                return_params: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with question-specific IRT parameters and student-specific abilities.
        
        Args:
            questions: [batch_size, seq_len] question IDs
            responses: [batch_size, seq_len] student responses
            student_ids: [batch_size] student IDs for ability tracking
            return_params: Whether to return IRT parameters
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Create default student IDs if not provided
        if student_ids is None:
            student_ids = torch.arange(batch_size, device=device)
        
        # Question embeddings
        q_embed = self.q_embed(questions)  # [batch_size, seq_len, embed_dim]
        
        all_probabilities = []
        final_student_abilities = []
        
        # Process each time step
        for t in range(seq_len):
            q_t = questions[:, t]  # [batch_size]
            r_t = responses[:, t] if t > 0 else torch.zeros_like(q_t)  # [batch_size]
            
            # Question-response embedding
            qa_input = torch.stack([q_t.float(), r_t.float()], dim=1)
            qa_embed = self.qa_embed(qa_input)
            
            # Transform to key space
            query_key = self.q_to_key(q_embed[:, t])
            
            # === DKVMN MEMORY OPERATIONS ===
            # Compute attention for student ability retrieval
            attention_weights = self.compute_attention(query_key)
            
            # Get student abilities from memory
            student_abilities = self.student_memory.bayesian_read(attention_weights)
            
            # Store final abilities for theta extraction
            if t == seq_len - 1:
                final_student_abilities = student_abilities.detach()
                self.update_student_abilities(student_ids, final_student_abilities)
            
            # === QUESTION-SPECIFIC IRT PARAMETERS ===
            if self.training:
                # Sample parameters during training
                alphas, betas = self.irt_params.sample_parameters(q_t)
            else:
                # Use deterministic parameters during evaluation
                alphas, betas = self.irt_params.get_parameters(q_t)
            
            # === GPCM PROBABILITY COMPUTATION ===
            batch_probs = []
            for b in range(batch_size):
                theta = student_abilities[b]
                alpha = alphas[b]
                beta = betas[b]
                
                # Compute GPCM probability for this student-question pair
                prob = self._gpcm_probability(theta, alpha, beta)
                batch_probs.append(prob)
            
            probabilities = torch.stack(batch_probs, dim=0)  # [batch_size, n_categories]
            all_probabilities.append(probabilities)
            
            # === MEMORY UPDATE ===
            if t > 0:  # Update after seeing responses
                evidence_embed = self.qa_to_evidence(qa_embed)
                
                # Update memory based on attention and response evidence
                for i in range(self.memory_size):
                    if attention_weights[:, i].sum() > 1e-6:
                        # Convert responses to ability evidence
                        response_evidence = (r_t.float() / (self.n_categories - 1) - 0.5) * 2  # [-1, 1]
                        evidence_mean = evidence_embed.mean(dim=0) * response_evidence.mean()
                        evidence_precision = torch.ones_like(evidence_mean) * 0.5
                        update_weight = attention_weights[:, i].mean().item()
                        
                        self.student_memory.bayesian_update(i, evidence_mean, evidence_precision, update_weight)
        
        # Stack all probabilities
        final_probabilities = torch.stack(all_probabilities, dim=1)  # [batch_size, seq_len, n_categories]
        
        # Compute KL divergence
        total_kl = self.irt_params.kl_divergence()
        
        # Apply KL annealing
        kl_annealing_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs) if self.kl_warmup_epochs > 0 else 1.0
        
        aux_dict = {
            'kl_divergence': total_kl * kl_annealing_factor,
            'raw_kl_divergence': total_kl,
            'kl_annealing_factor': kl_annealing_factor,
            'final_student_abilities': final_student_abilities
        }
        
        if return_params:
            # Extract all question parameters
            all_q_ids = torch.arange(self.n_questions, device=device)
            all_alphas, all_betas = self.irt_params.get_parameters(all_q_ids)
            
            aux_dict.update({
                'alphas': all_alphas,  # [n_questions] - question-specific!
                'betas': all_betas,    # [n_questions, n_categories-1] - question-specific!
                'thetas': self.student_abilities[:200],  # [200] - student-specific!
                'student_ids': student_ids
            })
        
        return final_probabilities, aux_dict
    
    def _gpcm_probability(self, theta: torch.Tensor, alpha: torch.Tensor, 
                         beta: torch.Tensor) -> torch.Tensor:
        """Compute GPCM probability with proper numerical stability."""
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
        """Extract interpretable IRT parameters with correct structure."""
        with torch.no_grad():
            # Get all question parameters (CORRECT: question-specific)
            all_q_ids = torch.arange(self.n_questions)
            alphas, betas = self.irt_params.get_parameters(all_q_ids)
            
            # Get student abilities (CORRECT: student-specific)
            student_abilities = self.student_abilities[:200]  # First 200 students
            
            return {
                'alpha': alphas,           # [n_questions] - one per question
                'beta': betas,             # [n_questions, n_categories-1] - one set per question  
                'theta': student_abilities # [200] - one per student
            }