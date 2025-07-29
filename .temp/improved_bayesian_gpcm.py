#!/usr/bin/env python3
"""
Improved Variational Bayesian GPCM Model for Better IRT Parameter Recovery

Key improvements over baseline:
1. Direct IRT parameter learning with proper constraints
2. Separate parameter recovery pathway 
3. Stronger IRT structure enforcement
4. Better memory-IRT integration
5. Avoid naive question_specific loss terms

Focus: Recover IRT parameters through architecture rather than loss engineering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional


class IRTParameterEncoder(nn.Module):
    """
    Dedicated IRT parameter encoder that learns to map from question/student representations
    to proper IRT parameter distributions.
    """
    
    def __init__(self, n_questions: int, n_students: int, n_categories: int = 4, 
                 embed_dim: int = 64):
        super().__init__()
        self.n_questions = n_questions
        self.n_students = n_students
        self.n_categories = n_categories
        self.embed_dim = embed_dim
        
        # Question embeddings for IRT parameter extraction
        self.q_irt_embed = nn.Embedding(n_questions + 1, embed_dim)
        
        # Discrimination parameter encoder (per question)
        self.alpha_encoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # mean and log_var
        )
        
        # Threshold parameter encoder (per question)
        self.beta_encoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, (n_categories - 1) * 2)  # mean and log_var for each threshold
        )
        
        # Student ability encoder (input from projected LSTM state)
        self.theta_encoder = nn.Sequential(
            nn.Linear(32, 32),  # Input from lstm_projection
            nn.ReLU(),
            nn.Linear(32, 2)  # mean and log_var
        )
        
        # Student response history encoder for theta estimation
        self.response_encoder = nn.LSTM(2, embed_dim // 2, batch_first=True, bidirectional=True)
        self.lstm_projection = nn.Linear(embed_dim, 32)  # Project LSTM output to consistent size
        
        # Initialize parameters
        self._initialize_irt_parameters()
    
    def _initialize_irt_parameters(self):
        """Initialize IRT parameter encoders with domain knowledge."""
        # Initialize question embeddings with small values
        nn.init.normal_(self.q_irt_embed.weight, mean=0, std=0.1)
        
        # Initialize alpha encoder to produce reasonable discrimination values
        for layer in self.alpha_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Bias alpha encoder to produce positive discriminations around 1.0
        self.alpha_encoder[-1].bias.data[0] = 0.0  # mean
        self.alpha_encoder[-1].bias.data[1] = np.log(0.3)  # log_var
        
        # Initialize beta encoder for reasonable threshold spacing
        for layer in self.beta_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def encode_question_parameters(self, question_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode question-specific IRT parameters.
        
        Returns:
            alpha_mean, alpha_log_var: [batch_size, 1] discrimination parameters
            beta_mean, beta_log_var: [batch_size, n_categories-1] threshold parameters
        """
        batch_size = question_ids.shape[0]
        device = question_ids.device
        
        # Get question embeddings
        q_embed = self.q_irt_embed(question_ids)  # [batch_size, embed_dim]
        
        # Encode discrimination parameters
        alpha_params = self.alpha_encoder(q_embed)  # [batch_size, 2]
        alpha_mean = alpha_params[:, 0:1]  # [batch_size, 1] - in log space
        alpha_log_var = alpha_params[:, 1:2]  # [batch_size, 1]
        
        # Encode threshold parameters  
        beta_params = self.beta_encoder(q_embed)  # [batch_size, (n_categories-1)*2]
        beta_mean = beta_params[:, :(self.n_categories-1)]  # [batch_size, n_categories-1]
        beta_log_var = beta_params[:, (self.n_categories-1):]  # [batch_size, n_categories-1]
        
        return alpha_mean, alpha_log_var, beta_mean, beta_log_var
    
    def encode_student_abilities(self, responses: torch.Tensor, questions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode student abilities from response history.
        
        Args:
            responses: [batch_size, seq_len] response history
            questions: [batch_size, seq_len] question history
            
        Returns:
            theta_mean, theta_log_var: [batch_size, 1] ability parameters
        """
        batch_size, seq_len = responses.shape
        device = responses.device
        
        # Create response features (correctness and performance)
        max_score = self.n_categories - 1
        correctness = (responses == max_score).float()  # [batch_size, seq_len]
        performance = responses.float() / max_score  # [batch_size, seq_len]
        
        # Stack features
        response_features = torch.stack([correctness, performance], dim=-1)  # [batch_size, seq_len, 2]
        
        # Encode with LSTM
        lstm_out, (hidden, _) = self.response_encoder(response_features)  # [batch_size, seq_len, embed_dim]
        
        # For bidirectional LSTM, hidden has shape [2, batch_size, hidden_size]
        # Take the last hidden state from both directions
        forward_hidden = hidden[0]  # [batch_size, hidden_size]
        backward_hidden = hidden[1]  # [batch_size, hidden_size]
        final_state = torch.cat([forward_hidden, backward_hidden], dim=1)  # [batch_size, embed_dim]
        
        # Project to consistent dimension for theta encoder
        projected_state = self.lstm_projection(final_state)  # [batch_size, 32]
        
        # Encode ability parameters
        theta_params = self.theta_encoder(projected_state)  # [batch_size, 2]
        theta_mean = theta_params[:, 0:1]  # [batch_size, 1]
        theta_log_var = theta_params[:, 1:2]  # [batch_size, 1]
        
        return theta_mean, theta_log_var


class ImprovedVariationalBayesianGPCM(nn.Module):
    """
    Improved Variational Bayesian GPCM with better IRT parameter recovery.
    
    Key improvements:
    1. Dedicated IRT parameter encoders
    2. Separate memory and IRT pathways
    3. Strong IRT structure enforcement
    4. Better parameter recovery architecture
    """
    
    def __init__(self, n_students: int, n_questions: int, n_categories: int = 4,
                 embed_dim: int = 64, memory_size: int = 20, memory_key_dim: int = 50,
                 memory_value_dim: int = 200, kl_weight: float = 1.0):
        super().__init__()
        
        self.n_students = n_students
        self.n_questions = n_questions
        self.n_categories = n_categories
        self.kl_weight = kl_weight
        self.embed_dim = embed_dim
        
        # KL annealing
        self.kl_warmup_epochs = 20
        self.current_epoch = 0
        
        # IRT parameter encoder
        self.irt_encoder = IRTParameterEncoder(n_questions, n_students, n_categories, embed_dim)
        
        # Memory network (reduced influence for better IRT structure)
        self.memory_size = memory_size
        self.memory_key_dim = memory_key_dim
        self.memory_value_dim = memory_value_dim
        
        # Standard DKVMN components
        self.q_embed = nn.Embedding(n_questions + 1, embed_dim)
        self.qa_embed = nn.Linear(2, embed_dim)
        
        self.key_memory = nn.Parameter(torch.randn(memory_size, memory_key_dim))
        self.value_memory = nn.Parameter(torch.randn(memory_size, memory_value_dim))
        
        self.q_to_key = nn.Linear(embed_dim, memory_key_dim)
        self.qa_to_value = nn.Linear(embed_dim, memory_value_dim)
        self.erase_layer = nn.Linear(memory_value_dim, memory_value_dim)
        self.add_layer = nn.Linear(memory_value_dim, memory_value_dim)
        
        # Combination layer (reduce memory influence)
        self.combination_layer = nn.Linear(memory_value_dim + embed_dim, embed_dim)
        
        # Initialize weights
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
        Forward pass with improved IRT parameter recovery.
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Encode IRT parameters for all time steps
        all_alpha_means = []
        all_alpha_log_vars = []
        all_beta_means = []
        all_beta_log_vars = []
        
        probabilities = []
        total_kl = 0.0
        
        # Memory state initialization
        memory_values = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        for t in range(seq_len):
            q_ids = questions[:, t]  # [batch_size]
            
            # Encode question-specific IRT parameters
            alpha_mean, alpha_log_var, beta_mean, beta_log_var = self.irt_encoder.encode_question_parameters(q_ids)
            
            # Sample IRT parameters using reparameterization trick
            alpha_std = torch.exp(0.5 * alpha_log_var)
            alpha_eps = torch.randn_like(alpha_std)
            alpha_log = alpha_mean + alpha_eps * alpha_std
            alpha = torch.exp(alpha_log)  # Ensure positive discrimination
            
            beta_std = torch.exp(0.5 * beta_log_var)
            beta_eps = torch.randn_like(beta_std)
            beta = beta_mean + beta_eps * beta_std
            
            # Ensure ordered thresholds
            beta = torch.sort(beta, dim=-1)[0]
            
            # Encode student abilities from response history up to current time
            if t > 0:
                hist_responses = responses[:, :t]
                hist_questions = questions[:, :t]
                theta_mean, theta_log_var = self.irt_encoder.encode_student_abilities(hist_responses, hist_questions)
                
                theta_std = torch.exp(0.5 * theta_log_var)
                theta_eps = torch.randn_like(theta_std)
                theta = theta_mean + theta_eps * theta_std
            else:
                # Initialize theta with prior for first time step
                theta = torch.randn(batch_size, 1, device=device) * 0.5
                theta_mean = torch.zeros_like(theta)
                theta_log_var = torch.zeros_like(theta)
            
            # Memory network update (with reduced influence)
            q_embed = self.q_embed(q_ids)
            q_key = self.q_to_key(q_embed)
            correlation = F.softmax(torch.matmul(q_key, self.key_memory.t()), dim=1)
            read_value = torch.matmul(correlation.unsqueeze(1), memory_values).squeeze(1)
            
            # Small memory adjustment to theta (limited influence)
            combined = torch.cat([read_value, q_embed], dim=1)
            memory_adjustment = torch.tanh(self.combination_layer(combined))
            theta_adjusted = theta.squeeze(-1) + 0.1 * memory_adjustment.mean(dim=-1)  # Small adjustment
            theta_adjusted = theta_adjusted.unsqueeze(-1)
            
            # Compute GPCM probabilities
            probs = self._gpcm_probability(theta_adjusted.squeeze(-1), alpha.squeeze(-1), beta)
            probabilities.append(probs)
            
            # Update memory (if not last step)
            if t < seq_len - 1:
                # Create response embedding for memory update
                r = responses[:, t]
                correctness = (r == self.n_categories - 1).float()
                score = r.float() / (self.n_categories - 1)
                qa_input = torch.stack([correctness, score], dim=1)
                qa_embed = self.qa_embed(qa_input)
                
                qa_value = self.qa_to_value(qa_embed)
                erase_weight = torch.sigmoid(self.erase_layer(qa_value))
                add_content = torch.tanh(self.add_layer(qa_value))
                
                # Update memory with reduced write strength
                memory_values = memory_values * (1 - correlation.unsqueeze(2) * erase_weight.unsqueeze(1) * 0.5)
                memory_values = memory_values + correlation.unsqueeze(2) * add_content.unsqueeze(1) * 0.5
            
            # Accumulate parameters for KL computation
            all_alpha_means.append(alpha_mean)
            all_alpha_log_vars.append(alpha_log_var)
            all_beta_means.append(beta_mean)
            all_beta_log_vars.append(beta_log_var)
            
            # KL divergence for this time step
            if t > 0:  # Only compute theta KL after first step
                # Alpha KL (log-normal prior: LogN(0, 0.3))
                alpha_dist = Normal(alpha_mean, alpha_std)
                alpha_prior = Normal(torch.zeros_like(alpha_mean), torch.ones_like(alpha_std) * 0.3)
                alpha_kl = kl_divergence(alpha_dist, alpha_prior).sum()
                
                # Beta KL (normal prior: N(0, 1))
                beta_dist = Normal(beta_mean, beta_std)
                beta_prior = Normal(torch.zeros_like(beta_mean), torch.ones_like(beta_std))
                beta_kl = kl_divergence(beta_dist, beta_prior).sum()
                
                # Theta KL (normal prior: N(0, 1))
                theta_dist = Normal(theta_mean, theta_std)
                theta_prior = Normal(torch.zeros_like(theta_mean), torch.ones_like(theta_std))
                theta_kl = kl_divergence(theta_dist, theta_prior).sum()
                
                total_kl += (alpha_kl + beta_kl + theta_kl)
        
        probabilities = torch.stack(probabilities, dim=1)  # [batch_size, seq_len, n_categories]
        
        # Apply KL annealing
        kl_annealing_factor = min(1.0, self.current_epoch / self.kl_warmup_epochs) if self.kl_warmup_epochs > 0 else 1.0
        effective_kl_weight = self.kl_weight * kl_annealing_factor
        
        aux_dict = {
            'kl_divergence': total_kl * effective_kl_weight,
            'raw_kl_divergence': total_kl,
            'kl_annealing_factor': kl_annealing_factor
        }
        
        if return_params:
            # Return final parameters for analysis
            aux_dict.update({
                'alpha_mean': all_alpha_means[-1],
                'alpha_log_var': all_alpha_log_vars[-1],
                'beta_mean': all_beta_means[-1],
                'beta_log_var': all_beta_log_vars[-1],
                'all_alpha_means': torch.stack(all_alpha_means, dim=1),
                'all_beta_means': torch.stack(all_beta_means, dim=1)
            })
        
        return probabilities, aux_dict
    
    def set_epoch(self, epoch: int):
        """Update current epoch for KL annealing."""
        self.current_epoch = epoch
    
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
        cum_logits = torch.zeros(batch_size, K, device=theta.device)
        
        for k in range(1, K):
            # Sum of alpha * (theta - beta_h) for h < k
            cum_sum = 0
            for h in range(k):
                cum_sum += alpha * (theta - beta[:, h])
            cum_logits[:, k] = cum_sum
        
        # Compute probabilities with numerical stability
        exp_logits = torch.exp(cum_logits - cum_logits.max(dim=1, keepdim=True)[0])
        probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        
        return probabilities
    
    def elbo_loss(self, probabilities: torch.Tensor, targets: torch.Tensor,
                  kl_divergence: torch.Tensor) -> torch.Tensor:
        """Compute ELBO loss."""
        batch_size, seq_len, n_categories = probabilities.shape
        
        # Negative log likelihood
        targets_long = targets.long()
        observed_probs = probabilities.gather(2, targets_long.unsqueeze(2)).squeeze(2)
        observed_probs = torch.clamp(observed_probs, min=1e-8, max=1.0)
        
        log_likelihood = torch.log(observed_probs).sum()
        nll = -log_likelihood / (batch_size * seq_len)
        
        # ELBO = NLL + KL/N
        total_observations = batch_size * seq_len
        elbo = nll + kl_divergence / total_observations
        
        return elbo
    
    def get_posterior_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get IRT parameter statistics for evaluation."""
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Sample parameters for all questions
            all_question_ids = torch.arange(self.n_questions, device=device)
            alpha_means, alpha_log_vars, beta_means, beta_log_vars = self.irt_encoder.encode_question_parameters(all_question_ids)
            
            # Generate some sample response sequences for theta estimation
            n_samples = min(50, self.n_students)
            sample_responses = torch.randint(0, self.n_categories, (n_samples, 10), device=device)
            sample_questions = torch.randint(0, self.n_questions, (n_samples, 10), device=device)
            
            theta_means, theta_log_vars = self.irt_encoder.encode_student_abilities(sample_responses, sample_questions)
            
            return {
                'theta': {
                    'mean': theta_means.squeeze(-1),
                    'std': torch.exp(0.5 * theta_log_vars.squeeze(-1)),
                },
                'alpha': {
                    'mean': torch.exp(alpha_means.squeeze(-1)),  # Convert from log space
                    'std': torch.exp(0.5 * alpha_log_vars.squeeze(-1)),
                },
                'beta': {
                    'mean': beta_means,
                    'std': torch.exp(0.5 * beta_log_vars),
                }
            }