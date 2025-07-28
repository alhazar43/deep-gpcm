# IRT Prior Modification Plan for Deep-GPCM

## Executive Summary

This plan addresses the disconnect between synthetic data generation parameters and learned model parameters by incorporating IRT priors into the Deep-GPCM architecture and creating comprehensive evaluation metrics.

## Current State Analysis

### 1. Data Generation Priors (data_gen.py)
```python
# Student abilities (theta): N(0, 1)
self.theta = np.random.normal(0, 1, n_students)

# Item discrimination (alpha): LogNormal(0, 0.3)  
self.alpha = np.random.lognormal(0, 0.3, n_questions)

# Item thresholds (beta): Ordered thresholds with base difficulty N(0, 1)
base_diff = np.random.normal(0, 1)
thresh = np.sort(np.random.normal(base_diff, 0.5, n_cats - 1))
```

### 2. Model Learning Issues
- **Theta Range Mismatch**: Generated θ ~ N(0,1), but learned θ ∈ [-7.40, 13.79] (baseline) or [-14.50, 11.91] (AKVMN)
- **Alpha Distribution**: Generated α ~ LogNormal(0, 0.3), but learned α shows different distribution
- **Beta Ordering**: No guarantee that learned thresholds maintain proper ordering
- **No Prior Knowledge**: Model has no awareness of expected parameter distributions

## Proposed Modifications

### Phase 1: Model Architecture Enhancements

#### IMPORTANT NOTE FOR IMPLEMENTATION
**Create a new file `models/baseline_bayesian.py`** - Do NOT modify the existing baseline.py. This ensures we can compare Bayesian vs non-Bayesian approaches cleanly.

#### 1.0 Research SOTA Bayesian Neural IRT Methods

Before implementing, research these advanced approaches:

1. **VTIRT (Variational Temporal IRT)**
   - Key idea: Model temporal dynamics of IRT parameters using variational inference
   - Captures time-varying student abilities and item characteristics
   - Uses structured variational distributions to maintain temporal coherence
   - Applications: Adaptive testing, learning curve modeling

2. **VIBO (Variational Inference Bayesian Optimization)**
   - Combines variational inference with Bayesian optimization
   - Optimizes hyperparameters while maintaining uncertainty estimates
   - Efficient exploration-exploitation tradeoff
   - Applications: Automated hyperparameter tuning for IRT models

3. **Bayesian Knowledge Tracing (BKT)**
   - Classic approach: Hidden Markov Model for skill mastery
   - Modern variants: Deep BKT, Variational BKT
   - Key papers:
     - "Bayesian Knowledge Tracing" (Corbett & Anderson, 1995)
     - "Deep Bayesian Knowledge Tracing" (Ghosh et al., 2020)
     - "Variational Bayesian Knowledge Tracing" (Wang et al., 2021)

4. **BOBCAT (Bayesian OBservation model for Competence Assessment and Tracking)**
   - Hierarchical Bayesian model for student assessment
   - Combines IRT with learning dynamics
   - Accounts for forgetting and skill dependencies

5. **Additional Methods to Research**:
   - **SPARFA** (Sparse Factor Analysis): Bayesian factor analysis for educational data
   - **Deep-IRT**: Neural network enhanced IRT with uncertainty quantification
   - **BayesGrad**: Bayesian gradient-based optimization for IRT
   - **MCMC-IRT**: Markov Chain Monte Carlo methods for IRT
   - **VI-HMM**: Variational Inference for Hidden Markov Models in education

#### 1.1 Variational Bayesian Deep-GPCM Model

Instead of simple prior regularization, implement proper variational inference:

```python
# In models/baseline_bayesian.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, kl_divergence
import numpy as np

class VariationalBayesianGPCM(nn.Module):
    """
    Variational Bayesian Deep-GPCM with proper posterior inference.
    
    Key differences from naive approach:
    1. Learn posterior distributions q(θ|x), q(α|x), q(β|x) instead of point estimates
    2. Use reparameterization trick for gradient propagation
    3. Hierarchical Bayesian structure with global and local parameters
    4. Proper ELBO optimization instead of simple regularization
    """
    
    def __init__(self, n_questions, n_cats=4, n_students=None,
                 # Prior hyperparameters
                 theta_prior_mean=0.0, theta_prior_std=1.0,
                 alpha_prior_mean=0.0, alpha_prior_std=0.3,
                 beta_prior_base_std=1.0,
                 # Variational parameters
                 posterior_hidden_dim=100,
                 n_samples=5,  # Monte Carlo samples for ELBO
                 kl_annealing_epochs=10,
                 **kwargs):
        super().__init__()
        
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.n_students = n_students
        self.n_samples = n_samples
        self.kl_annealing_epochs = kl_annealing_epochs
        
        # Prior distributions (fixed)
        self.register_buffer('theta_prior_loc', torch.tensor(theta_prior_mean))
        self.register_buffer('theta_prior_scale', torch.tensor(theta_prior_std))
        self.register_buffer('log_alpha_prior_loc', torch.tensor(alpha_prior_mean))
        self.register_buffer('log_alpha_prior_scale', torch.tensor(alpha_prior_std))
        self.register_buffer('beta_prior_scale', torch.tensor(beta_prior_base_std))
        
        # Variational posterior networks
        self.build_variational_networks(**kwargs)
        
        # Base DKVMN components (reuse from baseline)
        self.build_memory_networks(**kwargs)
        
    def build_variational_networks(self, key_dim=50, value_dim=200, 
                                   posterior_hidden_dim=100, **kwargs):
        """Build networks for variational posteriors"""
        
        # Student ability posterior q(θ_i | history_i)
        # Uses amortized inference: condition on student's response history
        self.theta_posterior_net = nn.Sequential(
            nn.Linear(value_dim + key_dim, posterior_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(posterior_hidden_dim, posterior_hidden_dim),
            nn.ReLU(),
            nn.Linear(posterior_hidden_dim, 2)  # mean and log_std
        )
        
        # Item discrimination posterior q(α_j | item_features_j)
        # Hierarchical: global prior + item-specific variation
        self.alpha_posterior_net = nn.ModuleDict({
            'global': nn.Sequential(
                nn.Linear(key_dim, posterior_hidden_dim),
                nn.ReLU(),
                nn.Linear(posterior_hidden_dim, 2)  # global mean and log_std
            ),
            'local': nn.Embedding(n_questions + 1, 2)  # item-specific offset
        })
        
        # Item threshold posterior q(β_j | item_features_j)
        self.beta_posterior_net = nn.Sequential(
            nn.Linear(key_dim, posterior_hidden_dim),
            nn.ReLU(),
            nn.Linear(posterior_hidden_dim, 2 * (self.n_cats - 1))  # mean and log_std for each threshold
        )
        
    def reparameterize(self, loc, scale, dist_type='normal'):
        """Reparameterization trick for gradient propagation"""
        if self.training:
            if dist_type == 'normal':
                eps = torch.randn_like(scale)
                return loc + eps * scale
            elif dist_type == 'lognormal':
                normal_sample = loc + torch.randn_like(scale) * scale
                return torch.exp(normal_sample)
        else:
            # During evaluation, use mean
            return loc if dist_type == 'normal' else torch.exp(loc)
    
    def compute_variational_parameters(self, summary_vector, q_embed, q_id):
        """Compute variational posterior parameters"""
        
        # Theta posterior (student ability)
        theta_features = torch.cat([summary_vector, q_embed], dim=-1)
        theta_params = self.theta_posterior_net(theta_features)
        theta_loc, theta_log_scale = theta_params.chunk(2, dim=-1)
        theta_scale = F.softplus(theta_log_scale) + 1e-4
        
        # Alpha posterior (discrimination) - hierarchical
        global_alpha_params = self.alpha_posterior_net['global'](q_embed)
        global_loc, global_log_scale = global_alpha_params.chunk(2, dim=-1)
        
        local_offset = self.alpha_posterior_net['local'](q_id)
        local_loc_offset, local_log_scale = local_offset.chunk(2, dim=-1)
        
        # Combine global and local
        alpha_loc = global_loc + local_loc_offset
        alpha_scale = F.softplus(global_log_scale + local_log_scale) + 1e-4
        
        # Beta posterior (thresholds)
        beta_params = self.beta_posterior_net(q_embed)
        beta_loc, beta_log_scale = beta_params.chunk(2, dim=-1)
        beta_scale = F.softplus(beta_log_scale) + 1e-4
        
        # Reshape beta parameters
        beta_loc = beta_loc.view(-1, self.n_cats - 1)
        beta_scale = beta_scale.view(-1, self.n_cats - 1)
        
        return {
            'theta': (theta_loc, theta_scale),
            'alpha': (alpha_loc, alpha_scale), 
            'beta': (beta_loc, beta_scale)
        }
    
    def sample_parameters(self, posterior_params, n_samples=None):
        """Sample from variational posteriors"""
        if n_samples is None:
            n_samples = self.n_samples
            
        theta_loc, theta_scale = posterior_params['theta']
        alpha_loc, alpha_scale = posterior_params['alpha']
        beta_loc, beta_scale = posterior_params['beta']
        
        # Sample multiple times for Monte Carlo ELBO
        theta_samples = []
        alpha_samples = []
        beta_samples = []
        
        for _ in range(n_samples):
            # Sample theta
            theta = self.reparameterize(theta_loc, theta_scale, 'normal')
            theta_samples.append(theta)
            
            # Sample alpha (ensure positive via lognormal)
            alpha = self.reparameterize(alpha_loc, alpha_scale, 'lognormal')
            alpha_samples.append(alpha)
            
            # Sample beta with ordering constraint
            beta_raw = self.reparameterize(beta_loc, beta_scale, 'normal')
            beta_ordered = self.enforce_threshold_ordering(beta_raw)
            beta_samples.append(beta_ordered)
            
        return {
            'theta': torch.stack(theta_samples),  # [n_samples, batch_size]
            'alpha': torch.stack(alpha_samples),
            'beta': torch.stack(beta_samples)     # [n_samples, batch_size, n_cats-1]
        }
    
    def compute_kl_divergence(self, posterior_params, epoch=0):
        """Compute KL divergence between posterior and prior"""
        
        # KL annealing for better optimization
        annealing_weight = min(1.0, epoch / self.kl_annealing_epochs)
        
        # Theta KL
        theta_loc, theta_scale = posterior_params['theta']
        theta_posterior = Normal(theta_loc, theta_scale)
        theta_prior = Normal(self.theta_prior_loc, self.theta_prior_scale)
        theta_kl = kl_divergence(theta_posterior, theta_prior).sum()
        
        # Alpha KL (log-normal)
        alpha_loc, alpha_scale = posterior_params['alpha']
        alpha_posterior = Normal(alpha_loc, alpha_scale)  # In log space
        alpha_prior = Normal(self.log_alpha_prior_loc, self.log_alpha_prior_scale)
        alpha_kl = kl_divergence(alpha_posterior, alpha_prior).sum()
        
        # Beta KL
        beta_loc, beta_scale = posterior_params['beta']
        beta_posterior = Normal(beta_loc.flatten(), beta_scale.flatten())
        beta_prior = Normal(torch.zeros_like(beta_loc.flatten()), 
                           self.beta_prior_scale.expand_as(beta_scale.flatten()))
        beta_kl = kl_divergence(beta_posterior, beta_prior).sum()
        
        total_kl = annealing_weight * (theta_kl + alpha_kl + beta_kl)
        
        return total_kl, {
            'theta_kl': theta_kl.item(),
            'alpha_kl': alpha_kl.item(),
            'beta_kl': beta_kl.item(),
            'annealing_weight': annealing_weight
        }
    
    def forward(self, q_data, r_data, return_kl=True, epoch=0):
        """Forward pass with variational inference"""
        batch_size, seq_len = q_data.shape
        
        # ... (memory network operations as in baseline) ...
        
        # Collect outputs
        all_probs = []
        all_kl = 0
        all_posterior_params = []
        
        for t in range(seq_len):
            # Get current question
            q_t = q_data[:, t]
            q_embed_t = self.q_embed(q_t)
            
            # ... (memory read operations) ...
            
            # Compute variational parameters
            posterior_params = self.compute_variational_parameters(
                summary_vector, q_embed_t, q_t
            )
            all_posterior_params.append(posterior_params)
            
            # Sample parameters
            param_samples = self.sample_parameters(posterior_params)
            
            # Compute GPCM probabilities for each sample
            probs_samples = []
            for s in range(self.n_samples):
                theta_s = param_samples['theta'][s]
                alpha_s = param_samples['alpha'][s]
                beta_s = param_samples['beta'][s]
                
                # GPCM probability
                probs_s = self.gpcm_probability(theta_s, alpha_s, beta_s)
                probs_samples.append(probs_s)
            
            # Average over samples
            probs_t = torch.stack(probs_samples).mean(0)
            all_probs.append(probs_t)
            
            # Accumulate KL divergence
            if return_kl:
                kl_t, kl_metrics = self.compute_kl_divergence(posterior_params, epoch)
                all_kl += kl_t
                
            # ... (memory write operations) ...
        
        # Stack outputs
        probs = torch.stack(all_probs, dim=1)
        
        if return_kl:
            return probs, all_kl / (batch_size * seq_len), all_posterior_params
        else:
            return probs

class VariationalGPCMLoss(nn.Module):
    """ELBO loss for Variational Bayesian GPCM"""
    
    def __init__(self, n_students, n_items):
        super().__init__()
        self.n_students = n_students
        self.n_items = n_items
        
    def forward(self, probs, targets, kl_div, kl_weight=1.0):
        """
        Compute ELBO = E_q[log p(y|θ,α,β)] - KL[q||p]
        """
        # Likelihood term (expected log probability)
        log_likelihood = F.cross_entropy(
            probs.view(-1, probs.size(-1)), 
            targets.view(-1),
            reduction='sum'
        )
        
        # ELBO (maximize ELBO = minimize negative ELBO)
        elbo = -log_likelihood + kl_weight * kl_div
        
        # Normalize by total observations
        n_observations = targets.numel()
        normalized_elbo = elbo / n_observations
        
        return normalized_elbo, {
            'log_likelihood': -log_likelihood.item() / n_observations,
            'kl_div': kl_div.item() / n_observations,
            'elbo': -normalized_elbo.item()
        }
```

#### 1.2 Advanced Techniques to Implement

1. **Hierarchical Variational Inference**
   - Global hyperpriors on population parameters
   - Local posteriors for individual students/items
   - Proper shrinkage and regularization

2. **Normalizing Flows for Flexible Posteriors**
   - Instead of assuming Gaussian posteriors
   - Learn complex posterior distributions
   - Better capture multi-modal ability distributions

3. **Variational Information Bottleneck**
   - Compress student history into minimal sufficient statistics
   - Preserve only task-relevant information
   - Improve generalization

4. **Uncertainty Quantification**
   - Epistemic uncertainty from parameter posteriors
   - Aleatoric uncertainty from response variability
   - Calibrated confidence intervals

#### 1.3 Research-Based Advanced Approaches

**Note to Future Self**: Don't just implement naive MAP or simple regularization. Consider these SOTA approaches:

1. **VTIRT (Variational Temporal IRT) Implementation**
   ```python
   class VTIRT(nn.Module):
       """
       Variational Temporal IRT - Models time-varying IRT parameters
       Key innovation: Temporal coherence in variational posteriors
       """
       def __init__(self, n_questions, n_students, n_timesteps):
           super().__init__()
           
           # Temporal prior: parameters evolve smoothly over time
           self.temporal_prior = nn.GRU(
               input_size=hidden_dim,
               hidden_size=hidden_dim,
               num_layers=2
           )
           
           # Variational posterior with temporal structure
           self.theta_posterior_rnn = nn.LSTM(
               input_size=feature_dim,
               hidden_size=hidden_dim,
               num_layers=2,
               bidirectional=True  # Use future information too
           )
           
       def temporal_kl_divergence(self, params_seq):
           """KL with temporal smoothness prior"""
           # Standard KL
           kl_standard = self.standard_kl(params_seq)
           
           # Temporal smoothness penalty
           temporal_diff = params_seq[1:] - params_seq[:-1]
           smoothness_penalty = torch.mean(temporal_diff ** 2)
           
           return kl_standard + self.lambda_smooth * smoothness_penalty
           
       def forward(self, student_sequences):
           # Process entire sequence jointly
           theta_seq, _ = self.theta_posterior_rnn(student_sequences)
           
           # Sample from structured posterior
           theta_samples = self.structured_sample(theta_seq)
           
           return theta_samples
   ```

2. **Bayesian Knowledge Tracing with Deep Learning**
   ```python
   class DeepBKT(nn.Module):
       """
       Modern Bayesian Knowledge Tracing with neural networks
       Combines HMM structure with deep learning flexibility
       """
       def __init__(self):
           super().__init__()
           
           # Skill mastery states (latent)
           self.n_skills = n_skills
           
           # Transition model P(mastered_t | mastered_{t-1}, context)
           self.transition_net = nn.Sequential(
               nn.Linear(context_dim + 1, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, 2)  # [P(stay), P(learn)]
           )
           
           # Emission model P(correct | mastered, item_features)
           self.emission_net = nn.Sequential(
               nn.Linear(item_dim + 1, hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim, n_categories)
           )
           
       def forward_filter(self, observations, contexts):
           """Forward filtering for online inference"""
           T = len(observations)
           alphas = []
           
           # Initialize
           alpha_0 = self.prior_mastery
           alphas.append(alpha_0)
           
           for t in range(T):
               # Predict
               transition_probs = self.transition_net(contexts[t])
               alpha_predict = self.hmm_predict(alphas[-1], transition_probs)
               
               # Update
               emission_probs = self.emission_net(observations[t])
               alpha_t = self.hmm_update(alpha_predict, emission_probs)
               alphas.append(alpha_t)
               
           return alphas
   ```

2. **Neural Process IRT (NP-IRT)**
   ```python
   class NeuralProcessIRT(nn.Module):
       """
       Based on "Neural Processes" (Garnelo et al., 2018)
       Handles variable number of observations per student/item
       """
       def __init__(self):
           super().__init__()
           # Encoder: aggregate context information
           self.context_encoder = SetEncoder()  
           # Decoder: predict at target points
           self.decoder = IRTDecoder()
           
       def forward(self, context_x, context_y, target_x):
           # Encode context into global representation
           r = self.context_encoder(context_x, context_y)
           
           # Sample latent variable
           z = self.sample_latent(r)
           
           # Decode at target points
           predictions = self.decoder(target_x, z, r)
           return predictions
   ```

3. **Attentive Neural Process for IRT (ANP-IRT)**
   ```python
   class AttentiveNeuralProcessIRT(nn.Module):
       """
       Extension with attention mechanism for better context aggregation
       """
       def __init__(self):
           super().__init__()
           self.cross_attention = MultiHeadAttention(...)
           
       def forward(self, context, target):
           # Use attention to selectively aggregate relevant context
           attended_context = self.cross_attention(
               query=target_embedding,
               key=context_embedding,
               value=context_values
           )
           return self.decode(attended_context)
   ```

4. **Hierarchical Variational Models**
   ```python
   class HierarchicalBayesianIRT(nn.Module):
       """
       Multi-level model with population, group, and individual parameters
       """
       def __init__(self):
           super().__init__()
           
           # Population level
           self.population_prior = Normal(0, 1)
           
           # Group level (e.g., school, course)
           self.group_posterior = VariationalPosterior(
               prior=self.population_prior
           )
           
           # Individual level
           self.individual_posterior = VariationalPosterior(
               prior=lambda: self.group_posterior.sample()
           )
   ```

5. **Flow-Based IRT Models**
   ```python
   class NormalizingFlowIRT(nn.Module):
       """
       Use normalizing flows for flexible posterior distributions
       """
       def __init__(self, n_flows=10):
           super().__init__()
           self.flows = nn.ModuleList([
               PlanarFlow(dim) for _ in range(n_flows)
           ])
           
       def transform_posterior(self, z0):
           z = z0
           log_det_sum = 0
           for flow in self.flows:
               z, log_det = flow(z)
               log_det_sum += log_det
           return z, log_det_sum
   ```

#### 1.4 Key Papers to Research Before Implementation

**Future Self: Read these papers thoroughly before coding!**

1. **Bayesian Deep Learning for Education**
   - "Uncertainty-Aware Deep Knowledge Tracing" (2022)
   - "A Bayesian Approach to Knowledge Tracing with Item Response Theory" (2021)
   - "Variational Deep Knowledge Tracing for Language Learning" (2020)

2. **Variational Inference for IRT**
   - "Variational Item Response Theory: Fast, Accurate, and Expressive" (2020)
   - "Deep Generative Models for Student Modeling" (2019)
   - "Scalable Bayesian IRT Models" (2021)

3. **Information Bottleneck Methods**
   - "Learning Parsimonious Deep Representations" (2018)
   - "The Information Bottleneck Method" (Tishby et al., 2000)
   - "Deep Variational Information Bottleneck" (Alemi et al., 2017)

4. **Neural Process Family**
   - "Conditional Neural Processes" (2018)
   - "Attentive Neural Processes" (2019)
   - "Meta-Learning Probabilistic Inference for Prediction" (2019)

#### 1.5 Key Differences from Naive Approaches

**Why these methods are better than simple MAP/regularization:**

1. **Proper Uncertainty Quantification**
   - Naive: Point estimates with no uncertainty
   - SOTA: Full posterior distributions with calibrated uncertainty

2. **Temporal Coherence (VTIRT)**
   - Naive: Independent parameters at each timestep
   - VTIRT: Structured temporal dependencies, smooth evolution

3. **Hierarchical Structure**
   - Naive: Flat priors for all parameters
   - SOTA: Multi-level models with population/group/individual levels

4. **Flexible Posteriors**
   - Naive: Gaussian assumptions
   - SOTA: Normalizing flows, mixtures, non-parametric

5. **Information Efficiency**
   - Naive: Use all available information equally
   - VIBO: Learn minimal sufficient statistics

#### 1.6 Implementation Priority Order

**Start with these in order:**

1. **Basic Variational Bayesian GPCM** (baseline_bayesian.py)
   - Implement core variational inference with proper ELBO
   - Use reparameterization trick correctly
   - Verify parameter recovery on synthetic data
   - Compare with ground truth from data generation

2. **VTIRT Extensions**
   - Add temporal structure to posteriors
   - Implement smoothness priors
   - Test on sequences with known temporal dynamics

3. **Bayesian Knowledge Tracing Integration**
   - Combine IRT parameters with skill mastery modeling
   - Hidden Markov structure for learning dynamics
   - Compare with classic BKT baselines

4. **Advanced Inference Methods**
   - VIBO for information compression
   - Normalizing flows for flexible posteriors
   - Hierarchical variational models

5. **Evaluation Framework**
   - Parameter recovery metrics
   - Uncertainty calibration tests
   - Temporal consistency evaluation
   - Comparison with non-Bayesian baseline
        self.register_buffer('theta_prior_mean', torch.tensor(theta_prior_mean))
        self.register_buffer('theta_prior_std', torch.tensor(theta_prior_std))
        self.register_buffer('alpha_prior_mean', torch.tensor(alpha_prior_mean))
        self.register_buffer('alpha_prior_std', torch.tensor(alpha_prior_std))
        self.register_buffer('beta_prior_mean', torch.tensor(beta_prior_mean))
        self.register_buffer('beta_prior_std', torch.tensor(beta_prior_std))
        
        self.prior_weight = prior_weight
        
        # Base GPCM model
        self.gpcm_model = DeepGpcmModel(n_questions, n_cats, **kwargs)
        
        # Prior-aware parameter networks
        self.theta_regularizer = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Bound correction term
        )
        
        # Question-specific prior embeddings
        self.alpha_prior_embed = nn.Embedding(n_questions + 1, 8)
        self.beta_prior_embed = nn.Embedding(n_questions + 1, 8)
```

#### 1.2 Prior-Regularized Parameter Prediction
```python
def predict_parameters_with_priors(self, summary_vector, q_embed_t, q_id):
    # Base predictions
    theta_base = self.student_ability_network(summary_vector).squeeze(-1) * self.ability_scale
    alpha_base = self.discrimination_network(torch.cat([summary_vector, q_embed_t], dim=-1)).squeeze(-1)
    beta_base = self.question_threshold_network(q_embed_t)
    
    # Apply prior regularization
    # Theta: Pull towards N(0,1)
    theta_correction = self.theta_regularizer(theta_base.unsqueeze(-1)).squeeze(-1)
    theta = theta_base + self.prior_weight * theta_correction * (self.theta_prior_mean - theta_base)
    
    # Alpha: Ensure positive and pull towards LogNormal(0, 0.3)
    alpha_prior_feat = self.alpha_prior_embed(q_id)
    alpha_correction = F.softplus(self.alpha_corrector(torch.cat([alpha_base.unsqueeze(-1), alpha_prior_feat], dim=-1)))
    alpha = alpha_base * (1 - self.prior_weight) + alpha_correction * self.prior_weight
    
    # Beta: Ensure ordering and pull towards prior
    beta_prior_feat = self.beta_prior_embed(q_id)
    beta_correction = self.beta_corrector(torch.cat([beta_base, beta_prior_feat], dim=-1))
    beta = self.enforce_threshold_ordering(beta_base + self.prior_weight * beta_correction)
    
    return theta, alpha, beta

def enforce_threshold_ordering(self, betas):
    """Ensure beta_1 < beta_2 < ... < beta_{K-1}"""
    # Use cumulative sum of positive values
    beta_diffs = F.softplus(betas[..., 1:] - betas[..., :-1])
    ordered_betas = torch.cumsum(torch.cat([betas[..., :1], beta_diffs], dim=-1), dim=-1)
    return ordered_betas
```

### Phase 2: Prior-Aware Loss Functions

#### 2.1 IRT Prior Regularization Loss
```python
class IRTPriorLoss(nn.Module):
    def __init__(self, theta_prior=(0.0, 1.0), alpha_prior=(0.0, 0.3), 
                 beta_prior=(0.0, 1.0), weight_schedule='linear'):
        super().__init__()
        self.theta_mean, self.theta_std = theta_prior
        self.alpha_mean, self.alpha_std = alpha_prior  
        self.beta_mean, self.beta_std = beta_prior
        self.weight_schedule = weight_schedule
        
    def forward(self, theta, alpha, beta, epoch=0, max_epochs=100):
        # Adaptive weighting: start strong, decay over time
        if self.weight_schedule == 'linear':
            w = 1.0 - (epoch / max_epochs)
        elif self.weight_schedule == 'exponential':
            w = np.exp(-5 * epoch / max_epochs)
        else:
            w = 1.0
            
        # Theta prior: KL divergence from N(theta_mean, theta_std)
        theta_loss = 0.5 * torch.mean(
            ((theta - self.theta_mean) / self.theta_std) ** 2 + 
            torch.log(self.theta_std) - torch.log(torch.std(theta))
        )
        
        # Alpha prior: KL divergence from LogNormal(alpha_mean, alpha_std)
        log_alpha = torch.log(alpha + 1e-8)
        alpha_loss = 0.5 * torch.mean(
            ((log_alpha - self.alpha_mean) / self.alpha_std) ** 2 +
            torch.log(self.alpha_std) - torch.log(torch.std(log_alpha))
        )
        
        # Beta prior: Penalize deviation from expected distribution
        beta_mean_per_q = torch.mean(beta, dim=-1)  # Mean threshold per question
        beta_loss = 0.5 * torch.mean(
            ((beta_mean_per_q - self.beta_mean) / self.beta_std) ** 2
        )
        
        # Beta ordering penalty
        beta_diffs = beta[..., 1:] - beta[..., :-1]
        ordering_loss = torch.mean(F.relu(-beta_diffs))  # Penalize negative differences
        
        total_prior_loss = w * (theta_loss + alpha_loss + beta_loss + 10 * ordering_loss)
        
        return total_prior_loss, {
            'theta_loss': theta_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'beta_loss': beta_loss.item(),
            'ordering_loss': ordering_loss.item(),
            'prior_weight': w
        }
```

#### 2.2 Combined Training Loss
```python
class GPCMWithPriorLoss(nn.Module):
    def __init__(self, gpcm_weight=1.0, prior_weight=0.1):
        super().__init__()
        self.gpcm_loss = nn.CrossEntropyLoss()
        self.prior_loss = IRTPriorLoss()
        self.gpcm_weight = gpcm_weight
        self.prior_weight = prior_weight
        
    def forward(self, predictions, targets, theta, alpha, beta, epoch=0):
        # Standard GPCM loss
        gpcm_loss = self.gpcm_loss(predictions.view(-1, predictions.size(-1)), targets.view(-1))
        
        # Prior regularization
        prior_loss, prior_metrics = self.prior_loss(theta, alpha, beta, epoch)
        
        # Combined loss
        total_loss = self.gpcm_weight * gpcm_loss + self.prior_weight * prior_loss
        
        return total_loss, {
            'gpcm_loss': gpcm_loss.item(),
            'prior_loss': prior_loss.item(),
            **prior_metrics
        }
```

### Phase 3: Ground Truth Storage and Comparison

#### 3.1 Enhanced Data Generation with Parameter Storage
```python
def save_with_ground_truth_params(self, sequences, data_dir):
    """Save data with ground truth IRT parameters"""
    # ... existing save logic ...
    
    # Save ground truth parameters
    gt_params = {
        'theta': self.theta.tolist(),  # Student abilities
        'alpha': self.alpha.tolist(),  # Item discriminations
        'beta': self.beta.tolist(),     # Item thresholds
        'generation_params': {
            'theta_dist': 'N(0, 1)',
            'alpha_dist': 'LogNormal(0, 0.3)',
            'beta_dist': 'Ordered N(base_diff, 0.5)'
        }
    }
    
    np.savez(data_dir / 'ground_truth_params.npz', **gt_params)
```

#### 3.2 IRT Parameter Comparison Metrics
```python
class IRTParameterEvaluator:
    def __init__(self, ground_truth_path):
        self.gt_params = np.load(ground_truth_path)
        self.gt_theta = self.gt_params['theta']
        self.gt_alpha = self.gt_params['alpha']
        self.gt_beta = self.gt_params['beta']
        
    def evaluate_learned_parameters(self, learned_theta, learned_alpha, learned_beta):
        """Compare learned parameters to ground truth"""
        metrics = {}
        
        # Theta comparison (per student)
        # Note: Need to align student IDs
        theta_corr = np.corrcoef(self.gt_theta, learned_theta)[0, 1]
        theta_rmse = np.sqrt(np.mean((self.gt_theta - learned_theta) ** 2))
        theta_bias = np.mean(learned_theta) - np.mean(self.gt_theta)
        
        metrics['theta'] = {
            'correlation': theta_corr,
            'rmse': theta_rmse,
            'bias': bias,
            'gt_mean': np.mean(self.gt_theta),
            'gt_std': np.std(self.gt_theta),
            'learned_mean': np.mean(learned_theta),
            'learned_std': np.std(learned_theta)
        }
        
        # Alpha comparison (per question)
        alpha_corr = np.corrcoef(self.gt_alpha, learned_alpha)[0, 1]
        alpha_rmse = np.sqrt(np.mean((self.gt_alpha - learned_alpha) ** 2))
        
        metrics['alpha'] = {
            'correlation': alpha_corr,
            'rmse': alpha_rmse,
            'gt_mean': np.mean(self.gt_alpha),
            'learned_mean': np.mean(learned_alpha)
        }
        
        # Beta comparison (per question, per threshold)
        beta_corr_per_threshold = []
        for k in range(self.gt_beta.shape[1]):
            corr = np.corrcoef(self.gt_beta[:, k], learned_beta[:, k])[0, 1]
            beta_corr_per_threshold.append(corr)
            
        metrics['beta'] = {
            'correlation_per_threshold': beta_corr_per_threshold,
            'mean_correlation': np.mean(beta_corr_per_threshold),
            'ordering_preserved': self.check_threshold_ordering(learned_beta)
        }
        
        # Distribution comparison
        metrics['distribution_tests'] = {
            'theta_ks_test': scipy.stats.ks_2samp(self.gt_theta, learned_theta),
            'alpha_ks_test': scipy.stats.ks_2samp(self.gt_alpha, learned_alpha),
            'theta_normal_test': scipy.stats.normaltest(learned_theta),
            'log_alpha_normal_test': scipy.stats.normaltest(np.log(learned_alpha + 1e-8))
        }
        
        return metrics
```

### Phase 4: Visualization Enhancements

#### 4.1 Ground Truth vs Learned Parameter Plots
```python
def plot_parameter_comparison(gt_params, learned_params, output_dir):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Theta comparisons
    # Scatter plot
    axes[0, 0].scatter(gt_params['theta'], learned_params['theta'], alpha=0.5)
    axes[0, 0].plot([-3, 3], [-3, 3], 'r--')
    axes[0, 0].set_xlabel('Ground Truth θ')
    axes[0, 0].set_ylabel('Learned θ')
    axes[0, 0].set_title('Student Ability Comparison')
    
    # Distribution comparison
    axes[0, 1].hist(gt_params['theta'], bins=30, alpha=0.5, label='Ground Truth', density=True)
    axes[0, 1].hist(learned_params['theta'], bins=30, alpha=0.5, label='Learned', density=True)
    axes[0, 1].set_xlabel('θ')
    axes[0, 1].set_title('Ability Distribution')
    axes[0, 1].legend()
    
    # Q-Q plot
    scipy.stats.probplot(learned_params['theta'], dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Theta Q-Q Plot (vs Normal)')
    
    # Row 2: Alpha comparisons
    # Similar plots for discrimination parameters
    
    # Row 3: Beta comparisons  
    # Plot per threshold
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_comparison.png', dpi=300)
```

#### 4.2 Prior Influence Visualization
```python
def plot_prior_influence_over_training(training_history, output_dir):
    """Show how prior influence changes during training"""
    epochs = range(len(training_history))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Prior weight decay
    ax1.plot(epochs, [h['prior_weight'] for h in training_history])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Prior Weight')
    ax1.set_title('Prior Influence Decay')
    
    # Parameter convergence to priors
    ax2.plot(epochs, [h['theta_bias'] for h in training_history], label='θ bias')
    ax2.plot(epochs, [h['alpha_bias'] for h in training_history], label='α bias')
    ax2.plot(epochs, [h['beta_bias'] for h in training_history], label='β bias')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Bias from Prior Mean')
    ax2.set_title('Parameter Convergence')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prior_influence.png', dpi=300)
```

### Phase 5: Implementation Roadmap

#### 5.1 Short Term (Immediate)
1. **Save Ground Truth Parameters**: Modify data_gen.py to save IRT parameters
2. **Create Evaluation Metrics**: Implement IRTParameterEvaluator class
3. **Basic Comparison Plots**: Add ground truth vs learned visualization

#### 5.2 Medium Term (1-2 weeks)
1. **Prior-Aware Model**: Implement PriorAwareDeepGPCM
2. **Prior Loss Functions**: Add IRTPriorLoss and combined loss
3. **Training Integration**: Update training scripts to use priors

#### 5.3 Long Term (2-4 weeks)
1. **Hyperparameter Tuning**: Find optimal prior weights
2. **Alternative Prior Strategies**: Test different prior incorporation methods
3. **Extensive Evaluation**: Compare prior-aware vs baseline on multiple datasets

## Expected Benefits

1. **Parameter Interpretability**: Learned parameters match expected IRT distributions
2. **Improved Generalization**: Prior knowledge prevents overfitting
3. **Faster Convergence**: Starting with good priors speeds training
4. **Better Stability**: Bounded parameters prevent extreme values
5. **Validation Capability**: Can now validate if model truly learns IRT structure

## Evaluation Metrics

1. **Parameter Recovery**:
   - Correlation between ground truth and learned parameters
   - RMSE for each parameter type
   - Distribution similarity (KS test, Q-Q plots)

2. **Predictive Performance**:
   - Cross-entropy loss
   - Accuracy metrics
   - Out-of-distribution generalization

3. **Prior Influence**:
   - Prior weight schedule effectiveness
   - Convergence speed comparison
   - Parameter stability over training

## Conclusion

This modification plan addresses the fundamental disconnect between synthetic data generation and model learning by:
1. Making models aware of expected parameter distributions
2. Adding prior regularization to guide learning
3. Creating comprehensive evaluation metrics
4. Enabling direct comparison with ground truth parameters

The implementation maintains backward compatibility while adding powerful new capabilities for IRT parameter learning and evaluation.