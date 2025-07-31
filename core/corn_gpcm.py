#!/usr/bin/env python3
"""
CORN-enhanced Deep GPCM model with superior categorical-ordinal balance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import DeepGPCM
from .corn_layer import CORNLayer, HybridOrdinalLoss, ProgressiveWeightScheduler


class CORNDeepGPCM(DeepGPCM):
    """CORN-enhanced Deep GPCM model.
    
    This model replaces the standard GPCM probability layer with CORN
    (Conditional Ordinal Regression for Neural Networks), which provides:
    - Better categorical accuracy than pure CORAL
    - Maintained ordinal consistency
    - Superior handling of class imbalance
    - No weight-sharing constraints for better expressivity
    """
    
    def __init__(self, n_questions: int, n_cats: int = 4, memory_size: int = 50,
                 key_dim: int = 50, value_dim: int = 200, final_fc_dim: int = 50,
                 embedding_strategy: str = "linear_decay", ability_scale: float = 1.0,
                 corn_dropout: float = 0.3, **kwargs):
        """
        Args:
            n_questions: Number of questions/items
            n_cats: Number of response categories  
            memory_size: Size of DKVMN memory
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
            final_fc_dim: Hidden dimension before CORN layer
            embedding_strategy: Response embedding strategy
            ability_scale: Scaling factor for student abilities
            corn_dropout: Dropout rate in CORN layer
        """
        super().__init__(n_questions, n_cats, memory_size, key_dim, value_dim,
                         final_fc_dim, embedding_strategy, ability_scale, **kwargs)
        
        # Replace GPCM probability layer with CORN layer
        self.probability_layer = CORNLayer(
            input_dim=final_fc_dim,
            num_classes=n_cats,
            dropout=corn_dropout,
            architecture=kwargs.get('corn_architecture', 'standard'),
            hidden_multiplier=kwargs.get('corn_hidden_multiplier', 0.5)
        )
        
        # Store CORN-specific logits for loss computation
        self.corn_logits = None
        
        # Initialize progressive weight scheduler
        self.weight_scheduler = None
        self.current_epoch = 0
    
    def set_weight_scheduler(self, total_epochs: int, **scheduler_kwargs):
        """Set up progressive weight scheduler for training."""
        self.weight_scheduler = ProgressiveWeightScheduler(
            total_epochs=total_epochs, **scheduler_kwargs
        )
    
    def update_epoch(self, epoch: int):
        """Update current epoch for weight scheduling."""
        self.current_epoch = epoch
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor):
        """Forward pass through CORN-enhanced Deep GPCM.
        
        Args:
            questions: Question indices [batch_size, seq_len]
            responses: Response categories [batch_size, seq_len]
            
        Returns:
            Tuple of (student_abilities, item_thresholds, discrimination_params, class_probabilities)
        """
        batch_size, seq_len = questions.shape
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Create embeddings
        gpcm_embeds = self.create_embeddings(questions, responses)  # (batch_size, seq_len, embed_dim)
        q_embeds = self.q_embed(questions)  # (batch_size, seq_len, key_dim)
        
        # Process embeddings (can be enhanced by subclasses)
        processed_embeds = self.process_embeddings(gpcm_embeds, q_embeds)
        
        # Sequential processing
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        corn_probs = []
        summary_vectors = []  # Collect summary vectors for CORN layer
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]  # (batch_size, key_dim)
            gpcm_embed_t = processed_embeds[:, t, :]  # (batch_size, embed_dim)
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            
            # Extract IRT parameters (for compatibility)
            theta_t, alpha_t, betas_t = self.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)  # (batch_size,)
            alpha_t = alpha_t.squeeze(1)  # (batch_size,)
            betas_t = betas_t.squeeze(1)  # (batch_size, K-1)
            
            # Store outputs and summary vector
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            summary_vectors.append(summary_vector)
            
            # Write to memory for next time step
            if t < seq_len - 1:
                self.memory.write(correlation_weight, value_embed_t)
        
        # Stack outputs
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        
        # Apply CORN layer to summary vectors
        summary_vectors = torch.stack(summary_vectors, dim=1)  # (batch_size, seq_len, final_fc_dim)
        final_features = summary_vectors.view(-1, summary_vectors.size(-1))
        
        # Get CORN probabilities
        class_probabilities = self.probability_layer(final_features)
        class_probabilities = class_probabilities.view(batch_size, seq_len, self.n_cats)
        
        # Store binary logits for CORN loss (if available)
        if hasattr(self.probability_layer, 'last_binary_logits'):
            self.corn_logits = self.probability_layer.last_binary_logits
            self.corn_logits = self.corn_logits.view(batch_size, seq_len, -1)
        
        return student_abilities, item_thresholds, discrimination_params, class_probabilities
    
    def get_corn_logits(self):
        """Get the last computed CORN binary logits for loss computation."""
        return self.corn_logits
    
    def compute_loss(self, logits, targets, mask=None, loss_type='hybrid'):
        """Compute CORN-optimized loss.
        
        Args:
            logits: Class probabilities from forward pass
            targets: True class labels
            mask: Optional sequence mask
            loss_type: 'ce', 'corn', 'emd', or 'hybrid'
        """
        if loss_type == 'hybrid':
            # Use hybrid loss with progressive weighting
            weights = {'ce': 0.5, 'emd': 0.3, 'corn': 0.2}
            
            if self.weight_scheduler and self.training:
                scheduled_weights = self.weight_scheduler.get_weights(self.current_epoch)
                # Adjust weights based on schedule
                weights['ce'] = scheduled_weights['categorical'] * 0.6
                weights['emd'] = scheduled_weights['ordinal'] * 0.5
                weights['corn'] = scheduled_weights['ordinal'] * 0.5
            
            loss_fn = HybridOrdinalLoss(
                num_classes=self.n_cats,
                ce_weight=weights['ce'],
                emd_weight=weights['emd'],
                corn_weight=weights['corn']
            )
            
            # Need both class probs and CORN logits for hybrid loss
            corn_logits = self.get_corn_logits()
            if corn_logits is None:
                # Fallback to CE loss if CORN logits not available
                return F.cross_entropy(torch.log(logits + 1e-8), targets)
            
            loss_dict = loss_fn(corn_logits, logits, targets, mask)
            return loss_dict['total_loss']
        
        elif loss_type == 'ce':
            # Standard cross-entropy
            log_probs = torch.log(logits + 1e-8)
            if mask is not None:
                loss = F.cross_entropy(log_probs.view(-1, self.n_cats), 
                                     targets.view(-1), reduction='none')
                loss = (loss * mask.view(-1).float()).sum() / mask.float().sum().clamp(min=1)
            else:
                loss = F.cross_entropy(log_probs.view(-1, self.n_cats), targets.view(-1))
            return loss
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class AdaptiveCORNGPCM(CORNDeepGPCM):
    """Adaptive CORN model with uncertainty-based weight adjustment."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(kwargs.get('final_fc_dim', 50), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor):
        """Forward pass with uncertainty estimation."""
        batch_size, seq_len = questions.shape
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Create embeddings
        gpcm_embeds = self.create_embeddings(questions, responses)  # (batch_size, seq_len, embed_dim)
        q_embeds = self.q_embed(questions)  # (batch_size, seq_len, key_dim)
        
        # Process embeddings (can be enhanced by subclasses)
        processed_embeds = self.process_embeddings(gpcm_embeds, q_embeds)
        
        # Sequential processing
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        summary_vectors = []  # Collect summary vectors for CORN layer
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]  # (batch_size, key_dim)
            gpcm_embed_t = processed_embeds[:, t, :]  # (batch_size, embed_dim)
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            
            # Extract IRT parameters (for compatibility)
            theta_t, alpha_t, betas_t = self.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)  # (batch_size,)
            alpha_t = alpha_t.squeeze(1)  # (batch_size,)
            betas_t = betas_t.squeeze(1)  # (batch_size, K-1)
            
            # Store outputs and summary vector
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            summary_vectors.append(summary_vector)
            
            # Write to memory for next time step
            if t < seq_len - 1:
                self.memory.write(correlation_weight, value_embed_t)
        
        # Stack outputs
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        
        # Apply CORN layer to summary vectors
        summary_vectors = torch.stack(summary_vectors, dim=1)  # (batch_size, seq_len, final_fc_dim)
        final_features = summary_vectors.view(-1, summary_vectors.size(-1))
        
        # Get CORN probabilities
        class_probabilities = self.probability_layer(final_features)
        class_probabilities = class_probabilities.view(batch_size, seq_len, self.n_cats)
        
        # Store binary logits for CORN loss (if available)
        if hasattr(self.probability_layer, 'last_binary_logits'):
            self.corn_logits = self.probability_layer.last_binary_logits
            self.corn_logits = self.corn_logits.view(batch_size, seq_len, -1)
        
        # Estimate prediction uncertainty using final features
        uncertainty = self.uncertainty_head(final_features)
        uncertainty = uncertainty.view(batch_size, seq_len, 1)
        
        # Store uncertainty for adaptive weighting
        self.last_uncertainty = uncertainty
        
        return student_abilities, item_thresholds, discrimination_params, class_probabilities
    
    def get_adaptive_weights(self, base_categorical: float = 0.6):
        """Get uncertainty-based adaptive weights."""
        if not hasattr(self, 'last_uncertainty') or self.last_uncertainty is None:
            return {'categorical': base_categorical, 'ordinal': 1.0 - base_categorical}
        
        # Higher uncertainty → more emphasis on ordinal consistency
        uncertainty = self.last_uncertainty.mean()
        ordinal_weight = base_categorical + (uncertainty * 0.4)
        categorical_weight = 1.0 - ordinal_weight
        
        return {
            'categorical': categorical_weight.item(),
            'ordinal': ordinal_weight.item(),
            'uncertainty': uncertainty.item()
        }


class MultiTaskCORNGPCM(DeepGPCM):
    """Multi-task CORN model with separate categorical and ordinal heads."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, memory_size: int = 50,
                 key_dim: int = 50, value_dim: int = 200, final_fc_dim: int = 50,
                 shared_dim: int = 256, **kwargs):
        super().__init__(n_questions, n_cats, memory_size, key_dim, value_dim,
                         final_fc_dim, **kwargs)
        
        # Shared representation encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(final_fc_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.categorical_head = nn.Linear(shared_dim, n_cats)
        self.ordinal_head = CORNLayer(shared_dim, n_cats)
        self.consistency_head = nn.Linear(shared_dim, 1)
        
        self.shared_dim = shared_dim
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor):
        """Multi-task forward pass."""
        batch_size, seq_len = questions.shape
        
        # Initialize memory
        self.memory.init_value_memory(batch_size, self.init_value_memory)
        
        # Create embeddings
        gpcm_embeds = self.create_embeddings(questions, responses)  # (batch_size, seq_len, embed_dim)
        q_embeds = self.q_embed(questions)  # (batch_size, seq_len, key_dim)
        
        # Process embeddings (can be enhanced by subclasses)
        processed_embeds = self.process_embeddings(gpcm_embeds, q_embeds)
        
        # Sequential processing
        student_abilities = []
        item_thresholds = []
        discrimination_params = []
        summary_vectors = []  # Collect summary vectors for multitask heads
        
        for t in range(seq_len):
            # Current embeddings
            q_embed_t = q_embeds[:, t, :]  # (batch_size, key_dim)
            gpcm_embed_t = processed_embeds[:, t, :]  # (batch_size, embed_dim)
            
            # Transform to value dimension
            value_embed_t = self.gpcm_value_embed(gpcm_embed_t)  # (batch_size, value_dim)
            
            # Memory operations
            correlation_weight = self.memory.attention(q_embed_t)
            read_content = self.memory.read(correlation_weight)
            
            # Create summary vector
            summary_input = torch.cat([read_content, q_embed_t], dim=-1)
            summary_vector = self.summary_network(summary_input)
            
            # Extract IRT parameters (for compatibility)
            theta_t, alpha_t, betas_t = self.irt_extractor(
                summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
            )
            theta_t = theta_t.squeeze(1)  # (batch_size,)
            alpha_t = alpha_t.squeeze(1)  # (batch_size,)
            betas_t = betas_t.squeeze(1)  # (batch_size, K-1)
            
            # Store outputs and summary vector
            student_abilities.append(theta_t)
            item_thresholds.append(betas_t)
            discrimination_params.append(alpha_t)
            summary_vectors.append(summary_vector)
            
            # Write to memory for next time step
            if t < seq_len - 1:
                self.memory.write(correlation_weight, value_embed_t)
        
        # Stack outputs
        student_abilities = torch.stack(student_abilities, dim=1)  # (batch_size, seq_len)
        item_thresholds = torch.stack(item_thresholds, dim=1)  # (batch_size, seq_len, K-1)
        discrimination_params = torch.stack(discrimination_params, dim=1)  # (batch_size, seq_len)
        
        # Apply shared encoder to summary vectors
        summary_vectors = torch.stack(summary_vectors, dim=1)  # (batch_size, seq_len, final_fc_dim)
        shared_features = self.shared_encoder(summary_vectors.view(-1, summary_vectors.size(-1)))
        shared_features = shared_features.view(batch_size, seq_len, self.shared_dim)
        
        # Task-specific predictions
        categorical_logits = self.categorical_head(shared_features)
        ordinal_probs = self.ordinal_head(shared_features.view(-1, self.shared_dim))
        ordinal_probs = ordinal_probs.view(batch_size, seq_len, self.n_cats)
        consistency_scores = torch.sigmoid(self.consistency_head(shared_features))
        
        # Combine predictions based on consistency
        alpha = consistency_scores
        combined_probs = alpha * F.softmax(categorical_logits, dim=-1) + (1 - alpha) * ordinal_probs
        
        # Store task-specific outputs
        self.categorical_logits = categorical_logits
        self.ordinal_probs = ordinal_probs
        self.consistency_scores = consistency_scores
        
        return student_abilities, item_thresholds, discrimination_params, combined_probs
    
    def compute_multitask_loss(self, targets, mask=None, 
                              categorical_weight=0.4, ordinal_weight=0.4, consistency_weight=0.2):
        """Compute multi-task loss."""
        if not hasattr(self, 'categorical_logits'):
            raise RuntimeError("Must call forward() before computing multi-task loss")
        
        # Categorical task loss
        cat_loss = F.cross_entropy(
            self.categorical_logits.view(-1, self.n_cats),
            targets.view(-1),
            reduction='none'
        )
        
        # Ordinal task loss (using CORN probabilities)
        ord_loss = F.cross_entropy(
            torch.log(self.ordinal_probs.view(-1, self.n_cats) + 1e-8),
            targets.view(-1),
            reduction='none'
        )
        
        # Consistency regularization
        cat_probs = F.softmax(self.categorical_logits, dim=-1)
        consistency_loss = F.mse_loss(
            cat_probs.view(-1, self.n_cats),
            self.ordinal_probs.view(-1, self.n_cats),
            reduction='none'
        ).mean(dim=-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1).float()
            cat_loss = (cat_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            ord_loss = (ord_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            consistency_loss = (consistency_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        else:
            cat_loss = cat_loss.mean()
            ord_loss = ord_loss.mean()
            consistency_loss = consistency_loss.mean()
        
        # Combined loss
        total_loss = (categorical_weight * cat_loss + 
                     ordinal_weight * ord_loss + 
                     consistency_weight * consistency_loss)
        
        return {
            'total_loss': total_loss,
            'categorical_loss': cat_loss,
            'ordinal_loss': ord_loss,
            'consistency_loss': consistency_loss
        }