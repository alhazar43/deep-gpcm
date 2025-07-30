# Implementation Plan: Temporal Attention-Memory Fusion (TAM)

## Overview

This document provides a concrete implementation plan for the Temporal Attention-Memory Fusion (TAM) architecture, identified as the most promising solution for achieving 5-7% performance improvement over baseline.

## Theoretical Justification

TAM is based on cognitive science research showing that human working memory and attention systems are deeply intertwined:
- Attention gates what enters working memory
- Working memory content guides attention allocation
- Both systems co-evolve during learning

## Detailed Architecture

### Core Components

```python
class TemporalAttentionMemoryFusion(nn.Module):
    """
    TAM: True co-evolution of attention and memory during sequential processing.
    
    Key innovations:
    1. Dual memory systems (episodic + working)
    2. Bidirectional cross-attention
    3. Uncertainty-guided processing
    4. Temporal consistency regularization
    """
    
    def __init__(self, n_questions, n_cats, embed_dim=64, memory_size=50,
                 key_dim=50, value_dim=200, working_memory_dim=128,
                 n_heads=8, dropout=0.1):
        super().__init__()
        
        # Embedding layers
        self.embedding = LinearDecayEmbedding(n_questions, n_cats, embed_dim)
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        
        # Dual memory systems
        self.episodic_memory = DKVMN(memory_size, key_dim, value_dim)
        self.working_memory = nn.LSTMCell(
            input_size=embed_dim + value_dim,
            hidden_size=working_memory_dim
        )
        
        # Cross-attention modules
        self.memory_to_attention = nn.MultiheadAttention(
            embed_dim=value_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_to_memory = nn.MultiheadAttention(
            embed_dim=working_memory_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion networks
        self.embed_fusion = nn.Sequential(
            nn.Linear(embed_dim + working_memory_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.memory_fusion = nn.Sequential(
            nn.Linear(value_dim + working_memory_dim, value_dim),
            nn.LayerNorm(value_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(value_dim + working_memory_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # IRT parameter extraction (enhanced)
        self.irt_extractor = EnhancedIRTExtractor(
            input_dim=value_dim + working_memory_dim,
            n_cats=n_cats,
            ability_scale=2.0
        )
        
        # GPCM layer
        self.gpcm_layer = GPCMProbabilityLayer()
```

### Forward Pass Implementation

```python
def forward(self, questions, responses):
    batch_size, seq_len = questions.shape
    device = questions.device
    
    # Initialize memories
    self.episodic_memory.init_value_memory(batch_size)
    h_t = torch.zeros(batch_size, self.working_memory_dim).to(device)
    c_t = torch.zeros(batch_size, self.working_memory_dim).to(device)
    
    # Create embeddings
    gpcm_embeds = self.embedding(questions, responses)
    q_embeds = self.q_embed(questions)
    
    # Storage for outputs
    abilities, thresholds, discriminations, probabilities = [], [], [], []
    attention_weights = []
    uncertainties = []
    
    for t in range(seq_len):
        # Current timestep embeddings
        gpcm_embed_t = gpcm_embeds[:, t, :]
        q_embed_t = q_embeds[:, t, :]
        
        # Step 1: Read from episodic memory
        correlation_weight = self.episodic_memory.attention(q_embed_t)
        episodic_read = self.episodic_memory.read(correlation_weight)
        
        # Step 2: Cross-attention from memory to working memory
        episodic_read_seq = episodic_read.unsqueeze(1)
        h_t_seq = h_t.unsqueeze(1)
        
        attended_memory, attn_weights = self.memory_to_attention(
            query=h_t_seq,
            key=episodic_read_seq,
            value=episodic_read_seq,
            need_weights=True
        )
        attended_memory = attended_memory.squeeze(1)
        attention_weights.append(attn_weights)
        
        # Step 3: Update working memory with attended information
        working_input = torch.cat([gpcm_embed_t, attended_memory], dim=-1)
        h_t, c_t = self.working_memory(working_input, (h_t, c_t))
        
        # Step 4: Estimate uncertainty
        uncertainty_input = torch.cat([episodic_read, h_t], dim=-1)
        uncertainty = self.uncertainty_net(uncertainty_input)
        uncertainties.append(uncertainty)
        
        # Step 5: Attention-guided embedding refinement
        if uncertainty.mean() > 0.5:  # High uncertainty: deep processing
            refined_embed = self.embed_fusion(
                torch.cat([gpcm_embed_t, h_t], dim=-1)
            )
        else:  # Low uncertainty: light processing
            refined_embed = gpcm_embed_t
        
        # Step 6: Generate memory write value
        write_value = self.memory_fusion(
            torch.cat([episodic_read, h_t], dim=-1)
        )
        
        # Step 7: Update episodic memory
        write_gate = 1.0 - uncertainty  # Less update when uncertain
        gated_write = write_gate * episodic_read + (1 - write_gate) * write_value
        self.episodic_memory.write(correlation_weight, gated_write)
        
        # Step 8: Extract IRT parameters
        irt_input = torch.cat([episodic_read, h_t], dim=-1)
        theta_t, alpha_t, betas_t = self.irt_extractor(irt_input, q_embed_t)
        
        # Step 9: Compute GPCM probabilities
        gpcm_prob_t = self.gpcm_layer(theta_t, alpha_t, betas_t)
        
        # Store outputs
        abilities.append(theta_t)
        thresholds.append(betas_t)
        discriminations.append(alpha_t)
        probabilities.append(gpcm_prob_t)
    
    # Stack outputs
    outputs = {
        'abilities': torch.stack(abilities, dim=1),
        'thresholds': torch.stack(thresholds, dim=1),
        'discriminations': torch.stack(discriminations, dim=1),
        'probabilities': torch.stack(probabilities, dim=1),
        'attention_weights': torch.stack(attention_weights, dim=1),
        'uncertainties': torch.stack(uncertainties, dim=1)
    }
    
    return outputs
```

### Training Strategy

```python
class TAMTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Multi-stage optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': model.embedding.parameters(), 'lr': 1e-3},
            {'params': model.episodic_memory.parameters(), 'lr': 1e-3},
            {'params': model.working_memory.parameters(), 'lr': 5e-4},
            {'params': model.memory_to_attention.parameters(), 'lr': 5e-4},
            {'params': model.attention_to_memory.parameters(), 'lr': 5e-4},
            {'params': model.irt_extractor.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-5)
        
        # Learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.consistency_loss = TemporalConsistencyLoss()
        self.uncertainty_loss = UncertaintyCalibrationLoss()
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        for questions, responses in train_loader:
            questions = questions.to(self.device)
            responses = responses.to(self.device)
            
            # Forward pass
            outputs = self.model(questions, responses)
            
            # Compute losses
            ce_loss = self.ce_loss(
                outputs['probabilities'].reshape(-1, self.model.n_cats),
                responses.reshape(-1)
            )
            
            consistency_loss = self.consistency_loss(
                outputs['abilities'],
                outputs['attention_weights']
            )
            
            uncertainty_loss = self.uncertainty_loss(
                outputs['uncertainties'],
                outputs['probabilities'],
                responses
            )
            
            # Combined loss with adaptive weighting
            if epoch < 10:
                # Early training: focus on basic prediction
                loss = ce_loss + 0.1 * consistency_loss
            else:
                # Later training: add uncertainty calibration
                loss = ce_loss + 0.2 * consistency_loss + 0.1 * uncertainty_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(train_loader)
```

## Implementation Timeline

### Week 1: Core Architecture
- [ ] Implement dual memory system
- [ ] Implement cross-attention modules
- [ ] Basic forward pass
- [ ] Unit tests

### Week 2: Training Infrastructure
- [ ] Custom loss functions
- [ ] Training loop with multi-stage optimization
- [ ] Validation metrics
- [ ] Checkpointing

### Week 3: Optimization & Debugging
- [ ] Profile performance bottlenecks
- [ ] Implement gradient checkpointing if needed
- [ ] Debug training instabilities
- [ ] Hyperparameter tuning

### Week 4: Evaluation & Analysis
- [ ] Comprehensive evaluation on multiple datasets
- [ ] Ablation studies
- [ ] Attention visualization
- [ ] Performance comparison

## Expected Challenges & Solutions

### Challenge 1: Training Instability
**Solution**: 
- Gradual complexity increase (curriculum learning)
- Proper initialization (Xavier/He)
- Gradient clipping and normalization

### Challenge 2: Memory Requirements
**Solution**:
- Gradient checkpointing for long sequences
- Mixed precision training
- Efficient attention implementation

### Challenge 3: Overfitting
**Solution**:
- Progressive dropout rates
- Data augmentation (sequence permutation)
- Regularization through consistency loss

## Evaluation Plan

### Metrics
1. **Primary**: Categorical accuracy, QWK, ordinal accuracy
2. **Secondary**: Training efficiency, inference speed
3. **Interpretability**: Attention coherence, uncertainty calibration

### Datasets
1. **Primary**: synthetic_OC (development)
2. **Validation**: assist2015, assist2017
3. **Test**: STATICS, EdNet

### Baselines
1. Current baseline GPCM
2. Current AKVMN implementations
3. State-of-the-art DKT models

## Success Criteria

### Minimum Viable Success
- 5% improvement over baseline on synthetic_OC
- Maintains performance across datasets
- Interpretable attention patterns

### Target Success
- 7-8% improvement over baseline
- Generalizes to all datasets
- Provides uncertainty estimates
- Attention patterns align with educational theory

### Stretch Goals
- 10%+ improvement
- Real-time inference capability
- Transfer learning to new domains

## Risk Mitigation

### Technical Risks
1. **Computational complexity**: O(n²) attention → Use local attention windows
2. **Memory scaling**: Linear memory growth → Implement memory pruning
3. **Gradient vanishing**: Deep architecture → Residual connections

### Research Risks
1. **No improvement**: Have simpler TAM variant ready
2. **Dataset-specific**: Validate early on multiple datasets
3. **Black box**: Maintain interpretability throughout

## Conclusion

The TAM architecture addresses the fundamental limitation of current AKVMN implementations by enabling true co-evolution of attention and memory. The implementation plan provides a clear path to achieving 5-7% performance improvement while maintaining interpretability and computational efficiency.