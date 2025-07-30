# Co-evolutionary AKVMN Implementation Plan

## Overview

This document provides a concrete implementation plan for the Co-evolutionary Memory-Attention Architecture, the most promising solution to fix AKVMN's underperformance.

## Core Architecture

### Key Principles
1. **Temporal Co-evolution**: Attention and memory states evolve together at each timestep
2. **Bidirectional Flow**: Memory informs attention, attention guides memory updates
3. **Adaptive Refinement**: Processing depth varies based on item difficulty
4. **Gradient Preservation**: Skip connections maintain gradient flow

## Detailed Implementation

### 1. Base Co-evolutionary Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class CoevolutionaryAttentionMemory(nn.Module):
    """Core module for co-evolutionary attention-memory processing."""
    
    def __init__(self, embed_dim: int, memory_size: int, key_dim: int, 
                 value_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Attention components
        self.attention_memory = nn.Parameter(torch.randn(1, memory_size, embed_dim))
        self.attention_gate = nn.Linear(embed_dim + value_dim, embed_dim)
        
        # Memory-guided attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        
        # Co-evolution GRU
        self.coevolution_cell = nn.GRUCell(embed_dim + value_dim, embed_dim)
        
        # Fusion layers
        self.pre_fusion = nn.Sequential(
            nn.Linear(embed_dim + value_dim + embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.post_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim + value_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with appropriate schemes."""
        nn.init.xavier_uniform_(self.attention_memory)
        nn.init.xavier_uniform_(self.attention_gate.weight)
        nn.init.zeros_(self.attention_gate.bias)
        
        for module in [self.pre_fusion, self.post_fusion, self.uncertainty_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, gpcm_embed: torch.Tensor, memory_read: torch.Tensor,
                attention_state: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Co-evolutionary forward pass.
        
        Args:
            gpcm_embed: Current GPCM embedding [batch, embed_dim]
            memory_read: Content read from DKVMN memory [batch, value_dim]
            attention_state: Current attention memory state [batch, memory_size, embed_dim]
            hidden_state: GRU hidden state [batch, embed_dim]
            
        Returns:
            refined_embed: Refined embedding [batch, embed_dim]
            new_attention_state: Updated attention state
            new_hidden_state: Updated hidden state
            uncertainty: Estimated uncertainty [batch, 1]
        """
        batch_size = gpcm_embed.size(0)
        
        # 1. Estimate uncertainty for adaptive processing
        uncertainty = self.uncertainty_net(torch.cat([gpcm_embed, memory_read], dim=-1))
        
        # 2. Memory-guided attention refinement
        # Query from current embedding, keys/values from attention memory
        gpcm_embed_expanded = gpcm_embed.unsqueeze(1)  # [batch, 1, embed_dim]
        refined_attn, attn_weights = self.memory_attention(
            gpcm_embed_expanded, attention_state, attention_state
        )
        refined_attn = refined_attn.squeeze(1)  # [batch, embed_dim]
        
        # 3. Gate the attention refinement based on memory content
        gate = torch.sigmoid(self.attention_gate(torch.cat([refined_attn, memory_read], dim=-1)))
        gated_embed = gate * refined_attn + (1 - gate) * gpcm_embed
        
        # 4. Pre-fusion combining all information
        combined = torch.cat([gated_embed, memory_read, attention_state.mean(dim=1)], dim=-1)
        pre_fused = self.pre_fusion(combined)
        
        # 5. Co-evolution through GRU
        new_hidden_state = self.coevolution_cell(
            torch.cat([gated_embed, memory_read], dim=-1),
            hidden_state
        )
        
        # 6. Post-fusion for final refinement
        refined_embed = self.post_fusion(pre_fused) + gpcm_embed  # Skip connection
        
        # 7. Update attention state (soft update based on uncertainty)
        update_rate = 0.1 * (1 + uncertainty)  # Higher uncertainty -> more update
        new_attention_state = attention_state + update_rate.unsqueeze(-1) * new_hidden_state.unsqueeze(1)
        
        return refined_embed, new_attention_state, new_hidden_state, uncertainty
```

### 2. Integrated Co-evolutionary AKVMN Model

```python
class CoevolutionaryAKVMN(nn.Module):
    """Full AKVMN model with co-evolutionary attention-memory integration."""
    
    def __init__(self, n_questions: int, n_cats: int = 4, embed_dim: int = 64,
                 memory_size: int = 50, key_dim: int = 50, value_dim: int = 200,
                 final_fc_dim: int = 50, n_heads: int = 4, dropout_rate: float = 0.1,
                 ability_scale: float = 2.0, min_cycles: int = 1, max_cycles: int = 3):
        super().__init__()
        
        self.n_questions = n_questions
        self.n_cats = n_cats
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.min_cycles = min_cycles
        self.max_cycles = max_cycles
        
        # Question embeddings
        self.q_embed = nn.Embedding(n_questions + 1, key_dim, padding_idx=0)
        
        # GPCM embedding with learnable weights
        self.gpcm_embedding = LearnableGPCMEmbedding(n_questions, n_cats, embed_dim)
        
        # DKVMN Memory
        self.memory = DKVMNMemory(memory_size, key_dim, value_dim)
        
        # Co-evolutionary attention-memory module
        self.coevolution = CoevolutionaryAttentionMemory(
            embed_dim, memory_size, key_dim, value_dim, n_heads, dropout_rate
        )
        
        # Multiple refinement cycles
        self.refinement_cycles = nn.ModuleList([
            CoevolutionaryAttentionMemory(
                embed_dim, memory_size, key_dim, value_dim, n_heads, dropout_rate
            ) for _ in range(max_cycles - 1)
        ])
        
        # Summary network
        self.summary_fc = nn.Sequential(
            nn.Linear(value_dim + embed_dim, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_fc_dim, final_fc_dim),
            nn.ReLU()
        )
        
        # IRT parameter extraction
        self.irt_extractor = IRTParameterExtractor(
            final_fc_dim, n_cats, ability_scale, True, dropout_rate, key_dim
        )
        
        # Prediction head
        self.prediction_fc = nn.Linear(final_fc_dim, 1)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.q_embed.weight)
        
        for module in self.summary_fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
                
        nn.init.xavier_uniform_(self.prediction_fc.weight)
        nn.init.zeros_(self.prediction_fc.bias)
    
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with co-evolutionary processing.
        
        Args:
            questions: Question indices [batch, seq_len]
            responses: Response values [batch, seq_len]
            
        Returns:
            predictions: Response predictions [batch, seq_len, n_cats]
            student_abilities: Estimated abilities [batch, seq_len]
            item_difficulties: Item difficulties [batch, seq_len, n_cats-1]
            discrimination: Discrimination parameters [batch, seq_len]
        """
        batch_size, seq_len = questions.shape
        device = questions.device
        
        # Initialize states
        attention_state = self.coevolution.attention_memory.expand(batch_size, -1, -1).clone()
        hidden_state = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # Storage for outputs
        predictions = []
        abilities = []
        difficulties = []
        discriminations = []
        
        for t in range(seq_len):
            # Get embeddings
            q_t = questions[:, t]
            r_t = responses[:, t]
            
            q_embed = self.q_embed(q_t)
            gpcm_embed = self.gpcm_embedding(q_t, r_t)
            
            # Read from DKVMN memory
            correlation_weight = self.memory.attention(q_embed)
            read_content = self.memory.read(correlation_weight)
            
            # Co-evolutionary refinement
            refined_embed = gpcm_embed
            cycle_uncertainties = []
            
            # First cycle (always executed)
            refined_embed, attention_state, hidden_state, uncertainty = self.coevolution(
                refined_embed, read_content, attention_state, hidden_state
            )
            cycle_uncertainties.append(uncertainty)
            
            # Additional cycles based on uncertainty
            n_extra_cycles = int(uncertainty.mean() * (self.max_cycles - self.min_cycles))
            for i in range(n_extra_cycles):
                if i < len(self.refinement_cycles):
                    refined_embed, attention_state, hidden_state, uncertainty = \
                        self.refinement_cycles[i](
                            refined_embed, read_content, attention_state, hidden_state
                        )
                    cycle_uncertainties.append(uncertainty)
                    
                    # Early stopping if confident
                    if uncertainty.mean() < 0.2:
                        break
            
            # Create summary vector
            summary = self.summary_fc(torch.cat([read_content, refined_embed], dim=-1))
            
            # Extract IRT parameters
            ability, difficulty, discrimination = self.irt_extractor(summary, q_embed)
            
            # Make prediction
            pred = self.predict_gpcm(ability, difficulty, discrimination)
            
            # Update DKVMN memory with co-evolved value
            # Value combines refined embedding with attention state
            value_update = torch.cat([
                refined_embed, 
                attention_state.mean(dim=1)
            ], dim=-1)[:, :self.memory.value_dim]
            
            self.memory.write(correlation_weight, value_update)
            
            # Store outputs
            predictions.append(pred)
            abilities.append(ability)
            difficulties.append(difficulty)
            discriminations.append(discrimination)
        
        # Stack outputs
        predictions = torch.stack(predictions, dim=1)
        abilities = torch.stack(abilities, dim=1)
        difficulties = torch.stack(difficulties, dim=1)
        discriminations = torch.stack(discriminations, dim=1)
        
        return predictions, abilities, difficulties, discriminations
    
    def predict_gpcm(self, ability: torch.Tensor, difficulty: torch.Tensor, 
                     discrimination: torch.Tensor) -> torch.Tensor:
        """GPCM response probability calculation."""
        # Implementation of GPCM probability function
        # P(X=k|θ,a,b) calculation
        pass
```

### 3. Supporting Components

```python
class LearnableGPCMEmbedding(nn.Module):
    """GPCM embedding with learnable response weights."""
    
    def __init__(self, n_questions: int, n_cats: int, embed_dim: int):
        super().__init__()
        self.n_questions = n_questions
        self.n_cats = n_cats
        
        # Learnable response weights
        self.response_weights = nn.Parameter(torch.ones(n_cats))
        
        # Embedding layers
        self.q_embed = nn.Embedding(n_questions + 1, embed_dim // 2, padding_idx=0)
        self.r_embed = nn.Linear(n_cats, embed_dim // 2)
        self.combine = nn.Linear(embed_dim, embed_dim)
        
        self._init_parameters()
        
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.q_embed.weight)
        nn.init.xavier_uniform_(self.r_embed.weight)
        nn.init.zeros_(self.r_embed.bias)
        nn.init.xavier_uniform_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)
        
    def forward(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create GPCM embeddings with learnable weights."""
        # Question embedding
        q_emb = self.q_embed(questions)
        
        # Response embedding with learnable weights
        r_onehot = F.one_hot(responses, num_classes=self.n_cats).float()
        r_weighted = r_onehot * F.softmax(self.response_weights, dim=0)
        r_emb = self.r_embed(r_weighted)
        
        # Combine
        combined = torch.cat([q_emb, r_emb], dim=-1)
        return self.combine(combined)
```

## Training Strategy

### 1. Progressive Training
```python
def progressive_training_schedule(epoch: int) -> dict:
    """Adjust training parameters progressively."""
    if epoch < 5:
        # Warm-up: train with minimum cycles
        return {
            'min_cycles': 1,
            'max_cycles': 1,
            'dropout': 0.0,
            'lr_scale': 0.1
        }
    elif epoch < 15:
        # Increase complexity
        return {
            'min_cycles': 1,
            'max_cycles': 2,
            'dropout': 0.05,
            'lr_scale': 1.0
        }
    else:
        # Full model
        return {
            'min_cycles': 1,
            'max_cycles': 3,
            'dropout': 0.1,
            'lr_scale': 0.5
        }
```

### 2. Loss Function with Regularization
```python
class CoevolutionaryLoss(nn.Module):
    """Loss function with co-evolution specific regularization."""
    
    def __init__(self, alpha: float = 0.01, beta: float = 0.001):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # Attention consistency weight
        self.beta = beta   # Complexity penalty weight
        
    def forward(self, predictions, targets, attention_states, uncertainties):
        # Main task loss
        task_loss = self.ce_loss(predictions.reshape(-1, predictions.size(-1)), 
                                 targets.reshape(-1))
        
        # Attention consistency loss (encourage smooth evolution)
        if len(attention_states) > 1:
            consistency_loss = sum(
                F.mse_loss(attention_states[i], attention_states[i+1])
                for i in range(len(attention_states) - 1)
            ) / (len(attention_states) - 1)
        else:
            consistency_loss = 0.0
            
        # Complexity penalty (encourage low uncertainty for easy items)
        complexity_loss = uncertainties.mean()
        
        total_loss = task_loss + self.alpha * consistency_loss + self.beta * complexity_loss
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'consistency_loss': consistency_loss.item() if consistency_loss else 0.0,
            'complexity_loss': complexity_loss.item()
        }
```

## Integration Steps

### Phase 1: Core Implementation (Days 1-3)
1. Implement `CoevolutionaryAttentionMemory` module
2. Create `LearnableGPCMEmbedding` with weight learning
3. Build base `CoevolutionaryAKVMN` model
4. Verify gradient flow and memory updates

### Phase 2: Training Pipeline (Days 4-5)
1. Implement progressive training schedule
2. Add co-evolutionary loss function
3. Create evaluation metrics for attention quality
4. Set up logging and visualization

### Phase 3: Testing & Optimization (Days 6-7)
1. Test on synthetic dataset
2. Profile computational performance
3. Tune hyperparameters
4. Compare with baseline and current AKVMN

### Phase 4: Full Evaluation (Week 2)
1. Run on all benchmark datasets
2. Conduct ablation studies
3. Analyze attention patterns
4. Document results and insights

## Expected Results

### Performance Improvements
- **Categorical Accuracy**: +5-7% over baseline
- **Ordinal Accuracy**: +4-6% improvement
- **QWK**: +0.06-0.10 increase
- **MAE**: -0.03-0.05 reduction

### Computational Overhead
- Training time: +20-30% due to iterative refinement
- Memory usage: +15% for attention states
- Inference time: +25% worst case (adaptive reduces average)

### Key Advantages
1. True integration of attention and memory
2. Adaptive computation based on difficulty
3. Interpretable attention evolution
4. Stable training with progressive schedule

## Validation Experiments

### 1. Ablation Studies
- Remove co-evolution → measure impact
- Fix cycles to 1 → test adaptive benefit
- Remove uncertainty estimation → assess efficiency
- Disable attention state updates → verify co-evolution importance

### 2. Attention Analysis
- Visualize attention evolution across timesteps
- Correlate uncertainty with item difficulty
- Track attention consistency metrics
- Analyze memory-attention interaction patterns

### 3. Comparative Analysis
- Baseline DKVMN
- Current AKVMN implementations
- Co-evolutionary AKVMN
- Statistical significance testing

## Conclusion

This implementation plan provides a concrete path to achieving the expected 5-10% improvement through true co-evolution of attention and memory. The key innovation is processing within the sequential loop where attention and memory mutually influence each other, rather than treating attention as a preprocessing step.