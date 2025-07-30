# AKVMN Fix Implementation Plan

## Overview
Based on the deep analysis, the current AKVMN models fail because they treat attention as a preprocessing step rather than integrating it deeply with memory operations. This plan outlines how to fix the implementation to achieve the expected 5-10% improvement over baseline.

## Key Architectural Changes Needed

### 1. Deep Memory-Attention Integration
Instead of:
```python
embeddings → attention_refinement → memory_operations → IRT_params
```

We need:
```python
embeddings → [attention ↔ memory]_iterative → IRT_params
```

### 2. Core Implementation Strategy

#### A. Create a new `IntegratedAttentionDKVMN` class
This will replace the separate attention and memory modules with an integrated approach:

```python
class IntegratedAttentionDKVMN(nn.Module):
    def __init__(self, memory_size, key_dim, value_dim, n_heads=4, n_cycles=2):
        super().__init__()
        # Standard DKVMN components
        self.memory = DKVMN(memory_size, key_dim, value_dim)
        
        # Attention components for memory-attention co-evolution
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(value_dim, n_heads, batch_first=True)
            for _ in range(n_cycles)
        ])
        
        # Memory-guided embedding refinement
        self.memory_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(value_dim + embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(n_cycles)
        ])
        
    def forward(self, q_embed, gpcm_embed):
        # Iterative refinement with memory feedback
        for cycle in range(self.n_cycles):
            # 1. Memory attention
            correlation_weight = self.memory.attention(q_embed)
            memory_content = self.memory.read(correlation_weight)
            
            # 2. Refine embeddings using memory content
            combined = torch.cat([gpcm_embed, memory_content], dim=-1)
            refined_embed = self.memory_refinement[cycle](combined)
            
            # 3. Apply attention with memory context
            attn_out, _ = self.attention_layers[cycle](
                refined_embed, memory_content.unsqueeze(1), memory_content.unsqueeze(1)
            )
            
            # 4. Update embedding for next cycle
            gpcm_embed = refined_embed + 0.5 * attn_out.squeeze(1)
            
        return gpcm_embed, memory_content, correlation_weight
```

#### B. Modify the forward pass to use integrated processing

```python
def forward(self, questions, responses):
    batch_size, seq_len = questions.shape
    
    # Initialize
    self.memory.init_value_memory(batch_size, self.init_value_memory)
    
    # Create initial embeddings
    gpcm_embeds = self.create_embeddings(questions, responses)
    q_embeds = self.q_embed(questions)
    
    # Sequential processing with deep integration
    outputs = []
    
    for t in range(seq_len):
        q_embed_t = q_embeds[:, t, :]
        gpcm_embed_t = gpcm_embeds[:, t, :]
        
        # CRITICAL: Integrated attention-memory processing
        # This is where the magic happens
        refined_embed, memory_read, correlation_weight = self.integrated_memory_attention(
            q_embed_t, gpcm_embed_t
        )
        
        # Create summary with refined features
        summary_input = torch.cat([memory_read, q_embed_t], dim=-1)
        summary_vector = self.summary_network(summary_input)
        
        # Extract IRT parameters (these can influence next iteration)
        theta_t, alpha_t, betas_t = self.irt_extractor(
            summary_vector.unsqueeze(1), q_embed_t.unsqueeze(1)
        )
        
        # GPCM probability
        gpcm_prob_t = self.gpcm_layer(theta_t, alpha_t, betas_t)
        
        # Store outputs
        outputs.append((theta_t, alpha_t, betas_t, gpcm_prob_t))
        
        # Update memory with refined information
        if t < seq_len - 1:
            # Transform refined embedding for memory update
            value_embed_t = self.gpcm_value_embed(refined_embed)
            self.memory.write(correlation_weight, value_embed_t)
```

### 3. Key Implementation Details

#### A. Attention-Memory Co-evolution
- Attention weights should be influenced by current memory state
- Memory updates should incorporate attention patterns
- Each refinement cycle should see updated memory content

#### B. Gradient Flow Improvements
- Add residual connections between refinement cycles
- Use gated updates to control information flow
- Ensure gradients can flow from IRT parameters back to attention

#### C. Adaptive Processing
- Allow variable number of refinement cycles based on uncertainty
- Skip refinement for high-confidence predictions
- Focus computation on difficult items

### 4. Specific Code Changes

#### A. Create new file: `core/integrated_attention.py`
```python
"""Deeply integrated attention-memory network for AKVMN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .model import DeepGPCM
from .memory_networks import DKVMN


class IntegratedAttentionGPCM(DeepGPCM):
    """AKVMN with true memory-attention integration."""
    
    def __init__(self, n_questions, n_cats=4, memory_size=50, 
                 key_dim=50, value_dim=200, final_fc_dim=50,
                 n_heads=4, n_cycles=2, embed_dim=64):
        # Initialize base model
        super().__init__(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=memory_size,
            key_dim=key_dim,
            value_dim=value_dim,
            final_fc_dim=final_fc_dim,
            embedding_strategy="linear_decay",
            ability_scale=2.0
        )
        
        self.model_name = "integrated_akvmn_gpcm"
        self.n_heads = n_heads
        self.n_cycles = n_cycles
        self.embed_dim = embed_dim
        
        # Memory-attention integration components
        self._build_integration_components()
        
    def _build_integration_components(self):
        """Build components for memory-attention integration."""
        # ... (implementation details)
```

#### B. Update model factory
Add the new integrated model to the factory:
```python
elif model_type == 'integrated_akvmn':
    from .integrated_attention import IntegratedAttentionGPCM
    model = IntegratedAttentionGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        # ... parameters
    )
```

### 5. Testing Strategy

1. **Baseline Comparison**: Train on synthetic_OC and compare with baseline
2. **Ablation Study**: Test with/without each component
3. **Gradient Analysis**: Monitor gradient flow through integrated components
4. **Performance Metrics**: Target 5-10% improvement in QWK and categorical accuracy

### 6. Implementation Timeline

1. **Phase 1** (2 hours): Implement `IntegratedAttentionGPCM` class
2. **Phase 2** (1 hour): Integration testing and debugging
3. **Phase 3** (3 hours): Training and evaluation
4. **Phase 4** (1 hour): Performance analysis and tuning

## Expected Outcomes

1. **Performance**: 5-10% improvement over baseline (QWK ~0.80-0.84)
2. **Convergence**: Faster convergence due to better gradient flow
3. **Stability**: More stable training with adaptive refinement
4. **Interpretability**: Attention weights provide insight into memory usage

## Risk Mitigation

1. **Complexity**: Keep implementation focused on core integration
2. **Overfitting**: Use dropout and early stopping
3. **Numerical Stability**: Add safety checks for NaN/inf values
4. **Memory Usage**: Monitor GPU memory with integrated approach

## Success Criteria

- Categorical accuracy > 74% (baseline: 70.46%)
- QWK > 0.80 (baseline: 0.761)
- Consistent improvements across multiple runs
- Stable training without degradation