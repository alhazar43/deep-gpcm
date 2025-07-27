# AKT Integration TODO

## Current Transformer Analysis
**Why it works better (+1.5% accuracy):**
- Uses **question-only embeddings** for transformer sequence modeling
- Preserves **linear decay in GPCM computation** (not in transformer)
- **Simple integration**: transformer → summary features → GPCM parameters
- **Bypasses embedding complexity** in attention mechanism

## Core Integration Strategy

### 1. AKT Embedding (Week 1)
- Create `AKTEmbedding` class with linear decay integration
- **Key insight**: Keep linear decay for GPCM, use projected embeddings for attention
- Test embedding compatibility with current transformer architecture

### 2. AKT Attention (Week 2)  
- Implement distance-aware attention from AKT
- **Critical**: Use question embeddings (like current transformer) + AKT attention enhancements
- Add gamma parameters for position-aware scaling

### 3. Model Integration (Week 3)
- Create `AKTTransformer` class combining:
  - Question-only transformer sequence modeling (proven approach)
  - AKT distance-aware attention 
  - Preserved GPCM probability computation
- Factory pattern integration

### 4. Testing & Optimization (Week 4)
- Benchmark against current transformer (55.5% baseline)
- Target: >57% categorical accuracy
- Ablation studies on distance-aware components

## Implementation Files

```
models/
├── akt_embedding.py       # AKT-compatible embedding with linear decay
├── akt_attention.py       # Distance-aware attention mechanism  
├── akt_transformer.py     # Main AKT model (follows current transformer pattern)
```

## Key Technical Decisions

1. **Follow Proven Pattern**: Question embeddings → transformer → GPCM (like current)
2. **Add AKT Enhancements**: Distance-aware attention, position effects
3. **Preserve Linear Decay**: In GPCM computation, not transformer attention
4. **Simple Naming**: `AKTEmbedding`, `AKTAttention`, `AKTTransformer`

## Success Criteria
- [ ] **Week 1**: AKT embedding functional, no performance regression
- [ ] **Week 2**: Distance-aware attention working, attention entropy metrics
- [ ] **Week 3**: Full model integration, >56% categorical accuracy  
- [ ] **Week 4**: >57% accuracy target, production-ready implementation

## Linear Decay Integration Strategy
Instead of complex GPCM-aware embeddings, use **dual-path approach**:
- **Path 1**: Question embeddings → AKT attention → summary features
- **Path 2**: Linear decay embeddings → GPCM probability computation
- **Integration**: Summary features influence GPCM parameters (θ, α, β)

This preserves what works (linear decay in GPCM) while adding what's proven (AKT attention for sequences).