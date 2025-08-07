# TODO: Ordinal-Aware Attention Module Implementation

## Overview
Systematic implementation plan for integrating ordinal-aware attention mechanisms into the deep-gpcm project while ensuring backward compatibility and minimal disruption to existing models.

## Phase 1: Foundation Setup (Week 1)

### 1.1 Core Infrastructure (Days 1-2)
- [ ] Create `models/attention/` directory structure
  - [ ] `__init__.py` - Module exports
  - [ ] `base.py` - Base classes (AttentionContext, AttentionMechanism, AttentionConfig)
  - [ ] `registry.py` - Plugin registry system
  - [ ] `pipeline.py` - AttentionPipeline for composing mechanisms
  - [ ] `utils.py` - Helper functions and utilities

- [ ] Implement base classes
  - [ ] AttentionContext dataclass with proper validation
  - [ ] Abstract AttentionMechanism base class
  - [ ] AttentionConfig with sensible defaults
  - [ ] Registry system with error handling

- [ ] Add compatibility layer
  - [ ] Create `models/attention_gpcm_modular.py` alongside existing `attention_gpcm.py`
  - [ ] Ensure imports don't break existing code
  - [ ] Add feature flags for gradual rollout

### 1.2 Individual Attention Mechanisms (Days 3-5)
- [ ] Implement core attention mechanisms
  - [ ] `ordinal_aware.py` - Ordinal distance-based attention scoring
  - [ ] `response_conditioned.py` - Response-level specific refinement
  - [ ] `ordinal_pattern.py` - Temporal pattern detection
  - [ ] `qwk_aligned.py` - QWK metric-aligned attention
  - [ ] `hierarchical_ordinal.py` - Two-level attention (binary + ordinal)

- [ ] Unit tests for each mechanism
  - [ ] `tests/test_attention_mechanisms.py`
  - [ ] Test forward pass, shapes, gradients
  - [ ] Test attention weight extraction
  - [ ] Test with edge cases (empty sequences, single item)

### 1.3 Integration Testing (Days 6-7)
- [ ] Create ModularAttentionGPCM class
  - [ ] Inherit from existing AttentionGPCM
  - [ ] Override only necessary methods
  - [ ] Maintain backward compatibility

- [ ] Integration tests
  - [ ] Test pipeline composition
  - [ ] Test different fusion methods
  - [ ] Test with existing training pipeline
  - [ ] Memory and performance profiling

**Success Criteria Phase 1:**
- All attention mechanisms pass unit tests
- ModularAttentionGPCM produces same outputs as AttentionGPCM when using basic attention
- No regression in existing model performance
- Memory usage within 10% of baseline

## Phase 2: Validation & Optimization (Week 2)

### 2.1 Synthetic Data Testing (Days 1-2)
- [ ] Create ordinal-specific test datasets
  - [ ] Perfect ordinal sequences (0→1→2→3)
  - [ ] Declining patterns (3→2→1→0)
  - [ ] Near-miss patterns (consistently ±1)
  - [ ] Random patterns for baseline

- [ ] Validate mechanism behavior
  - [ ] Ordinal-aware attention focuses on similar response levels
  - [ ] Pattern detection identifies sequences correctly
  - [ ] QWK-aligned attention penalizes large distances
  - [ ] Response conditioning shows different behaviors per level

### 2.2 Small-Scale Real Data Testing (Days 3-4)
- [ ] Test on subset of actual datasets
  - [ ] Use 10% of assist2009_updated for quick iterations
  - [ ] Compare metrics: QWK, ordinal accuracy, MAE
  - [ ] Analyze attention weight visualizations
  - [ ] Check for overfitting indicators

- [ ] Hyperparameter tuning
  - [ ] Distance penalty in ordinal-aware (0.1-1.0)
  - [ ] Window size for pattern detection (3-7)
  - [ ] Correct threshold for hierarchical (2-3)
  - [ ] Number of attention heads (4-16)

### 2.3 Performance Optimization (Days 5-7)
- [ ] Implement caching strategies
  - [ ] Cache attention weights during inference
  - [ ] Implement gradient checkpointing option
  - [ ] Add mixed precision support

- [ ] Profile and optimize
  - [ ] Identify bottlenecks with PyTorch profiler
  - [ ] Optimize matrix operations
  - [ ] Consider torch.compile for PyTorch 2.0+
  - [ ] Batch processing optimization

**Success Criteria Phase 2:**
- Synthetic data shows expected attention patterns
- Small-scale testing shows ≥2% QWK improvement
- Performance within 20% of baseline AttentionGPCM
- Memory usage optimized for large batches

## Phase 3: Full Integration (Week 3)

### 3.1 Configuration System (Days 1-2)
- [ ] Implement configuration management
  - [ ] YAML configuration loader
  - [ ] Command-line argument integration
  - [ ] Default configurations for each dataset
  - [ ] Configuration validation

- [ ] Add to existing training scripts
  - [ ] Update `train_gpcm.py` with modular option
  - [ ] Add `--attention-type` flag (basic/modular)
  - [ ] Add `--attention-mechanisms` flag
  - [ ] Maintain backward compatibility

### 3.2 Full Dataset Evaluation (Days 3-5)
- [ ] Run complete benchmarks
  - [ ] All 9 datasets with 5-fold CV
  - [ ] Compare against baseline AttentionGPCM
  - [ ] Generate comprehensive metrics report
  - [ ] Create visualization notebooks

- [ ] Ablation studies
  - [ ] Test each mechanism individually
  - [ ] Test combinations of 2-3 mechanisms
  - [ ] Test different fusion methods
  - [ ] Analyze computational vs accuracy trade-offs

### 3.3 Model Checkpoint Compatibility (Days 6-7)
- [ ] Implement checkpoint migration
  - [ ] Load legacy AttentionGPCM checkpoints
  - [ ] Convert to modular format
  - [ ] Save in new format with metadata
  - [ ] Backward compatibility for old scripts

- [ ] Documentation and examples
  - [ ] Update README with modular attention section
  - [ ] Create example notebooks
  - [ ] Document configuration options
  - [ ] Migration guide for existing users

**Success Criteria Phase 3:**
- Full benchmark shows consistent improvements:
  - QWK: +0.03 to +0.08 average improvement
  - Ordinal Accuracy: +2% to +4% improvement
  - Large error rate: 20-30% reduction
- All existing scripts work without modification
- New modular system fully documented

## Phase 4: Production Readiness (Week 4)

### 4.1 Robustness Testing (Days 1-3)
- [ ] Edge case testing
  - [ ] Empty sequences
  - [ ] Single-item sequences
  - [ ] Very long sequences (>1000 items)
  - [ ] Extreme class imbalance

- [ ] Stress testing
  - [ ] Large batch sizes (256+)
  - [ ] Mixed precision training
  - [ ] Multi-GPU training
  - [ ] Memory pressure scenarios

### 4.2 Final Integration (Days 4-5)
- [ ] Merge strategy
  - [ ] Create feature branch with all changes
  - [ ] Gradual merge with feature flags
  - [ ] A/B testing infrastructure
  - [ ] Rollback plan

- [ ] Production monitoring
  - [ ] Add metrics logging
  - [ ] Performance monitoring
  - [ ] Error tracking
  - [ ] Resource usage alerts

### 4.3 Release Preparation (Days 6-7)
- [ ] Final documentation
  - [ ] API reference
  - [ ] Architecture diagrams
  - [ ] Performance benchmarks
  - [ ] Known limitations

- [ ] Release checklist
  - [ ] All tests passing
  - [ ] Documentation complete
  - [ ] Benchmarks documented
  - [ ] Migration guide ready
  - [ ] Deprecation warnings added

**Success Criteria Phase 4:**
- Zero regression in existing functionality
- Robust under all test scenarios
- Performance meets production standards
- Complete documentation and examples

## Key Milestones

1. **Week 1 Milestone**: Working prototype with all mechanisms implemented
2. **Week 2 Milestone**: Validated improvements on synthetic and small-scale data
3. **Week 3 Milestone**: Full integration with proven benefits across all datasets
4. **Week 4 Milestone**: Production-ready system with complete documentation

## Risk Mitigation

### Technical Risks
- **Risk**: Overfitting to ordinal structure
  - **Mitigation**: Strong regularization, dropout, validation monitoring
  
- **Risk**: Computational overhead
  - **Mitigation**: Caching, optimization, optional mechanism selection

- **Risk**: Training instability
  - **Mitigation**: Careful initialization, gradient clipping, learning rate scheduling

### Integration Risks
- **Risk**: Breaking existing workflows
  - **Mitigation**: Feature flags, backward compatibility, extensive testing

- **Risk**: Checkpoint incompatibility
  - **Mitigation**: Migration tools, versioning, fallback options

## Testing Strategy

### Unit Testing
- Each attention mechanism tested in isolation
- Mock data for predictable outputs
- Gradient flow verification
- Edge case handling

### Integration Testing
- Full model training on small datasets
- Checkpoint save/load cycles
- Multi-GPU compatibility
- Memory leak detection

### Performance Testing
- Benchmark against baseline
- Profile critical paths
- Memory usage monitoring
- Scaling tests (batch size, sequence length)

### Validation Testing
- Metric improvements across datasets
- Attention weight interpretability
- Ablation study results
- Statistical significance tests

## Implementation Notes

### Design Principles
1. **Modularity**: Each mechanism is self-contained and testable
2. **Compatibility**: No breaking changes to existing code
3. **Performance**: Optimization without sacrificing accuracy
4. **Flexibility**: Easy to add new mechanisms or modify existing ones

### Code Organization
```
models/
├── attention/
│   ├── __init__.py
│   ├── base.py              # Base classes
│   ├── registry.py          # Plugin system
│   ├── pipeline.py          # Composition logic
│   ├── mechanisms/
│   │   ├── __init__.py
│   │   ├── ordinal_aware.py
│   │   ├── response_conditioned.py
│   │   ├── ordinal_pattern.py
│   │   ├── qwk_aligned.py
│   │   └── hierarchical_ordinal.py
│   └── utils.py
├── attention_gpcm.py        # Original (unchanged)
└── attention_gpcm_modular.py # New modular version
```

### Configuration Example
```yaml
attention:
  type: modular
  mechanisms:
    - ordinal_aware
    - qwk_aligned
  fusion_method: weighted
  config:
    distance_penalty: 0.5
    n_heads: 8
    dropout: 0.1
```

## Progress Tracking

- [ ] Phase 1: Foundation Setup (0/7 days)
- [ ] Phase 2: Validation & Optimization (0/7 days)
- [ ] Phase 3: Full Integration (0/7 days)
- [ ] Phase 4: Production Readiness (0/7 days)

Total: 0/28 days completed

## Next Steps

1. Start with Phase 1.1 - Create directory structure and base classes
2. Implement one attention mechanism as proof of concept
3. Validate approach with small-scale testing
4. Proceed with full implementation based on initial results