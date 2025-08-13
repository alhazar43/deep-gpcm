# TEMP TODO: Unified Prediction System Implementation

## Overview
Implement a unified prediction system that computes three types of predictions (hard/argmax, soft/expected value, threshold/cumulative) and allows metrics to choose the most appropriate prediction method for evaluation.

## Implementation Tasks

### Phase 0: Pre-Implementation (Priority: CRITICAL)
- [ ] Analyze existing model output formats across all architectures
- [ ] Document current evaluation pipeline flow
- [ ] Create comprehensive test dataset with edge cases
- [ ] Set up monitoring infrastructure for development

### Phase 1: Core Implementation (Priority: HIGH)

#### 1.1 Create `utils/predictions.py`
- [ ] Implement `compute_unified_predictions()` function
  - [ ] Hard predictions (argmax)
  - [ ] Soft predictions (expected value)
  - [ ] Threshold predictions (cumulative)
  - [ ] Proper mask handling for padding tokens
  - [ ] Edge case handling (all zeros, numerical instability)
  - [ ] GPU acceleration support
  - [ ] Batch prediction support
- [ ] Add `categorical_to_cumulative()` helper function
- [ ] Implement `compute_adaptive_thresholds()` for data-driven thresholds
- [ ] Add vectorized computation for efficiency
- [ ] Include prediction caching strategy with thread safety
- [ ] Add numerical stability checks (epsilon smoothing)
- [ ] Implement model output adapters for different architectures

#### 1.2 Create Prediction Strategy Classes
- [ ] Abstract `PredictionStrategy` base class
- [ ] `HardPrediction` strategy (argmax)
- [ ] `SoftPrediction` strategy (expected value)
- [ ] `ThresholdPrediction` strategy (cumulative)
- [ ] `PredictionConfig` dataclass for configuration

### Phase 2: Metric System Integration (Priority: HIGH)

#### 2.1 Update `utils/metrics.py`
- [ ] Add `compute_metrics_multimethod()` function
- [ ] Create `METRIC_METHOD_MAP` registry:
  ```python
  METRIC_METHOD_MAP = {
      'categorical_accuracy': 'hard',
      'ordinal_accuracy': 'threshold',
      'mean_absolute_error': 'soft',
      'quadratic_weighted_kappa': 'soft',
      'spearman_correlation': 'soft',
      'kendall_tau': 'soft',
      'pearson_correlation': 'soft',
      'cohen_kappa': 'hard',
      'precision': 'hard',
      'recall': 'hard',
      'f1_score': 'hard'
  }
  ```
- [ ] Maintain backward compatibility with existing `compute_metrics()`
- [ ] Add metric-specific handling for probability requirements
- [ ] Implement comparison metrics (soft vs hard improvements)

### Phase 2.5: Basic Integration (Priority: HIGH)
- [ ] Create minimal integration with evaluate.py
- [ ] End-to-end testing with one model
- [ ] Validate output format consistency
- [ ] Performance baseline measurement

### Phase 3: Evaluation Integration (Priority: MEDIUM)

#### 3.1 Update `evaluate.py`
- [ ] Add `evaluate_model_multimethod()` function
- [ ] Extract `gpcm_probs` from model output tuple correctly
- [ ] Add CLI arguments:
  - `--use_multimethod_eval`
  - `--thresholds` (list of floats)
  - `--prediction_methods` (which methods to compute)
- [ ] Integrate with existing evaluation pipeline
- [ ] Add detailed result saving with method annotations

#### 3.2 Update `train.py` validation
- [ ] Add optional multi-method validation during training
- [ ] Add arguments:
  - `--use_multimethod_eval`
  - `--schedule_metric` (which metric to use for scheduling)
  - `--schedule_method` (override method for scheduling metric)
- [ ] Ensure backward compatibility (default to current behavior)
- [ ] Add performance monitoring for prediction computation

### Phase 4: Testing & Validation (Priority: HIGH)

#### 4.1 Create `tests/test_predictions.py`
- [ ] Unit tests for each prediction method
- [ ] Test vectorized vs loop implementations
- [ ] Validate predictions are valid category indices
- [ ] Test mask handling correctness
- [ ] Test with different n_cats values (2, 3, 4, 5)
- [ ] Stress testing with large batches (>10K samples)
- [ ] Edge case testing (single category, probability ties)
- [ ] Numerical stability testing
- [ ] GPU/CPU consistency tests

#### 4.2 Create `tests/test_multimethod_metrics.py`
- [ ] Test metric computation with each method
- [ ] Verify metric-method mapping works correctly
- [ ] Test backward compatibility
- [ ] Performance benchmarks

#### 4.3 Cross-validation compatibility
- [ ] Test with k-fold CV in train.py
- [ ] Ensure predictions aggregate correctly across folds
- [ ] Verify no memory leaks with caching

### Phase 5: Documentation (Priority: MEDIUM)

#### 5.1 Create documentation
- [ ] Add docstrings to all new functions
- [ ] Create `docs/prediction_methods.md`:
  - When to use each prediction type
  - Performance implications
  - Theoretical justification
- [ ] Update README.md with new evaluation options

#### 5.2 Create example notebook
- [ ] `notebooks/multimethod_evaluation_demo.ipynb`
- [ ] Show differences between prediction methods
- [ ] Demonstrate metric improvements
- [ ] Visualize prediction distributions

### Phase 6: Performance Optimization (Priority: LOW)

#### 6.1 Optimize computations
- [ ] Profile prediction computation time
- [ ] Implement lazy evaluation option
- [ ] Add streaming support for large datasets
- [ ] Optimize memory usage with in-place operations

#### 6.2 Add caching layer
- [ ] Implement prediction cache with configurable size
- [ ] Add cache hit/miss statistics
- [ ] Support cache persistence between evaluations

### Phase 7: Advanced Features (Priority: LOW)

#### 7.1 Confidence intervals
- [ ] Add confidence/variance estimates for soft predictions
- [ ] Implement bootstrap confidence intervals
- [ ] Add uncertainty quantification metrics

#### 7.2 Custom threshold learning
- [ ] Implement threshold optimization on validation set
- [ ] Support per-category thresholds
- [ ] Add threshold visualization tools

## Implementation Order

1. **Week 1**: Phase 1 (Core Implementation) + Phase 4.1 (Basic Tests)
2. **Week 2**: Phase 2 (Metric Integration) + Phase 3 (Evaluation Integration)
3. **Week 3**: Phase 4.2-4.3 (Complete Testing) + Phase 5 (Documentation)
4. **Week 4**: Phase 6-7 (Optimization and Advanced Features)

## Success Criteria

- [ ] All existing tests pass (backward compatibility)
- [ ] Multi-method evaluation shows metric improvements
- [ ] Performance overhead < 5% for evaluation
- [ ] Documentation complete and clear
- [ ] At least 90% test coverage for new code

## Rollout Strategy

1. **Alpha**: Feature flag disabled by default
2. **Beta**: Enable for specific models with `--use_multimethod_eval`
3. **GA**: Make default if metrics improve consistently
4. **Deprecation**: Phase out single-method evaluation in future release

## Risk Mitigation

- **Risk**: Breaking existing evaluation
  - **Mitigation**: Comprehensive backward compatibility tests
  
- **Risk**: Performance regression
  - **Mitigation**: Benchmark before/after, optimize critical paths
  
- **Risk**: User confusion
  - **Mitigation**: Clear documentation, method annotations in output
  
- **Risk**: Memory issues with large datasets
  - **Mitigation**: Streaming support, configurable caching

- **Risk**: Numerical instability with extreme probabilities
  - **Mitigation**: Add epsilon smoothing, log-space computations
  
- **Risk**: Model output format incompatibility
  - **Mitigation**: Add model output adapters/normalizers
  
- **Risk**: Invalid metric-prediction combinations
  - **Mitigation**: Add validation warnings, compatibility matrix
  
- **Risk**: Cross-dataset threshold inconsistency
  - **Mitigation**: Dataset-specific threshold learning

## Dependencies

- NumPy >= 1.19.0 (for advanced indexing)
- PyTorch >= 1.9.0 (current requirement)
- SciPy >= 1.5.0 (for correlation metrics)
- scikit-learn >= 0.24.0 (current requirement)

## Additional Components (Based on Architect Review)

### Monitoring & Analytics
- [ ] Implement prediction statistics collector
- [ ] Add method selection analytics
- [ ] Track prediction distribution changes
- [ ] Log edge cases and warnings
- [ ] Create diagnostic tools for debugging

### Visualization Tools
- [ ] Prediction distribution plots by method
- [ ] Method comparison visualizations
- [ ] Threshold optimization curves
- [ ] Confidence interval plots
- [ ] Interactive comparison dashboard

### Migration Support
- [ ] Script to update existing results with new predictions
- [ ] Backward compatibility layer for old models
- [ ] Result format versioning
- [ ] Migration validation tools

### Integration with Existing Systems
- [ ] CORAL/CORN layer integration for ordinal-aware predictions
- [ ] Support for partial credit scoring
- [ ] Cross-dataset validation framework
- [ ] A/B testing infrastructure

## Notes

- Maintain backward compatibility throughout
- Default behavior should not change
- All new features should be opt-in initially
- Focus on code clarity over premature optimization
- Ensure comprehensive testing before deployment
- Consider thread safety for all caching operations
- Implement proper error handling for all edge cases
- Document performance implications clearly