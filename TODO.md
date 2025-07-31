# GPCM-CORAL Threshold Integration TODO

## Project Overview

Integration of GPCM β thresholds with CORAL ordinal thresholds to improve the coral_gpcm model. Focus on **simple, clean implementation** with proven value before adding complexity.

**Key Principle**: Start minimal, prove value, then expand if needed.

## Phase 0: Simplification (CRITICAL - Missing from Original Plan)

### 0.1 Archive Over-Engineered Implementation
- [ ] **Move current threshold_coupling.py to research/ directory**
  - Current implementation is 486 lines for simple threshold combination
  - 4 different mechanisms when 1-2 would suffice
  - Complex mathematical formulations without proven necessity
  - Inconsistent interfaces between couplers

### 0.2 Identify Core Requirements
- [ ] **Define minimal viable coupling**
  - Simple weighted combination of GPCM and CORAL thresholds
  - Learnable weights with proper initialization
  - Clean interface for integration
  - Backward compatibility

### 0.3 Clean Slate Implementation
- [ ] **Create new core/threshold_coupling.py**
  - Start with single LinearThresholdCoupler
  - Clean, consistent interface
  - Minimal dependencies
  - Comprehensive testing from start

## Phase 1: Clean Linear Coupling Implementation

### 1.1 Base Interface Design
- [ ] **Create ThresholdCoupler base class**
```python
class ThresholdCoupler(ABC):
    def couple(self, 
               gpcm_thresholds: torch.Tensor,
               coral_thresholds: torch.Tensor, 
               student_ability: torch.Tensor,
               item_discrimination: Optional[torch.Tensor] = None) -> torch.Tensor:
```

### 1.2 Linear Coupling Implementation
- [ ] **Implement LinearThresholdCoupler**
  - Simple weighted combination: `gpcm_weight * gpcm + coral_weight * coral`
  - Learnable weights (default: gpcm_weight=0.7, coral_weight=0.3)
  - Proper tensor broadcasting
  - Clean parameter validation

### 1.3 Factory Pattern
- [ ] **Create ThresholdCouplerFactory**
  - Configuration-driven coupler creation
  - Clean separation of concerns
  - Easy extension for future couplers

### 1.4 Testing Framework
- [ ] **Create comprehensive tests**
  - Unit tests for LinearThresholdCoupler
  - Shape consistency tests
  - Gradient flow validation
  - Performance benchmarks

## Phase 2: Integration with coral_gpcm

### 2.1 Integration Points
- [ ] **Modify CORALDeepGPCM class**
  - Add optional `threshold_coupler` parameter
  - Integrate coupling in forward pass
  - Maintain 100% backward compatibility
  - Clean error handling

### 2.2 Configuration System
- [ ] **Create simple configuration**
```python
@dataclass
class ThresholdCouplingConfig:
    enabled: bool = False
    coupling_type: str = "linear"
    gpcm_weight: float = 0.7
    coral_weight: float = 0.3
```

### 2.3 Training Integration
- [ ] **Update training pipeline**
  - Add coupling config to train.py
  - Update model_factory.py
  - Ensure training stability

## Phase 3: Validation and Performance

### 3.1 Performance Validation
- [ ] **Benchmark coupling vs no coupling**
  - Training time overhead (target: ≤5%)
  - Performance improvement (target: ≥2%)
  - Memory usage impact
  - Forward pass timing

### 3.2 Comprehensive Testing
- [ ] **End-to-end validation**
  - Train on synthetic_OC dataset
  - Compare metrics with/without coupling
  - Validate training stability
  - Test model save/load with coupling

### 3.3 Success Criteria Validation
- [ ] **Verify all targets met**
  - ≤5% training time overhead
  - ≥2% performance improvement on validation metrics
  - 100% backward compatibility maintained
  - >95% test coverage for coupling modules

## Phase 4: Documentation and Examples

### 4.1 Clean Documentation
- [ ] **Create practical documentation**
  - Simple usage guide
  - Configuration examples
  - When to enable coupling
  - Performance characteristics

### 4.2 Usage Examples
- [ ] **Create example scripts**
  - Basic coupling usage
  - Training with coupling enabled
  - Performance comparison script

## Improved Naming Conventions

**Professional, Specific Names:**
- `LinearThresholdCoupler` (not "BasicThresholdCoupler")
- `AttentionThresholdCoupler` (not "AttentionCoupler")
- `MLPThresholdCoupler` (not "NeuralCoupler")
- `EnsembleThresholdCoupler` (not "AdaptiveCoupler")

**Avoid:**
- Generic names ("Basic", "Neural", "Adaptive")
- Mathematical jargon ("Hierarchical", "Mathematical")
- Limiting implications ("Basic" implies inferiority)

## Implementation Priority

### Priority 1: Prove Value (Phases 0-2)
- Simple linear coupling only
- Clean interface and integration
- Performance validation
- **Decision Point**: Continue only if coupling shows ≥2% improvement

### Priority 2: Expand if Warranted (Phase 3+)
- Add AttentionThresholdCoupler if linear coupling successful
- Add MLPThresholdCoupler if attention shows further gains
- Only implement additional complexity if justified by results

### Priority 3: Advanced Features (Future)
- Ensemble coupling selection
- Adaptive weighting mechanisms
- Complex mathematical formulations

## Risk Mitigation

### Critical Risks Addressed
- **Over-engineering**: Start with minimal implementation
- **Performance degradation**: Benchmark at each step
- **Breaking changes**: Maintain backward compatibility
- **Complexity creep**: Require justification for each addition

### Success Gates
- **Phase 0 Gate**: Clean, simple implementation ready
- **Phase 1 Gate**: Linear coupling works and is tested
- **Phase 2 Gate**: Integration maintains compatibility
- **Phase 3 Gate**: Performance targets met

## Architecture Principles

### Code Quality
- **Modularity**: Clean separation of concerns
- **Testability**: Comprehensive test coverage
- **Maintainability**: Simple, readable code
- **Performance**: Minimal overhead

### Interface Design
- **Consistency**: Same interface pattern across couplers
- **Simplicity**: Minimal required parameters
- **Flexibility**: Easy to extend without breaking changes
- **Validation**: Clear error messages and parameter checking

## Success Metrics

**Technical Success:**
- ≤5% training time overhead for coupling
- ≥2% performance improvement on validation metrics  
- 100% backward compatibility maintained
- >95% test coverage for coupling modules

**Architectural Success:**
- Clean, maintainable interface
- Easy configuration and usage
- Minimal cognitive complexity
- Clear performance characteristics

---

**Estimated Effort**: 
- Phase 0: 3-4 hours (simplification)
- Phase 1: 4-5 hours (linear coupling)
- Phase 2: 3-4 hours (integration)
- Phase 3: 3-4 hours (validation)
- **Total**: 13-17 hours

**Critical Success Factor**: Phase 0 simplification must be completed first to avoid building on over-engineered foundation.