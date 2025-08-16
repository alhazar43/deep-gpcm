# Changelog

All notable changes to the Deep-GPCM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-08-15

### Added
- **⚖️ WeightedOrdinalLoss**: Class-balanced ordinal loss for educational assessment
  - **Automatic Class Weight Computation**: Dynamic balancing from training data using sqrt_balanced strategy
  - **Optional Ordinal Distance Penalty**: Configurable penalty for distant category mistakes
  - **Factory Integration**: Seamless configuration through model factory with `weighted_ordinal_weight` parameter
  - **Device Compatibility**: Enhanced GPU/CPU device handling for multi-device training
  - **Flexible Usage Modes**: Pure class-weighted CE or class-weighted + ordinal penalty modes
- **Comprehensive Documentation**: Mathematical formulation, architectural integration, and usage guides
  - **SUMMARY.md**: Complete implementation overview and usage instructions
  - **ARCHITECTURE.md**: Updated with WeightedOrdinalLoss architectural integration
  - **MATH.md**: Formal mathematical notation, theorems, and gradient analysis

### Changed
- **Enhanced Loss Function Architecture**: Extended CombinedLoss with WeightedOrdinalLoss integration
- **Updated Model Configurations**: All models now include WeightedOrdinalLoss with conservative default weights
- **Improved Training Pipeline**: Added automatic class weight computation and injection
- **Enhanced Device Management**: Fixed device mismatch issues for multi-GPU compatibility

## [2.1.0] - 2025-08-14

### Added
- **🚀 Adaptive Hyperparameter Optimization**: State-of-the-art intelligent parameter search
  - **3-Phase Adaptive Epochs**: Intelligent allocation (5→15→40 epochs) with early stopping
  - **Model-Aware Parameter Search**: 12-15 parameters vs original 5, filtered by model capabilities
  - **Enhanced Search Space**: Architectural (key_dim, value_dim, embed_dim, n_heads) + learning parameters (lr, weight_decay, batch_size, grad_clip, label_smoothing)
  - **Fallback Safety System**: Automatic degradation to original optimization if adaptive features fail
  - **AI-Generated Analysis**: Parameter importance ranking, performance insights, and optimization recommendations
  - **Resource Efficiency**: 60-70% computation reduction through intelligent early stopping
- **Comprehensive Documentation**: Architecture diagrams, implementation details, and future roadmap
  - **ARCHITECTURE.md**: Complete adaptive optimization architecture with multi-stage intelligence stack
  - **TODO.md**: Future enhancement roadmap with advanced search strategies and MLOps integration
- **Production Integration**: Seamless pipeline integration with backward compatibility
  - Enhanced `train.py` and `main.py` with unified adaptive argument parsing
  - Automatic type conversion safety for Optuna numpy types
  - Model-aware parameter filtering to prevent incompatible parameters

### Changed
- **Major**: Enhanced Bayesian optimization with TPE sampler and adaptive epoch allocation
- **Major**: Expanded search space from 5 to 12-15 parameters with model-specific filtering
- **Minor**: Updated README and SUMMARY with adaptive optimization features
- **Minor**: Improved argument parsing with adaptive configuration options

### Fixed
- **Critical**: Type conversion issues (numpy int64 → Python int) for PyTorch compatibility
- **Major**: Model parameter filtering to exclude training-only parameters from model constructor
- **Minor**: Enhanced error handling and recovery for optimization failures

## [2.0.0] - 2024-01-15

### Added
- **Comprehensive Documentation Restructure**: Created focused documentation system
  - `ARCHITECTURE.md`: System and model architecture documentation
  - `SUMMARY.md`: Project overview with performance metrics and IRT analysis
  - `MATH.md`: Mathematical foundations and theoretical formulations
  - Streamlined `README.md` to usage-only content
- **Bayesian Hyperparameter Optimization**: Full Optuna integration with Gaussian Process regression
- **Advanced Loss Weight Optimization**: Automatic tuning of combined loss components
- **Enhanced IRT Analysis Pipeline**: Parameter recovery with temporal modeling
- **CORAL Ordinal Regression**: Proper IRT-CORAL separation implementation
- **Comprehensive Visualization Suite**: 9-panel optimization dashboards

### Changed
- **Major**: Restructured documentation from 10+ scattered files to 5 focused documents
- **Major**: Enhanced mathematical formulations with rigorous theoretical foundations
- **Minor**: Improved README structure focusing solely on usage instructions
- **Minor**: Updated performance metrics with validated cross-validation results

### Fixed
- **Critical**: Resolved CORAL design flaw with proper τ threshold separation
- **Major**: Fixed gradient instability in DKVMN memory networks
- **Minor**: Corrected binary classification support for 2-category datasets
- **Minor**: Enhanced IRT parameter extraction accuracy

## [1.5.0] - 2024-01-10

### Added
- **Unified Training Pipeline**: Consolidated argument parsing with backward compatibility
- **Comprehensive Model Factory**: Dynamic model registry with configuration management
- **Advanced Memory Networks**: DKVMN architecture with attention mechanisms
- **Temporal IRT Modeling**: Dynamic parameter extraction with time-series analysis

### Changed
- **Major**: Unified data architecture with auto-detection for 9 datasets
- **Minor**: Streamlined cross-validation logic with simplified interface
- **Minor**: Enhanced model evaluation with 8+ comprehensive metrics

### Fixed
- **Critical**: 46% performance improvement (0.476 → 0.691 AUC) through gradient flow optimization
- **Major**: Eliminated data_style parameter confusion across all scripts
- **Minor**: Fixed Q-matrix auto-detection for per-KC vs global mode switching

## [1.0.0] - 2024-01-05

### Added
- **Core Deep-GPCM Implementation**: DKVMN + IRT + GPCM architecture
- **Multi-Model Support**: 6 model variants with different architectures
- **Cross-Validation Framework**: 5-fold CV with comprehensive evaluation
- **Dataset Support**: 9 educational datasets with auto-detection
- **Visualization Pipeline**: Training curves and performance plots
- **IRT Analysis Tools**: Parameter recovery and temporal analysis

### Initial Features
- Production-ready training pipeline
- Comprehensive evaluation metrics
- Model comparison framework
- Automated cleanup utilities
- Extensible architecture for research

---

## Version Numbering Convention

- **Major Version** (X.0.0): Significant architectural changes, breaking API changes, major feature additions
- **Minor Version** (X.Y.0): New features, enhancements, non-breaking changes, performance improvements
- **Patch Version** (X.Y.Z): Bug fixes, small optimizations, documentation updates

### Change Categories

- **Added**: New features, capabilities, or documentation
- **Changed**: Modifications to existing functionality or structure
- **Deprecated**: Features marked for removal in future versions
- **Removed**: Features that have been deleted
- **Fixed**: Bug fixes and error corrections
- **Security**: Vulnerability fixes and security improvements

### Impact Levels

- **Critical**: Core functionality changes, major performance impacts
- **Major**: Significant feature additions or substantial modifications
- **Minor**: Small enhancements, documentation improvements, minor fixes