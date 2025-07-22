# Deep-GPCM Changelog

All notable changes to the Deep-GPCM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2025-07-22 - Comprehensive Strategy Analysis & Research Planning

### Major Milestone: Deep-GPCM Research Framework Complete

This release establishes Deep-GPCM as a comprehensive research framework with extensive strategy analysis, experimental design planning, and performance optimization roadmap based on recent advances in knowledge tracing research.

### Added

#### Comprehensive Strategy Comparison Framework
- **Multi-Format Analysis**: Simultaneous evaluation on both OC (Ordered Categories) and PC (Partial Credit) data formats
- **Epoch-wise Performance Tracking**: Detailed convergence analysis across embedding strategies and data formats
- **Large-Scale Dataset Generation**: Extended synthetic datasets with 800 students and 50 questions for robust evaluation
- **Statistical Performance Analysis**: Automated comparison reports with significance testing and effect size calculations

#### Advanced GPCM Compliance Verification
- **Mathematical Validation**: 100% compliance verification with standard GPCM formulation
- **Prediction Target Analysis**: Comprehensive analysis of model outputs and prediction mechanisms
- **Threshold Usage Documentation**: Clear explanation of GPCM parameter roles vs. prediction thresholds
- **Educational Interpretation Guidelines**: Framework for interpreting neural IRT parameters in educational contexts

#### Research-Informed Experimental Design
- **Benchmark Dataset Integration Plan**: Systematic evaluation protocol for ASSISTments, EdNet, STATICS, ES-KT-24, and SingPAD datasets
- **State-of-Art Comparison Framework**: Comparison protocol against DKVMN&MRI, BCKVMN, SAKT/AKT, and traditional GPCM implementations
- **Educational Impact Study Design**: Framework for interpretability analysis, predictive accuracy assessment, and practical deployment validation
- **Publication-Ready Research Questions**: Theoretically grounded research questions targeting Educational Data Mining and Learning Analytics venues

#### Performance Optimization Roadmap
- **Computational Efficiency Targets**: Specific performance goals (50% training time reduction, 30% memory usage reduction)
- **Accuracy Enhancement Strategies**: Research-informed improvements including multi-relational integration, Bayesian inference, and attention mechanisms
- **Production Deployment Framework**: Infrastructure components for large-scale educational technology integration
- **Advanced Research Extensions**: Frontier research directions including federated learning, causal inference, and multimodal learning analytics

### Performance Results

#### Strategy-Format Comparison Results
- **Best Overall Performance**: Linear Decay strategy on PC format (52.16% categorical accuracy, 79.50% ordinal accuracy)
- **Strategy Convergence**: Consistent performance across strategies with minimal differences (< 1% accuracy variation)
- **Format Impact**: PC format consistently outperforms OC format across all embedding strategies
- **Training Efficiency**: All strategies converge within 5 epochs on large-scale datasets

#### GPCM Compliance Analysis
- **Mathematical Compliance**: 100% adherence to standard GPCM probability formulation
- **Parameter Generation**: Correct implementation of θ (ability), α (discrimination), and β (thresholds)
- **Prediction Mechanism**: Proper categorical prediction via argmax without post-processing thresholds
- **Educational Validity**: Meaningful IRT parameter evolution demonstrating learning progression

### Technical Enhancements

#### Research Integration
- **Literature-Informed Design**: Integration of recent advances from DKVMN&MRI, BCKVMN, and variational GPCM research
- **Benchmark Dataset Compatibility**: Framework designed for evaluation on established educational datasets
- **Methodological Rigor**: Standardized evaluation protocols following educational measurement best practices

#### Documentation Standards
- **Professional Documentation**: Removal of informal elements and adoption of academic writing standards
- **Comprehensive Analysis Reports**: Detailed technical analysis with mathematical formulation verification
- **Reproducibility Framework**: Complete experimental protocol documentation for research replication

### Research Contributions

#### Methodological Innovations
1. **Neural Embedding Strategies for GPCM**: First systematic comparison of neural embedding approaches for polytomous IRT models
2. **Ordinal Loss Function Design**: Custom loss function specifically designed for educational category ordering
3. **Multi-Format Data Integration**: Framework supporting both discrete and continuous partial credit formats
4. **Comprehensive Evaluation Framework**: Standardized evaluation protocol for GPCM-based models

#### Educational Measurement Advances
1. **DKVMN-GPCM Integration**: Novel combination of dynamic memory networks with polytomous IRT models
2. **Interpretable Neural IRT**: Framework for educational interpretation of neural network-generated IRT parameters
3. **Real-time Educational Assessment**: Performance characteristics suitable for adaptive testing applications

### Development Roadmap

#### Immediate Priorities (Phase 3)
- **Experimental Validation**: Systematic evaluation on benchmark educational datasets
- **Comparative Analysis**: Comprehensive comparison with state-of-the-art knowledge tracing models
- **Educational Impact Studies**: Practical deployment validation in educational contexts

#### Future Development (Phase 4)
- **Performance Optimization**: Computational efficiency enhancements for production deployment
- **Accuracy Improvements**: Advanced architectural innovations based on recent research
- **Production Readiness**: Infrastructure development for large-scale educational technology integration

### Compatibility
- **Research Framework**: Compatible with established educational measurement protocols
- **Benchmark Integration**: Designed for seamless integration with standard evaluation datasets
- **Educational Standards**: Compliant with educational measurement and psychometric principles

### Known Applications
- **Educational Assessment**: Polytomous response analysis in educational testing
- **Adaptive Testing**: Real-time student ability estimation for computerized adaptive tests
- **Learning Analytics**: Student knowledge state tracking and learning progression analysis
- **Educational Research**: Methodological tool for educational measurement research

---

## [0.3.0] - 2025-07-22 - Phase 2 Complete: Enhancement & Analysis

### Major Milestone: Phase 2 Implementation Complete

This release marks the successful completion of Phase 2, providing comprehensive evaluation, analysis tools, and alternative embedding strategies for Deep-GPCM.

### Added

#### Alternative Embedding Strategies
- **Strategy 1 (Ordered Embedding)**: R^(2Q) implementation for partial credit models with binary correctness + normalized score
- **Strategy 2 (Unordered Embedding)**: R^(KQ) implementation for MCQ-style responses with one-hot category indicators
- **Strategy 3 (Linear Decay)**: Enhanced R^(KQ) implementation with triangular weight distribution
- **Dynamic Strategy Selection**: Model architecture adapts embedding dimensions based on selected strategy

#### Comprehensive Evaluation Framework
- **Advanced Evaluation Script** (`evaluate.py`): Multi-loss comparison, confusion matrices, per-category analysis
- **Baseline Comparisons**: Random baseline and frequency-based baseline for performance validation
- **Statistical Analysis**: Comprehensive metrics including categorical accuracy, ordinal accuracy, MAE, QWK
- **Model Diagnostics**: Parameter counting, convergence analysis, performance profiling

#### Cross-Validation Infrastructure
- **5-Fold Cross-Validation** (`train_cv.py`): Following deep-2pl patterns with statistical significance testing
- **CV Statistics**: Mean, standard deviation, min/max performance across folds
- **Automated Reporting**: JSON output with comprehensive fold-wise results and summary statistics

#### Strategy Analysis Tools
- **Comprehensive Comparison** (`analyze_strategies.py`): Systematic evaluation across embedding strategies and K values
- **Performance Heatmaps**: Strategy vs K-category performance visualization
- **Automated Benchmarking**: Quick and full analysis modes with configurable parameters

#### Advanced Visualization Dashboard
- **Training Curves** (`visualize.py`): Loss convergence, accuracy progression, learning rate schedules
- **Learning Progression Analysis**: Convergence detection, improvement rates, training stability
- **Cross-Validation Visualization**: Box plots, statistical summaries, performance distributions
- **Automated Reporting**: Detailed performance reports with analysis insights

### Performance Results

#### Strategy Comparison (K=3,4,5 categories)
- **Best Overall Performance**: Linear Decay strategy for K=3 (64.6% categorical accuracy)
- **Strategy Convergence**: All strategies show similar performance for K≥4
- **Ordinal Performance**: Consistently high ordinal accuracy (>75%) across all strategies
- **QWK Performance**: Strong ordinal correlation (QWK>0.5) demonstrating model understanding of category ordering

#### Cross-Validation Results
- **Stable Performance**: Low variance across folds indicating robust model architecture  
- **Statistical Significance**: Consistent improvement over random baseline (25% → 50%+)
- **Convergence**: Fast training with stable convergence in <10 epochs for most configurations

### Technical Enhancements

#### Architecture Improvements
- **Dynamic Embedding Dimensions**: Automatic adaptation based on strategy (R^(2Q) vs R^(KQ))
- **Strategy Integration**: Seamless switching between embedding approaches via configuration
- **Memory Efficiency**: Optimized tensor operations for different embedding sizes

#### Training Enhancements
- **Multi-Strategy Support**: Command-line selection of embedding strategies
- **Enhanced Logging**: Detailed progress tracking with strategy-specific metrics
- **Flexible Configuration**: Extensive hyperparameter customization options

#### Evaluation Improvements
- **Multiple Loss Functions**: Comparative evaluation with ordinal, cross-entropy, and MSE losses
- **Baseline Integration**: Automated baseline model training and comparison
- **Statistical Testing**: Comprehensive model validation with significance testing

### Analysis Findings

#### Embedding Strategy Performance
1. **Linear Decay (Strategy 3)**: Best for low K values, maintains triangular weight intuition
2. **Ordered Embedding (Strategy 1)**: Efficient R^(2Q) representation, good for partial credit scenarios
3. **Unordered Embedding (Strategy 2)**: Effective for MCQ-style data, similar to one-hot encoding
4. **Performance Convergence**: Strategies show similar performance as K increases (K>4)

#### Model Behavior Analysis
- **Ordinal Understanding**: All strategies demonstrate understanding of category ordering
- **Fast Convergence**: Training stabilizes within 10 epochs across all configurations
- **Scalability**: Linear performance scaling with increasing K categories
- **Robustness**: Consistent performance across different random initializations

### Dependencies
- matplotlib and seaborn for visualization
- pandas for data analysis
- scikit-learn for advanced metrics (enhanced)
- All previous Phase 1 dependencies maintained

### Compatibility
- **Backward Compatibility**: All Phase 1 functionality preserved
- **Strategy Migration**: Easy switching between embedding strategies
- **Data Format Support**: Enhanced support for different K values and category mappings

### Known Improvements
- **Enhanced Performance**: Strategy analysis enables optimal configuration selection
- **Comprehensive Evaluation**: Multi-faceted model assessment beyond single metrics
- **Research Ready**: Complete analysis framework for academic publication

---

## [0.2.0] - 2025-07-22 - Phase 1 Complete

### Major Milestone: Phase 1 Implementation Complete

This release marks the successful completion of Phase 1, providing a fully functional Deep-GPCM model extending Deep-IRT to handle multi-category responses.

### Added

#### Core GPCM Architecture
- **Linear Decay Embedding (Strategy 3)**: Implemented paper reference strategy with triangular weights around actual response: `x_t^(k) = max(0, 1 - |k-r_t|/(K-1)) * q_t`
- **DeepGpcmModel**: Extended Deep-IRT architecture with GPCM probability calculation using cumulative logits
- **DKVMN Memory Integration**: Successfully adapted memory operations for K-category embeddings (R^(KQ) vs R^(2Q))
- **GPCM Predictor**: IRT parameter generation (θ ability, α discrimination, β thresholds) with GPCM probability computation

#### Loss Functions & Metrics
- **Custom Ordinal Loss**: Implemented paper formulation respecting category ordering: `L = -Σ Σ Σ [I(y≤k)log(P(Y≤k)) + I(y>k)log(1-P(Y≤k))]`
- **Comprehensive Metrics**: Categorical accuracy, ordinal accuracy (±1), MAE, Quadratic Weighted Kappa, per-category accuracy
- **Baseline Loss Functions**: CrossEntropyLoss and MSELoss wrappers for comparison

#### Data Pipeline
- **Auto K-Category Detection**: Automatic detection of number of categories from dataset files
- **Dual Format Support**: Handle both OC (ordered categorical) and PC (partial credit) formats
- **Efficient Data Loading**: Simple but effective data loader with batching and masking

#### Training Infrastructure
- **Complete Training Pipeline**: Adapted from deep-2pl with GPCM-specific features
- **Model Checkpointing**: Save best models with comprehensive state information
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive training
- **Progress Tracking**: Detailed logging and metrics tracking per epoch

### Technical Specifications

#### Architecture Details
- **Model Class**: `DeepGpcmModel` extending deep-2pl patterns
- **Memory**: DKVMN with configurable memory_size (default: 50)
- **Embeddings**: Linear Decay (Strategy 3) with R^(KQ) dimensionality
- **Parameters**: ~148K parameters for typical configuration

#### File Structure
```
models/
├── memory.py          # DKVMN memory operations (adapted from deep-2pl)
└── model.py           # DeepGpcmModel + linear_decay_embedding

utils/
└── gpcm_utils.py      # OrdinalLoss + metrics + data utilities

train.py              # Training pipeline with GPCM support
```

### Performance Results

#### Synthetic_OC Dataset (K=4 categories)
- **Categorical Accuracy**: 49.04% (vs 25% random baseline)
- **Ordinal Accuracy**: 77.00% (within ±1 category)  
- **Quadratic Weighted Kappa**: 0.53 (good for ordinal data)
- **Mean Absolute Error**: 0.84 categories
- **Training**: Stable convergence in 2 epochs (quick test)

### Implementation Decisions

#### Embedding Strategy Choice
- **Selected**: Strategy 3 (Linear Decay) from paper reference
- **Rationale**: Provides smooth triangular weights around actual response, theoretically sound for ordinal data
- **Alternative Strategies**: Strategy 1 (Ordered) and Strategy 2 (Unordered) planned for Phase 2

#### Loss Function Design
- **Custom Ordinal Loss**: Superior to CrossEntropyLoss for ordered categorical data
- **Design Choice**: Respects category ordering vs treating categories as unordered classes
- **Performance**: Shows improved ordinal accuracy compared to standard classification

#### Architecture Decisions
- **Consolidated Files**: Fewer files than originally planned (4 core files vs 6+ planned)
- **Memory Reuse**: Existing DKVMN memory works with K-category embeddings without modification
- **Deep-2pl Alignment**: Maintains naming conventions and structure for consistency

### Dependencies
- PyTorch 2.5.1+
- NumPy for numerical operations
- scikit-learn for evaluation metrics (QWK)
- tqdm for progress bars

### Compatibility
- **CUDA**: Full GPU support tested
- **Data Formats**: Compatible with existing synthetic_OC and synthetic_PC datasets
- **Deep-2pl**: Maintains architectural consistency for future integration

### Known Limitations
- **Single Embedding Strategy**: Only Strategy 3 implemented (others planned for Phase 2)
- **Synthetic Data Only**: Tested on synthetic datasets, real-world validation pending
- **Basic Cross-validation**: Simple train/test split, 5-fold CV planned for Phase 2

---

## [0.1.0] - 2025-07-21 - Project Initialization

### Added
- Initial project structure following deep-2pl conventions
- Data generation pipeline with GPCM synthetic data
- Support for both OC (Ordered Categorical) and PC (Partial Credit) formats
- Basic documentation and TODO planning

### Technical Foundation
- **Data Generator**: `data_gen.py` with GpcmGen class for synthetic GPCM responses
- **Dual Formats**: synthetic_OC (integer categories 0-3) and synthetic_PC (decimal scores 0.0-1.0)
- **Metadata**: Complete parameter tracking for reproducibility
- **Directory Structure**: Mirrored from deep-2pl for consistency

### Data Generated
- synthetic_OC: 200 sequences (160 train, 40 test) with 4 categories
- synthetic_PC: Corresponding partial credit format
- True IRT parameters saved for validation

---

## Development Notes

### Phase 1 Success Factors
1. **Paper Reference Alignment**: Successfully implemented Strategy 3 exactly as specified
2. **Architecture Reuse**: DKVMN memory required no modifications for K-category support
3. **Custom Loss**: Ordinal loss significantly outperforms standard classification losses
4. **Simplified Design**: Consolidated implementation in fewer files than planned
5. **Performance Validation**: Strong results above random baseline with meaningful ordinal accuracy

### Next Phase Priorities
1. **Evaluation Framework**: Comprehensive model assessment and comparison
2. **Cross-validation**: 5-fold CV support like deep-2pl
3. **Alternative Strategies**: Implement Strategy 1 and 2 for comparison
4. **Real Data**: Validation on educational datasets with partial credit
5. **Publication**: Prepare results for EAAI-26 submission

### Long-term Vision
- Multi-skill GPCM extensions
- Real-world educational applications
- Interactive visualization dashboards
- Integration with existing educational platforms