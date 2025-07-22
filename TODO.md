# Deep-GPCM Implementation Plan

## Phase 1: Core GPCM Components 

### 1.1 GPCM Embedding Strategies 
**File**: `models/model.py` (consolidated implementation)

- Strategy 1: Ordered Embedding (R^(2Q)) - Most intuitive for partial credit
- Strategy 2: Unordered Embedding (R^(KQ)) - For MCQ-style responses
- Strategy 3: Linear Decay Embedding (R^(KQ)) - Triangular weights around actual response  
- Strategy 4: Adjacent Weighted Embedding (R^(KQ)) - Focus on adjacent categories

### 1.2 GPCM Predictor 
**File**: `models/model.py` - `DeepGpcmModel`

- IRT parameter generation (θ, α, β thresholds)
- GPCM probability calculation with cumulative logits
- Integration with DKVMN memory
- Following deep-2pl architecture patterns

### 1.3 Ordinal Loss Function 
**File**: `utils/gpcm_utils.py` - `OrdinalLoss`

```python
class OrdinalLoss(nn.Module):
    """L = -Σ Σ Σ [I(y≤k)log(P(Y≤k)) + I(y>k)log(1-P(Y≤k))]"""
    # Custom ordinal loss respecting category ordering - WORKING
```

### 1.4 Data Pipeline 
**File**: `utils/gpcm_utils.py` + `train.py`

- Auto-detect K categories from responses
- Handle both PC and OC formats  
- Metadata extraction and validation
- Simple but effective data loading

## Phase 2: Optimization 

### 2.1 Training Pipeline ENHANCED
**Files**: `train.py`, `train_cv.py`

**Completed:**
- GPCM-aware training loop with ordinal loss
- Multi-metric tracking (categorical acc, ordinal acc, MAE, QWK)
- Model checkpointing and learning rate scheduling
- Cross-validation support (5-fold like deep-2pl)
- Embedding strategy selection support
- Advanced hyperparameter configuration

### 2.2 Evaluation Framework 
**File**: `evaluate.py`

- Comprehensive model evaluation script
- Baseline comparisons (random, frequency-based)
- Per-category analysis and confusion matrices
- Statistical significance testing
- Multiple loss function comparison (ordinal, CE, MSE)

### 2.3 Alternative Embedding Strategies 
**File**: `models/model.py` - Integrated implementation

- Strategy 1 (Ordered Embedding) - R^(2Q) for partial credit
- Strategy 2 (Unordered Embedding) - R^(KQ) for MCQ data  
- Strategy 3 (Linear Decay) - R^(KQ) with triangular weights
- Embedding strategy ablation study framework
- Performance comparison across strategies and K values

### 2.4 Analysis & Visualization Tools 
**Files**: `analyze_strategies.py`, `visualize.py`

**PRIORITY FIX NEEDED**: Current visualization framework provides poor comparison capabilities

**Requirements**:
- **Two-Column Comparative Analysis**: OC vs PC format comprehensive comparison
  - Left column: OC format (Ordered Categories {0,1,2,3})
  - Right column: PC format (Partial Credit {0.000,0.333,0.667,1.000})
  - Strategy performance across both formats with identical visualization scales
  - Direct visual comparison capability with unified legends and axes

- **Enhanced Comparison Framework**:
  - Strategy performance matrices for both data formats
  - Cross-format performance correlation analysis
  - Statistical significance testing between OC and PC results
  - Embedding strategy effectiveness across format types
  - Training convergence comparison plots (side-by-side)

**Current Issues to Address**:
- Existing plots lack proper comparative structure
- No unified scaling or formatting across data formats
- Missing direct OC vs PC performance visualization
- Insufficient strategy comparison depth

## Phase 3: Experimental Design & Validation

### 3.1 Benchmark Dataset Evaluation
**Objective**: Validate Deep-GPCM on established educational datasets for comprehensive performance comparison

**Datasets**:
- **ASSISTments 2009/2015**: Large-scale math tutoring data; need to convert to PC format
- **EdNet**: Comprehensive educational interaction dataset; need to convert to PC format
- **STATICS**: Engineering mechanics assessment with multi-level scoring


**Evaluation**:
- Standardized 5-fold cross-validation across all datasets
- Performance comparison against established baselines (BKT, DKT, DKVMN, SAKT)
- Statistical significance testing with effect size analysis


### 3.2 Comparative Model Analysis
**Objective**: Position Deep-GPCM for multi-step prediction

**Target Comparisons**:
- **DKVMN&MRI (2024)**: Multi-relational information integration approach
- **BCKVMN**: Bayesian cognitive-aware memory networks
- **SAKT/AKT**: Self-attention knowledge tracing models
- **Traditional GPCM**: Classical psychometric implementations (R/Python)

**Experimental Design**:
- Controlled comparison on identical train/test splits
- Ablation studies isolating embedding strategy contributions
- Computational efficiency analysis (training time, inference speed)
- Memory usage profiling and scalability assessment

### 3.3 Educational Impact Studies
**Objective**: Demonstrate practical educational applications and interpretability

**Study Components**:
- **Interpretability Analysis**: IRT parameter evolution visualization and educational meaning
- **Predictive Accuracy**: Early prediction capability for student intervention
- **Adaptive Assessment**: Integration with computerized adaptive testing frameworks
- **Teacher Dashboard**: Practical deployment in educational technology platforms

**Validation Methods**:
- Expert educator review of IRT parameter interpretations
- Longitudinal student outcome prediction validation
- A/B testing in controlled educational environments
- Qualitative analysis of teacher usage patterns

### 3.4 Misc Remarks
**Objective**: Establish theoretical and empirical contributions to educational measurement

**Research Questions**:
1. How do neural embedding strategies impact GPCM parameter interpretation?
2. What is the optimal memory architecture for multi-category knowledge tracing?
3. How does ordinal loss function design affect educational prediction accuracy?
4. What are the computational trade-offs of different embedding dimensions?

<!-- **Publication Strategy**:
- **Target Venue**: Educational Data Mining (EDM) 2025 or Learning Analytics & Knowledge (LAK) 2025
- **Reproducibility Package**: Complete codebase with benchmark implementations
- **Supplementary Materials**: Interactive visualization tools and educational case studies -->

## Phase 4: Performance Optimization & Production Readiness

### 4.1 Computational Efficiency Enhancements
**Objective**: Optimize training time and memory usage for large-scale deployment

**Optimization Strategies**:
- **Memory Architecture Improvements**: Implement sparse memory mechanisms inspired by DKVMN&MRI
- **Gradient Computation Optimization**: Custom CUDA kernels for GPCM probability calculations
- **Batch Processing Enhancement**: Dynamic batching with variable sequence lengths
- **Model Compression**: Knowledge distillation for deployment-ready lightweight models

**Performance Targets**:
- 50% reduction in training time through vectorized operations
- 30% memory usage reduction via sparse tensor operations
- Sub-second inference time for real-time educational applications
- Scalability to 100K+ student sequences without memory overflow

### 4.2 Model Accuracy Improvements
**Objective**: Enhance prediction accuracy through advanced architectural innovations

**Research-Informed Enhancements**:
- **Multi-Relational Integration**: Incorporate exercise-exercise and learning-forgetting relationships (DKVMN&MRI approach)
- **Bayesian Inference Layer**: Add uncertainty quantification using variational inference techniques
- **Attention Mechanisms**: Self-attention for long-term dependency modeling (SAKT/AKT insights)
- **Multi-Modal Extensions**: Support for textual exercise content and visual learning materials

**Architecture Innovations**:
- **Hierarchical Memory**: Multi-level knowledge representation (concept, skill, domain)
- **Adaptive Embedding Dimensions**: Dynamic dimensionality based on data complexity
- **Temporal Modeling**: Incorporation of learning curve and forgetting curve dynamics
- **Transfer Learning**: Pre-trained embeddings from large educational language models

### 4.3 Production Deployment Optimization
**Objective**: Prepare system for large-scale educational technology integration

**Infrastructure Components**:
- **Model Serving Pipeline**: FastAPI-based REST API with automatic scaling
- **Database Integration**: Efficient storage and retrieval of student interaction data
- **Monitoring & Analytics**: Real-time performance tracking and educational impact metrics
- **A/B Testing Framework**: Controlled experiment platform for educational interventions

**Quality Assurance**:
- **Automated Testing Suite**: Unit tests for all embedding strategies and loss functions
- **Performance Regression Testing**: Continuous integration with benchmark dataset evaluation
- **Educational Validity Checks**: Automated validation of IRT parameter reasonableness
- **Privacy & Security**: FERPA-compliant data handling and model privacy preservation

### 4.4 Advanced Research Extensions
**Objective**: Explore cutting-edge research directions for future development

**Frontier Research Areas**:
- **Federated Learning**: Distributed training across multiple educational institutions
- **Causal Inference**: Identification of causal relationships in learning progression
- **Multimodal Learning Analytics**: Integration of physiological and behavioral signals
- **Explainable AI**: Advanced interpretability techniques for educational stakeholders

**Collaboration Opportunities**:
- **Industry Partnerships**: Integration with major educational technology platforms
- **Academic Collaborations**: Joint research with educational measurement and learning science groups
- **Open Source Community**: Development of educational AI toolkit for broader adoption

## Implementation Status

### Phase 1 
- [x] Project setup and structure
- [x] Synthetic data generation (OC & PC formats)
- [x] Data migration to deep-gpcm
- [x] GPCM embedding layer (Strategy 3 - Linear Decay) 
- [x] GPCM predictor with DeepGpcmModel
- [x] Ordinal loss function and custom metrics
- [x] Data loader with auto K-category detection
- [x] Training pipeline adapted from deep-2pl
- [x] Evaluation metrics and testing

### Phase 2 
- [x] Comprehensive evaluation script (`evaluate.py`)
- [x] Cross-validation support (`train_cv.py`)
- [x] Baseline comparisons (random, frequency-based)
- [x] Alternative embedding strategies (Strategy 1 & 2)
- [x] Strategy analysis framework (`analyze_strategies.py`)
- [x] Advanced visualization tools (`visualize.py`)
- [x] Performance analysis across different K values

### Phase 2.5: Advanced Metrics & Analysis (COMPLETED)
- [x] **New Prediction Accuracy Metrics Implementation**
  - [x] Prediction Consistency Accuracy (cumulative method for ordinal training consistency)
  - [x] Ordinal Ranking Accuracy (Spearman correlation between predicted and true values)
  - [x] Distribution Consistency Score (probability distribution vs ordinal structure alignment)
- [x] **Comprehensive Analysis Framework Enhancement**
  - [x] Replace QWK with new prediction accuracy metrics in comprehensive analysis
  - [x] Update comprehensive_analysis.py with new metrics integration
  - [x] Fallback mechanisms for backward compatibility with existing data
- [x] **Embedding Strategy Comparison System**
  - [x] 10-epoch training comparison across all embedding strategies (ordered, unordered, linear_decay)
  - [x] 3-column visualization framework (categorical accuracy, ordinal accuracy, prediction consistency)
  - [x] Performance ranking system with best performer identification
  - [x] Training progression analysis and final epoch comparison
- [x] **Advanced Visualization Tools**
  - [x] 3-column embedding strategy comparison plots with clear differentiation
  - [x] Training curve visualization showing strategy differences over epochs
  - [x] Final epoch performance bars with gold highlighting for best performers
  - [x] Comprehensive performance analysis and ranking system
- [x] **Documentation and Usage Updates**
  - [x] README.md updated with new use cases and advanced analysis examples
  - [x] Key features section enhanced with new prediction metrics descriptions
  - [x] Usage examples for embedding strategy comparison and advanced analysis

## Phase 1 & 2 

**Major Achievements:**

**Phase 1 - Core Implementation:**
- Linear Decay Embedding (Strategy 3) successfully implemented
- DKVMN memory integrated with K-category support
- Custom ordinal loss respecting category ordering
- Training pipeline working with synthetic_OC dataset
- Performance: 49% categorical accuracy, 77% ordinal accuracy, QWK=0.53

**Phase 2 - Enhancement & Analysis:**
- All three embedding strategies implemented and tested
- Comprehensive evaluation and cross-validation frameworks
- Strategy analysis showing optimal performance configurations
- Advanced visualization and reporting tools
- Baseline comparisons validating model performance
- Analysis findings: Linear Decay performs best for K=3, strategies converge for higher K

## Development Roadmap

**Immediate Focus (Phase 3)**: Experimental validation on benchmark educational datasets
**Future Development (Phase 4)**: Performance optimization for production deployment
**Long-term Vision**: Establishment as standard tool for educational measurement research

## Data Resources

**Available Datasets**:
- `data/synthetic_OC/`: 4-category ordered responses {0,1,2,3}
- `data/synthetic_PC/`: Partial credit scores {0.000,0.333,0.667,1.000}
- `data/large/`: Extended datasets with 800 students, 50 questions
- True IRT parameters for validation and ground truth comparison

**Evaluation Metrics**:
- **Categorical Accuracy**: Exact category prediction accuracy
- **Ordinal Accuracy**: Accuracy within ±1 category (educational tolerance)
- **Quadratic Weighted Kappa**: Standard metric for ordinal data assessment
- **Mean Absolute Error**: Average category prediction error
- **Per-Category Accuracy**: Individual category performance analysis

**Model Performance Standards**:
- Random baseline: 25% (4-category uniform)
- Educational significance threshold: >50% categorical accuracy
- Ordinal tolerance target: >75% accuracy within ±1 category

---

## Phase 5: Multi-Step Prediction Framework (FUTURE DEVELOPMENT)

### 5.1 Multi-Step Prediction Architecture Design
**Objective**: Extend Deep-GPCM for sequential multi-step response prediction

**Core Components**:
- **Sequence Extension**: Predict next N responses instead of single next response
- **Temporal Modeling**: Enhanced memory architecture for longer prediction horizons
- **Uncertainty Quantification**: Confidence intervals for multi-step predictions
- **Adaptive Horizon**: Dynamic prediction length based on student performance patterns

**Architecture Requirements**:
- Modified DKVMN memory with extended temporal context
- Multi-output GPCM predictor for sequence generation
- Hierarchical attention mechanisms for long-term dependencies
- Regularization strategies to prevent error accumulation

### 5.2 Multi-Step Evaluation Metrics Framework
**Primary Metrics**:

**Sequence-Level Metrics**:
- **Sequential Accuracy (SA_n)**: Exact sequence match for n-step predictions
- **Partial Sequence Accuracy (PSA_n)**: Average accuracy across sequence positions
- **Sequence Weighted Kappa (SWK_n)**: QWK extended to sequence evaluation
- **Early Prediction Accuracy (EPA_k)**: Accuracy at position k in n-step sequence

**Temporal Degradation Metrics**:
- **Prediction Decay Rate (PDR)**: Accuracy decline rate over prediction steps
- **Confidence Calibration (CC_n)**: Reliability of uncertainty estimates across steps
- **Temporal Stability Index (TSI)**: Consistency of predictions over time horizons

**Educational Impact Metrics**:
- **Intervention Trigger Accuracy (ITA)**: Correct identification of students needing help
- **Learning Trajectory Prediction (LTP)**: Accuracy of predicted skill development paths
- **Adaptive Assessment Efficiency (AAE)**: Optimal stopping criteria for testing

### 5.3 Research Questions for Multi-Step Extensions

**Technical Research Questions**:
1. **Optimal Prediction Horizon**: What is the maximum effective prediction length for different student populations?
2. **Memory Architecture**: How should DKVMN memory be modified for multi-step temporal modeling?
3. **Error Propagation**: How to minimize accumulating errors in sequential predictions?
4. **Computational Efficiency**: What are the scalability limits for real-time multi-step prediction?

**Educational Research Questions**:
1. **Early Intervention**: How early can we reliably predict student struggle patterns?
2. **Personalized Trajectories**: Can multi-step prediction enable truly adaptive learning paths?
3. **Assessment Optimization**: How to balance prediction accuracy with assessment efficiency?
4. **Interpretability**: How to make multi-step predictions educationally interpretable?

### 5.4 Implementation Roadmap

**Phase 5a: Foundation (3-4 months)**
- Extend current DKVMN architecture for sequence prediction
- Implement basic 2-step and 3-step prediction capabilities
- Develop sequence evaluation metrics and testing framework
- Validate on existing synthetic datasets with extended sequences

**Phase 5b: Enhancement (4-6 months)**
- Advanced temporal modeling with attention mechanisms
- Uncertainty quantification and confidence interval estimation
- Multi-horizon prediction with adaptive stopping criteria
- Performance optimization for real-time deployment

**Phase 5c: Validation (6-8 months)**
- Large-scale evaluation on educational benchmark datasets
- Comparison with state-of-the-art sequential prediction models
- Educational impact studies with real classroom deployments
- Publication and open-source release preparation

**Success Criteria**:
- **Technical**: >70% accuracy for 3-step predictions, <500ms inference time
- **Educational**: Demonstrate improved student outcomes through early intervention
- **Research**: Novel contributions to educational sequence modeling literature

### 5.5 Educational Applications

**Primary Applications**:
- **Early Warning Systems**: Proactive identification of at-risk students
- **Adaptive Learning Platforms**: Personalized learning paths based on predicted trajectories
- **Computerized Adaptive Testing**: Dynamic test length optimization
- **Curriculum Planning**: Data-driven insights for instructional design

**Secondary Applications**:
- **Student Dropout Prediction**: Early identification of disengagement patterns
- **Cognitive Load Estimation**: Real-time assessment of student cognitive state
- **Educational Resource Recommendation**: Personalized content delivery
- **Teacher Support Tools**: Actionable insights for classroom instruction

### 5.6 Dependencies & Compatibility

**Dependencies**:
- **Time Series Libraries**: Potentially `sktime` for advanced temporal analysis
- **Probabilistic Modeling**: `pyro` or `edward` for Bayesian extensions
- **High-Performance Computing**: `cupy` for GPU-accelerated computations

**Compatibility**:
- **Existing Framework**: Fully backward compatible with single-step prediction
- **Educational Standards**: Aligned with xAPI and Caliper for data interoperability
- **LMS Integration**: Designed for seamless integration with Moodle, Canvas, etc.

### 5.7 Long-Term Vision

**Goal**: Establish Deep-GPCM as the leading open-source framework for multi-step educational prediction, enabling a new generation of adaptive and personalized learning technologies.

**Future Directions**:
- **Reinforcement Learning Integration**: Optimal policy learning for educational interventions
- **Causal Inference**: Understanding the causal impact of different learning activities
- **Multi-Modal Learning**: Incorporating text, audio, and video data into predictions
- **Ethical AI**: Ensuring fairness, accountability, and transparency in educational AI
### Improvement Plan for `deep-gpcm`

**[DONE]** 7. **Refactor for Separation of Concerns:** Move evaluation and metrics logic out of the model and utility files into a dedicated evaluation module to improve code clarity and maintainability.
    *   **Action:** Create a new `evaluation/metrics.py` file.
    *   **Action:** Move the `GpcmMetrics` class from `utils/gpcm_utils.py` to the new metrics file.
    *   **Action:** Remove prediction-related methods (`gpcm_predict_*`) from the `DeepGpcmModel` in `models/model.py`. The model's `forward` method will now only output the raw probability distribution.
    *   **Action:** Update `train.py` and `evaluate.py` to use the new metrics module and work with the refactored model.

1.  **Align Training and Inference:** The most critical improvement is to use a prediction method that is consistent with the `OrdinalLoss` used during training.
    *   **Action:** Change the default prediction method from `argmax` to `cumulative`. The `cumulative` method, which is based on cumulative probabilities, directly aligns with the `OrdinalLoss` function. This should lead to a significant improvement in prediction accuracy, especially for metrics that consider ordinality.
    *   **Experiment:** Run experiments comparing the `argmax`, `cumulative`, and `expected` prediction methods to quantify the impact on accuracy. The `train.py` script already supports a `--prediction_method` argument, so this is straightforward to implement.

2.  **Hyperparameter Tuning:** Systematically tune the model's hyperparameters to find the optimal configuration.
    *   **Action:** Use a hyperparameter tuning library like Optuna or Hyperopt to search for the best combination of `learning_rate`, `batch_size`, `memory_size`, `key_dim`, `value_dim`, and `final_fc_dim`.
    *   **Focus:** The tuning process should be guided by the `prediction_consistency_accuracy` and `ordinal_ranking_accuracy` metrics, as these are most relevant to the ordinal nature of the task.

3.  **Embedding Strategy Analysis:** The `README.md` mentions four embedding strategies, but only three are implemented in `models/model.py`.
    *   **Action:** Implement the fourth strategy, "Adjacent Weighted Embedding," as described in the `TODO.md`.
    *   **Experiment:** Run a comprehensive comparison of all four embedding strategies to determine which one performs best for different datasets and numbers of categories.

4.  **Loss Function Experimentation:** While `OrdinalLoss` is theoretically sound, it's worth exploring other loss functions.
    *   **Action:** Experiment with other loss functions that are suitable for ordinal regression, such as the "Cumulative Link Loss" or "Focal Loss" adapted for ordinal data.
    *   **Comparison:** Compare the performance of these loss functions against the existing `OrdinalLoss`, `CrossEntropyLoss`, and `MSELoss`.

5.  **Regularization:** Introduce stronger regularization techniques to prevent overfitting and improve generalization.
    *   **Action:** Increase the `dropout_rate` in the `DeepGpcmModel`. Experiment with other regularization techniques like L1/L2 weight decay or batch normalization.

6.  **Architectural Enhancements:** Explore modifications to the model architecture.
    *   **Action:** As suggested in the `TODO.md`, investigate more advanced memory architectures like the one used in DKVMN&MRI, which incorporates multi-relational information.
    *   **Experiment:** Implement and evaluate the impact of these architectural changes on model performance.