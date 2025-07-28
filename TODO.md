# TODO: IRT Parameter Extraction and Visualization

## Overview
Add functionality to extract IRT (Item Response Theory) parameters from both baseline and AKVMN models and create visualization tools for these parameters.

## Phase 1: Model Updates - IRT Parameter Extraction Methods

### 1.1 Update BaselineGPCM Model (`models/baseline.py`)
- [ ] Add `get_irt_params()` method to BaselineGPCM class that extracts:
  - **θ (theta)**: From `student_abilities` output (already computed in forward pass)
  - **β (beta)**: From `item_thresholds` output (K-1 thresholds per item from `question_threshold_network`)
  - **α (alpha)**: From `discrimination_params` output (from `discrimination_network`)
- [ ] Add `extract_item_parameters(question_ids)` method to get parameters for specific items:
  - Extract from `question_threshold_network` weights for beta values
  - Extract from `discrimination_network` for base discrimination values
- [ ] Add `extract_memory_states()` method to get DKVMN memory states:
  - Key memory matrix from `self.gpcm_model.memory.key_memory_matrix`
  - Value memory states from current batch

### 1.2 Update AKVMNGPCM Model (`models/akvmn_gpcm.py`)
- [ ] Add `get_irt_params()` method matching BaselineGPCM interface:
  - Extract same parameters but handle iterative refinement cycles
  - Average or select final cycle parameters
- [ ] Add `get_refinement_history()` method specific to AKVMN:
  - Track parameter evolution across refinement cycles
  - Extract attention weights from multi-head attention layers
- [ ] Ensure parameter scaling consistency:
  - Both models use `ability_scale` parameter
  - Both use same GPCM probability computation

## Phase 2: Parameter Extraction Infrastructure

### 2.1 Create IRT Utils Module (`utils/irt_utils.py`)
- [ ] Core extraction functions:
  ```python
  def extract_irt_params_from_checkpoint(checkpoint_path, device='cpu')
  def extract_irt_params_from_model(model, data_loader, device='cpu')
  def aggregate_cv_irt_params(fold_params_list)
  ```
- [ ] Parameter processing functions:
  ```python
  def compute_item_statistics(alphas, betas)  # Mean, std, percentiles
  def compute_ability_statistics(thetas)      # Distribution stats
  def compute_test_information(alphas, betas, theta_range)
  ```
- [ ] Save/Load utilities:
  ```python
  def save_irt_params(params_dict, filepath, format='npz')  # Support npz, json, csv
  def load_irt_params(filepath)
  def export_irt_for_analysis(params, output_dir)  # Multiple formats
  ```

### 2.2 Update Training Pipeline (`train.py`)
- [ ] Add command-line arguments:
  - `--extract_irt`: Enable IRT parameter extraction
  - `--irt_output_dir`: Directory for IRT parameter files (default: 'irt_params/')
  - `--irt_save_freq`: Save IRT params every N epochs (default: 5)
- [ ] Modify training loop to extract parameters:
  - After each epoch validation, optionally extract IRT params
  - Save to `{irt_output_dir}/{model}_{dataset}_epoch{epoch}_irt.npz`
- [ ] Add to final model saving:
  - Extract final IRT parameters
  - Include in training summary JSON
  - Save as `{model}_{dataset}_final_irt.npz`

## Phase 3: Visualization Tool (`plot_irt.py`)

### 3.1 Create plot_irt.py Script Structure
- [ ] Command-line interface:
  ```bash
  python plot_irt.py --model_path MODEL_PATH --plot_type TYPE --output_dir OUTPUT
  python plot_irt.py --irt_file IRT_PARAMS.npz --plot_type all
  python plot_irt.py --compare model1.pth model2.pth --dataset synthetic_OC
  ```
- [ ] Argument parsing:
  - `--model_path`: Path to trained model checkpoint
  - `--irt_file`: Path to saved IRT parameters file
  - `--compare`: Compare two models side-by-side
  - `--plot_type`: icc, iif, tif, wright, dist, all
  - `--items`: Specific item indices to plot (default: random sample)
  - `--output_dir`: Directory for saving plots

### 3.2 Plotting Functions to Implement
- [ ] **Item Characteristic Curves (ICC)**:
  ```python
  def plot_icc(alphas, betas, item_indices, n_cats=4)
  def plot_icc_grid(alphas, betas, n_items=9, n_cats=4)
  ```
- [ ] **Item Information Functions (IIF)**:
  ```python
  def plot_iif(alphas, betas, item_indices, theta_range=(-3, 3))
  def plot_iif_overlay(alphas, betas, n_items=10)
  ```
- [ ] **Test Information Function (TIF)**:
  ```python
  def plot_tif(alphas, betas, theta_range=(-3, 3))
  def plot_tif_comparison(params1, params2, labels=['Model 1', 'Model 2'])
  ```
- [ ] **Wright Map (Item-Person Map)**:
  ```python
  def plot_wright_map(thetas, betas, alphas)
  def plot_wright_map_horizontal(thetas, betas)  # Alternative layout
  ```
- [ ] **Parameter Distributions**:
  ```python
  def plot_discrimination_dist(alphas, model_name='')
  def plot_threshold_matrix(betas, n_cats=4)  # Heatmap of thresholds
  def plot_ability_evolution(thetas_by_epoch)  # If tracking over time
  ```

## Phase 4: Integration and Testing

### 4.1 Update Evaluation Pipeline (`evaluate.py`)
- [ ] Add command-line arguments:
  - `--extract_irt`: Extract and save IRT parameters
  - `--plot_irt`: Generate IRT visualizations
- [ ] Modify evaluation to include IRT analysis:
  ```python
  # After loading model and computing metrics
  if args.extract_irt:
      irt_params = extract_irt_params_from_model(model, test_loader, device)
      save_irt_params(irt_params, f"{output_dir}/eval_irt_params.npz")
      
      # Add IRT statistics to evaluation report
      irt_stats = compute_irt_statistics(irt_params)
      evaluation_results['irt_analysis'] = irt_stats
  ```
- [ ] Add IRT-based metrics:
  - Test information at different ability levels
  - Item discrimination statistics
  - Threshold ordering violations (if any)

### 4.2 Testing with Synthetic Data
- [ ] Create test script `test_irt_extraction.py`:
  - Generate synthetic data with known IRT parameters
  - Train model and extract parameters
  - Compare extracted vs true parameters
  - Compute recovery metrics (correlation, RMSE)
- [ ] Test cases:
  - Varying number of items (10, 30, 50)
  - Different discrimination ranges
  - Various threshold patterns
  - Multiple student ability distributions
- [ ] Validation checks:
  - Parameter scale consistency
  - Threshold ordering (β₁ < β₂ < β₃ for 4 categories)
  - Discrimination positivity
  - Ability distribution normality

## Phase 5: Documentation and Examples

### 5.1 Update README.md
- [ ] Add new section "IRT Parameter Analysis":
  ```markdown
  ## IRT Parameter Analysis
  
  ### Extracting IRT Parameters
  \```bash
  # During training
  python train.py --model baseline --extract_irt --irt_save_freq 10
  
  # From trained model
  python evaluate.py --model_path MODEL.pth --extract_irt
  
  # Visualize parameters
  python plot_irt.py --model_path MODEL.pth --plot_type all
  \```
  
  ### Parameter Interpretation
  - **θ (theta)**: Student ability on scale [-3, 3]
  - **α (alpha)**: Item discrimination (higher = better differentiation)
  - **β (beta)**: Category thresholds (K-1 values per item)
  ```

### 5.2 Create Example Scripts
- [ ] `examples/irt_extraction_example.py`:
  ```python
  # Example: Extract and analyze IRT parameters
  from utils.irt_utils import extract_irt_params_from_checkpoint
  from plot_irt import plot_icc_grid, plot_tif
  
  # Load and extract
  params = extract_irt_params_from_checkpoint('model.pth')
  
  # Analyze
  print(f"Mean discrimination: {params['alpha'].mean():.3f}")
  print(f"Ability range: [{params['theta'].min():.2f}, {params['theta'].max():.2f}]")
  
  # Visualize
  plot_icc_grid(params['alpha'], params['beta'])
  plot_tif(params['alpha'], params['beta'])
  ```
- [ ] `examples/compare_models_irt.py`:
  - Load baseline and AKVMN models
  - Extract parameters from both
  - Create comparison plots
  - Statistical comparison of parameters

### 5.3 Theoretical Documentation
- [ ] Create `docs/IRT_THEORY.md`:
  - GPCM mathematical formulation
  - Parameter interpretation in educational context
  - Relationship to classical test theory
  - Best practices for IRT analysis

## Implementation Notes

### Parameter Extraction Details from Code Analysis:

#### BaselineGPCM (models/baseline.py)
- **Current outputs in forward()**: Returns tuple `(student_abilities, item_thresholds, discrimination_params, gpcm_probs)`
- **θ extraction**: Already computed via `student_ability_network` (line 455)
- **β extraction**: From `question_threshold_network` output (line 456) 
- **α extraction**: From `discrimination_network` output (line 460)
- **Key networks to access**:
  - `self.gpcm_model.student_ability_network`
  - `self.gpcm_model.question_threshold_network`
  - `self.gpcm_model.discrimination_network`
  - `self.gpcm_model.memory.key_memory_matrix` (for memory states)

#### AKVMNGPCM (models/akvmn_gpcm.py)
- **Same output format**: Compatible with baseline
- **Key differences**:
  - Has `n_cycles` of iterative refinement
  - Multi-head attention layers for each cycle
  - Refinement gates and fusion layers
- **Additional extractable**:
  - Attention weights from `self.attention_layers[cycle]`
  - Refinement history across cycles
  - Gate activations showing update magnitudes

### Technical Considerations:
1. Both models scale ability with `ability_scale` parameter (default 2.0-3.0)
2. Both use same GPCM probability computation (cumulative logits)
3. Parameters are extracted per time step in sequence
4. Need to aggregate across sequences for item-level parameters
5. Memory states are batch-specific and change during inference

### IRT Parameter Formulas:
- **GPCM Probability**: P(X=k|θ,α,β) = exp(Σ[α(θ-β_h)]) / Σ exp(...)
- **Information Function**: I(θ) = Σ α² * P * (1-P)
- **Test Information**: TIF(θ) = Σ I_i(θ) across all items

## Priority Order
1. **Phase 1**: Model methods for parameter extraction (2-3 hours)
2. **Phase 2**: IRT utilities and training integration (3-4 hours)
3. **Phase 3**: Visualization tool (4-5 hours)
4. **Phase 4**: Testing and validation (2-3 hours)
5. **Phase 5**: Documentation and examples (1-2 hours)