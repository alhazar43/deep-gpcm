# Temporal IRT Parameter Analysis in Deep-GPCM

## Key Findings

### 1. Temporal Nature of Parameters

All learned parameters in Deep-GPCM are **temporal/time-indexed**, meaning they change as students progress through questions:

- **θ (Student Ability)**: Evolves continuously as students answer questions
- **α (Item Discrimination)**: Learned dynamically for each question occurrence  
- **β (Item Thresholds)**: Learned dynamically for each question occurrence

### 2. Parameter Extraction Methods

The current implementation uses **averaging** to handle temporal parameters:

- **Item Parameters (α, β)**: Averaged across all occurrences of each question
- **Student Ability (θ)**: Two options:
  - `last`: Final ability value at the end of the sequence
  - `average`: Temporal average across the entire sequence

### 3. Correlation Results

#### Using "last" θ method:
| Model | θ Correlation | α Correlation | β Avg Correlation |
|-------|--------------|---------------|-------------------|
| Baseline | -0.033 | 0.157 | 0.303 |
| AKVMN | -0.063 | 0.265 | 0.527 |

#### Using "average" θ method:
| Model | θ Correlation | α Correlation | β Avg Correlation |
|-------|--------------|---------------|-------------------|
| Baseline | -0.114 | 0.157 | 0.303 |
| AKVMN | -0.109 | 0.265 | 0.527 |

### 4. Key Insights

1. **Poor θ Recovery**: Both models show very low correlation for student abilities (-0.03 to -0.11)
   - This suggests the models learn different ability representations than the true IRT parameters
   - The temporal evolution of abilities makes direct comparison challenging

2. **Better Item Parameter Recovery**: AKVMN shows better recovery for item parameters
   - α correlation: 0.265 (AKVMN) vs 0.157 (Baseline)
   - β correlation: 0.527 (AKVMN) vs 0.303 (Baseline)

3. **Temporal Dynamics**: The temporal analysis plots show:
   - Student abilities evolve over time (learning trajectories)
   - Population mean ability tends to stabilize or slightly decrease
   - High variance in individual trajectories

4. **Method Impact**: The choice of aggregation method affects results:
   - "Last" θ captures final knowledge state
   - "Average" θ captures overall performance level
   - Both show poor correlation with true static abilities

## Technical Implementation

### Temporal Parameter Structure

```python
# Each parameter has shape [batch_size, sequence_length, ...]
student_abilities      # [B, T]
item_discriminations   # [B, T]  
item_thresholds       # [B, T, K-1]
```

### Aggregation Approaches

1. **Item Parameters**: Average across all temporal occurrences
   ```python
   for each question q:
       α[q] = mean(all temporal α values when question q appears)
       β[q] = mean(all temporal β values when question q appears)
   ```

2. **Student Abilities**: Two approaches
   ```python
   # Last method
   θ[student] = abilities[student, -1]  # Final time step
   
   # Average method  
   θ[student] = mean(abilities[student, :])  # All time steps
   ```

## Future Improvements

1. **Better Temporal Modeling**: Consider time-aware correlation metrics that account for learning trajectories
2. **Alternative Aggregations**: Weighted averages giving more importance to later time steps
3. **Trajectory Analysis**: Compare learning trajectories rather than static parameters
4. **Separate Static/Dynamic**: Model static item properties separately from dynamic student states

## Usage

```bash
# Analyze with last θ (final ability)
python irt_analysis.py --dataset synthetic_OC --theta_method last --analysis_types recovery temporal

# Analyze with average θ (overall ability)
python irt_analysis.py --dataset synthetic_OC --theta_method average --analysis_types recovery temporal

# Use different item aggregation
python irt_analysis.py --dataset synthetic_OC --item_method last --analysis_types recovery

# Complete analysis with all visualizations
python irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal irt_plots

# Save extracted parameters
python irt_analysis.py --dataset synthetic_OC --save_params
```

## Conclusion

The temporal nature of Deep-GPCM parameters represents a fundamental difference from traditional IRT models. While this allows capturing learning dynamics, it makes direct parameter recovery comparison challenging. The models appear to learn meaningful item characterizations (moderate α and β correlations) but use a different representation for student abilities than traditional IRT.