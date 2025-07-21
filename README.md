# Deep-GPCM: Generalized Partial Credit Model for Knowledge Tracing

Extension of Deep-IRT with support for polytomous (K-category) responses using the Generalized Partial Credit Model (GPCM).

## Overview

Deep-GPCM extends traditional binary knowledge tracing to handle:
- **Partial Credit Responses**: Decimal scores in [0,1] 
- **Ordered Categories**: Discrete K-category responses {0, 1, 2, ..., K-1}
- **Multiple Embedding Strategies**: 4 different approaches for Q-A encoding

## Key Features

- ✅ **GPCM Model**: Proper IRT-based polytomous prediction
- ✅ **Dual Data Formats**: PC (partial credit) and OC (ordered categories)
- ✅ **Multiple Embeddings**: Ordered, unordered, linear decay, adjacent weighted
- ✅ **Ordinal Loss**: Specialized loss function for ordered responses
- ✅ **Synthetic Data**: Generated datasets for testing and validation

## Quick Start

### Generate Synthetic Data
```bash
python generate_synthetic_gpcm.py --format both --categories 4
```

### Train GPCM Model
```bash
python train.py --dataset synthetic_OC --categories 4 --embedding ordered
```

### Evaluate Model
```bash
python evaluate.py --model_path models/gpcm_model.pth --dataset synthetic_OC
```

## Data Formats

### Ordered Categories (OC)
```
48
26,9,25,18,6,29,...
2,0,2,0,3,3,...     # Categories: 0,1,2,3
```

### Partial Credit (PC)  
```
48
26,9,25,18,6,29,...
0.667,0.000,0.667,0.000,1.000,1.000,...  # Scores: [0,1]
```

## Architecture

```
Input: (questions, responses) → Embedding → DKVMN → GPCM Predictor → K-category probabilities
```

### Embedding Strategies
1. **Ordered (2Q)**: `[low_component, high_component]` - Most intuitive for partial credit
2. **Unordered (KQ)**: One-hot style for each category
3. **Linear Decay (KQ)**: Triangular weights around actual response  
4. **Adjacent Weighted (KQ)**: Focus on actual + adjacent categories

## Implementation Status

- [x] Project setup and data generation
- [ ] GPCM embedding layer  
- [ ] GPCM predictor with probability calculation
- [ ] Ordinal loss function
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Model persistence

## Requirements

```
torch>=1.9.0
numpy>=1.20.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```