# Deep-GPCM Benchmark Summary - synthetic_OC

**Dataset**: synthetic_OC  
**Device**: cuda  
**Generated**: 2025-07-27 13:28:11  

## Performance Overview

| Model | Categorical Acc | Ordinal Acc | QWK | MAE | Parameters | Speed (ms) |
|-------|----------------|-------------|-----|-----|------------|------------|
| **Baseline** | 0.4922 | 0.8057 | 0.5667 | 0.7753 | 130,655 | 33.3 |
| **Transformer** | 0.4373 | 0.7352 | 0.4201 | 0.9382 | 276,113 | 15.5 |
| **AKT Transformer** | 0.4321 | 0.6995 | 0.3623 | 1.0105 | 1,737,597 | 23.7 |
| **Deep Integration** | 0.4913 | 1.0000 | 0.7802 | 0.5087 | 171,217 | 14.1 |

## Model Improvements

- **Transformer**: -0.0549 (-11.15%) vs baseline
- **AKT Transformer**: -0.0601 (-12.21%) vs baseline
- **Deep Integration**: -0.0009 (-0.18%) vs baseline
