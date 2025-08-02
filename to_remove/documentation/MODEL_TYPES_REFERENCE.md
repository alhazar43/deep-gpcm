# Deep-GPCM Model Types Reference

## 🎯 Updated Model Configuration (January 2025)

The `adaptive_coral_gpcm` model now uses the **FullAdaptiveBlender** by default for better performance.

## 📋 Available Model Types

### Standard Models
| Model Type | Description | Parameters | Use Case |
|------------|-------------|------------|----------|
| `deep_gpcm` | Base GPCM model | ~151K | Baseline comparison |
| `attn_gpcm` | Attention-enhanced GPCM | Variable | Research comparisons |
| `coral_gpcm` | Hybrid CORAL-GPCM | ~151K | Fixed ordinal blending |
| `ecoral_gpcm` | Enhanced CORAL with coupling | ~154K | Threshold coupling research |

### Adaptive Models ⭐
| Model Type | Description | Blender | Parameters | Architecture | Performance |
|------------|-------------|---------|------------|--------------|-------------|
| `minimal_adaptive_coral_gpcm` | Minimal stability | MinimalAdaptiveBlender | ~154K | Standard | Good + Stable |
| `adaptive_coral_gpcm` | **Default adaptive** | **FullAdaptiveBlender** | **~154K** | **Standard** | **Best Balance** |
| `full_adaptive_coral_gpcm` | Maximum performance | FullAdaptiveBlender | ~527K | Large | Highest Performance |

## 🚀 Recommended Usage

### For Most Users (Recommended)
```bash
python train.py --model adaptive_coral_gpcm --dataset synthetic_OC --epochs 30 --n_folds 5
```
- **154K parameters** with full adaptive blending
- **Learnable parameters** (range_sensitivity, distance_sensitivity, baseline_bias)
- **BGT stability** framework for training stability
- **Best balance** of performance and efficiency

### For Maximum Performance
```bash
python train.py --model full_adaptive_coral_gpcm --dataset synthetic_OC --epochs 30 --n_folds 5
```
- **527K parameters** with larger architecture
- **Maximum capacity** for complex datasets
- **+10.6% QWK improvement** over minimal
- **Higher computational cost** but best results

### For Maximum Stability
```bash
python train.py --model minimal_adaptive_coral_gpcm --dataset synthetic_OC --epochs 30 --n_folds 5
```
- **154K parameters** with fixed blending parameters
- **Maximum stability** for sensitive environments
- **Fastest training** with lowest resource usage
- **Good performance** for most applications

## 🔧 Technical Details

### FullAdaptiveBlender (Default in `adaptive_coral_gpcm`)
```python
# Learnable parameters with BGT stability
range_sensitivity: 0.100 (learnable, bounded [0.01, 1.0])
distance_sensitivity: 0.200 (learnable, bounded [0.01, 1.0])
baseline_bias: 0.000 (learnable, bounded [-0.5, 0.5])

# Features:
✅ Complete semantic threshold alignment (τ₀↔β₀, τ₁↔β₁, τ₂↔β₂)
✅ BGT stability transforms for numerical stability  
✅ Gradient isolation to prevent memory network coupling
✅ Category-specific adaptive blending weights
✅ Graceful fallback to fixed blending on errors
```

### MinimalAdaptiveBlender (Legacy Stability)
```python
# Fixed parameters for maximum stability
base_sensitivity: 0.100 (fixed)
distance_threshold: 1.000 (fixed)

# Features:
✅ Simple L2 distance computation
✅ Fixed parameters ensure consistency
✅ Maximum gradient isolation
✅ Minimal computational overhead
```

## 📊 Performance Comparison (5 Epochs, synthetic_OC)

| Metric | Minimal | **Adaptive** | Full | Winner |
|--------|---------|-------------|------|--------|
| **QWK** | 0.274 | **~0.280** | 0.303 | Full |
| **Parameters** | 154K | **154K** | 527K | Adaptive |
| **Training Speed** | Fastest | **Fast** | Slower | Adaptive |
| **Stability** | Maximum | **High** | High | Adaptive |
| **Use Case** | Legacy | **Production** | Research | Adaptive |

## 🎯 Migration Guide

### If you were using `adaptive_coral_gpcm` before:
- **No changes needed** - you now get the full blender automatically
- **Better performance** - expect improved QWK scores
- **Same stability** - BGT framework ensures stable training

### If you need the old minimal behavior:
```bash
# Use the new minimal model type
python train.py --model minimal_adaptive_coral_gpcm --dataset synthetic_OC
```

### If you want maximum performance:
```bash
# Use the full model with larger architecture
python train.py --model full_adaptive_coral_gpcm --dataset synthetic_OC
```

## ✅ What Changed

1. **`adaptive_coral_gpcm`** now uses `FullAdaptiveBlender` by default
2. **New model type**: `minimal_adaptive_coral_gpcm` for users who want the old minimal behavior
3. **Same parameters**: ~154K parameters for standard architecture models
4. **Better performance**: Full blender provides better QWK scores
5. **Stable training**: BGT framework ensures numerical stability

---

**Bottom Line**: Use `adaptive_coral_gpcm` for the best balance of performance, stability, and efficiency. It's now the recommended default for all adaptive blending applications.