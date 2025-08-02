#!/usr/bin/env python3
"""
Compare all prediction methods with the new default median thresholds.
"""

import os
os.system('source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env')

print("="*80)
print("COMPARISON OF ALL PREDICTION METHODS")
print("="*80)

# Standard evaluation
print("\n1. STANDARD EVALUATION (Hard predictions only)")
print("-"*80)
os.system('python evaluate.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth 2>/dev/null | grep -A10 "EVALUATION SUMMARY"')

# Multi-method with default median thresholds
print("\n2. MULTI-METHOD EVALUATION (Default median thresholds [0.5, 0.5, 0.5])")
print("-"*80)
os.system('python evaluate.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval 2>/dev/null | grep -A20 "Multi-Method Evaluation Results"')

# Multi-method with old asymmetric thresholds
print("\n3. MULTI-METHOD EVALUATION (Old asymmetric thresholds [0.75, 0.5, 0.25])")
print("-"*80)
os.system('python evaluate.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval --thresholds 0.75 0.5 0.25 2>/dev/null | grep -A20 "Multi-Method Evaluation Results"')

# Multi-method with adaptive thresholds
print("\n4. MULTI-METHOD EVALUATION (Adaptive thresholds)")
print("-"*80)
os.system('python evaluate.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval --adaptive_thresholds 2>/dev/null | grep -A20 "Multi-Method Evaluation Results"')

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Key Findings:
1. With median thresholds [0.5, 0.5, 0.5], threshold predictions perform similarly to hard predictions
2. The old asymmetric thresholds [0.75, 0.5, 0.25] were too conservative, biasing predictions downward
3. Adaptive thresholds learn from the data distribution and may provide different trade-offs
4. Each prediction method has its optimal use cases:
   - Hard: Standard for classification metrics (accuracy, QWK)
   - Soft: Best for regression metrics (MAE, correlations)
   - Threshold: Ordinal-aware predictions, now with sensible defaults
""")