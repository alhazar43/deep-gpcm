# Quick Reference Commands for Deep Integration Fix

## Immediate Next Steps

### 1. Check Current "Fixed" Model Issues
```bash
# See the problematic softmax line
grep -n "probs = F.softmax" models/deep_integration_fixed.py

# Check what GPCM computation looks like in baseline
grep -A 20 "def gpcm_probability" models/baseline.py

# See training results that were "too good to be true"
cat save_models/results_deep_integration_fixed_synthetic_OC.json | grep accuracy
```

### 2. Create Proper Implementation
```bash
# Copy baseline as starting point
cp models/baseline.py models/deep_integration_gpcm_proper.py

# Key edits needed:
# 1. Add attention mechanism (but keep it simple)
# 2. Add iterative refinement (2-3 cycles max)
# 3. KEEP the gpcm_probability method
# 4. Add parameter extractors for theta, alpha, beta
```

### 3. Test on Real Data
```bash
# Don't trust synthetic_OC results!
python train.py --model deep_integration_proper --dataset STATICS --epochs 30
python train.py --model deep_integration_proper --dataset assist2015 --epochs 30

# Compare with baseline fairly
python train.py --model baseline --dataset STATICS --epochs 30
```

### 4. Debug Commands
```bash
# Check for NaN values during training
python -c "
import torch
torch.autograd.set_detect_anomaly(True)
# Then run training
"

# Monitor gradient norms
# Add this to training loop:
print(f'Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))}')

# Test single batch
python -c "
from models.deep_integration_gpcm_proper import ProperDeepIntegrationGPCM
model = ProperDeepIntegrationGPCM(n_questions=100, n_cats=4)
# Create dummy batch
q = torch.randint(0, 100, (1, 10))
r = torch.randint(0, 4, (1, 10))
out = model(q, r)
print('Output shapes:', [x.shape for x in out])
"
```

### 5. Key Files to Reference
```bash
# Baseline with correct GPCM
models/baseline.py  # Line 353: gpcm_probability method

# Original broken Deep Integration  
models/deep_integration_simplified.py  # Has NaN issues

# Oversimplified "fix"
models/deep_integration_fixed.py  # Line 273: uses softmax instead of GPCM

# Training scripts
train_baseline_standalone.py  # Works correctly
train_deep_integration_fixed_standalone.py  # Gets 99.9% (suspicious!)
```

### 6. Expected Realistic Results
```python
# Synthetic dataset (50 questions, too easy):
# Baseline: ~70% accuracy
# Deep Integration: ~75-80% accuracy (NOT 99.9%!)

# STATICS dataset (1223 questions, real data):
# Baseline: ~55% accuracy  
# Deep Integration: ~58-62% accuracy

# Key insight: Real improvement should be 5-10%, not 30%!
```

### 7. Critical Code Pattern
```python
# WRONG (current "fixed" model):
logits = self.predictor(features)
probs = F.softmax(logits, dim=-1)  # Simple softmax!

# CORRECT (what we need):
theta = self.ability_net(features)  # Student ability
alpha = self.discrimination_net(features)  # Item discrimination
betas = self.threshold_net(features)  # Item thresholds
probs = self.gpcm_probability(theta, alpha, betas)  # Proper GPCM!
```

### 8. Testing Checklist
- [ ] Model uses gpcm_probability, not just softmax
- [ ] Tested on dataset with >100 questions
- [ ] Accuracy improvement is realistic (5-10%)
- [ ] No NaN values during training
- [ ] Gradient norms stay bounded (<10)
- [ ] Can train for 30+ epochs stably

### Remember
The goal is a STABLE model with REALISTIC improvements using PROPER GPCM computation!