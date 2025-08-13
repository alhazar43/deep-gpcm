# ATTN-GPCM Abstract
**Option 1**
## *Knowledge Tracing with Partial Credit: A Neural-IRT Approach for Ordinal Response Prediction*

**Option 2**
## *ATTN-GPCM: Attentive Memory Network for Knowledge Tracing with Partial Credit*

Knowledge tracing (KT) models student performance by predicting future interactions. While recent KT models excel at binary prediction, they fail to predict ordinal and graded responses required to model true competency levels in partial credit settings. To bridge this gap, we propose ATTN-GPCM, which combines a deep knowledge tracing model with the Generalized Partial Credit Model (GPCM) to make modern KT architectures compatible with ordinal responses. Specifically, our model uses a self-attention embedding to capture ordinal contexts in past interactions. The embedded states are processed sequentially through Dynamic Key-Value Memory Network (DKVMN) to trace student's knowledge state over time. We then use the GPCM to estimate the probability of each ordinal category. Experiments show that our model achieves strong performance on ordinal metrics and successfully recovers underlying psychometric item parameters, validating the synthesis of deep knowledge tracing with psychometric (use **Item-Response Theory** if go with title option 1) models.

---

## Literature Summary & Positioning

### Existing Schemes

**Performance Benchmarks:**
- DKVMN typically achieves AUC ~0.82 on binary classification tasks
- Attention-based models show 3-5% improvements over baseline DKVMN
- Deep learning approaches significantly outperform classical BKT models

**Research Gaps:**
- Overwhelming focus on binary correct/incorrect responses
- Limited integration of classical psychometric theory with neural architectures
- Absence of partial credit modeling (discrete, ordered categories) in recent deep knowledge tracing approaches
- No existing work combines DKVMN with GPCM for ordinal response prediction

### ATTN-GPCM's Contributions

1. **First neural-IRT integration for polytomous** knowledge tracing
2. **Temporal parameter extraction** enabling dynamic psychometric modeling
3. **Mathematical consistency** (in terms of IRT parameters recovery)
4. **First Ordinal Benchmark (OR Dataset)** for knowledge tracing with partial credit

### Overall Architecture
***Linear Decay Embedding*** -> ***Multi-Head Attention*** -> Projection -> DKVMN -> GPCM (IRT Extraction) -> labels (softmax)

(Bold Italic for architectural hightlight)


