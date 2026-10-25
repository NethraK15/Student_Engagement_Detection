# 🔬 Preliminary 3-Class Engagement Study (Review 1 Exploration)

## Overview
This directory preserves the initial research exploration conducted during **Panel Review 1**. 

During this preliminary phase, the team evaluated a **3-class engagement formulation**:
- Class 0: Engaged
- Class 1: Partially Engaged
- Class 2: Disengaged

### Explored Techniques in this Archive:
1. **`Baseline/`**: Initial single-frame appearance and feature extraction experiments.
2. **`daisee_transfer/`**: Transfer learning and domain adaptation using the public DAiSEE benchmark.
3. **`lightweight_temporal/`**: Preliminary trials with Attention Pooling, TCN, and single-layer GRU.
4. **`DPP_LSTM_Transformer_*.py`**: Early LSTM-Transformer fusion prototype.

---

## 📌 Progression to the Final Research Framework
Based on empirical observations of subtle transition boundaries between "partially engaged" and "disengaged" students in naturalistic classroom videos, the work was formalized into the **Dual-Stream Temporal Transformer Framework for Binary Engagement Detection** presented in our paper and implemented in:
- `src/`: Proposed Dual-Stream Pre-LN Transformer pipeline.
- `ablations/`: The 4-stage controlled ablation study (Stage 1 to Stage 4).
