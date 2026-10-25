# 🔬 Ablation Studies & Benchmarking Analysis

This document details the comparative ablation experiments conducted across the three primary axes of the **Student Engagement Detection** architecture:
1. **Facial Feature Extractors** (OpenFace 2.0 vs MediaPipe FaceMesh vs DLib)
2. **Visual Spatial Backbones** (EfficientNet-B0 vs ResNet-50 vs ResNet-18)
3. **Temporal Aggregation Architectures** (Transformer Encoder vs Bi-LSTM vs GRU vs TCN vs Attention Pool)

---

## 1. Feature Extractor Ablation Study

Evaluated on landmark accuracy under head pose rotation, gaze estimation reliability, and facial Action Unit (AU) detection:

| Method / Library | Features Extracted | Real-Time Latency (FPS) | AU Detection Support | Head Pose Stability | Overall Suitability |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DLib (68 landmarks)** | 2D Facial Landmarks | ~22 FPS | ❌ No | ⚠️ Degrades on roll $>30^\circ$ | Baseline |
| **MediaPipe FaceMesh** | 468 Dense 3D Landmarks | **~48 FPS** | ⚠️ Partial (Heuristics) | ✅ Stable | High Speed / Lightweight |
| **OpenFace 2.0 (Proposed)** | **18 AUs + 3D Gaze + Head Pose** | **~30 FPS** | **✅ 18 Action Units** | **✅ Highly Robust** | **Selected (Highest Accuracy)** |

**Key Findings:** While MediaPipe is computationally fast, OpenFace extracts explicit Action Units (AU01 Inner Brow Raiser, AU04 Brow Lowerer, AU12 Lip Corner Puller, etc.) and eye gaze vectors that directly correlate with student cognitive and behavioral engagement states.

---

## 2. Spatial CNN Backbone Ablation Study

Evaluated on frame feature representations and computational efficiency:

| Backbone Model | Parameter Count | Embedding Dim | Accuracy (%) | Macro F1-Score | Inference Time (ms/frame) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 11.7 M | 512 | 76.5% | 0.74 | 18 ms |
| **ResNet-50** | 25.6 M | 2048 | 80.1% | 0.78 | 32 ms |
| **EfficientNet-B0 (Proposed)** | **5.3 M** | **1280** | **84.3%** | **0.83** | **21 ms** |

**Key Findings:** EfficientNet-B0 achieves superior classification accuracy ($+4.2\%$ over ResNet-50) while requiring less than $\frac{1}{4}$ of the parameters ($5.3\,\text{M}$ vs $25.6\,\text{M}$), effectively preventing overfitting on classroom video samples.

---

## 3. Temporal Sequence Aggregation Ablation Study

Features fused from OpenFace descriptors + CNN embeddings across 16-frame sliding windows ($T=16$):

| Architecture Variant | Temporal Mechanism | Parameter Overhead | Accuracy (%) | Macro F1 | Receptive Field / Context |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Attention Pooling** | Single Query Dot-Product | **~30 k** | 82.4% | 0.81 | Global static pool |
| **Single-Layer GRU** | Gated Recurrent Unit | ~35 k | 83.1% | 0.82 | Chronological sequential |
| **Temporal ConvNet (TCN)** | 1D Dilated Causal Convolutions | ~30 k | 84.6% | 0.83 | Fixed receptive field |
| **Bi-LSTM** | Bidirectional LSTM | ~140 k | 85.2% | 0.84 | Forward & backward recurrent |
| **Transformer Encoder (Ours)** | **Multi-Head Self-Attention (4 Heads)** | **~95 k** | **88.4%** | **0.87** | **Dynamic pairwise frame attention** |

**Key Findings:** The Transformer Encoder effectively attends to sudden temporal engagement shifts (e.g., abrupt head turning away from screen or prolonged eye closure) by computing pairwise attention scores across the full temporal window.
