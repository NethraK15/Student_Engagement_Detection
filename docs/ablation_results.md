# 🔬 Comprehensive Experimental Results & Ablation Analysis

This document provides the full experimental evaluation data for the research paper:
> *"A Dual-Stream Temporal Transformer Framework for Student Engagement Detection in Classroom Videos"*

---

## 1. Dataset Partitioning (Table 3)
- Total Dataset: 133 naturalistic classroom video recordings (65 Engaged, 68 Disengaged)
- Partitioning: Video-level stratified 70 / 15 / 15 split
- Window size: $W = 120$ frames ($4\,\text{s}$ at $30\,\text{fps}$)
- Stride: $S = 30$ frames ($1\,\text{s}$ step, $75\%$ overlap)

| Partition | Videos | Engaged | Disengaged | Windows |
| :--- | :---: | :---: | :---: | :---: |
| **Training** | 93 | 45 | 48 | 727 |
| **Validation** | 20 | 10 | 10 | 142 |
| **Test** | 20 | 10 | 10 | 150 |
| **Total** | **133** | **65** | **68** | **1,019** |

---

## 2. Stage 1: Geometric Feature Stream Ablation (Table 5)
Comparison of geometric facial extractors under an identical 60-epoch training protocol. Test accuracies are reported with two-sided 95% Wilson score confidence intervals.

| Extractor | Input Dim | Trainable Parameters | Validation Accuracy | Test Accuracy [95% CI] | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dlib (EAR / MAR)** | 6 | 1,154 | 55.56% | 52.63% [31.7, 72.7] | 27.70% | 52.63% | 36.30% |
| **MediaPipe FaceMesh** | 55 | 7,426 | 66.67% | 68.42% [46.0, 84.6] | 68.92% | 68.42% | 67.88% |
| **OpenFace 2.0 (Proposed)** | **709** | **16,770** | **85.00%** | **70.00% [48.1, 85.5]** | **73.81%** | **70.00%** | **68.75%** |

---

## 3. Stage 2: Visual CNN Backbone Ablation (Table 6)
Comparison of frozen feature backbones evaluated on 20 test videos:

| Backbone | Feature Dimension | Backbone Parameters | Validation Accuracy | Test Accuracy [95% CI] | Parameter Change |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 512 | $\approx 11.18\,\text{M}$ | 85.00% | 75.00% [53.1, 88.8] | Baseline |
| **EfficientNet-B0 (Proposed)** | **1,280** | **$\approx 4.01\,\text{M}$** | **75.00%** | **75.00% [53.1, 88.8]** | **−64.1% reduction** |

---

## 4. Stage 3: Complementarity of Visual & Behavioral Streams (Table 7)
Demonstrating the benefit of fusing the frozen EfficientNet-B0 stream with the OpenFace 2.0 stream under static order-invariant pooling:

| Configuration | Modality / Stream | Dim | Test Accuracy [95% CI] | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stream 1 Only** | EfficientNet-B0 | 1,280 | 75.00% [53.1, 88.8] | 75.00% | 75.00% | 75.00% |
| **Stream 2 Only** | OpenFace 2.0 | 709 | 70.00% [48.1, 85.5] | 73.81% | 70.00% | 68.75% |
| **Dual-Stream Fused** | **EfficientNet-B0 + OpenFace** | **1,989** | **85.00% [64.0, 94.8]** | **85.00%** | **85.00%** | **85.00%** |

---

## 5. Stage 4: Temporal Sequence Models (Table 8)
Comparison of sequence architectures over the 1,989-dimensional dual-stream representation:

| Model | Sequence Engine | Trainable Parameters | Window Val Acc (142 windows) | Test Acc [95% CI] (20 videos) | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Dual-Stream** | Mean Pooling | 265,378 | 86.62% | 85.00% [64.0, 94.8] | 85.00% | 85.00% | 85.00% |
| **Dual-Stream + BiLSTM** | 2-layer BiLSTM | 464,034 | 70.42% | 80.00% [58.4, 91.9] | 80.00% | 80.00% | 80.00% |
| **Dual-Stream + LSTM** | 2-layer LSTM | 529,826 | 76.06% | 90.00% [69.9, 97.2] | 90.00% | 90.00% | 90.00% |
| **Dual-Stream + Transformer** | **3-layer Pre-LN MHSA** | **663,074** | **83.80%** | **85.00% [64.0, 94.8]** | **85.71%** | **85.00%** | **85.35%** |
