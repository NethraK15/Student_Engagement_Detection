# 🔄 Ablation Study: Dual-Stream Fusion + BiLSTM (Google Colab Guide)

This guide provides step-by-step instructions on how to run the **BiLSTM temporal modeling ablation study** on Google Colab.

---

## 📌 What is this Ablation Experiment?

In video-based student engagement classification, researchers evaluate which temporal modeling technique works best:

| Variant | Temporal Modeling Technique | Description | Trainable Params |
| :--- | :--- | :--- | :--- |
| **1. No Temporal Modeling** (`colab_binary_no_transformer`) | Masked Temporal Mean Pooling | Simply averages frame features across time without sequential dependencies | **265,378** |
| **2. Recurrent Modeling** (`colab_binary_bilstm`) | **2-Layer Bidirectional LSTM (BiLSTM)** | Models forward & backward temporal flow through gated recurrent states | **464,034** |
| **3. Attention Modeling** (`colab_binary_transformer`) | **3-Layer Transformer Encoder** | Multi-Head Self-Attention across all frame pairs simultaneously | **663,586** |

This gives your team a **complete, publication-quality 3-way temporal ablation comparison**!

---

## 📁 1. Google Drive Directory Structure

```text
My Drive/
└── Student_Engagement_DL/
    ├── sequence_fuse_features_binary/       ← Your preprocessed .pt files (ALREADY IN DRIVE)
    │   ├── engaged__VID20260730105033.pt
    │   ├── disengaged__VID20260730112511.pt
    │   └── ...
    │
    └── colab_binary_bilstm/                 ← UPLOAD THIS NEW FOLDER TO YOUR DRIVE
        ├── Ablation_Binary_BiLSTM_Colab.ipynb  (Interactive Colab Notebook)
        ├── config_binary_bilstm.py             (Configurations & paths)
        ├── models_binary_bilstm.py             (BiLSTM model & parameter counter)
        ├── data_loader_binary_bilstm.py        (Sliding windows & data loader)
        ├── train_binary_bilstm.py              (Training script with early stopping)
        ├── evaluate_binary_bilstm.py           (Video-level evaluation & metrics)
        └── Colab_BiLSTM_Guide.md               (This documentation)
```

---

## 🚀 2. How to Run in Google Colab

### **Step A: Upload `colab_binary_bilstm` to Google Drive**
1. Open [Google Drive](https://drive.google.com/).
2. Open your `Student_Engagement_DL` folder.
3. Drag and drop the `colab_binary_bilstm` folder from `c:\FYP\Binary classification\colab_binary_bilstm` into your Drive.

### **Step B: Open Notebook in Google Colab**
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Click **Upload** $\rightarrow$ select `Ablation_Binary_BiLSTM_Colab.ipynb` from your computer.

### **Step C: Enable Free T4 GPU**
* **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU** $\rightarrow$ **Save**.

### **Step D: Run Cells Sequentially**
1. **Cell 1:** Mount Drive (`drive.mount('/content/drive')`).
2. **Cell 2:** Set directory (`%cd /content/drive/MyDrive/Student_Engagement_DL/colab_binary_bilstm`).
3. **Cell 3:** Install packages (`pip install -q ...`).
4. **Cell 4:** Verify feature files in `sequence_fuse_features_binary`.
5. **Cell 5:** Check BiLSTM parameter breakdown.
6. **Cell 6:** Train BiLSTM model (`!python train_binary_bilstm.py`).
7. **Cell 7:** Evaluate on test videos (`!python evaluate_binary_bilstm.py`).
8. **Cell 8 & 9:** View Confusion Matrix, ROC-AUC curve, and predictions table.

---

## 📊 3. Parameter Count & Ablation Summary

```text
===========================================================================
        ARCHITECTURE PARAMETER COUNT: DUAL-STREAM + BiLSTM
===========================================================================
  Stream 1: EfficientNet-B0 Visual Extractor      : 4,007,548 params (Features only)
            [Full Standard EfficientNet-B0 Model] : 5,288,548 params (~5.29M)
  Stream 2: OpenFace Facial Behavior Stream       : 709 numerical features (0 PyTorch params)
  Dual-Stream Fused Input Dimension               : 1989 (1280 CNN + 709 OpenFace)
---------------------------------------------------------------------------
  BiLSTM ABLATION MODEL COMPONENTS:
    • Fusion & Projection (Linear + LayerNorm)    : 254,976 params
    • 2-Layer BiLSTM Stack (Hidden=64, Bidirect)  : 198,656 params
    • Temporal Pooling Layer                      : 0 params (Masked Mean Pooling)
    • MLP Classification Head (Binary 2-class)    : 10,402 params
    -------------------------------------------------------
    Total Trainable Parameters in BiLSTM Model    : 464,034
---------------------------------------------------------------------------
  ENTIRE END-TO-END PIPELINE (Backbone + BiLSTM Head):
    • Option A (Conv Features + BiLSTM Model)     : 4,471,582 params (~4.47M)
    • Option B (Full Standard B0 + BiLSTM Model)  : 5,752,582 params (~5.75M)
---------------------------------------------------------------------------
  TEMPORAL MODELING 3-WAY ABLATION COMPARISON:
    1. No Temporal Modeling (Direct Mean Pooling) : 265,378 trainable params
    2. BiLSTM Recurrent Modeling (This Model)     : 464,034 trainable params
    3. Transformer Self-Attention (Full Proposed) : 663,586 trainable params
===========================================================================
```

---

## 💾 4. Generated Outputs on Google Drive

Outputs will be saved in your Drive under:
* `models_binary_bilstm/binary_bilstm_weights.pth`: Best checkpoint.
* `evaluation_results_binary_bilstm/predictions_binary_bilstm.csv`: Video predictions.
* `evaluation_results_binary_bilstm/metrics_binary_bilstm.json`: Accuracy & F1 scores.
* `evaluation_results_binary_bilstm/confusion_matrix_binary_bilstm.png`: Confusion Matrix.
* `evaluation_results_binary_bilstm/roc_curve_binary_bilstm.png`: ROC-AUC plot.
* `evaluation_results_binary_bilstm/training_curves_binary_bilstm.png`: Loss & Accuracy curves.
