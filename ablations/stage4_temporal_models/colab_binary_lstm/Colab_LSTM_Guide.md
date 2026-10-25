# ⏳ Ablation Study: Dual-Stream Fusion + Unidirectional LSTM (Google Colab Guide)

This guide provides step-by-step instructions on how to run the **Unidirectional LSTM temporal modeling ablation study** on Google Colab.

---

## 📌 What is this Ablation Experiment?

In video-based student engagement classification, evaluating different sequential architectures proves why your final proposed model was selected:

| Variant | Temporal Modeling Technique | Description | Trainable Params |
| :--- | :--- | :--- | :--- |
| **1. No Temporal Modeling** (`colab_binary_no_transformer`) | Masked Temporal Mean Pooling | Averages frame features across time without sequential dependencies | **265,378** |
| **2. Unidirectional LSTM** (`colab_binary_lstm`) | **2-Layer Unidirectional LSTM** | Standard forward-only recurrent temporal processing | **529,826** |
| **3. Bidirectional LSTM** (`colab_binary_bilstm`) | **2-Layer Bidirectional LSTM (BiLSTM)** | Gated recurrent processing looking both forward & backward in time | **464,034** |
| **4. Attention Modeling** (`colab_binary_transformer`) | **3-Layer Transformer Encoder** | Multi-Head Self-Attention across all frame pairs simultaneously | **663,586** |

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
    └── colab_binary_lstm/                   ← UPLOAD THIS NEW FOLDER TO YOUR DRIVE
        ├── Ablation_Binary_LSTM_Colab.ipynb    (Interactive Colab Notebook)
        ├── config_binary_lstm.py               (Configurations & paths)
        ├── models_binary_lstm.py               (LSTM model & parameter counter)
        ├── data_loader_binary_lstm.py          (Sliding windows & data loader)
        ├── train_binary_lstm.py                (Training script with early stopping)
        ├── evaluate_binary_lstm.py             (Video-level evaluation & metrics)
        └── Colab_LSTM_Guide.md                 (This documentation)
```

---

## 🚀 2. How to Run in Google Colab

### **Step A: Upload `colab_binary_lstm` to Google Drive**
1. Open [Google Drive](https://drive.google.com/).
2. Open your `Student_Engagement_DL` folder.
3. Drag and drop the `colab_binary_lstm` folder from `c:\FYP\Binary classification\colab_binary_lstm` into your Drive.

### **Step B: Open Notebook in Google Colab**
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Click **Upload** $\rightarrow$ select `Ablation_Binary_LSTM_Colab.ipynb` from your computer.

### **Step C: Enable Free T4 GPU**
* **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU** $\rightarrow$ **Save**.

### **Step D: Run Cells Sequentially**
1. **Cell 1:** Mount Drive (`drive.mount('/content/drive')`).
2. **Cell 2:** Set directory (`%cd /content/drive/MyDrive/Student_Engagement_DL/colab_binary_lstm`).
3. **Cell 3:** Install packages (`pip install -q ...`).
4. **Cell 4:** Verify feature files in `sequence_fuse_features_binary`.
5. **Cell 5:** Check LSTM parameter breakdown.
6. **Cell 6:** Train LSTM model (`!python train_binary_lstm.py`).
7. **Cell 7:** Evaluate on test videos (`!python evaluate_binary_lstm.py`).
8. **Cell 8 & 9:** View Confusion Matrix, ROC-AUC curve, and predictions table.

---

## 📊 3. Parameter Count Breakdown

```text
===========================================================================
      ARCHITECTURE PARAMETER COUNT: DUAL-STREAM + UNIDIRECTIONAL LSTM
===========================================================================
  Stream 1: EfficientNet-B0 Visual Extractor      : 4,007,548 params (Features only)
            [Full Standard EfficientNet-B0 Model] : 5,288,548 params (~5.29M)
  Stream 2: OpenFace Facial Behavior Stream       : 709 numerical features (0 PyTorch params)
  Dual-Stream Fused Input Dimension               : 1989 (1280 CNN + 709 OpenFace)
---------------------------------------------------------------------------
  LSTM ABLATION MODEL COMPONENTS:
    • Fusion & Projection (Linear + LayerNorm)    : 254,976 params
    • 2-Layer Unidirectional LSTM (Hidden=128)    : 264,448 params
    • Temporal Pooling Layer                      : 0 params (Masked Mean Pooling)
    • MLP Classification Head (Binary 2-class)    : 10,402 params
    -------------------------------------------------------
    Total Trainable Parameters in LSTM Model      : 529,826
---------------------------------------------------------------------------
  ENTIRE END-TO-END PIPELINE (Backbone + LSTM Head):
    • Option A (Conv Features + LSTM Model)       : 4,537,374 params (~4.54M)
    • Option B (Full Standard B0 + LSTM Model)    : 5,818,374 params (~5.82M)
===========================================================================
```

---

## 💾 4. Generated Outputs on Google Drive

Outputs will be saved in your Drive under:
* `models_binary_lstm/binary_lstm_weights.pth`: Best checkpoint.
* `evaluation_results_binary_lstm/predictions_binary_lstm.csv`: Video predictions.
* `evaluation_results_binary_lstm/metrics_binary_lstm.json`: Accuracy & F1 scores.
* `evaluation_results_binary_lstm/confusion_matrix_binary_lstm.png`: Confusion Matrix.
* `evaluation_results_binary_lstm/roc_curve_binary_lstm.png`: ROC-AUC plot.
* `evaluation_results_binary_lstm/training_curves_binary_lstm.png`: Loss & Accuracy curves.
