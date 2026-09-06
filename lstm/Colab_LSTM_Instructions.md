# Experiment 2: BiLSTM Sequence Model Guide (Google Colab)

This guide describes how to run **Experiment 2 (Bidirectional LSTM Sequence Model)** in Google Colab.

> [!NOTE]
> **No re-extraction needed!** Experiment 2 reuses the pre-extracted `openface_features/` and `efficientnet_features/` from your previous run.

---

## 1. Directory Structure

Upload the `colab_lstm/` folder directly under your `Student_Engagement_DL/` directory in Google Drive:

```text
Google Drive/
└── Student_Engagement_DL/
    ├── Dataset/                         <-- Your video dataset
    ├── openface_features/               <-- Pre-extracted OpenFace features (reused!)
    ├── efficientnet_features/           <-- Pre-extracted EfficientNet features (reused!)
    └── colab_lstm/                      <-- Upload this folder
        ├── Colab_LSTM_Instructions.md
        ├── config_lstm.py
        ├── data_loader_lstm.py
        ├── models_lstm.py
        ├── train_lstm.py
        ├── evaluate_lstm.py
        └── preprocess/
            └── fuse_sequence_features.py
```

---

## 2. Notebook Execution Cells

Open your notebook (using GPU T4 runtime) and paste the cells in order:

### Cell 1: Mount Google Drive & Append Path
```python
from google.colab import drive
import os
import sys

drive.mount('/content/drive')

drive_path = "/content/drive/MyDrive/Student_Engagement_DL"
lstm_path = os.path.join(drive_path, "colab_lstm")

if not os.path.exists(lstm_path):
    raise FileNotFoundError(f"Could not find colab_lstm folder at {lstm_path}")

sys.path.append(lstm_path)
%cd $lstm_path
```

### Cell 2: Step 1 - Sequence Feature Fusion (Retains 2D Temporal Shape)
Concatenates frame features frame-by-frame without mean-pooling and saves them to `sequence_fused_features/`:
```bash
!python preprocess/fuse_sequence_features.py
```

### Cell 3: Step 2 - Train BiLSTM Sequence Classifier
Trains the 2-layer Bidirectional LSTM model with temporal self-attention:
```bash
!python train_lstm.py
```

### Cell 4: Step 3 - Evaluate BiLSTM Model & Save Metrics
Runs evaluation on the test set, outputs accuracy and metrics, and plots confusion matrix and ROC curves:
```bash
!python evaluate_lstm.py
```

---

## 3. Experiment 2 Saved Files

All outputs for Experiment 2 are stored separately under `/content/drive/MyDrive/Student_Engagement_DL/`:

1. `sequence_fused_features/`: Unified sequence tensors `[seq_len, 1989]`.
2. `models_lstm/lstm_engagement_weights.pth`: Best trained BiLSTM weights.
3. `evaluation_results_lstm/`:
   - `predictions_lstm.csv`: Video filenames, true labels, predicted labels, and class probabilities.
   - `metrics_lstm.json`: Accuracy, precision, recall, and F1-scores.
   - `confusion_matrix_lstm.png`: BiLSTM confusion matrix heatmap.
   - `roc_curve_lstm.png`: Multi-class One-vs-Rest ROC curve chart.
