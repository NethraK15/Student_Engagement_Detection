# Refined Lightweight GRU Model Guide (`colab_lstm_gru`)

This guide describes how to run the **Refined Lightweight GRU Model (`colab_lstm_gru`)** in Google Colab.

> [!NOTE]
> **No re-extraction needed!** Reuses the pre-extracted `openface_features/` and `efficientnet_features/` from your previous run.

---

## 1. Directory Structure

Upload the `colab_lstm_gru/` folder directly under your `Student_Engagement_DL/` directory in Google Drive:

```text
Google Drive/
└── Student_Engagement_DL/
    ├── Dataset/                         <-- Your video dataset
    ├── openface_features/               <-- Pre-extracted OpenFace features (reused!)
    ├── efficientnet_features/           <-- Pre-extracted EfficientNet features (reused!)
    └── colab_lstm_gru/                  <-- Upload this folder
        ├── Colab_GRU_Instructions.md
        ├── config_gru.py
        ├── data_loader_gru.py
        ├── models_gru.py
        ├── train_gru.py
        ├── evaluate_gru.py
        └── preprocess/
            └── fuse_sequence_features.py
```

---

## 2. Notebook Execution Cells

Open your notebook (using GPU T4 runtime) and paste the cells in order:

### Cell 1: Mount Google Drive & Append Path
```python
from google.colab import drive
import os, sys

drive.mount('/content/drive')

gru_path = "/content/drive/MyDrive/Student_Engagement_DL/colab_lstm_gru"
sys.path.append(gru_path)
%cd $gru_path
```

### Cell 2: Step 1 - Sequence Feature Fusion (Retains 2D Sequences)
Fuses OpenFace and EfficientNet features into 2D sequence tensors without temporal averaging:
```bash
!python preprocess/fuse_sequence_features.py
```

### Cell 3: Step 2 - Train Refined Lightweight GRU Classifier
Trains the model (featuring $1989 \rightarrow 128$ Linear Bottleneck, 1-layer BiGRU, and Gradient Clipping):
```bash
!python train_gru.py
```

### Cell 4: Step 3 - Evaluate GRU Model & Save Metrics/Plots
Runs evaluation on the test set, outputs accuracy and metrics, and plots confusion matrix and ROC curves:
```bash
!python evaluate_gru.py
```

---

## 3. Refined GRU Saved Files

All outputs for `colab_lstm_gru` are stored separately under `/content/drive/MyDrive/Student_Engagement_DL/`:

1. `sequence_fused_features_gru/`: Unified sequence tensors `[seq_len, 1989]`.
2. `models_gru/gru_engagement_weights.pth`: Best trained GRU weights.
3. `evaluation_results_gru/`:
   - `predictions_gru.csv`: Video filenames, true labels, predicted labels, and class probabilities.
   - `metrics_gru.json`: Accuracy, precision, recall, and F1-scores.
   - `confusion_matrix_gru.png`: GRU confusion matrix heatmap.
   - `roc_curve_gru.png`: Multi-class One-vs-Rest ROC curve chart.
