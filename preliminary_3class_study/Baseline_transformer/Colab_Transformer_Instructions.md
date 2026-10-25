# Experiment 3: Temporal Transformer Model Guide (`colab_transformer`)

This guide describes how to run **Experiment 3: Temporal Transformer with Multi-Head Self-Attention (`colab_transformer`)** in Google Colab.

> [!NOTE]
> **No re-extraction needed!** Reuses the pre-extracted `sequence_fused_features` (or `openface_features/` and `efficientnet_features/`) from your previous run in Google Drive!

---

## 1. Directory Structure

Upload the `colab_transformer/` folder directly under your `Student_Engagement_DL/` directory in Google Drive:

```text
Google Drive/
└── Student_Engagement_DL/
    ├── Dataset/                         <-- Your video dataset (or New_Datset_Crafted)
    ├── sequence_fused_features/         <-- Pre-extracted sequence features (reused!)
    └── colab_transformer/               <-- Upload this folder
        ├── Colab_Transformer_Instructions.md
        ├── config_transformer.py
        ├── data_loader_transformer.py
        ├── models_transformer.py
        ├── train_transformer.py
        ├── evaluate_transformer.py
        └── preprocess/
            └── fuse_sequence_features.py
```

---

## 2. Notebook Execution Cells

Open your Google Colab notebook (using GPU or free CPU runtime) and run the cells in order:

### Cell 1: Mount Google Drive & Append Path
```python
from google.colab import drive
import os, sys

drive.mount('/content/drive')

transformer_path = "/content/drive/MyDrive/Student_Engagement_DL/colab_transformer"
sys.path.append(transformer_path)
%cd $transformer_path
```

### Cell 2: Step 1 - Fuse Sequence Features (Optional - Auto-detected!)
*(If you already have 189 `.pt` files in `sequence_fused_features`, this script will auto-detect and skip!)*
```bash
!python preprocess/fuse_sequence_features.py
```

### Cell 3: Step 2 - Train Temporal Transformer Model
Trains the Temporal Transformer featuring **Multi-Head Self-Attention**, **Sinusoidal Positional Encodings**, and **Early Stopping**:
```bash
!python train_transformer.py
```

### Cell 4: Step 3 - Evaluate Transformer Model & Save Metrics/Plots
Runs evaluation on the test set, outputs metrics, and plots confusion matrix and ROC curves:
```bash
!python evaluate_transformer.py
```

---

### Cell 5: Step 4 - Test Pipeline on Custom Multi-Student Classroom Video
Upload your classroom video (e.g. `classroom_sample.mp4`) to your Google Drive folder or `/content/`, then run YOLOv8 Face Detection + ByteTrack + Baseline Transformer engagement inference:
```bash
!pip install -q ultralytics
!python test_classroom_pipeline_transformer.py --video "/content/drive/MyDrive/Student_Engagement_DL/Classroom_Recordings/sample.mp4" --model "/content/drive/MyDrive/Student_Engagement_DL/models_transformer/transformer_engagement_weights.pth"
```

---

### Cell 6: Step 5 - Launch Interactive Web Dashboard
Run the interactive dashboard to upload videos directly via a web UI and view real-time multi-student metrics, confusion matrices, and timeline charts:
```bash
!pip install -q gradio ultralytics
!python dashboard_transformer.py
```

---

## 3. Temporal Transformer Saved Files

All outputs for `colab_transformer` are stored under `/content/drive/MyDrive/Student_Engagement_DL/`:

1. `models_transformer/transformer_engagement_weights.pth`: Best trained Transformer weights.
2. `evaluation_results_transformer/`:
   - `predictions_transformer.csv`: Video filenames, true labels, predicted labels, and probabilities.
   - `metrics_transformer.json`: Accuracy, precision, recall, and F1-scores.
   - `confusion_matrix_transformer.png`: Transformer confusion matrix heatmap.
   - `roc_curve_transformer.png`: Multi-class One-vs-Rest ROC curve chart.
   - `<video_name>_transformer_results/`:
     - `<video_name>_transformer_annotated.mp4`: Bounding box annotated video with student track IDs and Transformer engagement labels.
     - `<video_name>_transformer_details.csv`: Frame-by-frame detailed predictions and confidence scores per student.
     - `<video_name>_transformer_report.txt`: Summary text report of engagement percentages and average score.
     - `<video_name>_transformer_chart.png`: Stacked bar chart visualization of engagement distribution across students.
3. `colab_transformer/dashboard_outputs/`:
   - Interactive dashboard generated outputs (annotated MP4, confusion matrix, timeline charts, and student distribution charts).


