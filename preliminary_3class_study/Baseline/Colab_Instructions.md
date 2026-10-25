# Rebuilt Baseline Student Engagement Detection: Google Colab Setup & Run Guide

This guide describes how to run the rebuilt, modular student engagement detection pipeline in Google Colab using your Google Drive dataset.

---

## 1. Google Drive Directory Structure

To run the pipeline, organize your Google Drive (`MyDrive/`) as follows:

```text
Google Drive/
└── Student_Engagement_DL/
    ├── Dataset/                         <-- Your dataset folder
    │   ├── Engaged/                     <-- Videos of engaged students
    │   ├── Partial/                     <-- Videos of partially engaged students
    │   └── Disengaged/                  <-- Videos of disengaged students
    └── colab_baseline/                  <-- Upload this entire folder here
        ├── Colab_Instructions.md
        ├── config.py
        ├── data_loader.py
        ├── models.py
        ├── train.py
        ├── evaluate.py
        └── preprocess/
            ├── __init__.py
            ├── extract_frames.py
            ├── extract_openface.py
            ├── extract_efficientnet.py
            └── fuse_features.py
```

### Steps to upload:
1. Ensure your video dataset is structured under `Student_Engagement_DL/Dataset/` in your Drive.
2. Upload the `colab_baseline/` folder directly under `Student_Engagement_DL/`.

---

## 2. Google Colab Notebook Cells

Create a new Google Colab notebook (using a **GPU T4** runtime for fast feature extraction and training) and paste the following cells in order.

### Cell 1: Mount Google Drive
```python
from google.colab import drive
import os
import sys

# Mount Google Drive
drive.mount('/content/drive')

# Verify paths and set project directory
drive_path = "/content/drive/MyDrive/Student_Engagement_DL"
baseline_path = os.path.join(drive_path, "colab_baseline")

if not os.path.exists(baseline_path):
    raise FileNotFoundError(f"Could not find colab_baseline directory at: {baseline_path}. "
                            "Please check your Google Drive path structure.")

# Append colab_baseline to path so Python finds imports
sys.path.append(baseline_path)
%cd $baseline_path
```

### Cell 2: Install OpenFace (Optional, ~30 mins)
> [!NOTE]
> Compiling OpenFace takes about 30–40 minutes in Colab. 
> **If you skip this step**, the OpenFace extraction script (`preprocess/extract_openface.py`) will automatically fall back to 0-filled vectors and the pipeline will train successfully using **EfficientNet-B0 visual features only**. This provides a strong visual baseline with zero setup time!
>
> Run the cell below only if you want to compile and run OpenFace for Action Units and landmarks.

```bash
# Clone OpenFace repository
!git clone -q --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git /content/OpenFace
%cd /content/OpenFace

# Update apt packages and install dependencies
!sudo apt-get update
!sudo apt-get install -y build-essential g++-8 cmake libboost-all-dev libopenblas-dev

# Download OpenFace models
!bash ./download_models.sh

# Build and install OpenFace
!sudo bash ./install.sh

# Go back to the project folder
%cd /content/drive/MyDrive/Student_Engagement_DL/colab_baseline
```

### Cell 3: Copy Dataset to Colab Local SSD (Highly Recommended)
Copying the dataset from Google Drive to Colab's local scratch storage avoids network read latency, making the frame extraction **5x to 10x faster**.

```bash
# Copy dataset to local high-speed disk (takes a few seconds)
!cp -r "/content/drive/MyDrive/Student_Engagement_DL/Dataset" "/content/Dataset"
```

### Cell 4: Step 1 - Frame Extraction & Resizing
Extracts all video frames, resizes them to 224x224, and stores them locally on the SSD as `.npy` files.
```bash
!python preprocess/extract_frames.py
```

### Cell 5: Step 2 - OpenFace Feature Extraction
Runs OpenFace on the videos to extract AUs and facial coordinates. Automatically falls back to zeros if OpenFace was not compiled in Cell 2.
```bash
!python preprocess/extract_openface.py
```

### Cell 6: Step 3 - EfficientNet Feature Extraction
Runs EfficientNet-B0 visual feature extraction on the local preprocessed frames and saves them.
```bash
!python preprocess/extract_efficientnet.py
```

### Cell 7: Step 4 - Feature Fusion & Temporal Pooling
Concatenates OpenFace and EfficientNet features frame-by-frame and averages them across all frames to create a single video-level `.pt` feature representation.
```bash
!python preprocess/fuse_features.py
```

### Cell 8: Step 5 - Model Training
Splits fused features into Train (70%), Val (15%), and Test (15%), trains the fully connected classifier, and saves the best weights.
```bash
!python train.py
```

### Cell 9: Step 6 - Model Evaluation & Performance Plots
Runs evaluation on the test split, outputs performance metrics (Accuracy, Precision, Recall, F1), generates a classification report, and saves the Confusion Matrix and ROC curve.
```bash
!python evaluate.py
```

---

## 3. Training Outputs & Saved Files

All evaluation files will be saved persistently in your Google Drive under `/content/drive/MyDrive/Student_Engagement_DL/`:

1. **Pre-extracted Fused Features**: `fused_features/` - Unified `.pt` files containing fused vectors for each video.
2. **Saved Weights**: `models/baseline_engagement_weights.pth` - PyTorch weights of the best performing model on the validation split.
3. **Evaluation Folder**: `evaluation_results/` containing:
   - `predictions.csv`: Table containing columns `video_filename`, `true_label`, `predicted_label`, and class probabilities.
   - `metrics.json`: JSON output containing precision, recall, accuracy, and F1 metrics.
   - `confusion_matrix.png`: Matplotlib heatmap visual of prediction accuracy.
   - `roc_curve.png`: Multi-class ROC curves showing One-vs-Rest AUC performance.
