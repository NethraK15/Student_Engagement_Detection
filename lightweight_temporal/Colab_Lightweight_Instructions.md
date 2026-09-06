# Google Colab Instructions: Lightweight Temporal Experiment

This experiment drops the heavy **Full Self-Attention Transformer Encoder** (~300k+ parameters) in favor of lightweight temporal architectures optimized for small datasets (~100 video clips):

1. **Option 1: Attention Pooling** (`--model attention_pool`) - A single learned query attending over the frame vectors. High capacity to weight "distracted" frames with minimal parameters (~25k-40k params).
2. **Option 2: 1-2 Layer TCN** (`--model tcn`) - Temporal Convolutional Network. Cheap to train, robust local motion pattern detector (~30k params).
3. **Option 3: Single-Layer GRU** (`--model gru`) - Single unidirectional GRU layer (~35k params).

---

## 📌 Step 1: Mount Google Drive & Set Up Directory

Run this in your first Google Colab cell:

```python
from google.colab import drive
import os

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Define Project Root in Drive
PROJECT_ROOT = "/content/drive/MyDrive/Student_Engagement_DL"
LIGHTWEIGHT_DIR = os.path.join(PROJECT_ROOT, "colab_lightweight_temporal")

# 3. Create project directories if they don't exist
os.makedirs(LIGHTWEIGHT_DIR, exist_ok=True)

# 4. Change current working directory to lightweight experiment folder
%cd {LIGHTWEIGHT_DIR}
print("Current Working Directory:", os.getcwd())
```

---

## 📌 Step 2: Install Dependencies & Verify Files

Run this cell to install required dependencies and verify that feature folders exist:

```python
!pip install -q torch torchvision pandas scikit-learn matplotlib seaborn

import sys
sys.path.append(LIGHTWEIGHT_DIR)

import config_lightweight

print("--- DIRECTORY CHECK ---")
print(f"OpenFace Features Exist:     {os.path.exists(config_lightweight.OPENFACE_FEATURES_DIR)}")
print(f"EfficientNet Features Exist: {os.path.exists(config_lightweight.EFFICIENTNET_FEATURES_DIR)}")
print(f"Sequence Fused Features Dir: {config_lightweight.SEQUENCE_FUSED_DIR}")
```

---

## 📌 Step 3: Run Preprocessing (Fuse Sequence Features)

If you haven't fused OpenFace + EfficientNet features for this experiment yet, run:

```python
!python preprocess/fuse_sequence_features.py
```

---

## 📌 Step 4: Test Pipeline (Mock Run)

Run the test script to ensure all 3 models compile and execute forward/backward passes correctly:

```python
!python test_pipeline_lightweight.py
```

---

## 📌 Step 5: Train Model Variants

You can train any (or all) of the 3 lightweight model variants!

### 🔹 Option A: Train Attention Pooling Model (Recommended First)
```python
!python train_lightweight.py --model attention_pool --epochs 50 --lr 0.001
```

### 🔹 Option B: Train 1-2 Layer TCN (Temporal ConvNet)
```python
!python train_lightweight.py --model tcn --epochs 50 --lr 0.001
```

### 🔹 Option C: Train Single-Layer GRU
```python
!python train_lightweight.py --model gru --epochs 50 --lr 0.001
```

---

## 📌 Step 6: Evaluate Trained Models & Generate Reports

Run evaluation for the model variant you trained to get Accuracy, F1-Score, Classification Report, Confusion Matrix, and ROC Curves:

### 🔹 Evaluate Attention Pooling
```python
!python evaluate_lightweight.py --model attention_pool
```

### 🔹 Evaluate TCN Model
```python
!python evaluate_lightweight.py --model tcn
```

### 🔹 Evaluate GRU Model
```python
!python evaluate_lightweight.py --model gru
```

---

## 📊 Summary of Model Variants & Parameters

| Architecture | Model Flag (`--model`) | Core Idea | Params Footprint | Best Use Case |
|---|---|---|---|---|
| **Attention Pooling** | `attention_pool` | Single learned query attending over sequence frames | ~30k - 40k | Tiny datasets (~100 clips), highlights key distracted frames |
| **Temporal ConvNet** | `tcn` | 1-2 Conv1D layers with BatchNorm & Avg Pooling | ~25k - 35k | Strong local temporal/motion pattern extraction |
| **Single-Layer GRU** | `gru` | Light 1-layer GRU with masked average pooling | ~35k - 45k | Light recurrent baseline with low parameter budget |
