# Google Colab Instructions: Temporal Transformer DAiSEE Transfer Learning

This experiment pre-trains the **Temporal Transformer Model** (from `colab_transformer`) on the public **DAiSEE** dataset (thousands of clips), then transfers and fine-tunes the learned Transformer weights on your target 100 classroom clips dataset!

---

## 📌 Step 1: Mount Google Drive & Navigate to Experiment Folder

```python
from google.colab import drive
import os
import sys

drive.mount('/content/drive')

PROJECT_ROOT = "/content/drive/MyDrive/Student_Engagement_DL"
TRANSFER_DIR = os.path.join(PROJECT_ROOT, "colab_daisee_transfer")

%cd {TRANSFER_DIR}
sys.path.append(TRANSFER_DIR)
print("Current Working Directory:", os.getcwd())
```

---

## 📌 Step 2: Install Dependencies & Check Paths

```python
!pip install -q torch torchvision pandas scikit-learn matplotlib seaborn tqdm

import config_daisee

print("--- PATH & DATASET STATUS ---")
print(f"DAiSEE Zip Path Detected:  {config_daisee.DAISEE_ZIP_PATH}")
print(f"Target Fused Features Dir: {config_daisee.TARGET_FUSED_DIR}")
```

---

## 📌 Step 3: Test Pipeline

```python
!python test_pipeline_transfer.py
```

---

## 📌 Step 4: Phase 1 — Pre-train Temporal Transformer Head on DAiSEE

Pre-trains the 2-layer Multi-Head Self-Attention Transformer Encoder (`d_model=128`, `nhead=4`) on DAiSEE:

```python
!python pretrain_daisee.py --model transformer --epochs 30 --lr 0.0005
```

---

## 📌 Step 5: Phase 2 — Fine-Tune Pretrained Transformer on Target Dataset (~100 Clips)

Transfers DAiSEE Transformer weights, adapts the classification head to 3 classes (`Engaged`, `Partially Engaged`, `Disengaged`), and fine-tunes on your target dataset clips:

```python
!python finetune_target.py --model transformer --epochs 40 --lr 0.0001
```

---

## 📌 Step 6: Evaluate Fine-Tuned Transformer & View Results

```python
!python evaluate_transfer.py --model transformer
```
