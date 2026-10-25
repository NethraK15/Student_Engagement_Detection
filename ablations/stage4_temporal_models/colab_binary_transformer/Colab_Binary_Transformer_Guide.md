# Binary Engagement Transformer — Google Colab Guide

## What Changed vs Original `colab_transformer`

| Feature | Original | This (`colab_binary_transformer`) |
|---|---|---|
| **Classes** | 3 (Engaged / Partial / Disengaged) | **2 only (Engaged / Disengaged)** |
| **Partial Engaged** | Included | **Completely ignored — never seen** |
| **Frame coverage** | First 120 frames only | **Full video via sliding windows** |
| **Window** | Single truncated seq | **120-frame window, stride 30** |
| **Inference** | Single sequence → 1 pred | **Soft-vote all windows → 1 pred** |
| **Preprocessing** | 4 separate scripts | **1 all-in-one script** |

---

## 1. Google Drive Directory Structure

Upload **only** `colab_binary_transformer/` to your Drive:

```text
Google Drive/
└── Student_Engagement_DL/
    ├── New_Dataset_Crafted/             ← Your video dataset (must already be here)
    │   ├── engaged/                     ← Used ✓  (or any folder with "engaged" in name)
    │   ├── disengaged/                  ← Used ✓  (or any folder with "disengaged" in name)
    │   └── partially engaged/           ← IGNORED ✗  (any folder with "partial" is skipped)
    └── colab_binary_transformer/        ← Upload THIS folder
        ├── Colab_Binary_Transformer_Guide.md
        ├── config_binary_transformer.py
        ├── data_loader_binary_transformer.py
        ├── models_binary_transformer.py
        ├── train_binary_transformer.py
        ├── evaluate_binary_transformer.py
        ├── test_classroom_binary_transformer.py
        └── preprocess/
            ├── __init__.py
            └── fuse_sequence_features_binary.py
```

> **Note**: `New_Dataset_Crafted` is auto-detected by the config — no manual path changes needed.
> The preprocessing script reads raw videos directly and extracts all features itself.

---

## 2. Colab Notebook Cells

Open a Colab notebook with **GPU runtime** (Runtime → Change runtime type → T4 GPU).

---

### Cell 1 — Mount Drive & Set Path
```python
from google.colab import drive
import os, sys

drive.mount('/content/drive')

binary_path = "/content/drive/MyDrive/Student_Engagement_DL/colab_binary_transformer"
sys.path.insert(0, binary_path)
%cd $binary_path
print("Working directory:", os.getcwd())
```

---

### Cell 2 — Install All Dependencies
```bash
!pip install -q torch torchvision tqdm scikit-learn pandas matplotlib seaborn opencv-python-headless
```

---

### Cell 3 — ⚠️ Install OpenFace (MANDATORY — run this before preprocessing)

> OpenFace is **required** for AU + landmark features. The preprocessing script will exit with an error if it's not installed.  
> This cell takes **8–12 minutes** on Colab. Run it once per session (Colab resets on reconnect, but your `.pt` feature files are safely saved to Drive).

```bash
%%bash
set -e   # exit on any error

echo "=== Updating package repositories ==="
apt-get update -qq

echo "=== Installing system dependencies ==="
apt-get install -y -q --fix-missing \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-all-dev \
    libdlib-dev \
    libtbb-dev \
    libopencv-dev

echo "=== Cloning OpenFace ==="
if [ ! -d "/content/OpenFace" ]; then
    git clone --depth=1 https://github.com/TadasBaltrusaitis/OpenFace.git /content/OpenFace
fi

echo "=== Downloading face models ==="
cd /content/OpenFace
bash download_models.sh

echo "=== Building OpenFace ==="
mkdir -p /content/OpenFace/build
cd /content/OpenFace/build
cmake -D CMAKE_BUILD_TYPE=RELEASE .. -DCMAKE_CXX_FLAGS="-O2"
make -j$(nproc)

echo "=== Verifying installation ==="
ls -la /content/OpenFace/build/bin/FeatureExtraction
echo "✅ OpenFace installed successfully!"
```

**Verify it works on one test video:**
```bash
# Test OpenFace on a sample video from your dataset
!SAMPLE=$(find /content/drive/MyDrive/Student_Engagement_DL/New_Dataset_Crafted -name "*.mp4" | head -1) && \
 echo "Testing on: $SAMPLE" && \
 /content/OpenFace/build/bin/FeatureExtraction -f "$SAMPLE" -out_dir /tmp/of_test && \
 echo "✅ OpenFace test passed!" && \
 ls /tmp/of_test/

```

---

### Cell 4 — Step 1: Run Complete Preprocessing Pipeline
> This ONE script does everything:
> - Scans dataset → **skips "partially engaged" folders entirely**
> - Extracts ALL frames from each video (no 120-frame limit)
> - Extracts EfficientNet-B0 CNN features (1280-dim per frame)
> - Extracts OpenFace features (709-dim per frame, or zeros if not installed)
> - Fuses CNN + OpenFace → saves full temporal sequence as `.pt`
>
> ✅ **Resume-safe**: Already processed videos are skipped automatically.
```bash
!python preprocess/fuse_sequence_features_binary.py
```

**Expected output:**
```
[Step 1] Skipped partially-engaged folder(s): ['PARTIALLY ENGAGED', 'L_Partially engaged']
[Step 1] Found 54 binary videos (engaged: 27, disengaged: 27)
[Step 3] Loading EfficientNet-B0 CNN extractor...
         Device: cuda

Binary preprocessing: 100%|████████| 54/54 [08:32<00:00]

PREPROCESSING COMPLETE
  Successfully processed : 54
  Resumed / Skipped      : 0
  Failed                 : 0
  Total .pt files saved  : 54
```

---

### Cell 5 — Step 2: Train the Binary Transformer
> Sliding windows (size=120, stride=30) are generated **on-the-fly** during training.
> For a 300-frame video → 7 training windows covering the entire video.
> AdamW + CosineAnnealingLR + early stopping (patience=10).
```bash
!python train_binary_transformer.py
```

**Expected per-epoch output:**
```
Epoch [001/060] | Train Loss: 0.5120  Acc: 74.31% | Val Loss: 0.4821  Acc: 77.14% ✓ Best
Epoch [002/060] | Train Loss: 0.4830  Acc: 78.52% | Val Loss: 0.4612  Acc: 80.00% ✓ Best
...
```

---

### Cell 6 — Step 3: Evaluate the Binary Transformer
> Uses **soft-vote aggregation** across all video windows → one prediction per video.
```bash
!python evaluate_binary_transformer.py
```

**Expected output:**
```
BINARY TRANSFORMER — VIDEO-LEVEL METRICS
  Accuracy   :  85.71%
  Precision  :  86.20%  (weighted)
  Recall     :  85.71%  (weighted)
  F1-score   :  85.83%  (weighted)
```

---

### Cell 7 — Step 4: Test on Classroom Video (Annotated Output)
> Detects multiple students with YOLO + ByteTrack, enhances lighting (CLAHE + Sharpening), extracts EfficientNet visual features, and runs the Binary Engagement Transformer.
> - 🟩 **GREEN Box**: **Engaged**
> - 🟥 **RED Box**: **Disengaged**
> - Auto-detects your video from `clasroom_recording` on Google Drive!

```bash
# If running for the first time, make sure ultralytics is installed:
!pip install -q ultralytics

# Run inference on classroom video (auto-picks from clasroom_recording in Drive)
!python test_classroom_binary_transformer.py

# Or specify a custom video path directly:
# !python test_classroom_binary_transformer.py --video "/content/drive/MyDrive/Student_Engagement_DL/clasroom_recording/your_video.mp4"
```

**Outputs generated in Drive (`evaluation_results_binary_transformer/<video_name>_binary_results/`):**
1. 🎥 **`<video_name>_annotated_binary.mp4`** — Annotated video with Green (Engaged) & Red (Disengaged) bounding boxes + top HUD classroom analytics banner.
2. 📊 **`<video_name>_binary_chart.png`** — Student engagement distribution bar chart & engagement timeline curve over the video.
3. 📄 **`<video_name>_binary_report.txt`** — Overall classroom score & per-student engagement breakdown.
4. 📋 **`<video_name>_binary_details.csv`** — Frame-by-frame engagement tracking log.

---

```text
Student_Engagement_DL/
├── sequence_fused_features_binary/          ← Full video .pt files
│   ├── engaged__VID20260730105033.pt
│   ├── disengaged__VID20260730112511.pt
│   └── ...
├── models_binary_transformer/
│   └── binary_transformer_weights.pth       ← Best model checkpoint
└── evaluation_results_binary_transformer/
    ├── metrics_binary_transformer.json
    ├── confusion_matrix_binary_transformer.png
    ├── roc_curve_binary_transformer.png
    └── predictions_binary_transformer.csv
```

---

## 4. How Sliding Window Works

For a **10-second video at 30fps = ~300 frames**:

```
Frames total: 300
Window size : 120 frames
Stride      : 30 frames

Window 1: frames [  0 → 119]
Window 2: frames [ 30 → 149]
Window 3: frames [ 60 → 179]
Window 4: frames [ 90 → 209]
Window 5: frames [120 → 239]
Window 6: frames [150 → 269]
Window 7: frames [180 → 299]
                           ↑ Every frame is covered!
```

**During training**: Each window = one sample with label inherited from the video.  
**At inference**: All 7 windows are scored → softmax probabilities averaged → `argmax` = final label.

> The model sees **every single frame** of every video — not just the first 120!

---

## 5. Config Quick Reference

Edit `config_binary_transformer.py` if needed:

| Parameter | Default | Effect |
|---|---|---|
| `WINDOW_SIZE` | 120 | Frames per training window |
| `WINDOW_STRIDE` | 30 | Step between windows (smaller = more windows) |
| `BATCH_SIZE` | 16 | Reduce to 8 if CUDA OOM |
| `EPOCHS` | 60 | Max training epochs |
| `PATIENCE` | 10 | Early stopping threshold |
| `D_MODEL` | 128 | Transformer embedding dim |
| `NUM_LAYERS` | 3 | Transformer encoder depth |

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `OpenFace FeatureExtraction NOT FOUND` | Run Cell 3 (OpenFace install) **before** Cell 4 |
| `Unable to fetch some archives` | Run `!apt-get update -qq` before `apt-get install` (included in updated Cell 3) |
| `cmake: not found` | Run `!apt-get install -y cmake` manually first |
| `make: Error` during build | Re-run Cell 3 — sometimes a dependency install lags |
| `No binary videos found` | Check dataset folder name — config auto-detects `New_Dataset_Crafted` |
| `No .pt files found` | Run Cell 4 (preprocessing) before training |
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 8 in `config_binary_transformer.py` |
| `Stratified split failed` | Harmless — automatic fallback to random split |
| OpenFace lost after Colab restart | Re-run Cell 3 — takes ~10 min. Your `.pt` files in Drive are safe, preprocessing resumes from where it left off |

### ⚠️ Important: OpenFace resets each Colab session

Colab's `/content/` is erased on every disconnect. But your **Drive is persistent**:
- Run order each new session: **Cell 1 → Cell 2 → Cell 3 → Cell 4 (skip already done) → Cell 5 → Cell 6**
- Cell 4 preprocessing is **resume-safe** — already processed `.pt` files are skipped automatically
- Once all `.pt` files exist in Drive, you only need Cells 1, 2, 5, 6 in future sessions
