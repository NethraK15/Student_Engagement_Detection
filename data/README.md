# 📊 Dataset Organization & Download Instructions

This project evaluates student engagement using video frames annotated across three primary classes:
- **0: Engaged** (Focused eye gaze, upright head posture, active listening)
- **1: Partially Engaged** (Nominal focus, mild gaze drift, intermittent head movement)
- **2: Disengaged** (Persistent head turning, closed eyes / yawning, off-task behavior)

---

## ⚠️ Repository Storage Policy
In accordance with GitHub repository limits and deep learning reproducibility standards, **raw datasets and high-resolution video frames ($>100\text{ MB}$) are hosted on external cloud storage** rather than being tracked directly in Git history.

A lightweight set of sample frames is provided in `data/sample/` to verify pipeline syntax and data loading without downloading the complete dataset.

---

## 📥 How to Download the Full Dataset

### Option 1: Automated Download Script (Recommended)
Run the automated fetch script:
```bash
pip install gdown
python scripts/download_dataset.py
```

### Option 2: Direct Cloud Download
Download the zipped dataset archive from our project storage:
- **Google Drive Link**: `https://drive.google.com/drive/folders/YOUR_DRIVE_FOLDER_ID_HERE` *(Replace with team drive link)*
- **Kaggle / Hugging Face**: *(Optional mirror)*

Once downloaded, extract the archive into the `dataset/` directory so the folder structure matches:

```text
dataset/
├── images_flat/                    # Extracted cropped student face frames
│   ├── student01_frame001.jpg
│   └── student01_frame002.jpg
├── openface/                       # OpenFace extracted CSV features
│   └── images_flat.csv
└── labels.csv                      # Frame-level / sequence-level ground truth labels
```

---

## 🏷️ Annotations Format (`labels.csv`)

| Column Name | Description | Example Values |
| :--- | :--- | :--- |
| `frame_id` / `filename` | Identifier linking to the image frame | `student01_frame001.jpg` |
| `label` | Engagement ground truth class index | `0` (Engaged), `1` (Partially), `2` (Disengaged) |
| `confidence` | Inter-annotator agreement score | `1.0`, `0.85` |
