# 📘 Beginner's Guide: Running the OpenFace Binary Ablation Study in Google Colab

This guide walks you through every single step to run your ablation study in Google Colab, even if you have never used Google Colab before!

---

## 🎯 What We Are Doing in this Ablation Study

In your original pipeline:
1. **Input**: Student face video stream
2. **Dual-Stream Feature Extraction**:
   - Appearance Stream: **EfficientNet-B0** (1280-dim)
   - Geometric Stream: **OpenFace** (128-dim head pose, eye gaze, action units, landmarks)
3. **Temporal Modeling**: **Transformer Encoder**
4. **Classification Layer**: Mean Pooling $\to$ Fully Connected (ReLU) $\to$ Dropout $\to$ Fully Connected $\to$ Softmax
5. **Classes**: Originally 3 classes (`Engaged`, `Partially Engaged`, `Disengaged`)

### ✂️ Changes for this Ablation Study:
- ❌ **EfficientNet-B0 is REMOVED** (no visual appearance features).
- ❌ **Transformer Encoder is REMOVED** (no multi-head self-attention).
- ✅ **OpenFace ALONE is USED** (pure facial geometry).
- ✅ **Classification layer remains the same**: Temporal Mean Pooling $\to$ FC $\to$ ReLU $\to$ Dropout $\to$ FC $\to$ Softmax.
- ✅ **Brought down to 2 classes**: `Engaged` vs `Disengaged`.

---

## 📂 Step 1: Getting your Dataset from OneDrive to Google Colab

Your features are stored on OneDrive inside the folder `sequence_fused_features_binary`.
Here is the easiest, most reliable way to bring it to Google Colab:

### Recommended Method (Via Google Drive):
1. **Download from OneDrive**:
   - Open your OneDrive link in your browser.
   - Right-click the folder `sequence_fused_features_binary` and click **Download** (it will download as a `.zip` file, e.g., `sequence_fused_features_binary.zip`).
2. **Upload to Google Drive**:
   - Go to [Google Drive](https://drive.google.com).
   - Click **+ New** $\to$ **File upload**, and select `sequence_fused_features_binary.zip` (or upload the unzipped folder directly).
   - Place it directly inside `My Drive` or inside a folder like `My Drive/Student_Engagement_DL`.

*(Alternative method: You can also upload `sequence_fused_features_binary.zip` directly into Colab's left sidebar Files tab).*

---

## 🚀 Step 2: Opening the Notebook in Google Colab

1. Open your web browser and go to: **[colab.research.google.com](https://colab.research.google.com/)**
2. In the pop-up window, click on the **Upload** tab.
3. Click **Browse** and select the file from your project:
   `ablation_study/OpenFace_Binary_Ablation_Colab.ipynb`
4. The notebook will open in Colab!

---

## ⚙️ Step 3: Enable Free GPU (Optional but Recommended)

1. In Google Colab's top menu bar, click on **Runtime** $\to$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or CPU if GPU quota is exhausted; CPU also works quickly since OpenFace features are lightweight!).
3. Click **Save**.

---

## ▶️ Step 4: Running the Notebook Step-by-Step

In Google Colab, each grey box is a **cell**. You can run a cell by clicking the **Play (▶️)** button on its left, or pressing `Shift + Enter`.

Run each cell in order:

### 1. Run Step 1 (Install Dependencies):
- Installs PyTorch, scikit-learn, and matplotlib.

### 2. Run Step 2 (Connect to Google Drive & Locate Dataset):
- A pop-up will ask: *"Permit this notebook to access your Google Drive files?"*
- Click **Connect to Google Drive** and log in with your Google account.
- The script automatically checks for `sequence_fused_features_binary` and unzips it if it's a `.zip`.

### 3. Run Step 3 (Inspect Features & Extract OpenFace):
- Automatically loads the feature files.
- Ensures only the OpenFace features are used (dropping EfficientNet if present).
- Converts labels to binary:
  - `0`: **Engaged**
  - `1`: **Disengaged**
- Prints the exact count of samples per class.

### 4. Run Step 4 (Build Ablation Model):
- Instantiates the OpenFace classifier with Mean Pooling and the 2-layer classification head.
- Displays model architecture and total parameters.

### 5. Run Step 5 (Train Model):
- Splits the dataset into:
  - **70% Training**
  - **15% Validation**
  - **15% Testing**
- Trains for 35 epochs and automatically saves the checkpoint that achieved the **Best Validation Accuracy**.

### 6. Run Step 6 (Evaluate on Test Set & Print Table):
- Evaluates the best model on unseen test data.
- Automatically calculates and prints the table with all 5 metrics!

---

## 📊 Step 5: Copying Your Results into Your Ablation Table

Once Step 6 finishes, Colab will print out the formatted table:

```
===============================================================================================
             📊 ABLATION STUDY RESULTS TABLE (FOR YOUR REPORT/PAPER) 📊              
===============================================================================================
Model                               | Best Validation Accuracy | Testing Accuracy | Recall     | F1-Score   | Precision 
-----------------------------------------------------------------------------------------------
OpenFace Alone (Binary Ablation)    |               XX.XX%     |           XX.XX% |     XX.XX% |     XX.XX% |     XX.XX%
===============================================================================================
```

You can directly copy the numbers from this row and add them to your ablation table:

| Model | Best Validation Accuracy | Testing Accuracy | Recall | F1-Score | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline Model | 52.47% | 61.17% | 61.32% | 59.61% | 62.52% |
| BiLSTM + Baseline | 52.33% | 64.62% | 62.31% | 64.72% | 70.02% |
| Transformer + Baseline | 53.43% | 69.07% | 67.12% | 69.34% | 69.55% |
| Illumination + LSTM Baseline | 59.57% | 54.28% | 48.18% | 53.13% | 52.57% |
| Illumination + BiLSTM Baseline | 48.86% | 47.38% | 46.36% | 38.99% | 48.04% |
| Illumination + Transformer Baseline | 41.71% | 37.03% | 38.18% | 20.70% | 14.63% |
| **OpenFace Alone (Binary)** | **[From Output]** | **[From Output]** | **[From Output]** | **[From Output]** | **[From Output]** |

---

## 💡 Troubleshooting FAQs

- **Q: Colab says `No .pt files found`?**
  - Make sure the folder is unzipped and located in your Google Drive or at `/content/sequence_fused_features_binary`.
  - In Cell 2, check the path printed under `DATASET_DIR`.
- **Q: Does my session disconnect if I close my laptop?**
  - Training takes only 1-2 minutes! Keep your tab open until Step 6 finishes so your Colab runtime stays active.
