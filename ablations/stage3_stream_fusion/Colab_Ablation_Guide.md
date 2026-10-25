# 🧪 Ablation Study: Dual-Stream Fusion WITHOUT Transformer (Google Colab Guide)

This guide is designed for **first-time Google Colab users**. It provides step-by-step instructions on how to run the ablation study on Google Colab without needing to run anything locally on your PC.

---

## 📌 What is this Ablation Study?

In your original pipeline (`colab_binary_transformer`), you have:
1. **Stream 1**: EfficientNet-B0 visual features (1280-dim per frame)
2. **Stream 2**: OpenFace Action Units, gaze, head pose (709-dim per frame)
3. **Fusion**: Concatenation into a 1989-dim feature vector per frame
4. **Temporal Modeling**: **Transformer Encoder Stack** (Self-Attention across frames)
5. **Classification Head**: MLP Head (Engaged vs Disengaged)

In this **Ablation Study** (`colab_binary_no_transformer`):
- We keep **Stream 1 (EfficientNet-B0)**, **Stream 2 (OpenFace)**, their **Fusion (1989-dim)**, and the **identical Classification Head**.
- We **REMOVE the Transformer** (and Positional Encoding).
- We use direct **Masked Temporal Mean Pooling** across frames to aggregate video representations.
- This isolates the exact contribution of the Transformer to your research.

---

## 📁 1. Your Google Drive Structure

In your Google Drive, make sure your files are organized as follows:

```text
My Drive/
└── Student_Engagement_DL/
    ├── sequence_fuse_features_binary/       ← Your preprocessed .pt files (ALREADY IN YOUR DRIVE)
    │   ├── engaged__VID20260730105033.pt
    │   ├── disengaged__VID20260730112511.pt
    │   └── ...
    │
    └── colab_binary_no_transformer/         ← UPLOAD THIS NEW FOLDER TO YOUR DRIVE
        ├── Ablation_Binary_Fusion_Colab.ipynb  (Interactive Colab Notebook)
        ├── config_binary_fusion.py             (Configurations & paths)
        ├── models_binary_fusion.py             (Ablation model & parameter counter)
        ├── data_loader_binary_fusion.py        (Sliding windows & data loader)
        ├── train_binary_fusion.py              (60-Epoch training script)
        ├── evaluate_binary_fusion.py           (Video-level evaluation & metrics)
        └── Colab_Ablation_Guide.md             (This documentation)
```

> **Note on Folder Spelling:**  
> The scripts automatically check for both `sequence_fuse_features_binary` and `sequence_fused_features_binary`!

---

## 🚀 2. Step-by-Step Instructions for Google Colab

### **Step A: Upload the files to Google Drive**
1. Open [Google Drive](https://drive.google.com/) in your web browser.
2. Navigate to your `Student_Engagement_DL` folder.
3. Drag and drop the `colab_binary_no_transformer` folder from your PC (`c:\FYP\Binary classification\colab_binary_no_transformer`) into `Student_Engagement_DL` in Google Drive.

---

### **Step B: Open the Notebook in Google Colab**
1. Open [Google Colab](https://colab.research.google.com/).
2. A window will pop up asking to open a notebook:
   - Click the **"Upload"** tab.
   - Click **"Browse"** and select `Ablation_Binary_Fusion_Colab.ipynb` from `c:\FYP\Binary classification\colab_binary_no_transformer`.
   *(Alternatively, if you uploaded the folder to Google Drive, click the "Google Drive" tab and select `Ablation_Binary_Fusion_Colab.ipynb`).*

---

### **Step C: Enable Free GPU (T4 GPU)**
Before running any code, make sure Colab is using a GPU:
1. In the top menu bar of Google Colab, click:  
   **Runtime** $\rightarrow$ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

---

### **Step D: Run the Cells Step-by-Step**

The notebook is divided into sequential numbered steps. To run a cell, click on it and click the **Play ▶** button on its left (or press `Shift + Enter`):

#### **Cell 1: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```
*When prompted, click **"Connect to Google Drive"**, choose your Google account, and grant permission.*

#### **Cell 2: Set Working Directory**
```python
%cd /content/drive/MyDrive/Student_Engagement_DL/colab_binary_no_transformer
```

#### **Cell 3: Install Dependencies**
```bash
!pip install -q torch torchvision scikit-learn pandas matplotlib seaborn
```

#### **Cell 4: Verify Input Data**
*Confirms that your `.pt` files in `sequence_fuse_features_binary` are accessible and have the correct 1989-dim feature shape.*

#### **Cell 5: Display Parameter Count of Architecture**
*Displays the exact parameter count of EfficientNet-B0, OpenFace, Fusion layer, Classification head, and the parameter reduction vs Transformer.*

#### **Cell 6: Train for 60 Epochs**
```bash
!python train_binary_fusion.py
```
*Trains for 60 epochs. Shows progress, Train Loss, Train Acc, Val Loss, Val Acc, and saves the best model checkpoint to Google Drive.*

#### **Cell 7: Evaluate on Held-Out Test Set**
```bash
!python evaluate_binary_fusion.py
```
*Computes video-level Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curve.*

#### **Cell 8: View Visualizations**
*Displays the Confusion Matrix, ROC Curve, and Loss/Accuracy plots directly in the Colab notebook.*

---

## 📊 3. Parameter Count & Ablation Analysis

The code automatically calculates and displays this table before training and at the end of execution:

```text
======================================================================
        ARCHITECTURE PARAMETER COUNT & ABLATION SUMMARY
======================================================================
  Stream 1: EfficientNet-B0 Visual Extractor : 4,007,548 params (Pretrained/Frozen)
  Stream 2: OpenFace Facial Behavior Stream  : 709 features (Algorithmic / 0 PyTorch params)
  Dual-Stream Fused Input Dimension          : 1989 (1280 CNN + 709 OpenFace)
----------------------------------------------------------------------
  ABLATION MODEL (Without Transformer):
    • Fusion & Projection (Linear + LayerNorm): 254,976 params
    • Temporal Pooling Layer                   : 0 params (Non-parametric Mean Pooling)
    • MLP Classification Head (Binary 2-class) : 10,402 params
    -------------------------------------------------------
    Total Trainable Parameters                 : 265,378
    Total Ablation Model Parameters            : 265,378
----------------------------------------------------------------------
  ENTIRE END-TO-END PIPELINE (Backbone + Fusion Head):
    Grand Total Architecture Parameters        : 4,272,926 (~4.27M)
----------------------------------------------------------------------
  ABLATION COMPARISON (Transformer vs Non-Transformer):
    • Full Model with Transformer Stack        : ~663,586 trainable params
    • Ablation Model (No Transformer)          : 265,378 trainable params
    • Parameter Reduction                      : -398,208 params (-60.0%)
======================================================================
```

### **Ablation Takeaway for Your Report / Paper:**
- **Full Model (with Transformer)**: ~663k trainable parameters. Captures temporal sequence attention across frames.
- **Ablation Model (without Transformer)**: ~265k trainable parameters (**60% fewer trainable parameters**). Tests how well the dual-stream feature fusion performs with basic temporal averaging.
- Comparing test accuracy and F1-score between these two models proves the exact quantitative value added by the Transformer module!

---

## 📂 4. Generated Output Files on Google Drive

After running training and evaluation, the following outputs will be automatically saved directly in your Google Drive:

1. **`models_binary_fusion_ablation/`**:
   - `binary_fusion_ablation_weights.pth`: Best model checkpoint weights.

2. **`evaluation_results_binary_fusion_ablation/`**:
   - `predictions_binary_fusion_ablation.csv`: Detailed video-level test predictions with probabilities.
   - `metrics_binary_fusion_ablation.json`: Quantitative metrics (Accuracy, F1, Precision, Recall, AUC).
   - `confusion_matrix_binary_fusion_ablation.png`: Confusion Matrix chart.
   - `roc_curve_binary_fusion_ablation.png`: ROC Curve and AUC plot.
   - `training_curves_binary_fusion_ablation.png`: 60-epoch Loss and Accuracy curves over time.

---

## ❓ Frequently Asked Questions (FAQ)

### **Q: Do I need to run OpenFace in Colab for this ablation study?**
**No!** Because you already have `sequence_fuse_features_binary` in your Google Drive, all the EfficientNet-B0 and OpenFace features are already pre-extracted. You **do not need** to install or run OpenFace again.

### **Q: Can I close Colab while it's training?**
Keep the Colab browser tab open while training. Colab gives ~12 hours of continuous GPU time for free, and training 60 epochs on pre-extracted features takes only **2–5 minutes**!

### **Q: What if Colab says "Folder not found"?**
Check Cell 4 in the notebook. It tests all common path locations and will tell you if your Drive folder is named `sequence_fuse_features_binary` or `sequence_fused_features_binary`.
