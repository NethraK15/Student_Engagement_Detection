# 🔬 Deep Learning Ablation Study: Visual Backbones (Colab Guide)

This directory contains the standalone pipelines and Google Colab execution notebooks for the **Visual Backbone Ablation Study** in the **Student Engagement Detection** project.

---

## 📁 Directory Structure

```
Ablation_dl/
│
├── Ablation_EfficientNet_Binary.ipynb   # Colab Notebook runner for EfficientNet-B0
├── Ablation_ResNet18_Binary.ipynb       # Colab Notebook runner for ResNet-18
├── README_Colab_Guide.md                # Quick guide & execution instructions
│
├── ablation_efficientnet_binary.py      # Standalone Python pipeline for EfficientNet-B0
└── ablation_resnet18_binary.py          # Standalone Python pipeline for ResNet-18
```

---

## 🎯 Experimental Objective

Evaluate the representational power and parameter efficiency of unimodal deep learning visual backbones under a strictly controlled **Binary Engagement Classification** setting:
- **Classes**:
  - `0`: Engaged
  - `1`: Disengaged
  - *(Partially Engaged is strictly excluded)*
- **Data Partition**:
  - Train: **93 samples**
  - Validation: **20 samples**
  - Test: **20 samples**
  - Total: **133 samples** (65 Engaged, 68 Disengaged)

---

## 🚀 How to Run in Google Colab

### Option A: Using the Notebooks
1. Upload the `Ablation_dl/` folder to your Google Drive.
2. Open either:
   - [`Ablation_EfficientNet_Binary.ipynb`](file:///c:/Users/Admin/OneDrive/Desktop/Final%20Yr%20Project/Student_Engagement_Detection/Ablation_dl/Ablation_EfficientNet_Binary.ipynb)
   - [`Ablation_ResNet18_Binary.ipynb`](file:///c:/Users/Admin/OneDrive/Desktop/Final%20Yr%20Project/Student_Engagement_Detection/Ablation_dl/Ablation_ResNet18_Binary.ipynb)
3. Run all cells sequentially. The notebooks automatically mount Drive, run the `.py` script via CLI command, and display the resulting loss/accuracy curves and confusion matrices inline.

### Option B: Using the Command Line
```bash
# EfficientNet-B0 visual stream ablation (reads sequence_fused_features_binary)
python ablation_efficientnet_binary.py

# ResNet-18 visual backbone ablation (extracts 512-D from New_Dataset_Crafted)
python ablation_resnet18_binary.py
```

---

## 📊 Summary of Experimental Results

| Backbone | Feature Dim. | Backbone Params | Val Acc. | Test Acc. [95% CI] | Binary Precision | Binary Recall | Binary F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 512 | $\approx 11.18\text{ M}$ | 85.00% | **70.00% [48.1, 85.5]** | 70.00% | 70.00% | 70.00% |
| **EfficientNet-B0** | 1,280 | $\approx 5.29\text{ M}$ | 75.00% | **75.00% [53.1, 88.8]** | **72.73%** | **80.00%** | **76.19%** |

### Output Artifacts Saved:
- `ablation_results_efficientnet/`:
  - `best_efficientnet_model.pth`
  - `efficientnet_training_curves.png`
  - `efficientnet_confusion_matrix.png`
  - `efficientnet_ablation_metrics.json`
- `ablation_results_resnet/`:
  - `best_resnet18_model.pth`
  - `resnet18_training_curves.png`
  - `resnet18_confusion_matrix.png`
  - `resnet18_ablation_metrics.json`
