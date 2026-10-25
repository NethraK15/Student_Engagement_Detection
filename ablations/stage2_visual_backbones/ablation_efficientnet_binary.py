#!/usr/bin/env python3
"""
=============================================================================
EFFICIENTNET-B0 BINARY CLASSIFICATION ABLATION PIPELINE
=============================================================================
Project: Student Engagement Detection
Architecture: Data -> EfficientNet-B0 (1280-D) -> Binary Classification
Classes:
  - 0: Engaged
  - 1: Disengaged
  (Partially Engaged is strictly EXCLUDED)
Dataset Split:
  - Train: 93 samples
  - Validation: 20 samples
  - Test: 20 samples
  - Total: 133 samples
=============================================================================
"""

import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# 1. Configuration & Auto-Detection of Google Drive Folder
# ---------------------------------------------------------------------------

DEFAULT_DRIVE_FOLDER_NAME = "sequence_fused_features_binary"
CLASS_NAMES = ["Engaged", "Disengaged"]
NUM_CLASSES = 2
EFFICIENTNET_DIM = 1280
OUTPUT_DIR = "./ablation_results_efficientnet"
NUM_EPOCHS = 60


def find_data_directory(user_specified_dir=None):
    """
    Auto-detects the preprocessed feature directory in Google Drive or locally.
    """
    if user_specified_dir and os.path.exists(user_specified_dir):
        return user_specified_dir

    candidate_paths = [
        f"/content/drive/MyDrive/{DEFAULT_DRIVE_FOLDER_NAME}",
        f"/content/drive/MyDrive/Student_Engagement_DL/{DEFAULT_DRIVE_FOLDER_NAME}",
        f"/content/drive/MyDrive/Student_Engagement_Detection/{DEFAULT_DRIVE_FOLDER_NAME}",
        f"/content/{DEFAULT_DRIVE_FOLDER_NAME}",
        f"./{DEFAULT_DRIVE_FOLDER_NAME}",
        f"../{DEFAULT_DRIVE_FOLDER_NAME}"
    ]

    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            return p

    # Dynamic search within /content/drive if mounted on Colab
    if os.path.exists("/content/drive"):
        print("[Auto-Detect] Searching for folder in /content/drive...")
        for root, dirs, _ in os.walk("/content/drive"):
            for d in dirs:
                if d.lower() == DEFAULT_DRIVE_FOLDER_NAME.lower():
                    return os.path.join(root, d)

    return None


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset for EfficientNet Modality Ablation
# ---------------------------------------------------------------------------

class EfficientNetAblationDataset(Dataset):
    """
    Loads preprocessed sequence files and extracts ONLY the visual stream
    features (first 1280 dimensions), performing temporal average pooling.
    """
    def __init__(self, file_paths, eff_dim=EFFICIENTNET_DIM):
        self.file_paths = file_paths
        self.eff_dim = eff_dim

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu")

        raw = data.get("sequence", data.get("feature", data.get("features", list(data.values())[0]))) if isinstance(data, dict) else data
        raw = raw.float() if torch.is_tensor(raw) else torch.tensor(raw, dtype=torch.float32)

        # Modality Ablation: Slices ONLY EfficientNet-B0 visual stream
        if raw.dim() == 2:
            eff_stream = raw[:, :self.eff_dim]
            pooled = eff_stream.mean(dim=0)
        else:
            pooled = raw.view(-1)[:self.eff_dim]

        # Z-score normalization
        norm_feat = (pooled - pooled.mean()) / (pooled.std() + 1e-6)

        # Label resolution (0: Engaged, 1: Disengaged)
        fn = os.path.basename(file_path).lower()
        if fn.startswith("disengaged") or "__disengaged" in fn:
            label = 1
        elif fn.startswith("engaged") or "__engaged" in fn:
            label = 0
        elif isinstance(data, dict) and "label" in data:
            label = 1 if int(data["label"]) in [1, 2] else 0
        else:
            label = 1 if "disengaged" in fn else 0

        return norm_feat, label


# ---------------------------------------------------------------------------
# 3. Dedicated EfficientNet Classification Head
# ---------------------------------------------------------------------------

class EfficientNetClassifier(nn.Module):
    """
    3-Layer Dense Classifier Head for EfficientNet-B0 Visual Ablation:
    1280 -> 512 -> 128 -> 2
    """
    def __init__(self, input_dim=EFFICIENTNET_DIM, num_classes=2):
        super(EfficientNetClassifier, self).__init__()
        self.net = nn.Sequential(
            # Layer 1: 1280 -> 512
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            # Layer 2: 512 -> 128
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 128 -> 2
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. Main Execution Function
# ---------------------------------------------------------------------------

def run_ablation(data_dir=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  STUDENT ENGAGEMENT DETECTION: EFFICIENTNET ABLATION STUDY")
    print("  Pipeline: Data -> EfficientNet-B0 (1280-D) -> Binary Classification")
    print("=" * 65)

    resolved_dir = find_data_directory(data_dir)
    if not resolved_dir:
        raise FileNotFoundError(
            f"Could not locate '{DEFAULT_DRIVE_FOLDER_NAME}'. Please ensure Drive is mounted."
        )
    print(f"[Dataset] Using feature folder: {resolved_dir}")

    # Gather binary .pt files (excluding partially engaged)
    raw_files = sorted(glob.glob(os.path.join(resolved_dir, "*.pt")))
    all_files = [f for f in raw_files if "partial" not in os.path.basename(f).lower()]
    print(f"[Dataset] Total binary .pt files found: {len(all_files)}")

    temp_ds = EfficientNetAblationDataset(all_files)
    all_labels = [temp_ds[i][1] for i in range(len(temp_ds))]
    print(f"[Distribution] Engaged (0): {all_labels.count(0)} | Disengaged (1): {all_labels.count(1)}")

    # Model Setup & Summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier(input_dim=EFFICIENTNET_DIM, num_classes=NUM_CLASSES).to(device)

    print("\n" + "=" * 65)
    print("     EFFICIENTNET ABLATION: MODEL PARAMETER SUMMARY       ")
    print("=" * 65)
    print(f"{'Layer / Module Name':<35} | {'Weight Shape':<15} | {'Parameters':<10}")
    print("-" * 65)
    total_head_params = 0
    for name, param in model.named_parameters():
        num_p = param.numel()
        total_head_params += num_p
        shape_str = str(list(param.shape))
        print(f"{name:<35} | {shape_str:<15} | {num_p:<10,}")

    EFFICIENTNET_B0_BACKBONE_PARAMS = 5_288_548
    print("-" * 65)
    print(f"Classifier Head Parameters (Trainable): {total_head_params:,}")
    print(f"Pretrained EfficientNet-B0 Backbone:    {EFFICIENTNET_B0_BACKBONE_PARAMS:,} (Frozen)")
    print(f"Total Full-Pipeline Parameters:        {total_head_params + EFFICIENTNET_B0_BACKBONE_PARAMS:,}")
    print("=" * 65)

    # Stratified Split (93 Train / 20 Val / 20 Test)
    idx = np.arange(len(all_files))
    if len(all_files) >= 133:
        test_size = 20
        val_size = 20
        tr_idx, temp_idx = train_test_split(idx, test_size=(test_size + val_size), random_state=42, stratify=all_labels)
        val_idx, te_idx = train_test_split(temp_idx, test_size=test_size, random_state=42, stratify=[all_labels[i] for i in temp_idx])
    else:
        tr_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=all_labels)
        val_idx, te_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=[all_labels[i] for i in temp_idx])

    print(f"\n[Splits] Train: {len(tr_idx)} | Val: {len(val_idx)} | Test: {len(te_idx)}")

    train_loader = DataLoader(EfficientNetAblationDataset([all_files[i] for i in tr_idx]), batch_size=16, shuffle=True)
    val_loader   = DataLoader(EfficientNetAblationDataset([all_files[i] for i in val_idx]), batch_size=16, shuffle=False)
    test_loader  = DataLoader(EfficientNetAblationDataset([all_files[i] for i in te_idx]), batch_size=16, shuffle=False)

    # Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_v_loss = float("inf")
    best_weights_path = os.path.join(output_dir, "best_efficientnet_model.pth")
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print(f"\nStarting EfficientNet Training for {NUM_EPOCHS} Epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_loss, tr_corr, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        tr_loss /= total
        tr_acc = (tr_corr / total) * 100

        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * x.size(0)
                v_corr += (out.argmax(1) == y).sum().item()
                v_total += y.size(0)
        v_loss /= v_total
        v_acc = (v_corr / v_total) * 100

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)

        if v_loss < best_v_loss:
            best_v_loss = v_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Train Acc: {tr_acc:5.1f}% | Val Acc: {v_acc:5.1f}%")

    # Final Evaluation
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(model(x).argmax(1).cpu().numpy())

    test_acc = accuracy_score(y_true, y_pred) * 100
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2] * 100

    print("\n" + "=" * 65)
    print("     FINAL EFFICIENTNET ABLATION RESULTS (TEST SET: 20 CLIPS)    ")
    print("=" * 65)
    print(f"  Test Accuracy:     {test_acc:.2f}%")
    print(f"  Binary Precision:  {p*100:.2f}%")
    print(f"  Binary Recall:     {r*100:.2f}%")
    print(f"  Binary F1-Score:   {f1*100:.2f}%")
    print(f"  Macro F1-Score:    {macro_f1:.2f}%")
    print("=" * 65)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    # Save Plots
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    ax1.set_title("EfficientNet Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], 'b-', label='Train Acc (%)')
    ax2.plot(epochs, history["val_acc"], 'g-', label='Val Acc (%)')
    ax2.set_title("EfficientNet Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "efficientnet_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("EfficientNet Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES)
    plt.yticks(ticks, CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if cm[i, j] > 5 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "efficientnet_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save JSON metrics
    metrics_path = os.path.join(output_dir, "efficientnet_ablation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model": "EfficientNet_B0_Binary_Ablation",
            "feature_dim": EFFICIENTNET_DIM,
            "train_samples": len(tr_idx),
            "val_samples": len(val_idx),
            "test_samples": len(te_idx),
            "test_accuracy": test_acc,
            "precision": p * 100,
            "recall": r * 100,
            "f1_score": f1 * 100,
            "macro_f1": macro_f1
        }, f, indent=4)

    print(f"[Done] Artifacts saved to '{output_dir}':")
    print(f"  - Model Weights:    {best_weights_path}")
    print(f"  - Training Curves:  {curves_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - Metrics JSON:     {metrics_path}")


if __name__ == "__main__":
    run_ablation()
