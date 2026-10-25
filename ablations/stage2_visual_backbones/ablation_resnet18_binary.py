#!/usr/bin/env python3
"""
=============================================================================
RESNET-18 BINARY CLASSIFICATION ABLATION PIPELINE
=============================================================================
Project: Student Engagement Detection
Architecture: Raw Videos / Face Crops -> ResNet-18 (512-D) -> Binary Classification
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
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------------------------
# 1. Configuration & Dataset Discovery
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Engaged", "Disengaged"]
NUM_CLASSES = 2
RESNET_FEATURE_DIM = 512
NUM_EPOCHS = 60
IMAGE_SIZE = (224, 224)
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
OUTPUT_DIR = "./ablation_results_resnet"


def find_dataset_folder(user_path=None):
    """
    Locates New_Dataset_Crafted or newly_crafted_dataset in Google Drive / locally.
    """
    if user_path and os.path.exists(user_path):
        return user_path

    candidates = [
        "/content/drive/MyDrive/New_Dataset_Crafted",
        "/content/drive/MyDrive/newly_crafted_dataset",
        "/content/drive/MyDrive/Student_Engagement_DL/New_Dataset_Crafted",
        "/content/drive/MyDrive/Student_Engagement_DL/newly_crafted_dataset",
        "/content/drive/MyDrive/Student_Engagement_DL/Dataset",
        "/content/New_Dataset_Crafted",
        "/content/newly_crafted_dataset",
        "./New_Dataset_Crafted",
        "./newly_crafted_dataset",
        "../New_Dataset_Crafted",
        "../newly_crafted_dataset"
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return c

    # Search dynamically in Drive
    if os.path.exists("/content/drive"):
        for root, dirs, _ in os.walk("/content/drive"):
            for d in dirs:
                if d.lower() in ["new_dataset_crafted", "newly_crafted_dataset"]:
                    return os.path.join(root, d)
    return None


def discover_binary_videos(dataset_dir):
    """
    Scans dataset folder for video clips.
    Strictly includes ONLY 'engaged' and 'disengaged'.
    Strictly EXCLUDES 'partially engaged'.
    """
    video_list = []
    if not os.path.exists(dataset_dir):
        return video_list

    for root, _, files in os.walk(dataset_dir):
        for f in sorted(files):
            if not f.lower().endswith(VIDEO_EXTENSIONS):
                continue

            full_path = os.path.join(root, f)
            norm_path = full_path.lower().replace(" ", "").replace("_", "").replace("-", "")

            # Strict exclusion of partially engaged
            if "partial" in norm_path or "partially" in norm_path:
                continue

            # Class determination
            if "disengaged" in norm_path or "disengage" in norm_path:
                label = 1
                class_name = "Disengaged"
            elif "engaged" in norm_path or "engage" in norm_path:
                label = 0
                class_name = "Engaged"
            else:
                continue

            video_list.append({
                "video_path": full_path,
                "filename": f,
                "base_name": os.path.splitext(f)[0],
                "label": label,
                "class_name": class_name
            })

    return video_list


# ---------------------------------------------------------------------------
# 2. ResNet-18 Feature Extractor
# ---------------------------------------------------------------------------

class ResNet18FeatureExtractor:
    """
    Loads pretrained ResNet-18 and replaces the final fully connected layer 
    with nn.Identity() to output raw 512-dimensional visual representations.
    """
    def __init__(self, device):
        self.device = device
        print(f"[ResNet-18] Initializing feature extractor on: {self.device}")

        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)

        model.fc = nn.Identity()  # Outputs 512-D
        model.eval()
        model.to(self.device)

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def extract_from_video(self, video_path, target_size=IMAGE_SIZE, batch_size=32):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, target_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"0 frames extracted from {video_path}")

        frames_np = np.array(frames, dtype=np.uint8)
        tensor = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0

        feats_list = []
        with torch.no_grad():
            for idx in range(0, len(tensor), batch_size):
                batch = tensor[idx : idx + batch_size]
                batch_norm = torch.stack([self.normalize(img) for img in batch]).to(self.device)
                batch_feats = self.model(batch_norm)
                feats_list.append(batch_feats.cpu().numpy())

        video_feats = np.concatenate(feats_list, axis=0)
        pooled_feat = video_feats.mean(axis=0)
        return pooled_feat


# ---------------------------------------------------------------------------
# 3. ResNet Classification Head
# ---------------------------------------------------------------------------

class UniversalEngagementClassifier(nn.Module):
    """
    Universal 2-Stage Classification Head for Controlled Ablation Studies:
    - Stage 1 (Projection Step): input_dim -> 128 (LayerNorm + ReLU + Dropout)
    - Stage 2 (Fixed 3-Layer Classifier): 128 -> 64 -> 32 -> 2 (num_classes)
    """
    def __init__(self, in_features=RESNET_FEATURE_DIM, num_classes=2, dropout=0.3):
        super(UniversalEngagementClassifier, self).__init__()
        
        # Stage 1: Projection Step (input_dim -> 128)
        self.projection = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stage 2: Fixed 3-Layer Classifier (128 -> 64 -> 32 -> 2)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        projected = self.projection(x)
        logits = self.classifier(projected)
        return logits


ResNetEngagementClassifier = UniversalEngagementClassifier


# ---------------------------------------------------------------------------
# 4. In-Memory Dataset
# ---------------------------------------------------------------------------

class ResNetVectorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        norm_feat = (feat - feat.mean()) / (feat.std() + 1e-6)
        return torch.tensor(norm_feat, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------------------------

def run_resnet_ablation(dataset_dir=None, cache_dir="./resnet18_extracted_features", output_dir=OUTPUT_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = find_dataset_folder(dataset_dir)
    if not data_dir:
        raise FileNotFoundError("Could not find New_Dataset_Crafted. Please specify dataset_dir path.")
    print(f"[Dataset] Using video folder: {data_dir}")

    videos = discover_binary_videos(data_dir)
    print(f"[Dataset] Total binary videos found (Partially Engaged excluded): {len(videos)}")

    labels = [v["label"] for v in videos]
    print(f"[Distribution] Engaged (0): {labels.count(0)} | Disengaged (1): {labels.count(1)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ResNet18FeatureExtractor(device)

    all_features = []
    all_valid_labels = []

    print("\n[Feature Extraction] Extracting/Loading ResNet-18 (512-D) features...")
    for v in videos:
        cache_file = os.path.join(cache_dir, f"{v['class_name']}__{v['base_name']}.npy")
        if os.path.exists(cache_file):
            feat = np.load(cache_file)
        else:
            try:
                feat = extractor.extract_from_video(v["video_path"])
                np.save(cache_file, feat)
            except Exception as e:
                print(f"[Warning] Failed {v['filename']}: {e}")
                continue

        all_features.append(feat)
        all_valid_labels.append(v["label"])

    all_features = np.array(all_features, dtype=np.float32)
    all_valid_labels = np.array(all_valid_labels, dtype=np.int64)
    total_valid = len(all_valid_labels)
    print(f"[Feature Extraction] Complete! Total usable samples: {total_valid}")

    # Strict Split: 93 Train / 20 Val / 20 Test
    idx = np.arange(total_valid)
    if total_valid >= 133:
        test_size = 20
        val_size = 20
        train_idx, temp_idx = train_test_split(idx, test_size=(test_size + val_size), random_state=42, stratify=all_valid_labels)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size, random_state=42, stratify=all_valid_labels[temp_idx])
    else:
        train_idx, temp_idx = train_test_split(idx, test_size=0.30, random_state=42, stratify=all_valid_labels)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=all_valid_labels[temp_idx])

    print(f"\n[Splits] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    train_loader = DataLoader(ResNetVectorDataset(all_features[train_idx], all_valid_labels[train_idx]), batch_size=16, shuffle=True)
    val_loader   = DataLoader(ResNetVectorDataset(all_features[val_idx], all_valid_labels[val_idx]), batch_size=16, shuffle=False)
    test_loader  = DataLoader(ResNetVectorDataset(all_features[test_idx], all_valid_labels[test_idx]), batch_size=16, shuffle=False)

    model = ResNetEngagementClassifier(in_features=RESNET_FEATURE_DIM, num_classes=NUM_CLASSES).to(device)

    print("\n" + "=" * 65)
    print("      RESNET-18 BINARY ABLATION: PARAMETER SUMMARY       ")
    print("=" * 65)
    print(f"{'Layer / Module Name':<35} | {'Weight Shape':<15} | {'Parameters':<10}")
    print("-" * 65)
    head_params = 0
    for name, param in model.named_parameters():
        p_count = param.numel()
        head_params += p_count
        print(f"{name:<35} | {str(list(param.shape)):<15} | {p_count:<10,}")

    RESNET18_BACKBONE_PARAMS = 11_176_512
    print("-" * 65)
    print(f"Trainable Classifier Parameters:       {head_params:,}")
    print(f"Pretrained ResNet-18 Backbone (Frozen): {RESNET18_BACKBONE_PARAMS:,}")
    print(f"Total Full-Pipeline Parameters:        {head_params + RESNET18_BACKBONE_PARAMS:,}")
    print("=" * 65)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_v_loss = float("inf")
    best_weights_path = os.path.join(output_dir, "best_resnet18_model.pth")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nStarting ResNet-18 Training for {NUM_EPOCHS} Epochs...")
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
            print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Train Loss: {tr_loss:.4f} Acc: {tr_acc:5.1f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:5.1f}%")

    # Evaluation on Test Set
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
    print("      FINAL RESNET-18 ABLATION RESULTS (TEST SET: 20 CLIPS)      ")
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
    ax1.set_title("ResNet-18 Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], 'b-', label='Train Acc (%)')
    ax2.plot(epochs, history["val_acc"], 'g-', label='Val Acc (%)')
    ax2.set_title("ResNet-18 Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "resnet18_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()

    # Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("ResNet-18 Confusion Matrix")
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
    cm_path = os.path.join(output_dir, "resnet18_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save Metrics JSON
    metrics_path = os.path.join(output_dir, "resnet18_ablation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "model": "ResNet18_Binary_Ablation",
            "feature_dim": RESNET_FEATURE_DIM,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "test_samples": len(test_idx),
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
    run_resnet_ablation()
