import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve, auc
)

import config_binary_bilstm as config
from data_loader_binary_bilstm import get_binary_dataloaders
from models_binary_bilstm import BinaryEngagementBiLSTM, display_parameter_summary


def predict_video_level(model, test_files, global_mean, global_std, device):
    """
    Evaluates test videos by aggregating sliding windows using soft-voting.
    """
    model.eval()
    video_targets = []
    video_preds   = []
    video_probs   = []
    video_names   = []

    with torch.no_grad():
        for fp in test_files:
            try:
                data     = torch.load(fp, map_location="cpu")
                sequence = data["sequence"]   # [N_frames, feat_dim]
                label    = int(data["label"])
                n_frames = sequence.shape[0]
                feat_dim = sequence.shape[1]

                if global_mean is not None and global_std is not None:
                    mean = global_mean.to(sequence.device)
                    std  = global_std.to(sequence.device)
                    sequence = (sequence - mean) / std

                window_probs_list = []
                start = 0
                while True:
                    end = start + config.WINDOW_SIZE
                    actual_len = min(config.WINDOW_SIZE, n_frames - start)
                    if actual_len <= 0:
                        break

                    window = sequence[start : start + actual_len]

                    if actual_len < config.WINDOW_SIZE:
                        pad = torch.zeros((config.WINDOW_SIZE - actual_len, feat_dim), dtype=torch.float32)
                        window = torch.cat([window, pad], dim=0)

                    window_tensor = window.unsqueeze(0).to(device)   # [1, Window_Size, Feat_Dim]
                    seq_len_t     = torch.tensor([actual_len], device=device)

                    logits = model(window_tensor, seq_len_t)
                    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    window_probs_list.append(probs)

                    start += config.WINDOW_STRIDE
                    if start >= n_frames:
                        break

                if not window_probs_list:
                    continue

                avg_probs  = np.mean(window_probs_list, axis=0)
                pred_label = int(np.argmax(avg_probs))

                video_targets.append(label)
                video_preds.append(pred_label)
                video_probs.append(avg_probs)
                video_names.append(os.path.basename(fp))

            except Exception as e:
                print(f"[Evaluate] Warning: Failed to evaluate '{os.path.basename(fp)}': {e}")

    return (
        np.array(video_targets),
        np.array(video_preds),
        np.array(video_probs),
        video_names
    )


def save_confusion_matrix(targets, preds, class_names, save_path):
    labels_range = list(range(len(class_names)))
    cm = confusion_matrix(targets, preds, labels=labels_range)
    plt.figure(figsize=(7, 6))
    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 15, "weight": "bold"}
        )
    else:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=12)
        plt.yticks(tick_marks, class_names, fontsize=12)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, weight="bold")

    plt.title("Ablation Study: Confusion Matrix (BiLSTM)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to : {save_path}")


def save_roc_curve(targets, probs, save_path):
    pos_probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(targets, pos_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"BiLSTM ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    plt.title("Ablation Study: ROC Curve (BiLSTM)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to        : {save_path}")
    return float(roc_auc)


def main():
    print("=" * 70)
    print("      ABLATION STUDY: VIDEO-LEVEL TEST EVALUATION (BiLSTM)")
    print("      Dual-Stream Fusion + 2-Layer BiLSTM")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device : {device}\n")

    if not os.path.exists(config.MODEL_WEIGHTS_PATH):
        print(f"[Error] Model checkpoint not found at:\n  {config.MODEL_WEIGHTS_PATH}")
        print("Please run 'train_binary_bilstm.py' first.")
        sys.exit(1)

    train_loader, val_loader, test_loader, test_files, global_mean, global_std = get_binary_dataloaders(
        sequence_fused_dir=config.SEQUENCE_FUSED_DIR,
        batch_size=config.BATCH_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Loading weights from: {config.MODEL_WEIGHTS_PATH}")
    checkpoint = torch.load(config.MODEL_WEIGHTS_PATH, map_location=device)
    input_dim = checkpoint.get("input_dim", config.FUSED_INPUT_DIM)

    model = BinaryEngagementBiLSTM(
        input_dim=input_dim,
        d_model=config.D_MODEL,
        lstm_hidden=config.LSTM_HIDDEN,
        num_layers=config.LSTM_LAYERS,
        bidirectional=config.BIDIRECTIONAL,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nRunning video-level evaluation on {len(test_files)} held-out test videos...")
    targets, preds, probs, names = predict_video_level(
        model, test_files, global_mean, global_std, device
    )

    acc = accuracy_score(targets, preds)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(targets, preds, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)

    print("\n" + "=" * 70)
    print("      BiLSTM ABLATION STUDY: TEST RESULTS")
    print("=" * 70)
    print(f"  Test Accuracy           : {acc * 100:.2f}%")
    print(f"  Weighted F1-Score       : {f1_w * 100:.2f}%")
    print(f"  Weighted Precision      : {prec_w * 100:.2f}%")
    print(f"  Weighted Recall         : {rec_w * 100:.2f}%")
    print(f"  Macro F1-Score          : {f1_m * 100:.2f}%")
    print("-" * 70)
    print("\nPer-Class Classification Report:")
    print(classification_report(targets, preds, target_names=config.CLASS_NAMES, digits=4))

    os.makedirs(config.EVAL_OUT_DIR, exist_ok=True)
    save_confusion_matrix(targets, preds, config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)
    roc_auc = save_roc_curve(targets, probs, config.ROC_CURVE_PATH)

    df_pred = pd.DataFrame({
        "video_file"       : names,
        "true_label"       : targets,
        "true_class"       : [config.CLASS_NAMES[t] for t in targets],
        "predicted_label"  : preds,
        "predicted_class"  : [config.CLASS_NAMES[p] for p in preds],
        "prob_engaged"     : probs[:, 0],
        "prob_disengaged"  : probs[:, 1],
    })
    df_pred.to_csv(config.PREDICTIONS_CSV_PATH, index=False)
    print(f"Predictions CSV saved to  : {config.PREDICTIONS_CSV_PATH}")

    metrics_summary = {
        "model_type"               : "Ablation Study: EfficientNet-B0 + OpenFace + BiLSTM",
        "test_accuracy"            : float(acc),
        "weighted_f1"              : float(f1_w),
        "weighted_precision"       : float(prec_w),
        "weighted_recall"          : float(rec_w),
        "macro_f1"                 : float(f1_m),
        "roc_auc"                  : float(roc_auc),
        "total_test_videos"        : len(targets),
        "training_epochs"          : config.EPOCHS,
    }
    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics JSON saved to     : {config.METRICS_JSON_PATH}")

    print("\n[Final Architecture Verification] Parameter count of the entire BiLSTM architecture:")
    display_parameter_summary(model, input_dim=input_dim)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
