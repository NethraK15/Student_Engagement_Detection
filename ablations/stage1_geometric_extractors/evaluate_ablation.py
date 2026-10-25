import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve, auc
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config_ablation as config
from dataset_openface import get_ablation_dataloaders
from model_openface import OpenFaceAblationClassifier

def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for seqs, lens, labels in test_loader:
            seqs, lens = seqs.to(device), lens.to(device)
            outputs = model(seqs, lens)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

def save_confusion_matrix(targets, preds, class_names, save_path):
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Ablation Study: OpenFace Alone Confusion Matrix', fontsize=12, pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            plt.text(j, i, str(val), horizontalalignment="center", verticalalignment="center", color=color, fontsize=12)

    plt.xlabel('Predicted Class', fontsize=11)
    plt.ylabel('Ground Truth Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Confusion matrix saved: {save_path}")

def save_roc_curve(targets, probs, save_path):
    plt.figure(figsize=(6, 5))
    if len(np.unique(targets)) > 1:
        # Binary ROC on positive class (Disengaged = 1)
        fpr, tpr, _ = roc_curve(targets, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'OpenFace Alone (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve - OpenFace Alone Binary Engagement', fontsize=12, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] ROC curve saved: {save_path}")

def run_evaluation(dataset_dir=None):
    dataset_path = dataset_dir if dataset_dir else config.SEQUENCE_FUSED_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {config.CHECKPOINT_PATH}. Please train the model first.")

    # 1. Load Checkpoint
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    feat_dim = checkpoint.get("input_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_classes = checkpoint.get("num_classes", 2)

    # 2. Load Test Data
    _, _, test_loader, _ = get_ablation_dataloaders(
        dataset_path, batch_size=config.BATCH_SIZE, random_state=config.RANDOM_STATE
    )

    # 3. Initialize Model and Load State Dict
    model = OpenFaceAblationClassifier(
        input_dim=feat_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Evaluate Test Set
    targets, preds, probs = evaluate_test_set(model, test_loader, device)

    test_acc = accuracy_score(targets, preds) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='weighted', zero_division=0)
    precision *= 100.0
    recall *= 100.0
    f1 *= 100.0

    # Macro averages as well for complete research verification
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    macro_precision *= 100.0
    macro_recall *= 100.0
    macro_f1 *= 100.0

    # 5. Print Formatted Table (Matching Project Paper Format)
    print("\n" + "=" * 90)
    print("               🏆 ABLATION STUDY RESULTS TABLE (MATCHING YOUR PAPER) 🏆               ")
    print("=" * 90)
    header = f"{'Model':<35} | {'Best Val Acc':<14} | {'Testing Acc':<12} | {'Recall':<10} | {'F1-Score':<10} | {'Precision':<10}"
    divider = "-" * len(header)
    print(header)
    print(divider)
    row_weighted = f"{'OpenFace Alone (Binary Ablation)':<35} | {best_val_acc:>12.2f}% | {test_acc:>10.2f}% | {recall:>8.2f}% | {f1:>8.2f}% | {precision:>8.2f}%"
    print(row_weighted)
    print(divider)
    print(f"{'(Macro-averaged metrics reference)':<35} | {'-':>14} | {test_acc:>10.2f}% | {macro_recall:>8.2f}% | {macro_f1:>8.2f}% | {macro_precision:>8.2f}%")
    print("=" * 90 + "\n")

    # Classification Report
    print("Detailed Classification Report:")
    print(classification_report(targets, preds, target_names=config.CLASS_NAMES, digits=4, zero_division=0))

    # Save artifacts
    save_confusion_matrix(targets, preds, config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)
    save_roc_curve(targets, probs, config.ROC_CURVE_PATH)

    metrics_dict = {
        "model": "OpenFace Alone (Binary Ablation)",
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "recall_weighted": float(recall),
        "f1_score_weighted": float(f1),
        "precision_weighted": float(precision),
        "recall_macro": float(macro_recall),
        "f1_score_macro": float(macro_f1),
        "precision_macro": float(macro_precision)
    }

    with open(config.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[Evaluation] Metrics saved to: {config.METRICS_JSON_PATH}")

    return metrics_dict

if __name__ == "__main__":
    run_evaluation()
