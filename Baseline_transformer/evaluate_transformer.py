import os
import sys
import json
import pandas as pd
import numpy as np
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
from sklearn.preprocessing import label_binarize

import config_transformer
from data_loader_transformer import get_sequence_dataloaders
from models_transformer import EngagementTransformerClassifier

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    video_names = []

    with torch.no_grad():
        for sequences, seq_lens, labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences, seq_lens)
            probs = F.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    return all_targets, all_preds, all_probs

def save_confusion_matrix(targets, preds, class_names, save_path):
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.title('Temporal Transformer Confusion Matrix', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Transformer Confusion Matrix saved: {save_path}")

def save_roc_curves(targets, probs, class_names, save_path):
    n_classes = len(class_names)
    targets_bin = label_binarize(targets, classes=[0, 1, 2])
    
    if n_classes == 2 or targets_bin.shape[1] == 1:
        targets_bin = np.hstack([1 - targets_bin, targets_bin])

    plt.figure(figsize=(8, 6))
    colors = ['#2ca02c', '#ff7f0e', '#d62728']

    for i in range(n_classes):
        if len(np.unique(targets_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        else:
            print(f"[Evaluation] Warning: Skipping ROC curve for class '{class_names[i]}' (requires both positive and negative test samples).")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Temporal Transformer Multi-Class ROC Curves', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Transformer ROC Curve saved: {save_path}")

def main():
    print("==================================================")
    print("      EVALUATING TEMPORAL TRANSFORMER MODEL       ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(config_transformer.MODEL_WEIGHTS_PATH):
        print(f"[Error] Model weights not found at: {config_transformer.MODEL_WEIGHTS_PATH}")
        print("Please train the model first by running train_transformer.py.")
        sys.exit(1)

    checkpoint = torch.load(config_transformer.MODEL_WEIGHTS_PATH, map_location=device)
    input_dim = checkpoint.get('input_dim', 1989)
    d_model = checkpoint.get('d_model', config_transformer.D_MODEL)
    nhead = checkpoint.get('nhead', config_transformer.NHEAD)
    num_layers = checkpoint.get('num_layers', config_transformer.NUM_LAYERS)

    _, _, test_loader, test_files = get_sequence_dataloaders(
        sequence_fused_dir=config_transformer.SEQUENCE_FUSED_DIR,
        batch_size=config_transformer.BATCH_SIZE,
        random_state=config_transformer.RANDOM_STATE
    )

    model = EngagementTransformerClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=config_transformer.DIM_FEEDFORWARD,
        dropout=config_transformer.DROPOUT,
        num_classes=len(config_transformer.CLASSES)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded Transformer checkpoint with input dim: {input_dim} (d_model: {d_model})\n")

    targets, preds, probs = evaluate_model(model, test_loader, device, config_transformer.CLASS_NAMES)

    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='weighted', zero_division=0)

    print("================ METRICS SUMMARY (TRANSFORMER) ================")
    print(f"Accuracy:  {acc * 100:6.2f}%")
    print(f"Precision: {precision * 100:6.2f}% (weighted)")
    print(f"Recall:    {recall * 100:6.2f}% (weighted)")
    print(f"F1-score:  {f1 * 100:6.2f}% (weighted)")
    print("================================================================\n")

    report_text = classification_report(targets, preds, labels=[0, 1, 2], target_names=config_transformer.CLASS_NAMES, zero_division=0)
    print("Classification Report:")
    print(report_text)

    os.makedirs(config_transformer.EVAL_OUT_DIR, exist_ok=True)

    metrics_data = {
        "model": "Temporal Transformer",
        "accuracy": float(acc),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classification_report": classification_report(targets, preds, labels=[0, 1, 2], target_names=config_transformer.CLASS_NAMES, output_dict=True, zero_division=0)
    }

    with open(config_transformer.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[Evaluation] Saved JSON metrics to: {config_transformer.METRICS_JSON_PATH}")

    save_confusion_matrix(targets, preds, config_transformer.CLASS_NAMES, config_transformer.CONFUSION_MATRIX_PATH)
    save_roc_curves(targets, probs, config_transformer.CLASS_NAMES, config_transformer.ROC_CURVE_PATH)

    video_names = [os.path.basename(f) for f in test_files[:len(targets)]]
    df_preds = pd.DataFrame({
        "video_file": video_names,
        "true_label_idx": targets,
        "true_label": [config_transformer.CLASS_NAMES[t] for t in targets],
        "pred_label_idx": preds,
        "pred_label": [config_transformer.CLASS_NAMES[p] for p in preds],
        "prob_engaged": probs[:, 0],
        "prob_partially_engaged": probs[:, 1],
        "prob_disengaged": probs[:, 2]
    })
    df_preds.to_csv(config_transformer.PREDICTIONS_CSV_PATH, index=False)
    print(f"[Evaluation] Predictions CSV saved to: {config_transformer.PREDICTIONS_CSV_PATH}")

    print("\n==================================================")
    print("       TRANSFORMER EVALUATION COMPLETE            ")
    print("==================================================")

if __name__ == "__main__":
    main()
