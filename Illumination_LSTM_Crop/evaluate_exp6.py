import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

try:
    import seaborn as sns
except ImportError:
    sns = None

import config_exp6
from data_loader_exp6 import get_sequence_dataloaders
from models_exp6 import EngagementLSTMClassifier

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plots and saves the confusion matrix heatmap for Experiment 6.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(8, 6))
    
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_percent = cm.astype('float') / row_sums * 100
    
    labels = [f"{count}\n({pct:.1f}%)" for count, pct in zip(cm.flatten(), cm_percent.flatten())]
    labels_grid = np.asarray(labels).reshape(cm.shape)

    if sns is not None:
        sns.heatmap(
            cm, 
            annot=labels_grid, 
            fmt="", 
            cmap="Blues", 
            xticklabels=classes, 
            yticklabels=classes,
            cbar=True,
            annot_kws={"size": 12}
        )
    else:
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, labels_grid[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10
                )
                
    plt.title("Confusion Matrix - Face Cropped Illuminated BiLSTM (Exp 6)", fontsize=13, pad=15)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Experiment 6 Confusion Matrix saved to Drive: {save_path}")

def plot_roc_curves(y_true, y_probs, classes, save_path):
    """
    Plots and saves multi-class One-vs-Rest ROC curves.
    """
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    plt.figure(figsize=(8, 6))
    has_plotted = False
    
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) < 2:
            print(f"[Evaluation] Warning: Skipping ROC curve for class '{classes[i]}' "
                  f"(requires both positive and negative test samples).")
            continue
            
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve: {classes[i]} (AUC = {roc_auc:.2f})')
        has_plotted = True
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Face Cropped Illuminated BiLSTM Model (One-vs-Rest)', fontsize=13, pad=15)
    
    if has_plotted:
        plt.legend(loc="lower right", fontsize=10)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Experiment 6 ROC Curve saved to Drive: {save_path}")

def main():
    print("==================================================")
    print("      EVALUATING EXPERIMENT 6 BiLSTM MODEL        ")
    print("==================================================")

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        _, _, test_loader, test_files = get_sequence_dataloaders(
            sequence_fused_dir=config_exp6.SEQUENCE_FUSED_DIR,
            batch_size=config_exp6.BATCH_SIZE,
            random_state=config_exp6.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Error] Failed to load test data: {e}")
        sys.exit(1)

    os.makedirs(config_exp6.EVAL_OUT_DIR, exist_ok=True)

    # 2. Load Model weights from Google Drive
    if not os.path.exists(config_exp6.MODEL_WEIGHTS_PATH):
        print(f"[Error] Saved Experiment 6 weights not found at '{config_exp6.MODEL_WEIGHTS_PATH}'.")
        print("Please train the model first by running train_exp6.py.")
        sys.exit(1)

    checkpoint = torch.load(config_exp6.MODEL_WEIGHTS_PATH, map_location=device)
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint.get('hidden_dim', config_exp6.HIDDEN_DIM)
    num_layers = checkpoint.get('num_layers', config_exp6.NUM_LAYERS)
    
    print(f"Loading Experiment 6 BiLSTM checkpoint with input dim: {input_dim}")
    
    model = EngagementLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=len(config_exp6.CLASSES)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Running Inference
    all_labels = []
    all_probs = []
    all_preds = []
    
    softmax = torch.nn.Softmax(dim=1)

    print("\nRunning Experiment 6 BiLSTM inference on test set...")
    with torch.no_grad():
        for sequences, seq_lens, labels in test_loader:
            sequences = sequences.to(device)
            logits = model(sequences, seq_lens)
            probs = softmax(logits)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = np.array(all_preds)

    # 4. Calculate metrics
    class_indices = list(range(len(config_exp6.CLASSES)))
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, average='weighted', zero_division=0
    )
    
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_indices, average=None, zero_division=0
    )

    print("\n================ METRICS SUMMARY (EXP 6: Face Cropped BiLSTM) ================")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}% (weighted)")
    print(f"Recall:    {recall*100:.2f}% (weighted)")
    print(f"F1-score:  {f1*100:.2f}% (weighted)")
    print("==============================================================================")

    report = classification_report(
        y_true, y_pred, labels=class_indices, target_names=config_exp6.CLASS_NAMES, zero_division=0
    )
    print("\nClassification Report:")
    print(report)

    # Save metrics JSON
    metrics_dict = {
        "model_type": "Face_Cropped_Illuminated_BiLSTM_Attention",
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "class_metrics": {}
    }
    for idx, c_name in enumerate(config_exp6.CLASS_NAMES):
        metrics_dict["class_metrics"][c_name] = {
            "precision": float(precision_class[idx]),
            "recall": float(recall_class[idx]),
            "f1": float(f1_class[idx])
        }

    with open(config_exp6.METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[Evaluation] Saved JSON metrics to Drive: {config_exp6.METRICS_JSON_PATH}")

    # 5. Plot Figures
    plot_confusion_matrix(y_true, y_pred, config_exp6.CLASS_NAMES, config_exp6.CONFUSION_MATRIX_PATH)
    plot_roc_curves(y_true, y_probs, config_exp6.CLASS_NAMES, config_exp6.ROC_CURVE_PATH)

    # 6. Save predictions CSV
    video_names = []
    for filepath in test_files:
        data = torch.load(filepath, map_location="cpu")
        video_names.append(data["video_name"])

    predictions_df = pd.DataFrame({
        "video_filename": video_names,
        "true_label": [config_exp6.CLASS_NAMES[l] for l in y_true],
        "predicted_label": [config_exp6.CLASS_NAMES[p] for p in y_pred],
        "true_label_idx": y_true,
        "predicted_label_idx": y_pred,
        "prob_engaged": y_probs[:, 0],
        "prob_partially_engaged": y_probs[:, 1],
        "prob_disengaged": y_probs[:, 2]
    })
    
    predictions_df.to_csv(config_exp6.PREDICTIONS_CSV_PATH, index=False)
    print(f"[Evaluation] Predictions CSV saved to Drive: {config_exp6.PREDICTIONS_CSV_PATH}")

    print("\n==================================================")
    print("        EXPERIMENT 6 EVALUATION COMPLETE          ")
    print("==================================================")

if __name__ == "__main__":
    main()
