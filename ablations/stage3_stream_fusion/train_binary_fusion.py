import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_binary_fusion as config
from data_loader_binary_fusion import get_binary_dataloaders
from models_binary_fusion import BinaryEngagementFusionModel, display_parameter_summary

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=config.MAX_GRAD_NORM):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, seq_lens, labels in dataloader:
        sequences = sequences.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences, seq_lens)
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, seq_lens, labels in dataloader:
            sequences = sequences.to(device)
            labels    = labels.to(device)

            outputs = model(sequences, seq_lens)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#ff7f0e", lw=2)
    ax1.set_title("Ablation Study: Loss Curve (No Transformer)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Accuracy",   color="#d62728", lw=2)
    ax2.set_title("Ablation Study: Accuracy Curve (No Transformer)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = os.path.join(save_dir, "training_curves_binary_fusion_ablation.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {chart_path}")


def main():
    print("=" * 70)
    print("  ABLATION EXPERIMENT: DUAL-STREAM FUSION WITHOUT TRANSFORMER")
    print("  Stream 1: EfficientNet-B0 Visual Backbone (1280-dim)")
    print("  Stream 2: OpenFace Facial Behavior Stream (709-dim)")
    print("  Fusion  : Concatenation (1989-dim) -> Projection -> Temporal Pooling")
    print("  Head    : 2-Class MLP Classification Head (Engaged vs Disengaged)")
    print("  Training Duration: 60 Epochs")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU Model      : {torch.cuda.get_device_name(0)}")

    # ── 1. Load DataLoaders ──────────────────────────────────────────
    print(f"\nLooking for sequence features in:\n  {config.SEQUENCE_FUSED_DIR}")
    try:
        train_loader, val_loader, _, _, global_mean, global_std = get_binary_dataloaders(
            sequence_fused_dir=config.SEQUENCE_FUSED_DIR,
            batch_size=config.BATCH_SIZE,
            random_state=config.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Fatal Error] Could not load dataset: {e}")
        sys.exit(1)

    # ── 2. Detect feature dimension ──────────────────────────────────
    seq_files = [f for f in os.listdir(config.SEQUENCE_FUSED_DIR) if f.endswith(".pt")]
    sample_pt = torch.load(os.path.join(config.SEQUENCE_FUSED_DIR, seq_files[0]), map_location="cpu")
    input_dim = sample_pt["sequence"].shape[1]

    # ── 3. Build Ablation Model (No Transformer) ─────────────────────
    model = BinaryEngagementFusionModel(
        input_dim=input_dim,
        d_model=config.D_MODEL,
        dropout=config.DROPOUT,
        num_classes=config.NUM_CLASSES
    ).to(device)

    # ── 4. Calculate & Display Parameter Count (Before Training) ─────
    print("\n[Architecture Initialisation] Calculating parameter counts...")
    param_info = display_parameter_summary(model, input_dim=input_dim)

    # ── 5. Loss, Optimizer, Cosine Scheduler ─────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(config.MODEL_WEIGHTS_PATH), exist_ok=True)
    os.makedirs(config.EVAL_OUT_DIR, exist_ok=True)

    # ── 6. Run Training for exactly 60 Epochs ─────────────────────────
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Beginning 60-epoch ablation training on {device}...\n")

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)

        status_flag = " ⭐ Best Model Saved" if improved else ""
        print(
            f"Epoch [{epoch+1:02d}/{config.EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:5.2f}%"
            + status_flag
        )

        if improved:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch"                 : epoch + 1,
                "model_state_dict"      : model.state_dict(),
                "optimizer_state_dict"  : optimizer.state_dict(),
                "val_acc"               : val_acc,
                "val_loss"              : val_loss,
                "input_dim"             : input_dim,
                "d_model"               : config.D_MODEL,
                "window_size"           : config.WINDOW_SIZE,
                "window_stride"         : config.WINDOW_STRIDE,
                "num_classes"           : config.NUM_CLASSES,
                "global_mean"           : global_mean,
                "global_std"            : global_std,
                "architecture_params"   : param_info,
            }, config.MODEL_WEIGHTS_PATH)

    # ── 7. Save Loss / Accuracy curves ───────────────────────────────
    plot_training_curves(history, config.EVAL_OUT_DIR)

    # ── 8. Display Final Summary & Parameter Count at the End ────────
    print("\n" + "=" * 70)
    print("      ABLATION STUDY: 60-EPOCH TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Weights saved to         : {config.MODEL_WEIGHTS_PATH}")
    print(f"  Training curves saved to : {config.EVAL_OUT_DIR}")
    
    # Required: Calculate and display parameter count of the entire architecture at the end
    print("\n[Final Architecture Verification] Displaying final parameter counts:")
    display_parameter_summary(model, input_dim=input_dim)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
