import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config_ablation as config
from dataset_openface import get_ablation_dataloaders
from model_openface import OpenFaceAblationClassifier

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for seqs, lens, labels in loader:
        seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(seqs, lens)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * seqs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / max(total, 1), (correct / max(total, 1)) * 100.0

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for seqs, lens, labels in loader:
            seqs, lens, labels = seqs.to(device), lens.to(device), labels.to(device)
            outputs = model(seqs, lens)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * seqs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return running_loss / max(total, 1), (correct / max(total, 1)) * 100.0

def run_training(dataset_dir=None, epochs=config.EPOCHS, lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE):
    dataset_path = dataset_dir if dataset_dir else config.SEQUENCE_FUSED_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> Running Ablation Training on Device: {device}")
    print(f">>> Dataset Source: {dataset_path}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, test_loader, feat_dim = get_ablation_dataloaders(
        dataset_path, batch_size=batch_size, random_state=config.RANDOM_STATE
    )

    # 2. Initialize Model
    model = OpenFaceAblationClassifier(
        input_dim=feat_dim,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)

    best_val_acc = 0.0
    epochs_no_improve = 0

    print("==================================================")
    print("      TRAINING: OPENFACE-ALONE BINARY ABLATION    ")
    print("==================================================")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "input_dim": feat_dim,
                "hidden_dim": config.HIDDEN_DIM,
                "num_classes": config.NUM_CLASSES
            }, config.CHECKPOINT_PATH)
            star = " 🌟 [NEW BEST]"
        else:
            epochs_no_improve += 1
            star = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%{star}")

        if epochs_no_improve >= config.PATIENCE:
            print(f"\n[Early Stopping] No improvement in validation accuracy for {config.PATIENCE} consecutive epochs.")
            break

    print("\n==================================================")
    print(f"🎉 Training Completed!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"📁 Best Model Checkpoint Saved to: {config.CHECKPOINT_PATH}")
    print("==================================================")

    return best_val_acc

if __name__ == "__main__":
    run_training()
