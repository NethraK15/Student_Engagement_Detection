import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import config_lightweight
from data_loader_lightweight import get_sequence_dataloaders
from models_lightweight import get_lightweight_model

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=config_lightweight.MAX_GRAD_NORM):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for sequences, seq_lens, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

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

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, seq_lens, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences, seq_lens)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Lightweight Temporal Model for Student Engagement")
    parser.add_argument("--model", type=str, default=config_lightweight.DEFAULT_MODEL_TYPE,
                        choices=["attention_pool", "tcn", "gru"],
                        help="Lightweight model architecture to train ('attention_pool', 'tcn', or 'gru')")
    parser.add_argument("--epochs", type=int, default=config_lightweight.EPOCHS, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config_lightweight.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=config_lightweight.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    model_type = args.model.lower().strip()
    paths = config_lightweight.get_model_paths(model_type)

    print("==================================================")
    print(f"  LIGHTWEIGHT TEMPORAL MODEL TRAINING ({model_type.upper()}) ")
    print("==================================================")

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load sequence dataloaders
    try:
        train_loader, val_loader, _, _ = get_sequence_dataloaders(
            sequence_fused_dir=config_lightweight.SEQUENCE_FUSED_DIR,
            batch_size=args.batch_size,
            random_state=config_lightweight.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Error] Failed to load sequence data: {e}")
        print("Please check if sequence feature files exist or run preprocess/fuse_sequence_features.py.")
        sys.exit(1)

    # 3. Detect input feature dimension
    seq_files = [f for f in os.listdir(config_lightweight.SEQUENCE_FUSED_DIR) if f.endswith(".pt")]
    if not seq_files:
        print("[Error] Sequence fused features folder is empty.")
        sys.exit(1)
        
    sample_path = os.path.join(config_lightweight.SEQUENCE_FUSED_DIR, seq_files[0])
    sample = torch.load(sample_path, map_location="cpu")
    input_dim = sample["sequence"].shape[1]
    
    print(f"Auto-detected Sequence Feature Dimension: {input_dim}")
    print(f"Selected Model Type:                      {model_type}")
    print(f"Embedding Dimension (d_model):            {config_lightweight.D_MODEL}")

    # 4. Model initialization
    model = get_lightweight_model(
        model_type=model_type,
        input_dim=input_dim,
        d_model=config_lightweight.D_MODEL,
        dropout=config_lightweight.DROPOUT,
        num_classes=len(config_lightweight.CLASSES)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Model Parameters:        {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config_lightweight.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    # 5. Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    epochs = args.epochs
    
    os.makedirs(os.path.dirname(paths["weights"]), exist_ok=True)

    print(f"\nStarting {model_type.upper()} training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:5.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:5.2f}%)")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'input_dim': input_dim,
                'd_model': config_lightweight.D_MODEL
            }, paths["weights"])
            print(f"  --> Saved new best {model_type} model weights (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config_lightweight.PATIENCE:
                print(f"\n[Early Stopping] Triggered after {epoch+1} epochs with no improvement.")
                break

    print("\n==================================================")
    print(f"   {model_type.upper()} TRAINING COMPLETE            ")
    print("==================================================")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model weights saved: {paths['weights']}")
    print("==================================================")

if __name__ == "__main__":
    main()
