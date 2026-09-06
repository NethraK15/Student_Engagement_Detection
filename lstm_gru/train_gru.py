import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import config_gru
from data_loader_gru import get_sequence_dataloaders
from models_gru import EngagementGRUClassifier

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=config_gru.MAX_GRAD_NORM):
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

        # Apply Gradient Norm Clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * sequences.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
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

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    print("==================================================")
    print("    REFINED GRU: TRAINING LIGHTWEIGHT MODEL      ")
    print("==================================================")

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load sequence dataloaders
    try:
        train_loader, val_loader, _, _ = get_sequence_dataloaders(
            sequence_fused_dir=config_gru.SEQUENCE_FUSED_DIR,
            batch_size=config_gru.BATCH_SIZE,
            random_state=config_gru.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Error] Failed to load sequence data: {e}")
        print("Please check if you ran fuse_sequence_features.py first.")
        sys.exit(1)

    # 3. Detect input feature dimension
    seq_files = [f for f in os.listdir(config_gru.SEQUENCE_FUSED_DIR) if f.endswith(".pt")]
    if not seq_files:
        print("[Error] Sequence fused features folder is empty.")
        sys.exit(1)
        
    sample_path = os.path.join(config_gru.SEQUENCE_FUSED_DIR, seq_files[0])
    sample = torch.load(sample_path, map_location="cpu")
    input_dim = sample["sequence"].shape[1]
    
    print(f"Auto-detected Sequence Feature Dimension: {input_dim}")
    print(f"Linear Bottleneck Projection:             {input_dim} -> {config_gru.PROJECTION_DIM}")
    print(f"Lightweight BiGRU Hidden Size:            {config_gru.HIDDEN_DIM}")
    print(f"Max Gradient Norm Clipping:               {config_gru.MAX_GRAD_NORM}")

    # 4. Model initialization
    model = EngagementGRUClassifier(
        input_dim=input_dim,
        proj_dim=config_gru.PROJECTION_DIM,
        hidden_dim=config_gru.HIDDEN_DIM,
        num_classes=len(config_gru.CLASSES)
    ).to(device)

    # Print total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Model Parameters:        {total_params:,} (Reduced from 3.2M!)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config_gru.LEARNING_RATE, weight_decay=config_gru.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 5. Training loop with Early Stopping & Validation Loss Checkpointing
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = getattr(config_gru, 'PATIENCE', 8)
    patience_counter = 0
    epochs = config_gru.EPOCHS
    
    os.makedirs(os.path.dirname(config_gru.MODEL_WEIGHTS_PATH), exist_ok=True)

    print("\nStarting Lightweight GRU training with Early Stopping...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:5.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:5.2f}%)")

        # Save checkpoint based on Validation Loss (prevents saving overfitted late epochs!)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'input_dim': input_dim,
                'proj_dim': config_gru.PROJECTION_DIM,
                'hidden_dim': config_gru.HIDDEN_DIM
            }, config_gru.MODEL_WEIGHTS_PATH)
            print(f"  --> Saved new best GRU model weights (Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered at epoch {epoch+1}. Stopped before overfitting!")
                break

    print("\n==================================================")
    print("             GRU TRAINING COMPLETE                ")
    print("==================================================")
    print(f"Best Validation Loss    : {best_val_loss:.4f}")
    print(f"Validation Acc at Best  : {best_val_acc*100:.2f}%")
    print(f"GRU model weights saved: {config_gru.MODEL_WEIGHTS_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()
