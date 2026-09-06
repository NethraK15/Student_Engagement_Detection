import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import config_transformer
from data_loader_transformer import get_sequence_dataloaders
from models_transformer import EngagementTransformerClassifier

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=config_transformer.MAX_GRAD_NORM):
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
    print("  EXPERIMENT 3: STANDARD TEMPORAL TRANSFORMER    ")
    print("==================================================")

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load sequence dataloaders
    try:
        train_loader, val_loader, _, _ = get_sequence_dataloaders(
            sequence_fused_dir=config_transformer.SEQUENCE_FUSED_DIR,
            batch_size=config_transformer.BATCH_SIZE,
            random_state=config_transformer.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Error] Failed to load sequence data: {e}")
        print("Please check if sequence feature files exist.")
        sys.exit(1)

    # 3. Detect input feature dimension
    seq_files = [f for f in os.listdir(config_transformer.SEQUENCE_FUSED_DIR) if f.endswith(".pt")]
    if not seq_files:
        print("[Error] Sequence fused features folder is empty.")
        sys.exit(1)
        
    sample_path = os.path.join(config_transformer.SEQUENCE_FUSED_DIR, seq_files[0])
    sample = torch.load(sample_path, map_location="cpu")
    input_dim = sample["sequence"].shape[1]
    
    print(f"Auto-detected Sequence Feature Dimension: {input_dim}")
    print(f"Transformer d_model Dimension:            {config_transformer.D_MODEL}")
    print(f"Number of Attention Heads (nhead):        {config_transformer.NHEAD}")
    print(f"Number of Transformer Layers:             {config_transformer.NUM_LAYERS}")
    print(f"Feedforward Dimension:                    {config_transformer.DIM_FEEDFORWARD}")

    # 4. Model initialization
    model = EngagementTransformerClassifier(
        input_dim=input_dim,
        d_model=config_transformer.D_MODEL,
        nhead=config_transformer.NHEAD,
        num_layers=config_transformer.NUM_LAYERS,
        dim_feedforward=config_transformer.DIM_FEEDFORWARD,
        dropout=config_transformer.DROPOUT,
        num_classes=len(config_transformer.CLASSES)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Model Parameters:        {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_transformer.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 5. Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs = config_transformer.EPOCHS
    
    os.makedirs(os.path.dirname(config_transformer.MODEL_WEIGHTS_PATH), exist_ok=True)

    print("\nStarting Standard Temporal Transformer training...")
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
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'input_dim': input_dim,
                'd_model': config_transformer.D_MODEL,
                'nhead': config_transformer.NHEAD,
                'num_layers': config_transformer.NUM_LAYERS
            }, config_transformer.MODEL_WEIGHTS_PATH)
            print(f"  --> Saved new best Transformer model weights (Val Acc: {val_acc*100:.2f}%)")

    print("\n==================================================")
    print("      TRANSFORMER TRAINING COMPLETE              ")
    print("==================================================")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Transformer weights saved: {config_transformer.MODEL_WEIGHTS_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()
