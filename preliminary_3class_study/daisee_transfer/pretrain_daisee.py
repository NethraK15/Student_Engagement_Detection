import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import config_daisee
from data_loader_daisee import get_daisee_dataloaders
from models_transfer import get_transfer_model

def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=config_daisee.MAX_GRAD_NORM):
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
    parser = argparse.ArgumentParser(description="Phase 1: Pre-train Temporal Transformer Head on DAiSEE Dataset")
    parser.add_argument("--model", type=str, default=config_daisee.DEFAULT_MODEL_TYPE,
                        choices=["transformer", "attention_pool", "tcn", "gru"],
                        help="Temporal architecture to pretrain ('transformer', 'attention_pool', 'tcn', 'gru')")
    parser.add_argument("--epochs", type=int, default=config_daisee.PRETRAIN_EPOCHS)
    parser.add_argument("--lr", type=float, default=config_daisee.PRETRAIN_LR)
    args = parser.parse_args()

    model_type = args.model.lower().strip()

    print("==================================================")
    print(f"  PHASE 1: PRE-TRAINING TEMPORAL HEAD ON DAISEE    ")
    print(f"               Architecture: {model_type.upper()}   ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DAiSEE DataLoaders
    train_loader, val_loader = get_daisee_dataloaders()

    # Detect input dimension
    daisee_files = [f for f in os.listdir(config_daisee.DAISEE_FUSED_DIR) if f.endswith(".pt")]
    sample_data = torch.load(os.path.join(config_daisee.DAISEE_FUSED_DIR, daisee_files[0]), map_location="cpu")
    seq_tensor = sample_data["sequence"] if "sequence" in sample_data else sample_data["features"]
    input_dim = seq_tensor.shape[1]

    print(f"Detected Feature Dimension: {input_dim}")
    print(f"DAiSEE Pre-training Classes: {config_daisee.DAISEE_NUM_CLASSES}")

    # Initialize model with DAiSEE 4-class head
    model = get_transfer_model(
        model_type=model_type,
        input_dim=input_dim,
        d_model=config_daisee.D_MODEL,
        dropout=config_daisee.DROPOUT,
        num_classes=config_daisee.DAISEE_NUM_CLASSES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters:    {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    os.makedirs(os.path.dirname(config_daisee.PRETRAINED_WEIGHTS_PATH), exist_ok=True)

    print(f"\nStarting Phase 1 DAiSEE Pre-training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:5.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:5.2f}%)")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            torch.save({
                'epoch': epoch + 1,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'input_dim': input_dim,
                'd_model': config_daisee.D_MODEL
            }, config_daisee.PRETRAINED_WEIGHTS_PATH)
            print(f"  --> Saved DAiSEE Pre-trained Checkpoint (Val Acc: {val_acc*100:.2f}%)")

    print("\n==================================================")
    print("  PHASE 1: DAISEE PRE-TRAINING COMPLETE           ")
    print(f"  Best DAiSEE Val Acc: {best_val_acc*100:.2f}%")
    print(f"  Weights saved to:    {config_daisee.PRETRAINED_WEIGHTS_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()
