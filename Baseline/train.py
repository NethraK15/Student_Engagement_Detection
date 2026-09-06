import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import config
from data_loader import get_dataloaders
from models import BaselineClassifier

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single training epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Runs evaluation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    print("==================================================")
    print("         TRAINING BASELINE ENGAGEMENT MODEL       ")
    print("==================================================")

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Get DataLoaders
    try:
        train_loader, val_loader, _, _ = get_dataloaders(
            fused_dir=config.FUSED_FEATURES_DIR,
            batch_size=config.BATCH_SIZE,
            random_state=config.RANDOM_STATE
        )
    except Exception as e:
        print(f"\n[Error] Failed to load data: {e}")
        print("Please check if you successfully completed preprocessing steps 1-4.")
        sys.exit(1)

    # 3. Auto-detect input dimension
    # Look at the first saved fused feature file
    fused_files = [f for f in os.listdir(config.FUSED_FEATURES_DIR) if f.endswith(".pt")]
    if not fused_files:
        print("[Error] Fused features directory is empty. Run Step 4 first.")
        sys.exit(1)
        
    sample_file = os.path.join(config.FUSED_FEATURES_DIR, fused_files[0])
    sample = torch.load(sample_file, map_location="cpu")
    input_dim = sample["feature"].shape[0]
    
    print(f"Auto-detected Input Feature Dimension: {input_dim}")
    print(f"  - EfficientNet sub-dimension: {sample.get('cnn_dim', 'N/A')}")
    print(f"  - OpenFace sub-dimension:     {sample.get('openface_dim', 'N/A')}")

    # 4. Initialize model components
    model = BaselineClassifier(input_dim=input_dim, num_classes=len(config.CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # 5. Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs = config.EPOCHS
    
    os.makedirs(os.path.dirname(config.MODEL_WEIGHTS_PATH), exist_ok=True)

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc*100:5.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:5.2f}%)")

        # Checkpointing: Save weights if validation accuracy increases (or loss decreases on accuracy tie)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'input_dim': input_dim
            }, config.MODEL_WEIGHTS_PATH)
            print(f"  --> Saved new best model weights (Val Acc: {val_acc*100:.2f}%)")

    print("\n==================================================")
    print("                TRAINING COMPLETE                 ")
    print("==================================================")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model weights saved to:   {config.MODEL_WEIGHTS_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()
