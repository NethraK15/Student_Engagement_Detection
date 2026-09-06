import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import config_daisee
from data_loader_daisee import get_target_dataloaders
from models_transfer import get_transfer_model, load_pretrained_daisee_weights

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
    parser = argparse.ArgumentParser(description="Phase 2: Fine-Tune Pre-trained Temporal Transformer Head on Target Dataset")
    parser.add_argument("--model", type=str, default=config_daisee.DEFAULT_MODEL_TYPE,
                        choices=["transformer", "attention_pool", "tcn", "gru"],
                        help="Temporal model architecture ('transformer', 'attention_pool', 'tcn', 'gru')")
    parser.add_argument("--epochs", type=int, default=config_daisee.FINETUNE_EPOCHS)
    parser.add_argument("--lr", type=float, default=config_daisee.FINETUNE_LR)
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze pretrained temporal layers and train top classifier only")
    args = parser.parse_args()

    model_type = args.model.lower().strip()

    print("==================================================")
    print(f"   PHASE 2: FINE-TUNING ON TARGET DATASET (~100 CLIPS)")
    print(f"               Architecture: {model_type.upper()}   ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config_daisee.PRETRAINED_WEIGHTS_PATH):
        print(f"[Warning] Pretrained DAiSEE weights not found at: {config_daisee.PRETRAINED_WEIGHTS_PATH}")
        print("          Please run pretrain_daisee.py first to generate DAiSEE pretrained weights.")
        sys.exit(1)

    # Load Target Dataset DataLoaders
    try:
        train_loader, val_loader, _, _ = get_target_dataloaders(
            target_fused_dir=config_daisee.TARGET_FUSED_DIR,
            batch_size=config_daisee.FINETUNE_BATCH_SIZE,
            random_state=config_daisee.RANDOM_STATE
        )
    except Exception as e:
        print(f"[Error] Failed to load target sequence data: {e}")
        sys.exit(1)

    # Detect feature dimension
    target_files = [f for f in os.listdir(config_daisee.TARGET_FUSED_DIR) if f.endswith(".pt")]
    sample_data = torch.load(os.path.join(config_daisee.TARGET_FUSED_DIR, target_files[0]), map_location="cpu")
    seq_tensor = sample_data["sequence"] if "sequence" in sample_data else sample_data["features"]
    input_dim = seq_tensor.shape[1]

    print(f"Target Feature Dimension: {input_dim}")
    print(f"Target Output Classes:    {config_daisee.NUM_TARGET_CLASSES} ({config_daisee.TARGET_CLASSES})")

    # Initialize model with 3-class target head
    model = get_transfer_model(
        model_type=model_type,
        input_dim=input_dim,
        d_model=config_daisee.D_MODEL,
        dropout=config_daisee.DROPOUT,
        num_classes=config_daisee.NUM_TARGET_CLASSES
    ).to(device)

    # Load pretrained weights from DAiSEE Phase 1
    model = load_pretrained_daisee_weights(
        model=model,
        pretrained_weights_path=config_daisee.PRETRAINED_WEIGHTS_PATH,
        num_target_classes=config_daisee.NUM_TARGET_CLASSES,
        freeze_backbone=args.freeze_backbone
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters for Fine-Tuning: {total_params:,}")

    # Calculate Class Weights to prevent model from collapsing to predicting a single class
    train_labels_list = [l for _, _, l in train_loader.dataset]
    class_counts = torch.bincount(torch.tensor(train_labels_list), minlength=config_daisee.NUM_TARGET_CLASSES).float()
    total_samples = len(train_labels_list)
    class_weights = total_samples / (config_daisee.NUM_TARGET_CLASSES * class_counts.clamp(min=1.0))
    # Normalize class weights and clamp to prevent gradient explosions / NaN loss
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.clamp(class_weights, min=0.5, max=3.0)
    class_weights = class_weights.to(device)
    print(f"Computed Class Weights to prevent single-class collapse: {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(os.path.dirname(config_daisee.FINETUNED_WEIGHTS_PATH), exist_ok=True)

    print(f"\nStarting Phase 2 Target Dataset Fine-Tuning...")
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
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'input_dim': input_dim,
                'd_model': config_daisee.D_MODEL
            }, config_daisee.FINETUNED_WEIGHTS_PATH)
            print(f"  --> Saved new best Fine-Tuned Model Weights (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config_daisee.FINETUNE_PATIENCE:
                print(f"\n[Early Stopping] Triggered after {epoch+1} epochs.")
                break

    print("\n==================================================")
    print("  PHASE 2: FINE-TUNING COMPLETE                   ")
    print(f"  Best Target Val Acc: {best_val_acc*100:.2f}%")
    print(f"  Fine-tuned weights saved to: {config_daisee.FINETUNED_WEIGHTS_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()
